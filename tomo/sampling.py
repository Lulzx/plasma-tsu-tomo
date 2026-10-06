"""thrml-backed Ising sampling infrastructure shared by all EBM variants.

Conventions
-----------
* Spins are +/-1 (stored as ``bool`` with ``True == +1`` as in thrml).
* ``IsingProblem`` has energy ``E(s) = -sum_i h_i s_i - sum_{i<j} J_ij s_i s_j + offset``
  and the Boltzmann law is ``p(s) ~ exp(-beta E(s))``.
* Block Gibbs update of spin i given the others:
  ``p(s_i = +1) = sigmoid(2 beta (h_i + sum_j J_ij s_j))``.

Two interchangeable backends implement the same colour-block Gibbs sweep:

``'thrml'`` (default / reference)
    ``IsingEBM`` + ``IsingSamplingProgram`` + ``sample_states``.  The thrml program is
    built once at ``beta = 1`` and *re-weighted* by a traced ``beta`` inside ``jit`` (all
    interaction weights are linear in beta), so annealing and tempering never re-build or
    re-compile.  Cost note: thrml pads every node's interaction list to the maximum degree
    of its block and builds the index tables with Python loops, so build time and memory
    scale like ``n_edges`` (Python) and ``n_nodes * max_degree`` (arrays).

``'jax'``
    A pure-JAX sampler with the same colour blocks and conditional.  Local fields are a
    dense per-block matvec (dense-ish graphs) or a padded gather (sparse graphs).

Float precision: everything is float32 (JAX default).  Energies of very large problems are
therefore accurate to ~1e-6 relative, which is irrelevant for sampling.

Public API: ``IsingProblem``, ``bits_to_spins_qubo``, ``greedy_coloring``, ``balance_coloring``,
``check_coloring``, ``IsingSampler``, ``run_ising``, ``anneal_ising``,
``parallel_tempering``, ``ising_energy``, ``geometric_betas``, ``SCHEDULES``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from thrml import Block, SamplingSchedule, sample_states
from thrml.models.discrete_ebm import DiscreteEBMInteraction
from thrml.models.ising import IsingEBM, IsingSamplingProgram, hinton_init

__all__ = [
    "IsingProblem", "bits_to_spins_qubo", "greedy_coloring", "balance_coloring", "check_coloring",
    "IsingSampler", "run_ising", "anneal_ising", "parallel_tempering",
    "ising_energy", "geometric_betas", "SCHEDULES", "ANNEAL", "TEMPERING",
]

# Presets mirroring configs/default.yaml (schedule section).
SCHEDULES = {
    "posterior": SamplingSchedule(n_warmup=2000, n_samples=500, steps_per_sample=10),
    "quick": SamplingSchedule(n_warmup=200, n_samples=100, steps_per_sample=5),
}
ANNEAL = dict(beta_min=0.1, beta_max=50.0, n_betas=30, sweeps_per_beta=20)
TEMPERING = dict(n_replicas=8, beta_min=0.2)


# --------------------------------------------------------------------------------------
# Problem container and conversions
# --------------------------------------------------------------------------------------
@dataclass
class IsingProblem:
    """Ising model ``E(s) = -h.s - sum_{i<j} J_ij s_i s_j + offset`` (+/-1 spins).

    ``rows, cols, vals`` is a COO list of couplings with ``rows < cols`` (each pair once).
    """

    h: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    vals: np.ndarray
    offset: float = 0.0

    def __post_init__(self):
        self.h = np.asarray(self.h, dtype=np.float64).ravel()
        self.rows = np.asarray(self.rows, dtype=np.int64).ravel()
        self.cols = np.asarray(self.cols, dtype=np.int64).ravel()
        self.vals = np.asarray(self.vals, dtype=np.float64).ravel()
        self.offset = float(self.offset)
        if not (len(self.rows) == len(self.cols) == len(self.vals)):
            raise ValueError("rows, cols, vals must have equal length")
        if len(self.rows) and not np.all(self.rows < self.cols):
            raise ValueError("couplings must satisfy rows < cols")
        if len(self.rows) and (self.rows.min() < 0 or self.cols.max() >= len(self.h)):
            raise ValueError("coupling index out of range")

    @property
    def n(self) -> int:
        return len(self.h)

    @property
    def n_edges(self) -> int:
        return len(self.rows)

    @property
    def edges(self) -> np.ndarray:
        """(E, 2) int array of coupled pairs (i < j)."""
        return np.stack([self.rows, self.cols], axis=1)

    @property
    def degrees(self) -> np.ndarray:
        return np.bincount(np.concatenate([self.rows, self.cols]), minlength=self.n)

    @classmethod
    def from_dense(cls, h, Jd, offset: float = 0.0) -> "IsingProblem":
        """Build from a dense symmetric coupling matrix (only the strict upper triangle is used)."""
        Jd = np.asarray(Jd, dtype=np.float64)
        r, c = np.nonzero(np.triu(Jd, 1))
        return cls(h, r, c, Jd[r, c], offset)

    def J_sparse(self) -> sp.csr_matrix:
        """Symmetric (n, n) CSR coupling matrix."""
        U = sp.coo_matrix((self.vals, (self.rows, self.cols)), shape=(self.n, self.n))
        return (U + U.T).tocsr()

    def energy(self, spins) -> np.ndarray:
        """float64 NumPy energy of spins (..., n) given as bool or +/-1."""
        s = np.asarray(spins)
        s = np.where(s, 1.0, -1.0) if s.dtype == bool else s.astype(np.float64)
        e = -(s @ self.h)
        if self.n_edges:
            e = e - np.sum(s[..., self.rows] * s[..., self.cols] * self.vals, axis=-1)
        return e + self.offset


def bits_to_spins_qubo(Qb, cb, const: float = 0.0) -> IsingProblem:
    """Convert a bit energy ``E(u) = u^T Qb u + cb.u + const`` (u in {0,1}) to +/-1 spins.

    Uses ``u = (s + 1) / 2``.  ``Qb`` may be dense or scipy sparse; it is symmetrised as
    ``(Qb + Qb^T) / 2`` (so a non-symmetric input gives the same energy).  Diagonal terms
    ``Q_ii u_i^2 = Q_ii u_i`` are folded into the linear term first.  Derivation, with
    ``c'_i = c_i + Q_ii``::

        J_ij   = -Q_ij / 2                       (i < j, Q_ij the symmetric entry)
        h_i    = -(c'_i / 2 + 1/2 sum_{j != i} Q_ij)
        offset =  const + sum_i c'_i / 2 + 1/2 sum_{i<j} Q_ij
    """
    if sp.issparse(Qb):
        Q = sp.csr_matrix(Qb, dtype=np.float64)
    else:
        Q = sp.csr_matrix(np.asarray(Qb, dtype=np.float64))
    n = Q.shape[0]
    if Q.shape != (n, n):
        raise ValueError("Qb must be square")
    Q = ((Q + Q.T) * 0.5).tocsr()
    cb = np.asarray(cb, dtype=np.float64).ravel()
    if cb.shape != (n,):
        raise ValueError("cb has wrong length")
    d = Q.diagonal()
    cp = cb + d
    Qo = (Q - sp.diags(d)).tocsr()  # off-diagonal, symmetric
    Qo.eliminate_zeros()
    rowsum = np.asarray(Qo.sum(axis=1)).ravel()
    U = sp.triu(Qo, k=1).tocoo()
    h = -(0.5 * cp + 0.5 * rowsum)
    off = const + 0.5 * cp.sum() + 0.5 * U.data.sum()
    return IsingProblem(h, U.row, U.col, -0.5 * U.data, off)


# --------------------------------------------------------------------------------------
# Graph colouring
# --------------------------------------------------------------------------------------
def _csr_adj(n_nodes: int, edges: np.ndarray) -> sp.csr_matrix:
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    A = sp.coo_matrix((np.ones(len(e), dtype=np.int8), (e[:, 0], e[:, 1])), shape=(n_nodes, n_nodes))
    A = (A + A.T).tocsr()
    A.data[:] = 1
    return A


def greedy_coloring(n_nodes: int, edges) -> np.ndarray:
    """Greedy colouring, largest-degree-first. Returns (n_nodes,) int colours 0..k-1.

    No two nodes joined by an edge share a colour (self-loops are rejected).
    ``edges`` is an (E, 2) integer array (either orientation, duplicates allowed).
    """
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if len(e) and np.any(e[:, 0] == e[:, 1]):
        raise ValueError("self-loop in edge list")
    A = _csr_adj(n_nodes, e)
    deg = np.diff(A.indptr)
    order = np.argsort(-deg, kind="stable")
    colors = -np.ones(n_nodes, dtype=np.int64)
    for v in order:
        nb = colors[A.indices[A.indptr[v]:A.indptr[v + 1]]]
        used = np.zeros(deg[v] + 2, dtype=bool)
        nb = nb[(nb >= 0) & (nb < len(used))]
        used[nb] = True
        colors[v] = np.argmin(used)  # first False
    return colors


def balance_coloring(n_nodes: int, edges, colors, max_passes: int = 20) -> np.ndarray:
    """Equalise colour-class sizes without adding colours, keeping the colouring proper.

    Greedy colouring leaves a few large classes and many small ones. A padded block sweep (the JAX
    backends) costs n_blocks * max_block_size rows, so on TCV the imbalance costs 2.5x. Each pass moves
    vertices out of oversized classes into the smallest class that none of their neighbours uses.
    """
    colors = np.asarray(colors, dtype=np.int64).copy()
    A = _csr_adj(n_nodes, np.asarray(edges, dtype=np.int64).reshape(-1, 2))
    k = int(colors.max()) + 1 if n_nodes else 0
    target = int(np.ceil(n_nodes / max(k, 1)))
    for _ in range(max_passes):
        sizes = np.bincount(colors, minlength=k)
        moved = 0
        for v in np.argsort(-sizes[colors], kind="stable"):
            if sizes[colors[v]] <= target:
                continue
            used = np.zeros(k, dtype=bool)
            used[colors[A.indices[A.indptr[v]:A.indptr[v + 1]]]] = True
            cand = np.flatnonzero(~used & (sizes < sizes[colors[v]] - 1))
            if len(cand):
                c = cand[np.argmin(sizes[cand])]
                sizes[colors[v]] -= 1
                sizes[c] += 1
                colors[v] = c
                moved += 1
        if not moved:
            break
    return colors


def check_coloring(n_nodes: int, edges, colors) -> bool:
    """True iff ``colors`` is a proper colouring of the graph."""
    colors = np.asarray(colors)
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    return colors.shape == (n_nodes,) and bool(np.all(colors[e[:, 0]] != colors[e[:, 1]]))


# --------------------------------------------------------------------------------------
# Energy
# --------------------------------------------------------------------------------------
@partial(jax.jit, static_argnames=("chunk",))
def _energy_core(h, rows, cols, vals, offset, s, chunk):
    """Energy of a batch of +/-1 float spins s (B, n) -> (B,)."""
    def one(si):
        return -jnp.dot(h, si) - jnp.dot(vals, si[rows] * si[cols]) + offset
    return jax.lax.map(one, s, batch_size=chunk)


def _energy_args(prob: IsingProblem):
    return (jnp.asarray(prob.h, jnp.float32), jnp.asarray(prob.rows, jnp.int32),
            jnp.asarray(prob.cols, jnp.int32), jnp.asarray(prob.vals, jnp.float32),
            jnp.float32(prob.offset))


def ising_energy(prob: IsingProblem, spins, _args=None) -> jnp.ndarray:
    """Efficient JAX energy of spins (..., n) (bool or +/-1); returns (...,) float32."""
    spins = jnp.asarray(spins)
    lead = spins.shape[:-1]
    s = jnp.where(spins, 1.0, -1.0).astype(jnp.float32) if spins.dtype == jnp.bool_ else spins.astype(jnp.float32)
    s = s.reshape(-1, prob.n)
    args = _energy_args(prob) if _args is None else _args
    chunk = int(max(1, min(256, 2e7 // max(prob.n_edges, 1))))
    chunk = min(chunk, s.shape[0]) if s.shape[0] else 1
    return _energy_core(*args, s, chunk).reshape(lead)


def geometric_betas(beta_min: float, beta_max: float, n: int) -> np.ndarray:
    """Geometric inverse-temperature ladder (n,) from beta_min to beta_max."""
    return np.geomspace(beta_min, beta_max, n)


# --------------------------------------------------------------------------------------
# Sampler (backend abstraction)
# --------------------------------------------------------------------------------------
def _as_schedule(schedule) -> SamplingSchedule:
    if isinstance(schedule, SamplingSchedule):
        return schedule
    if isinstance(schedule, dict):
        return SamplingSchedule(int(schedule["n_warmup"]), int(schedule["n_samples"]),
                                int(schedule["steps_per_sample"]))
    n_w, n_s, n_t = schedule
    return SamplingSchedule(int(n_w), int(n_s), int(n_t))


def _scale_program(program, beta):
    """Multiply every float interaction weight of a thrml program by (traced) beta."""
    def scale(x):
        if isinstance(x, jax.Array) and jnp.issubdtype(x.dtype, jnp.floating):
            return x * beta
        return x
    return eqx.tree_at(lambda p: p.per_block_interactions, program,
                       jax.tree.map(scale, program.per_block_interactions))


class IsingSampler:
    """Compiled colour-block Gibbs sampler for one ``IsingProblem`` and colouring.

    Build once, then call :meth:`run` many times (different keys / betas / schedules);
    compiled executables are cached per static configuration.  ``beta`` is traced, so
    changing it never recompiles.
    """

    def __init__(self, prob: IsingProblem, colors, backend: str = "thrml", dense: bool | None = None,
                 threads: int | None = None):
        if backend not in ("thrml", "jax"):
            raise ValueError("backend must be 'thrml' or 'jax'")
        t0 = time.perf_counter()
        self.prob, self.backend = prob, backend
        n = prob.n
        colors = np.asarray(colors)
        if not check_coloring(n, prob.edges, colors):
            raise ValueError("colors is not a proper colouring of the coupling graph")
        # compact colour labels -> block ids (drop empty colours)
        _, cid = np.unique(colors, return_inverse=True)
        self.blocks_idx = [np.flatnonzero(cid == c) for c in range(cid.max() + 1)]
        self.n_blocks = len(self.blocks_idx)
        self.max_degree = int(prob.degrees.max()) if n else 0
        self._energy_args = _energy_args(prob)
        self._cache: dict = {}
        self.threads = threads  # jax backend: chain groups run concurrently (None -> auto)
        if backend == "thrml":
            self._build_thrml()
        else:
            self._build_jax(dense)
        self.build_time = time.perf_counter() - t0

    # ---- thrml ----
    def _build_thrml(self):
        from thrml import SpinNode
        p = self.prob
        nodes = [SpinNode() for _ in range(p.n)]
        edges = list(zip([nodes[i] for i in p.rows], [nodes[j] for j in p.cols]))
        ebm = IsingEBM(nodes, edges, jnp.asarray(p.h, jnp.float32), jnp.asarray(p.vals, jnp.float32),
                       jnp.float32(1.0))
        self.nodes, self.ebm = nodes, ebm
        self.free_blocks = [Block([nodes[i] for i in b]) for b in self.blocks_idx]
        self.program = IsingSamplingProgram(ebm, self.free_blocks, [])
        self._data = self.program
        self._idx = [jnp.asarray(b, jnp.int32) for b in self.blocks_idx]

    def _sample_thrml(self, data, key, state, beta, sched: SamplingSchedule):
        program = _scale_program(data, beta)
        init = [state[i] for i in self._idx]
        out = sample_states(key, program, sched, init, [], self.free_blocks)
        res = jnp.zeros((sched.n_samples, self.prob.n), jnp.bool_)
        for i, o in zip(self._idx, out):
            res = res.at[:, i].set(o)
        return res

    # ---- pure jax ----
    # Two code paths with identical conditionals and colour-order updates:
    #  * ``_sample_jax``  : single chain (vmappable); used by anneal / tempering via ``sample_single``.
    #  * ``_sample_jax_batched`` : all chains at once with state laid out as (n, C) in colour-block
    #    order (so each block is a static contiguous row slice).  Dense blocks use one BLAS matmul
    #    ``J_block (nb, n) @ S (n, C)``; sparse blocks use degree-bucketed ELL gathers of whole rows
    #    ``S[nbr] (nb, d, C)`` (a row of C chains is contiguous -> cache-friendly gather).  Used by ``run``.
    @property
    def _data(self):
        """Per-chain block data (lazy; thrml: the program)."""
        if self.backend == "thrml":
            return self._thrml_data
        if self._legacy is None:
            self._legacy = self._build_jax_legacy()
        return self._legacy

    @_data.setter
    def _data(self, v):
        self._thrml_data = v

    def _build_jax_legacy(self):
        p = self.prob
        n = p.n
        J = p.J_sparse()
        h32 = np.asarray(p.h, np.float32)
        blocks = []
        for b in self.blocks_idx:
            Jb = J[b]  # (nb, n) csr
            if self._dense:
                blocks.append(dict(idx=jnp.asarray(b, jnp.int32), h=jnp.asarray(h32[b]),
                                   Jd=jnp.asarray(Jb.toarray(), jnp.float32)))
            else:
                deg = np.diff(Jb.indptr)
                d = max(int(deg.max()) if len(deg) else 0, 1)
                nbr = np.zeros((len(b), d), np.int32)
                val = np.zeros((len(b), d), np.float32)
                pos = np.arange(Jb.nnz) - np.repeat(Jb.indptr[:-1], deg)
                r = np.repeat(np.arange(len(b)), deg)
                nbr[r, pos] = Jb.indices
                val[r, pos] = Jb.data
                blocks.append(dict(idx=jnp.asarray(b, jnp.int32), h=jnp.asarray(h32[b]),
                                   nbr=jnp.asarray(nbr), val=jnp.asarray(val)))
        return blocks

    _bucket_min = 32   # ELL: colour blocks with fewer nodes are not split by degree

    @staticmethod
    def _degree_buckets(deg, ratio=0.6, max_buckets=8):
        """Split nodes (sorted by degree, descending) into groups whose padding waste is bounded."""
        order = np.argsort(-deg, kind="stable")
        groups, cur, top = [], [], None
        for i in order:
            if top is None:
                top = max(deg[i], 1)
            if deg[i] < ratio * top and len(groups) < max_buckets - 1:
                groups.append(cur)
                cur, top = [], max(deg[i], 1)
            cur.append(i)
        if cur:
            groups.append(cur)
        return [np.asarray(g, np.int64) for g in groups]

    def _build_jax(self, dense):
        p = self.prob
        n = p.n
        J = p.J_sparse()
        if dense is None:
            dense = (J.nnz > 0.05 * n * n) and (n * n * 4 <= 2e9)
        self._dense = dense
        self._legacy = None
        h32 = np.asarray(p.h, np.float32)
        deg_all = np.diff(J.indptr)
        groups = []                                   # node-id arrays, in sweep order
        for b in self.blocks_idx:
            if dense or len(b) < self._bucket_min:
                groups.append(b)
            else:
                groups += [b[g] for g in self._degree_buckets(deg_all[b], max_buckets=min(8, max(1, len(b) // (self._bucket_min // 2))))]
        perm = np.concatenate(groups) if groups else np.zeros(0, np.int64)
        inv = np.empty(n, np.int64)
        inv[perm] = np.arange(n)
        Jp = J[perm][:, perm].tocsr()
        self._perm, self._inv = perm, inv
        self._bounds, data = [], []
        a = 0
        for g in groups:
            bnd = a + len(g)
            Jb = Jp[a:bnd]
            hb = jnp.asarray(h32[g])[:, None]
            if dense:
                data.append(dict(h=hb, Jd=jnp.asarray(Jb.toarray(), jnp.float32)))
            else:
                deg = np.diff(Jb.indptr)
                d = max(int(deg.max()) if len(deg) else 0, 1)
                nbr = np.zeros((len(g), d), np.int32)
                val = np.zeros((len(g), d), np.float32)
                pos = np.arange(Jb.nnz) - np.repeat(Jb.indptr[:-1], deg)
                r = np.repeat(np.arange(len(g)), deg)
                nbr[r, pos] = Jb.indices
                val[r, pos] = Jb.data
                data.append(dict(h=hb, nbr=jnp.asarray(nbr), val=jnp.asarray(val)[:, :, None]))
            self._bounds.append((a, bnd))
            a = bnd
        self._bdata = data

    def _sweep_jax(self, data, key, s, beta):
        """One sweep over all colour blocks of a single chain. s: (n,) float +/-1."""
        blocks = data
        keys = jax.random.split(key, len(blocks))
        for blk, k in zip(blocks, keys):
            if "Jd" in blk:
                field = blk["h"] + blk["Jd"] @ s
            else:
                field = blk["h"] + jnp.sum(blk["val"] * s[blk["nbr"]], axis=-1)
            pup = jax.nn.sigmoid(2.0 * beta * field)
            new = jnp.where(jax.random.uniform(k, pup.shape) < pup, 1.0, -1.0)
            s = s.at[blk["idx"]].set(new)
        return s

    def _sample_jax(self, data, key, state, beta, sched: SamplingSchedule):
        s0 = jnp.where(state, 1.0, -1.0).astype(jnp.float32)
        kw, ks = jax.random.split(key)

        def sweeps(s, k, m):
            if m == 0:
                return s
            return jax.lax.scan(lambda c, kk: (self._sweep_jax(data, kk, c, beta), None), s,
                                jax.random.split(k, m))[0]

        s = sweeps(s0, kw, sched.n_warmup)
        if sched.n_samples <= 1:
            return (s > 0)[None]

        def body(c, k):
            c = sweeps(c, k, sched.steps_per_sample)
            return c, c > 0
        _, rest = jax.lax.scan(body, s, jax.random.split(ks, sched.n_samples - 1))
        return jnp.concatenate([(s > 0)[None], rest], axis=0)

    def _sweep_batched(self, data, key, S, beta):
        """One sweep for all chains. S: (n, C) float +/-1 in colour-block order."""
        U = jax.random.uniform(key, S.shape)
        for blk, (a, b) in zip(data, self._bounds):
            if "Jd" in blk:
                field = blk["h"] + blk["Jd"] @ S
            else:
                field = blk["h"] + jnp.einsum("bdo,bdc->bc", blk["val"], S[blk["nbr"]])
            pup = jax.nn.sigmoid(2.0 * beta * field)
            S = S.at[a:b].set(jnp.where(U[a:b] < pup, 1.0, -1.0))
        return S

    def _sample_jax_batched(self, data, keys, state, beta, sched: SamplingSchedule):
        """All chains at once: keys (C,), state (C, n) bool -> (C, n_samples, n) bool."""
        kd = jax.random.key_data(keys) if jnp.issubdtype(keys.dtype, jax.dtypes.prng_key) else keys
        key = jax.random.wrap_key_data(jnp.bitwise_xor.reduce(kd, axis=0).astype(jnp.uint32))
        key = jax.random.fold_in(key, kd.shape[0])
        perm, inv = jnp.asarray(self._perm), jnp.asarray(self._inv)
        S0 = jnp.where(state, 1.0, -1.0).astype(jnp.float32).T[perm]
        kw, ks = jax.random.split(key)

        def sweeps(S, k, m):
            if m == 0:
                return S
            return jax.lax.scan(lambda c, kk: (self._sweep_batched(data, kk, c, beta), None), S,
                                jax.random.split(k, m))[0]

        S = sweeps(S0, kw, sched.n_warmup)
        if sched.n_samples <= 1:
            out = (S > 0)[None]
        else:
            def body(c, k):
                c = sweeps(c, k, sched.steps_per_sample)
                return c, c > 0
            _, rest = jax.lax.scan(body, S, jax.random.split(ks, sched.n_samples - 1))
            out = jnp.concatenate([(S > 0)[None], rest], axis=0)    # (S, n, C)
        return jnp.transpose(out[:, inv, :], (2, 0, 1))

    # ---- common ----
    def sample_single(self, key, state, beta, sched: SamplingSchedule):
        """Traceable single-chain sampler: state (n,) bool -> (n_samples, n) bool."""
        f = self._sample_thrml if self.backend == "thrml" else self._sample_jax
        return f(self._data, key, state, beta, sched)

    def _compiled(self, name, build, *args):
        """Cache of AOT-compiled executables keyed by (name, arg shapes)."""
        sig = (name,) + tuple((tuple(a.shape), str(a.dtype)) for a in jax.tree.leaves(args[1:]))
        if sig not in self._cache:
            t0 = time.perf_counter()
            dyn, stat = eqx.partition(args[0], eqx.is_array)  # non-array leaves stay static
            fn = jax.jit(lambda d, *rest: build(eqx.combine(d, stat), *rest))
            exe = fn.lower(dyn, *args[1:]).compile()
            self._cache[sig] = (lambda data, *rest, _e=exe: _e(eqx.filter(data, eqx.is_array), *rest),
                                time.perf_counter() - t0)
        return self._cache[sig]

    def init_state(self, init, n_chains: int, key, beta: float = 1.0) -> jnp.ndarray:
        """Initial (C, n) bool state: array, ``'hinton'`` (p=sigmoid(beta h)) or ``'random'``."""
        n = self.prob.n
        if init is None:
            init = "hinton"
        if isinstance(init, str):
            if init == "random":
                return jax.random.bernoulli(key, 0.5, (n_chains, n))
            if init == "hinton":
                p = jax.nn.sigmoid(beta * jnp.asarray(self.prob.h, jnp.float32))
                return jax.random.bernoulli(key, p, (n_chains, n))
            raise ValueError(f"unknown init {init!r}")
        init = jnp.asarray(init)
        if init.dtype != jnp.bool_:
            init = init > 0
        if init.ndim == 1:
            init = jnp.broadcast_to(init, (n_chains, n))
        if init.shape != (n_chains, n):
            raise ValueError(f"init must have shape ({n_chains}, {n})")
        return init

    def run(self, keys, init, beta, sched: SamplingSchedule, compile_only: bool = False):
        """Run vmapped chains. keys (C,), init (C, n) bool -> samples (C, S, n) bool (device), compile_s."""
        beta = jnp.asarray(beta, jnp.float32)

        if self.backend == "jax":
            C = int(keys.shape[0])
            D = self._n_groups(C)
            Cg = C // D

            def f(data, keys, init, beta):
                return self._sample_jax_batched(data, keys, init, beta, sched)
            data = self._bdata
            name = ("run", sched.n_warmup, sched.n_samples, sched.steps_per_sample)
            exe, ct = self._compiled(name, f, data, keys[:Cg], init[:Cg], beta)
            if compile_only:
                return None, ct
            if D == 1:
                return exe(data, keys, init, beta), ct
            from concurrent.futures import ThreadPoolExecutor

            def go(g):
                sl = slice(g * Cg, (g + 1) * Cg)
                return exe(data, keys[sl], init[sl], beta).block_until_ready()
            with ThreadPoolExecutor(D) as ex:   # XLA releases the GIL while executing
                parts = list(ex.map(go, range(D)))
            return jnp.concatenate(parts, axis=0), ct

        def f(data, keys, init, beta):
            return jax.vmap(lambda k, s: self._single(data, k, s, beta, sched))(keys, init)
        exe, ct = self._compiled(("run", sched.n_warmup, sched.n_samples, sched.steps_per_sample),
                                 f, self._data, keys, init, beta)
        if compile_only:
            return None, ct
        return exe(self._data, keys, init, beta), ct

    def _n_groups(self, C: int) -> int:
        """Number of equally sized chain groups executed concurrently (a divisor of C)."""
        import os
        target = self.threads if self.threads is not None else min(C, max(1, (os.cpu_count() or 1) // 2))
        target = max(1, min(int(target), C))
        return max(d for d in range(1, target + 1) if C % d == 0)

    def _single(self, data, key, state, beta, sched):
        f = self._sample_thrml if self.backend == "thrml" else self._sample_jax
        return f(data, key, state, beta, sched)


# --------------------------------------------------------------------------------------
# Public drivers
# --------------------------------------------------------------------------------------
def _get_sampler(prob, colors, backend, sampler):
    return sampler if sampler is not None else IsingSampler(prob, colors, backend=backend)


def _energy_trace(sm: IsingSampler, samples, max_evals: int = 200_000_000):
    """Energy of every sample (C,S,n)->(C,S); strided over S if too expensive."""
    C, S, n = samples.shape
    stride = int(max(1, np.ceil(C * S * max(sm.prob.n_edges, n) / max_evals)))
    e = ising_energy(sm.prob, samples[:, ::stride], _args=sm._energy_args)
    return np.asarray(e), stride


def run_ising(prob: IsingProblem, colors, schedule, n_chains: int, key, beta: float = 1.0,
              init=None, backend: str = "thrml", sampler: IsingSampler | None = None) -> dict:
    """Block-Gibbs sample ``p(s) ~ exp(-beta E(s))`` with ``n_chains`` vmapped chains.

    Parameters
    ----------
    schedule : ``thrml.SamplingSchedule`` | dict | (n_warmup, n_samples, steps_per_sample)
    init : None/'hinton' | 'random' | bool array (C, n) (or (n,), broadcast)
    backend : 'thrml' (reference) or 'jax' (equivalent pure-JAX sampler)
    sampler : optional pre-built :class:`IsingSampler` to reuse (skips program build).

    Returns dict: ``samples`` (C,S,n) bool; ``energy_trace`` (C,S') float32 (true energy E,
    not beta*E; strided by ``energy_stride`` only for huge problems); ``time`` wall seconds
    of the sampling call excluding compilation; ``compile_time``; ``build_time``;
    ``n_blocks``; ``max_degree``; ``backend``.
    """
    sched = _as_schedule(schedule)
    sm = _get_sampler(prob, colors, backend, sampler)
    k_init, k_run = jax.random.split(key)
    state = sm.init_state(init, n_chains, k_init, beta)
    keys = jax.random.split(k_run, n_chains)
    sm.run(keys, state, beta, sched, compile_only=True)  # AOT compile (cached) outside the timer
    t0 = time.perf_counter()
    samples, ct = sm.run(keys, state, beta, sched)
    samples.block_until_ready()
    wall = time.perf_counter() - t0
    e, stride = _energy_trace(sm, samples)
    return dict(samples=np.asarray(samples), energy_trace=e, energy_stride=stride, time=wall,
                compile_time=ct, build_time=sm.build_time, n_blocks=sm.n_blocks,
                max_degree=sm.max_degree, backend=sm.backend)


def anneal_ising(prob: IsingProblem, colors, betas, sweeps_per_beta: int, n_chains: int, key,
                 init=None, backend: str = "thrml", sampler: IsingSampler | None = None) -> dict:
    """Simulated annealing over an inverse-temperature ladder ``betas`` (e.g. :func:`geometric_betas`).

    At each beta every chain does ``sweeps_per_beta`` block-Gibbs sweeps, carrying its state to
    the next beta.  Beta is a traced scalar inside one ``jit``/``scan``: no re-building or
    re-compilation per step.

    Returns ``final`` (C,n) bool, ``energy_trace`` (C, n_betas) (energy at the end of each
    beta step), ``best`` (C,n) bool / ``best_energy`` (C,) (lowest energy seen at a step end),
    ``betas``, ``time``, ``compile_time``, ``n_blocks``, ``max_degree``.
    """
    betas = jnp.asarray(betas, jnp.float32)
    sm = _get_sampler(prob, colors, backend, sampler)
    sched = SamplingSchedule(int(sweeps_per_beta), 1, 1)
    k_init, k_run = jax.random.split(key)
    state = sm.init_state(init, n_chains, k_init, float(betas[0]))
    keys = jax.random.split(k_run, n_chains)
    nb = len(betas)

    def f(data, keys, state, betas):
        def chain(k, s0):
            ks = jax.random.split(k, nb)
            e0 = ising_energy_core(s0)

            def step(c, inp):
                s, best_s, best_e = c
                kk, b = inp
                s = sm._single(data, kk, s, b, sched)[0]
                e = ising_energy_core(s)
                better = e < best_e
                best_s = jnp.where(better, s, best_s)
                best_e = jnp.where(better, e, best_e)
                return (s, best_s, best_e), e

            (s, bs, be), es = jax.lax.scan(step, (s0, s0, e0), (ks, betas))
            return s, bs, be, es
        return jax.vmap(chain)(keys, state)

    h, rows, cols, vals, off = sm._energy_args

    def ising_energy_core(s):
        x = jnp.where(s, 1.0, -1.0)
        return -jnp.dot(h, x) - jnp.dot(vals, x[rows] * x[cols]) + off

    exe, ct = sm._compiled(("anneal", int(sweeps_per_beta), nb), f, sm._data, keys, state, betas)
    t0 = time.perf_counter()
    final, best, best_e, tr = exe(sm._data, keys, state, betas)
    final.block_until_ready()
    wall = time.perf_counter() - t0
    return dict(final=np.asarray(final), energy_trace=np.asarray(tr), best=np.asarray(best),
                best_energy=np.asarray(best_e), betas=np.asarray(betas), time=wall, compile_time=ct,
                build_time=sm.build_time, n_blocks=sm.n_blocks, max_degree=sm.max_degree,
                backend=sm.backend)


def parallel_tempering(prob: IsingProblem, colors, betas, n_sweeps: int, swap_every: int, key,
                       n_chains: int = 1, init=None, backend: str = "thrml",
                       sampler: IsingSampler | None = None) -> dict:
    """Replica-exchange MCMC.

    ``betas`` (R,) are the replica inverse temperatures (sorted ascending internally).  The
    *target* replica, whose states are returned in ``samples``, is the one with beta closest to 1
    (the physical posterior if 1 is on the ladder).  Each round runs ``swap_every`` sweeps per replica, then attempts swaps of
    adjacent (in beta-sorted order) replicas with acceptance
    ``min(1, exp((b_i - b_j)(E_i - E_j)))``, alternating even/odd pairs between rounds.
    ``n_sweeps // swap_every`` rounds are executed.  State slot r always sits at ``betas[r]``;
    swaps exchange states.

    Returns ``samples`` (C, rounds, n) bool (target replica after each round),
    ``energy_trace`` (C, rounds, R) (energy of each slot), ``swap_rate`` (R-1,),
    ``final`` (C, R, n), ``betas`` (sorted ascending), ``target`` index, ``time``, ...
    """
    betas = np.sort(np.asarray(betas, np.float64))
    R = len(betas)
    target = int(np.argmin(np.abs(betas - 1.0)))
    n_rounds = int(n_sweeps) // int(swap_every)
    sm = _get_sampler(prob, colors, backend, sampler)
    sched = SamplingSchedule(int(swap_every), 1, 1)
    bj = jnp.asarray(betas, jnp.float32)
    k_init, k_run = jax.random.split(key)
    # replica init: each slot hinton-initialised at its own beta
    st = jnp.stack([sm.init_state(init, n_chains, jax.random.fold_in(k_init, r), float(betas[r]))
                    for r in range(R)], axis=1)  # (C,R,n)
    keys = jax.random.split(k_run, n_chains)
    h, rows, cols, vals, off = sm._energy_args

    def energy1(s):
        x = jnp.where(s, 1.0, -1.0)
        return -jnp.dot(h, x) - jnp.dot(vals, x[rows] * x[cols]) + off

    def f(data, keys, st, bj):
        def chain(k, S):  # S (R, n)
            def rnd(c, inp):
                S, acc, att = c
                kk, parity = inp
                k1, k2 = jax.random.split(kk)
                ks = jax.random.split(k1, R)
                S = jax.vmap(lambda ki, si, bi: sm._single(data, ki, si, bi, sched)[0])(ks, S, bj)
                E = jax.vmap(energy1)(S)
                # swap proposals between slots (i, i+1), i = parity, parity+2, ...
                i = jnp.arange(R - 1)
                active = (i % 2) == parity
                logr = (bj[i] - bj[i + 1]) * (E[i] - E[i + 1])
                u = jnp.log(jax.random.uniform(k2, (R - 1,)))
                accept = active & (u < logr)
                perm = jnp.arange(R)
                # build permutation: accepted pairs exchange; pairs are disjoint within a parity
                perm = perm.at[i].set(jnp.where(accept, i + 1, perm[i]))
                perm = perm.at[i + 1].set(jnp.where(accept, i, perm[i + 1]))
                S_new = S[perm]
                E_rec = E  # energies before the swap (slot energies at sampling time)
                return (S_new, acc + accept, att + active), (S_new[target], E_rec)
            parities = jnp.arange(n_rounds) % 2
            ks = jax.random.split(k, n_rounds)
            (Sf, acc, att), (tr_s, tr_e) = jax.lax.scan(
                rnd, (S, jnp.zeros(R - 1, jnp.int32), jnp.zeros(R - 1, jnp.int32)), (ks, parities))
            return Sf, tr_s, tr_e, acc, att
        return jax.vmap(chain)(keys, st)

    exe, ct = sm._compiled(("pt", int(swap_every), n_rounds, R), f, sm._data, keys, st, bj)
    t0 = time.perf_counter()
    Sf, tr_s, tr_e, acc, att = exe(sm._data, keys, st, bj)
    Sf.block_until_ready()
    wall = time.perf_counter() - t0
    acc, att = np.asarray(acc).sum(0), np.asarray(att).sum(0)
    return dict(samples=np.asarray(tr_s), energy_trace=np.asarray(tr_e),
                swap_rate=acc / np.maximum(att, 1), final=np.asarray(Sf), betas=betas, target=target,
                n_rounds=n_rounds, time=wall, compile_time=ct, build_time=sm.build_time,
                n_blocks=sm.n_blocks, max_degree=sm.max_degree, backend=sm.backend)
