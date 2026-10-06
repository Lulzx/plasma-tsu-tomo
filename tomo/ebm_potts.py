"""Variant P: Potts reference model for tomography (one K-state categorical variable per pixel).

Energy (levels x_j in {0..K-1}):  E(x) = 1/2 x^T Q x - c^T x + const   (see ``ebm_common``).

thrml encoding
--------------
thrml's discrete factors have energy ``-sum W[state...]`` (negative sign!).  Hence

* pair factor (j<k), ``SquareCategoricalEBMFactor``:  ``W_jk[a, b] = -Q_jk a b``  (shape (E, K, K)),
* unary factor, ``CategoricalEBMFactor``:             ``W_j[a] = -(1/2 Q_jj a^2 - c_j a)`` (shape (N, K)),

and ``CategoricalGibbsConditional`` samples ``p(a) ~ exp(theta_a)`` with theta the summed weights,
i.e. the Boltzmann law ``exp(-beta E)`` once all weights are multiplied by ``beta`` (done by a traced
scalar inside ``jit``, so annealing never rebuilds the program).

Blocking
--------
Q = Delta^2 (T^T W T + lam L) couples every pair of pixels that share a chord (plus 4-neighbours),
so the 2-colour checkerboard of the spec table is NOT a valid Gibbs blocking.  Blocks are the colour
classes of a proper colouring of the actual Q non-zero graph (``greedy_coloring``, validity asserted).

Backends (identical colour-block Gibbs sweep, same conditional)
---------------------------------------------------------------
``'thrml'``  reference: ``FactorSamplingProgram`` + ``CategoricalGibbsConditional`` + ``sample_states``.
``'jax'``    equivalent pure-JAX sampler: per block the local field is a dense matvec
             ``f = Qoff[block] @ x - c[block]``, ``theta_a = -beta (a f + 1/2 Q_jj a^2)``.
Both are validated against exact enumeration / each other in ``tests/test_ebm_potts.py``.

Public API: ``PottsModel``, ``build_potts``, ``build_potts_from_Q``, ``PottsSampler``,
``sample_potts``, ``coupling_split``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from thrml import Block, BlockGibbsSpec, CategoricalNode, SamplingSchedule, sample_states
from thrml.factor import FactorSamplingProgram
from thrml.models.discrete_ebm import (CategoricalEBMFactor, CategoricalGibbsConditional,
                                       SquareCategoricalEBMFactor)
from thrml.models.ebm import FactorizedEBM

from .ebm_common import EBMSetup, make_result, prepare, quadratic_form
from .sampling import check_coloring, geometric_betas, greedy_coloring

__all__ = ["PottsModel", "build_potts", "build_potts_from_Q", "PottsSampler", "sample_potts",
           "coupling_split"]


# --------------------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------------------
@dataclass
class PottsModel:
    """Potts EBM ``E(x) = 1/2 x^T Q x - c^T x + const`` plus its thrml objects and block structure."""

    Q: np.ndarray
    c: np.ndarray
    const: float
    K: int
    beta: float
    edges: np.ndarray            # (E, 2) pixel pairs j<k with Q_jk != 0
    colors: np.ndarray           # (N,) proper colouring of the coupling graph
    blocks_idx: list             # list of index arrays, one per colour (block)
    nodes: list = field(default=None, repr=False)
    free_blocks: list = field(default=None, repr=False)
    factors: list = field(default=None, repr=False)
    ebm: Any = field(default=None, repr=False)
    program: Any = field(default=None, repr=False)
    setup: Any = field(default=None, repr=False)
    build_time: float = 0.0

    @property
    def N(self):
        return len(self.c)

    @property
    def n_blocks(self):
        return len(self.blocks_idx)

    @property
    def degrees(self):
        return np.bincount(self.edges.ravel(), minlength=self.N)

    @property
    def max_degree(self):
        return int(self.degrees.max()) if self.N else 0

    def energy(self, x):
        """float64 energy of levels x (..., N)."""
        x = np.asarray(x, float)
        return 0.5 * np.einsum("...i,ij,...j->...", x, self.Q, x) - x @ self.c + self.const

    def thrml_energy(self, x):
        """Energy of one level vector (N,) from the thrml factor model (excludes ``const``; beta-scaled
        weights are divided back out so this is the unit-beta energy up to the constant)."""
        st = [jnp.asarray(np.asarray(x)[b], jnp.uint8) for b in self.blocks_idx]
        return float(self.ebm.energy(st, self.free_blocks)) / self.beta


def coupling_split(setup: EBMSetup) -> dict:
    """Share of Q's off-diagonal magnitude carried by the Laplacian vs the T^T W T part."""
    p, D = setup.problem, setup.Delta
    T = np.asarray(p.T, float)
    G = D ** 2 * (T.T * setup.w) @ T
    import scipy.sparse as sp
    L = p.L.toarray() if sp.issparse(p.L) else np.asarray(p.L, float)
    Lm = D ** 2 * setup.lam * L
    off = lambda A: np.abs(A - np.diag(np.diag(A))).sum()  # noqa: E731
    g, l = off(G), off(Lm)
    return dict(frac_laplacian=l / (g + l), frac_TtT=g / (g + l), offdiag_l1_TtT=g, offdiag_l1_lap=l)


def build_potts_from_Q(Q, c, const, K, beta=1.0, colors=None, build_thrml=True, tol=0.0) -> PottsModel:
    """Build the Potts model (colouring + thrml program) from an explicit quadratic form."""
    t0 = time.perf_counter()
    Q = np.asarray(Q, float)
    c = np.asarray(c, float)
    N = len(c)
    off = np.abs(Q - np.diag(np.diag(Q)))
    iu, ku = np.nonzero(np.triu(off > tol * off.max(), 1))
    edges = np.stack([iu, ku], axis=1)
    if colors is None:
        colors = greedy_coloring(N, edges)
    if not check_coloring(N, edges, colors):
        raise AssertionError("colouring is not proper for the coupling graph of Q")
    _, cid = np.unique(colors, return_inverse=True)
    blocks_idx = [np.flatnonzero(cid == k) for k in range(cid.max() + 1)]
    m = PottsModel(Q=Q, c=c, const=float(const), K=int(K), beta=float(beta), edges=edges,
                   colors=cid, blocks_idx=blocks_idx)
    if build_thrml:
        a = np.arange(K, dtype=np.float64)
        nodes = [CategoricalNode() for _ in range(N)]
        Wu = -(0.5 * np.diag(Q)[:, None] * a[None] ** 2 - c[:, None] * a[None])          # (N, K)
        fac = [CategoricalEBMFactor([Block(nodes)], jnp.asarray(beta * Wu, jnp.float32))]
        if len(edges):
            Wp = -Q[iu, ku][:, None, None] * (a[:, None] * a[None, :])[None]              # (E, K, K)
            fac.append(SquareCategoricalEBMFactor(
                [Block([nodes[j] for j in iu]), Block([nodes[k] for k in ku])],
                jnp.asarray(beta * Wp, jnp.float32)))
        free = [Block([nodes[i] for i in b]) for b in blocks_idx]
        ebm = FactorizedEBM(fac)
        spec = BlockGibbsSpec(free, [])
        program = FactorSamplingProgram(spec, [CategoricalGibbsConditional(K) for _ in free], fac, [])
        m.nodes, m.free_blocks, m.factors, m.ebm, m.program = nodes, free, fac, ebm, program
    m.build_time = time.perf_counter() - t0
    return m


def build_potts(problem, K, lam=None, beta=1.0, setup: EBMSetup | None = None, eps_max_factor=1.2,
                build_thrml=True) -> PottsModel:
    """Potts model of the tomography posterior (``setup`` from ``ebm_common.prepare`` is reused if given)."""
    if setup is None:
        setup = prepare(problem, K, lam, eps_max_factor)
    m = build_potts_from_Q(setup.Q, setup.c, setup.const, K, beta, build_thrml=build_thrml)
    m.setup = setup
    return m


# --------------------------------------------------------------------------------------
# sampler
# --------------------------------------------------------------------------------------
def _scale_program(program, f):
    def scale(x):
        if isinstance(x, jax.Array) and jnp.issubdtype(x.dtype, jnp.floating):
            return x * f
        return x
    return eqx.tree_at(lambda p: p.per_block_interactions, program,
                       jax.tree.map(scale, program.per_block_interactions))


class PottsSampler:
    """Compiled colour-block Gibbs sampler for a :class:`PottsModel` (``backend`` 'thrml' or 'jax').

    States are integer levels (C, N). ``beta`` is the absolute inverse temperature (traced).
    """

    def __init__(self, model: PottsModel, backend: str = "jax"):
        if backend not in ("thrml", "jax"):
            raise ValueError("backend must be 'thrml' or 'jax'")
        if backend == "thrml" and model.program is None:
            raise ValueError("model was built without thrml objects")
        self.model, self.backend = model, backend
        self._cache = {}
        K, N = model.K, model.N
        self._a = jnp.arange(K, dtype=jnp.float32)
        Q = model.Q
        self._Qj = jnp.asarray(Q, jnp.float32)
        self._cj = jnp.asarray(model.c, jnp.float32)
        self._const = jnp.float32(model.const)
        if backend == "thrml":
            self._data = model.program
            self._idx = [jnp.asarray(b, jnp.int32) for b in model.blocks_idx]
        else:
            # blocks padded to a common size so a sweep is a lax.scan over blocks (fast compile);
            # padding rows point at a dummy slot N and have zero couplings.
            Qoff = Q - np.diag(np.diag(Q))
            nbk, bm = model.n_blocks, max(len(b) for b in model.blocks_idx)
            idx = np.full((nbk, bm), N, np.int64)
            Qs = np.zeros((nbk, bm, N + 1))
            cs, ds = np.zeros((nbk, bm)), np.zeros((nbk, bm))
            for i, b in enumerate(model.blocks_idx):
                idx[i, :len(b)] = b
                Qs[i, :len(b), :N] = Qoff[b]
                cs[i, :len(b)] = model.c[b]
                ds[i, :len(b)] = np.diag(Q)[b]
            self._data = dict(idx=jnp.asarray(idx, jnp.int32), Q=jnp.asarray(Qs, jnp.float32),
                              c=jnp.asarray(cs, jnp.float32), d=jnp.asarray(ds, jnp.float32))

    # ---- energy (float32, traceable) ----
    def energy_core(self, x):
        x = x.astype(jnp.float32)
        return 0.5 * x @ (self._Qj @ x) - self._cj @ x + self._const

    # ---- jax backend ----
    def _sweep_jax(self, data, key, x, beta):
        """One sweep over all colour blocks. x: (N,) float levels."""
        a = self._a
        xe = jnp.concatenate([x, jnp.zeros(1, x.dtype)])  # dummy slot N absorbs padding writes

        def blk(xe, inp):
            idx, Qb, cb, db, k = inp
            f = Qb @ xe - cb
            theta = -beta * (f[:, None] * a[None] + 0.5 * db[:, None] * a[None] ** 2)
            new = jax.random.categorical(k, theta, axis=-1).astype(jnp.float32)
            return xe.at[idx].set(new), None
        keys = jax.random.split(key, data["idx"].shape[0])
        xe, _ = jax.lax.scan(blk, xe, (data["idx"], data["Q"], data["c"], data["d"], keys))
        return xe[:-1]

    def _sample_jax(self, data, key, state, beta, sched):
        x0 = state.astype(jnp.float32)
        kw, ks = jax.random.split(key)

        def sweeps(x, k, m):
            if m == 0:
                return x
            return jax.lax.scan(lambda c, kk: (self._sweep_jax(data, kk, c, beta), None), x,
                                jax.random.split(k, m))[0]
        x = sweeps(x0, kw, sched.n_warmup)
        if sched.n_samples <= 1:
            return x.astype(jnp.uint8)[None]

        def body(c, k):
            c = sweeps(c, k, sched.steps_per_sample)
            return c, c.astype(jnp.uint8)
        _, rest = jax.lax.scan(body, x, jax.random.split(ks, sched.n_samples - 1))
        return jnp.concatenate([x.astype(jnp.uint8)[None], rest], axis=0)

    # ---- thrml backend ----
    def _sample_thrml(self, data, key, state, beta, sched):
        program = _scale_program(data, beta / self.model.beta)
        init = [state[i].astype(jnp.uint8) for i in self._idx]
        out = sample_states(key, program, sched, init, [], self.model.free_blocks)
        res = jnp.zeros((sched.n_samples, self.model.N), jnp.uint8)
        for i, o in zip(self._idx, out):
            res = res.at[:, i].set(o.astype(jnp.uint8))
        return res

    def _single(self, data, key, state, beta, sched):
        f = self._sample_thrml if self.backend == "thrml" else self._sample_jax
        return f(data, key, state, beta, sched)

    # ---- compile cache ----
    def _compiled(self, name, fn, *args):
        sig = (name,) + tuple((tuple(a.shape), str(a.dtype)) for a in jax.tree.leaves(args[1:]))
        if sig not in self._cache:
            t0 = time.perf_counter()
            dyn, stat = eqx.partition(args[0], eqx.is_array)
            jf = jax.jit(lambda d, *rest: fn(eqx.combine(d, stat), *rest))
            exe = jf.lower(dyn, *args[1:]).compile()
            self._cache[sig] = (lambda data, *rest, _e=exe: _e(eqx.filter(data, eqx.is_array), *rest),
                                time.perf_counter() - t0)
        return self._cache[sig]

    def init_state(self, init, n_chains, key, x0=None, jitter=0.3):
        """(C, N) int32 levels. ``init``: 'tikhonov' (x0 rounded levels, each pixel shifted by +-1 with
        prob ``jitter`` per chain for over-dispersion), 'random'/'hinton' (uniform levels) or an array."""
        K, N = self.model.K, self.model.N
        if not isinstance(init, str) and init is not None:
            init = jnp.asarray(init, jnp.int32)
            return jnp.broadcast_to(init, (n_chains, N)) if init.ndim == 1 else init
        if init in (None, "random", "hinton") or x0 is None:
            return jax.random.randint(key, (n_chains, N), 0, K)
        if init != "tikhonov":
            raise ValueError(f"unknown init {init!r}")
        k1, k2 = jax.random.split(key)
        x0 = jnp.asarray(x0, jnp.int32)
        shift = jnp.where(jax.random.bernoulli(k1, jitter, (n_chains, N)),
                          jnp.where(jax.random.bernoulli(k2, 0.5, (n_chains, N)), 1, -1), 0)
        return jnp.clip(x0[None] + shift, 0, K - 1)

    def run(self, keys, init, beta, sched: SamplingSchedule, compile_only=False):
        """vmapped chains -> samples (C, S, N) uint8 (device), compile seconds."""
        beta = jnp.asarray(beta, jnp.float32)

        def f(data, keys, init, beta):
            return jax.vmap(lambda k, s: self._single(data, k, s, beta, sched))(keys, init)
        exe, ct = self._compiled(("run", sched.n_warmup, sched.n_samples, sched.steps_per_sample),
                                 f, self._data, keys, init, beta)
        if compile_only:
            return None, ct
        return exe(self._data, keys, init, beta), ct

    def energies(self, samples, chunk=64):
        """float32 energies of samples (..., N) via dense matmul."""
        s = jnp.asarray(samples)
        lead = s.shape[:-1]
        e = jax.lax.map(self.energy_core, s.reshape(-1, s.shape[-1]), batch_size=chunk)
        return np.asarray(e).reshape(lead)

    def anneal(self, key, init, betas, sweeps_per_beta):
        """Simulated annealing. Returns dict(final (C,N), best (C,N), best_energy (C,), energy_trace
        (C, n_betas), time, compile_time)."""
        betas = jnp.asarray(betas, jnp.float32)
        sched = SamplingSchedule(int(sweeps_per_beta), 1, 1)
        C = init.shape[0]
        keys = jax.random.split(key, C)
        nb = len(betas)

        def f(data, keys, state, betas):
            def chain(k, s0):
                ks = jax.random.split(k, nb)

                def step(c, inp):
                    s, bs, be = c
                    kk, b = inp
                    s = self._single(data, kk, s, b, sched)[0].astype(jnp.int32)
                    e = self.energy_core(s)
                    better = e < be
                    return (s, jnp.where(better, s, bs), jnp.where(better, e, be)), e
                s0 = s0.astype(jnp.int32)
                (s, bs, be), es = jax.lax.scan(step, (s0, s0, self.energy_core(s0)), (ks, betas))
                return s, bs, be, es
            return jax.vmap(chain)(keys, state)

        exe, ct = self._compiled(("anneal", int(sweeps_per_beta), nb), f, self._data, keys, init, betas)
        t0 = time.perf_counter()
        final, best, be, tr = exe(self._data, keys, init, betas)
        final.block_until_ready()
        return dict(final=np.asarray(final), best=np.asarray(best), best_energy=np.asarray(be),
                    energy_trace=np.asarray(tr), time=time.perf_counter() - t0, compile_time=ct)


# --------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------
def sample_potts(problem, K, lam, cfg, key, backend: str = "jax", setup: EBMSetup | None = None,
                 n_chains: int | None = None, posterior: dict | None = None, anneal: dict | None = None,
                 init: str | None = None, jitter: float = 0.3, do_map: bool = True,
                 model: PottsModel | None = None, eps_max_factor: float = 1.2,
                 frac_random: float = 0.25) -> dict:
    """Posterior (beta=1) + annealed-MAP reconstruction with the Potts model.

    Schedules default to ``cfg['schedule']`` (override with ``posterior``/``anneal`` dicts). Returns the
    standard EBM result dict (``ebm_common.make_result``) plus ``map_energy``, ``time_posterior``,
    ``time_anneal``, ``compile_time``, ``build_time``, ``setup``, ``backend``, ``lam``, ``delta`` (level step),
    ``levels`` (C,S,N uint8 samples), ``map_lev`` (MAP levels), ``frac_top`` (fraction of samples at the
    top level K-1: eps_max truncation diagnostic), ``eps_max``, ``n_random_chains``. ``frac_random`` of the chains
    start from uniformly random levels (rest: jittered Tikhonov). ``eps_max_factor`` scales eps_max.  ``time`` = sampling wall time (posterior +
    anneal, excluding compilation and model build).
    """
    sc = cfg["schedule"]
    post = dict(sc["posterior"], **(posterior or {}))
    ann = dict(sc["anneal"], **(anneal or {}))
    C = int(n_chains or sc["n_chains"])
    init = init or sc.get("init", "tikhonov")
    if model is None:
        model = build_potts(problem, K, lam, setup=setup, eps_max_factor=eps_max_factor,
                            build_thrml=(backend == "thrml"))
    setup = model.setup
    sm = PottsSampler(model, backend)
    k_init, k_post, k_ann = jax.random.split(key, 3)
    state = sm.init_state(init, C, k_init, x0=setup.x0, jitter=jitter)
    n_rand = int(np.ceil(frac_random * C)) if (init == "tikhonov" and frac_random > 0 and C > 1) else 0
    if n_rand:  # overdispersed chains: uniformly random levels, so R-hat is not flattered by a common start
        kr = jax.random.fold_in(k_init, 7)
        state = state.at[:n_rand].set(jax.random.randint(kr, (n_rand, model.N), 0, K))
    sched = SamplingSchedule(int(post["n_warmup"]), int(post["n_samples"]), int(post["steps_per_sample"]))
    keys = jax.random.split(k_post, C)
    _, ct = sm.run(keys, state, 1.0, sched, compile_only=True)
    t0 = time.perf_counter()
    samples, _ = sm.run(keys, state, 1.0, sched)
    samples.block_until_ready()
    t_post = time.perf_counter() - t0
    lev = np.asarray(samples)
    etrace = sm.energies(samples)
    res_map, t_ann, ct2, map_lev, map_e = None, 0.0, 0.0, None, None
    if do_map:
        betas = geometric_betas(ann["beta_min"], ann["beta_max"], int(ann["n_betas"]))
        res_map = sm.anneal(k_ann, state, betas, int(ann["sweeps_per_beta"]))
        t_ann, ct2 = res_map["time"], res_map["compile_time"]
        i = int(np.argmin(res_map["best_energy"]))
        map_lev = res_map["best"][i]
        map_e = float(model.energy(map_lev))  # float64 recomputation
    frac_top = float(np.mean(lev == K - 1))
    out = make_result(lev, setup.Delta, map_lev, energy_trace=etrace, n_blocks=model.n_blocks,
                      n_spins=model.N, max_degree=model.max_degree, time=t_post + t_ann,
                      invalid_frac=0.0, map_energy=map_e, time_posterior=t_post, time_anneal=t_ann,
                      compile_time=ct + ct2, build_time=model.build_time, setup=setup, backend=backend,
                      lam=setup.lam, delta=setup.Delta, levels=lev, map_lev=map_lev,
                      frac_top=frac_top, frac_top_map=None if map_lev is None else float(np.mean(map_lev == K - 1)),
                      eps_max=setup.eps_max, n_random_chains=n_rand,
                      anneal_trace=None if res_map is None else res_map["energy_trace"])
    return out
