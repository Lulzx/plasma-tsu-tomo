"""Binary (power-of-two) encoding of the pixel-only dense Ising model (analogue of I-dense).

Levels x_j in {0..K-1}, K = 2^nb, x_j = sum_m 2^m u_{j,m} (bit m of pixel j is spin j*nb+m, LSB first).
Every bit pattern is a valid level, so there is no domain-wall penalty and ``invalid_frac`` is 0.  The
energy is the same quadratic form E(x) = 1/2 x^T Q x - c^T x + const of ``ebm_common.prepare`` (so the
bit energy equals the pixel energy of the decoded levels exactly), expanded by ``embed.binary_bit_qubo``
and converted by ``sampling.bits_to_spins_qubo``.  Couplings scale as 2^(a+b), which costs up to
2*(nb-1) bits of extra coupling dynamic range; single-bit Gibbs moves cannot step a level by +-1 across
a carry (3 -> 4 = 011 -> 100), the mixing question studied in experiments/binary_encoding.py.

Gray code: x = gray_decode(g) has bit x_b = XOR_{b'>=b} g_b', i.e. x is a polynomial of degree nb in the
spins, so E(x(g)) is not quadratic in g in general; there is no exact quadratic Gray variant here.
"""
from __future__ import annotations

import jax
import numpy as np

from . import ebm_common as ec
from .config import load_config
from .embed import binary_bit_qubo
from .sampling import (IsingSampler, bits_to_spins_qubo, check_coloring, greedy_coloring, run_ising)

__all__ = ["n_bits", "binary_encode", "binary_decode", "build_ising_binary", "sample_binary"]


def n_bits(K):
    nb = int(round(np.log2(K)))
    if 2 ** nb != K or K < 2:
        raise ValueError("K must be a power of two >= 2")
    return nb


def binary_encode(x, K):
    """Levels (..., N) -> bits (..., N, nb), LSB first."""
    return ((np.asarray(x)[..., None] >> np.arange(n_bits(K))) & 1).astype(np.int8)


def binary_decode(u, K=None):
    """Bits (..., N, nb) -> levels (..., N)."""
    u = np.asarray(u).astype(np.int64)
    return (u << np.arange(u.shape[-1])).sum(-1)


def _color(prob, N, nb, pix_edges):
    cands = {"greedy_bits": greedy_coloring(prob.n, prob.edges)}
    pc = greedy_coloring(N, pix_edges)
    cands["pixel_x_bit"] = np.unique(np.repeat(pc, nb) * nb + np.tile(np.arange(nb), N), return_inverse=True)[1]
    counts = {k: int(len(np.unique(v))) for k, v in cands.items()}
    best = min(counts, key=counts.get)
    colors = cands[best]
    assert check_coloring(prob.n, prob.edges, colors), "invalid colouring"
    return colors, dict(coloring=best, colors_by_method=counts, n_blocks=counts[best])


def build_ising_binary(problem, K, lam=None, setup=None, eps_max_factor=1.2):
    """Dense binary-encoded Ising model -> (IsingProblem, meta); meta has colors, n_blocks, max_degree, setup, nb, ..."""
    nb = n_bits(K)
    if setup is None or setup.K != K:
        setup = ec.prepare(problem, K, lam, eps_max_factor)
    Qb, cb, constb, rowvar, off = binary_bit_qubo(setup.Q, setup.c, setup.const, nb)
    prob = bits_to_spins_qubo(Qb, cb, constb)
    import scipy.sparse as sp
    Qo = sp.triu(sp.csr_matrix(setup.Q - np.diag(np.diag(setup.Q))), k=1).tocoo()
    m = Qo.data != 0
    pix_edges = np.stack([Qo.row[m], Qo.col[m]], 1)
    colors, cinfo = _color(prob, setup.N, nb, pix_edges)
    deg = prob.degrees
    meta = dict(setup=setup, K=K, nb=nb, N=setup.N, Qb=Qb, cb=cb, constb=constb, colors=colors,
                n_spins=prob.n, n_edges=prob.n_edges, max_degree=int(deg.max()), mean_degree=float(deg.mean()),
                Delta=setup.Delta, eps_max=setup.eps_max, lam=setup.lam, variant="binary", **cinfo)
    return prob, meta


def sample_binary(problem, cfg, key, backend="jax", setup=None, schedule=None, n_chains=None,
                  overdispersed=True, sampler=None, K=None):
    """Posterior sampling of the binary model; standard result dict (see ``ebm_ising.sample_ising_variant``).

    Half the chains (odd ones) start from random bits when ``overdispersed``, the rest from the Tikhonov
    levels.  Extras: meta, setup, levels (C,S,N), levels_mean, rhat_max, time_sample, compile_time, build_time.
    """
    cfg = load_config(cfg)
    m, sc = cfg["model"], cfg["schedule"]
    K = int(K or m["K"])
    schedule = schedule or sc["posterior"]
    C = int(n_chains or sc["n_chains"])
    prob, meta = build_ising_binary(problem, K, m.get("lam"), setup=setup, eps_max_factor=m.get("eps_max_factor", 1.2))
    setup, nb = meta["setup"], meta["nb"]
    sm = sampler or IsingSampler(prob, meta["colors"], backend=backend)
    init = binary_encode(setup.x0, K).reshape(-1).astype(bool)
    if overdispersed:
        rnd = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**30))).random((C, prob.n)) < 0.5
        init = np.where((np.arange(C) % 2 == 0)[:, None], init[None, :], rnd)
    r = run_ising(prob, meta["colors"], schedule, C, key, beta=1.0, init=init, backend=backend, sampler=sm)
    lev = binary_decode(r["samples"].reshape(r["samples"].shape[:-1] + (setup.N, nb)))
    res = ec.make_result(lev, setup.Delta, None, energy_trace=r["energy_trace"], n_blocks=sm.n_blocks,
                         n_spins=prob.n, max_degree=sm.max_degree, time=r["time"], invalid_frac=0.0,
                         meta=meta, setup=setup, variant="binary", backend=backend, levels=lev,
                         levels_mean=lev.reshape(-1, lev.shape[-1]).mean(0), time_sample=r["time"],
                         compile_time=r["compile_time"], build_time=sm.build_time)
    rh = np.asarray(res["rhat"])
    res["rhat_max"] = float(np.nanmax(rh)) if (np.isfinite(rh).any() or np.isinf(rh).any()) else float("nan")
    return res
