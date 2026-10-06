"""Variant I: domain-wall (thermometer) Ising model of the tomography energy.

Each pixel j has K-1 bits u_{j,m} = [x_j > m] (``ebm_common.dw_encode``); the pixel energy
E(x) = 1/2 x^T Q x - c^T x + const with x_j = sum_m u_{j,m} plus a domain-wall penalty
A*#(u_{m+1}=1, u_m=0) is a QUBO (``ebm_common.pixel_quadratic_to_bit_qubo``), converted to +/-1
spins by ``sampling.bits_to_spins_qubo``.  Spin i corresponds to bit i = j*(K-1)+m (True == bit 1).

Variants
--------
``dense``   all T^T T / Laplacian couplings (every pair of pixels sharing a chord is coupled).
``sparse``  pixel-pair couplings dropped when |Q_jk| < threshold * max_offdiag |Q| or when the pixel
            centres are further than ``radius`` metres apart; Laplacian (4-neighbour) pairs and
            intra-pixel bit couplings are always kept.  The model is then *approximate*.
``chain``   I-chain (``tomo.ebm_chain``; lazily imported).

Blocking: a block must be an independent set of the *actual* coupling graph, so the 2-colour
pixel checkerboard of the spec is not valid here.  We colour the bit graph with greedy colouring
and, as a competitor, with the structured colouring ``pixel_colour * (K-1) + bit`` (pixel colours
from greedy colouring of the kept pixel graph), and keep whichever has fewer blocks.  Both are
asserted valid.

Caveats: the sparse variant is approximate in posterior spread/coverage (linearisation drops the
covariance of dropped pairs; std ~0.93 of dense measured), use it for means only.  The Tikhonov rel-L2
quoted with Ising results uses the discrepancy-principle lambda of ``ec.prepare`` (not the GCV lambda of
baselines.py).  ``overdispersed=True`` starts half of the chains from random bits for an honest R-hat.

Decoding invalid samples: a pixel whose bits are not of the form 1..10..0 is decoded as its bit
sum (level = sum_m u_m) and counted in ``invalid_frac``.  Any A > 0 makes ground states valid.
"""
from __future__ import annotations

import time

import jax
import numpy as np
import scipy.sparse as sp

from . import ebm_common as ec
from .config import load_config
from .metrics import rel_l2
from .sampling import (IsingSampler, anneal_ising, bits_to_spins_qubo, check_coloring, geometric_betas,
                       greedy_coloring, run_ising)

__all__ = ["dw_encode", "dw_decode", "build_ising_dense", "build_ising_sparse", "build_ising_chain",
           "sample_ising_variant", "color_ising", "pixel_pair_mask", "posterior_bias", "bits_to_levels"]

dw_encode = ec.dw_encode
dw_decode = ec.dw_decode


# ----------------------------------------------------------------------------------------
# colouring
# ----------------------------------------------------------------------------------------
def _smallest_last(N, pixel_edges):
    """Smallest-last greedy pixel colouring (networkx), or None if unavailable."""
    try:
        import networkx as nx
    except ImportError:
        return None
    g = nx.Graph()
    g.add_nodes_from(range(N))
    g.add_edges_from(map(tuple, np.asarray(pixel_edges).tolist()))
    d = nx.coloring.greedy_color(g, strategy="smallest_last")
    return np.array([d[i] for i in range(N)])


def color_ising(prob, K, pixel_edges=None):
    """Proper colouring of the Ising coupling graph (asserted). Returns (colors, info).

    Tries greedy colouring of the spin graph and, if ``pixel_edges`` (P,2) is given, the structured
    colouring pixel_colour*(K-1)+bit; returns the one with fewer colours.
    """
    n, nb = prob.n, K - 1
    cands = {"greedy_bits": greedy_coloring(n, prob.edges)}
    sl = _smallest_last(n // nb, pixel_edges) if pixel_edges is not None else None
    if sl is not None:
        cands["pixel_sl_x_bit"] = np.unique(np.repeat(sl, nb) * nb + np.tile(np.arange(nb), n // nb),
                                            return_inverse=True)[1]
    if pixel_edges is not None:
        N = n // nb
        pc = greedy_coloring(N, pixel_edges)
        cands["pixel_x_bit"] = (np.repeat(pc, nb) * nb + np.tile(np.arange(nb), N))
        cands["pixel_x_bit"] = np.unique(cands["pixel_x_bit"], return_inverse=True)[1]
    counts = {k: int(len(np.unique(v))) for k, v in cands.items()}
    best = min(counts, key=counts.get)
    colors = cands[best]
    assert check_coloring(n, prob.edges, colors), "invalid colouring"
    return colors, dict(coloring=best, colors_by_method=counts, n_blocks=counts[best])


# ----------------------------------------------------------------------------------------
# builders
# ----------------------------------------------------------------------------------------
def _prepare(problem, K, lam, setup, eps_max_factor=1.2):
    if setup is None or setup.K != K:
        setup = ec.prepare(problem, K, lam, eps_max_factor)
    return setup


def _finish(setup, Qb, cb, constb, A, pixel_pairs, extra):
    K = setup.K
    prob = bits_to_spins_qubo(Qb, cb, constb)
    pe = np.asarray(pixel_pairs).reshape(-1, 2)
    colors, cinfo = color_ising(prob, K, pe)
    deg = prob.degrees
    meta = dict(setup=setup, K=K, N=setup.N, A=A, Qb=Qb, cb=cb, constb=constb, colors=colors,
                n_spins=prob.n, n_edges=prob.n_edges, max_degree=int(deg.max()),
                mean_degree=float(deg.mean()), n_pixel_pairs=int(len(pe)),
                Delta=setup.Delta, eps_max=setup.eps_max, lam=setup.lam, **cinfo, **extra)
    return prob, meta


def _offdiag_pairs(Q):
    U = sp.triu(sp.csr_matrix(Q - np.diag(np.diag(Q))), k=1).tocoo()
    m = U.data != 0
    return U.row[m], U.col[m], U.data[m]


def build_ising_dense(problem, K, lam=None, A=None, setup=None, eps_max_factor=1.2):
    """Full domain-wall Ising model -> (IsingProblem, meta).

    ``lam=None`` uses the Tikhonov (discrepancy) lambda from ``ebm_common.prepare``; ``A=None`` the
    ``auto_A`` bound.  meta has n_blocks, max_degree, mean_degree, n_edges, colors, setup, ...
    """
    setup = _prepare(problem, K, lam, setup, eps_max_factor)
    A = ec.auto_A(setup.Q, setup.c, K) if A is None else float(A)
    Qb, cb, constb = ec.pixel_quadratic_to_bit_qubo(setup.Q, setup.c, setup.const, K, A)
    r, c, _ = _offdiag_pairs(setup.Q)
    return _finish(setup, Qb, cb, constb, A, np.stack([r, c], 1),
                   dict(variant="dense", threshold=None, radius=None, kept_fraction=1.0))


def pixel_pair_mask(problem, Q, threshold=None, radius=None):
    """Kept pixel pairs (P,2) (i<j) for I-sparse and the fraction kept (of the nonzero pairs of Q) and the number of Laplacian pairs.

    Keep (j,k) iff it is a Laplacian edge, or [|Q_jk| >= threshold * max_offdiag|Q|] and
    [dist(j,k) <= radius] (a criterion left as None is not applied).
    """
    r, c, v = _offdiag_pairs(Q)
    keep = np.ones(len(r), bool)
    if threshold is not None:
        keep &= np.abs(v) >= threshold * np.abs(v).max()
    if radius is not None:
        X = np.asarray(problem.grid.X)[problem.mask]
        Y = np.asarray(problem.grid.Y)[problem.mask]
        keep &= np.hypot(X[r] - X[c], Y[r] - Y[c]) <= radius + 1e-12
    e = np.sort(np.asarray(problem.edges).reshape(-1, 2), axis=1)
    N = Q.shape[0]
    lap = np.isin(r * N + c, e[:, 0] * N + e[:, 1])
    keep |= lap
    return np.stack([r[keep], c[keep]], 1), float(keep.mean()), int(lap.sum())


def build_ising_sparse(problem, K, lam=None, A=None, threshold=None, radius=None, setup=None,
                       eps_max_factor=1.2, compensate="linear"):
    """Sparsified Ising model -> (IsingProblem, meta) with meta['kept_fraction'] of pixel-pair couplings.

    threshold=None and radius=None -> threshold 0.05 (configs/default.yaml model.sparse_threshold).
    threshold=0 keeps every coupling, i.e. reproduces ``build_ising_dense`` exactly.

    ``compensate``: dropping the (positive) T^T W T couplings while keeping the data term c leaves every
    pixel trying to explain its chords alone, so levels overshoot badly (measured rel-L2 > 1).
    'linear' (default) replaces each dropped term 1/2 Q_jk x_j x_k by its linearisation about the
    Tikhonov levels m (clipped to [0,K-1]), i.e. c <- c - Q_drop m, const <- const - 1/2 m^T Q_drop m;
    this keeps the energy gradient exact at m and adds no couplings.  'none' is the raw truncation.
    """
    if threshold is None and radius is None:
        threshold = 0.05
    if compensate not in ("linear", "none"):
        raise ValueError("compensate must be 'linear' or 'none'")
    setup = _prepare(problem, K, lam, setup, eps_max_factor)
    pairs, frac, n_lap = pixel_pair_mask(problem, setup.Q, threshold, radius)
    c, const = setup.c, setup.const
    N = setup.N
    kept = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(N, N))
    kept = (kept + kept.T).tocsr()
    Qo = sp.csr_matrix(setup.Q - np.diag(np.diag(setup.Q)))
    if compensate == "linear" and frac < 1.0:
        Qd = Qo - Qo.multiply(kept > 0)
        m = np.clip(setup.eps_tik / setup.Delta, 0, K - 1)
        r = Qd @ m
        c, const = c - r, const - 0.5 * float(m @ r)
    if A is None:   # bound for the model actually sampled (kept couplings, compensated c)
        Qk = (Qo.multiply(kept > 0) + sp.diags(np.diag(setup.Q))).toarray()
        A = ec.auto_A(Qk, c, K)
    A = float(A)
    Qb, cb, constb = ec.pixel_quadratic_to_bit_qubo(setup.Q, c, const, K, A, mask_pairs=pairs)
    return _finish(setup, Qb, cb, constb, A, pairs,
                   dict(variant="sparse", threshold=threshold, radius=radius, kept_fraction=frac,
                        compensate=compensate))


def build_ising_chain(problem, K, lam, A, tau, Kz, **kw):
    """I-chain model (main contribution); implemented in ``tomo.ebm_chain`` (imported lazily)."""
    from .ebm_chain import build_ising_chain as _b
    return _b(problem, K, lam, A, tau, Kz, **kw)


# ----------------------------------------------------------------------------------------
# decoding / diagnostics
# ----------------------------------------------------------------------------------------
def bits_to_levels(bits, K):
    """Spins/bits (..., N*(K-1)) -> (levels (...,N), valid (...,N)); invalid pixels decode as sum of bits."""
    b = np.asarray(bits)
    return ec.dw_decode(b.reshape(b.shape[:-1] + (-1, K - 1)).astype(np.int8), K)


def posterior_bias(res, ref, truth=None):
    """Compare a result's posterior mean with a reference result (e.g. dense Ising or Potts).

    Returns rel_l2 between means, signed relative total-power bias, and (if truth) rel-L2 of both.
    """
    d = res["mean"] - ref["mean"]
    out = dict(rel_l2_vs_ref=rel_l2(res["mean"], ref["mean"]),
               power_bias=float(d.sum() / ref["mean"].sum()),
               max_abs_diff=float(np.abs(d).max()))
    if truth is not None:
        out.update(rel_l2=rel_l2(res["mean"], truth), rel_l2_ref=rel_l2(ref["mean"], truth))
    return out


# ----------------------------------------------------------------------------------------
# sampling driver
# ----------------------------------------------------------------------------------------
def sample_ising_variant(problem, variant, cfg, key, backend=None, setup=None, schedule=None, anneal=None,
                         n_chains=None, do_map=True, threshold=None, radius=None, A=None, sampler=None,
                         compensate="linear", overdispersed=False, **kw):
    """Posterior sampling + annealed MAP for variant in {'dense','sparse','chain'}.

    Returns the standard result dict (mean, std, map, samples (C,S,N) emissivity, rhat, energy_trace,
    n_blocks, n_spins, max_degree, time, invalid_frac) plus extras: meta, setup, levels_mean,
    rhat_max, map_energy, time_sample, time_anneal, compile_time, build_time, map_levels,
    mean_levels_samples (C,S,N levels).  ``schedule``/``anneal`` override cfg['schedule'] entries
    (dicts n_warmup/n_samples/steps_per_sample, resp. beta_min/beta_max/n_betas/sweeps_per_beta).
    Backend defaults to 'jax' (exactly equivalent to 'thrml', faster).
    For 'chain', dispatches to ``tomo.ebm_chain.sample_chain(problem, cfg, key, **kw)``.
    """
    if variant == "chain":
        from .ebm_chain import sample_chain
        if backend is not None:
            kw["backend"] = backend
        return sample_chain(problem, cfg, key, **kw)
    if variant not in ("dense", "sparse"):
        raise ValueError(f"unknown variant {variant!r}")
    backend = backend or "jax"
    cfg = load_config(cfg)
    m, sc = cfg["model"], cfg["schedule"]
    K = int(m["K"])
    A = m.get("A") if A is None else A
    schedule = schedule or sc["posterior"]
    anneal = anneal or sc["anneal"]
    C = int(n_chains or sc["n_chains"])
    setup = _prepare(problem, K, m.get("lam"), setup, m.get("eps_max_factor", 1.2))
    if variant == "dense":
        prob, meta = build_ising_dense(problem, K, setup.lam, A, setup=setup)
    else:
        th = m.get("sparse_threshold") if (threshold is None and radius is None) else threshold
        rad = m.get("sparse_radius") if radius is None and threshold is None else radius
        prob, meta = build_ising_sparse(problem, K, setup.lam, A, threshold=th, radius=rad, setup=setup,
                                         compensate=compensate)
    colors = meta["colors"]
    sm = sampler or IsingSampler(prob, colors, backend=backend)

    k_post, k_map = jax.random.split(key)
    mode = sc.get("init", "tikhonov")
    if mode == "tikhonov":
        init = ec.dw_encode(setup.x0, K).reshape(-1).astype(bool)
    elif mode in ("hinton", "random"):
        init = mode
    else:
        raise ValueError(f"unknown init {mode!r}")

    if overdispersed and not isinstance(init, str):
        # half the chains keep the Tikhonov warm start, half start from random bits: R-hat is then
        # a meaningful convergence diagnostic (identical starts understate non-convergence)
        rnd = np.random.default_rng(int(jax.random.randint(k_post, (), 0, 2**30))).random((C, prob.n)) < 0.5
        init = np.where((np.arange(C) % 2 == 0)[:, None], np.asarray(init)[None, :], rnd)
    r = run_ising(prob, colors, schedule, C, k_post, beta=1.0, init=init, backend=backend, sampler=sm)
    lev, valid = bits_to_levels(r["samples"], K)            # (C,S,N)
    inv = ec.invalid_fraction(valid)
    t_anneal, map_lev, map_e = 0.0, None, None
    ctime = r["compile_time"]
    if do_map:
        betas = geometric_betas(anneal["beta_min"], anneal["beta_max"], int(anneal["n_betas"]))
        a = anneal_ising(prob, colors, betas, int(anneal["sweeps_per_beta"]), C, k_map, init=init,
                         backend=backend, sampler=sm)
        i = int(np.argmin(a["best_energy"]))
        map_lev, _ = bits_to_levels(a["best"][i], K)
        map_e = float(a["best_energy"][i])
        t_anneal, ctime = a["time"], ctime + a["compile_time"]
    res = ec.make_result(lev, setup.Delta, map_lev, energy_trace=r["energy_trace"],
                         n_blocks=sm.n_blocks, n_spins=prob.n, max_degree=sm.max_degree,
                         time=r["time"] + t_anneal, invalid_frac=inv,
                         meta=meta, setup=setup, variant=variant, backend=backend,
                         levels_mean=lev.reshape(-1, lev.shape[-1]).mean(0), map_energy=map_e,
                         time_sample=r["time"], time_anneal=t_anneal, compile_time=ctime,
                         build_time=sm.build_time, levels=lev)
    rh = np.asarray(res["rhat"])
    # inf (W==0, B>0: chains stuck at different levels) is the worst case and must count
    res["rhat_max"] = float(np.nanmax(rh)) if np.isfinite(rh).any() or np.isinf(rh).any() else float("nan")
    res["rhat_n_nonfinite"] = int((~np.isfinite(rh)).sum())
    return res
