r"""I-tree: domain-wall Ising tomography with a balanced *binary tree* of partial sums per chord.

Same idea as I-chain (``tomo.ebm_chain``: remove the dense ``T^T W T`` coupling with auxiliary partial sums that carry
the data term only at one end), but the running sums are organised as a totalizer-style balanced binary tree instead of
a linear chain.  For chord i with pixels j_1..j_n (ordered along the chord, so subtrees are spatially contiguous) the
leaves are the pixel terms  m_k = T_{i j_k} Delta x_{j_k}  and every internal node v (n-1 of them, one fewer than the
chain's n; a chord with a single pixel gets one unary node) carries an auxiliary level z_v and the energy

    E = sum_i [ sum_{v in tree_i} (z_v - c_l(v) - c_r(v))^2 / (2 tau_v^2)  +  (b_i - z_{root_i})^2 / (2 s_i^2) ]
        + lam Delta^2 / 2 sum_<jk> (x_j - x_k)^2  +  A_pix (#pixel DW violations) + A_z (#z DW violations)       (1)

with c(child) = z_child for an internal child and = m_k for a pixel child.  Everything else (thermometer encoding of
x in K levels and z in Kz levels, domain-wall penalties, local z windows, tau modes, variance compensation) is shared
with I-chain; see the ``ebm_chain`` docstring.  z windows are centred on the warm-start *subtree* sums
S0_v = sum_{k in subtree(v)} m_k(x0), with half width H_v = max(std_mult sd_v, sigma_mult sigma_i), sd_v the std of the
subtree sum under the Gaussian relaxation (Q^{-1}) plus rounding variance.

Noise accounting.  Marginalising continuous z the residuals e_v = z_v - c_l - c_r are independent N(0, tau_v^2) and
add up at the root (z_root = sum_k m_k + sum_v e_v), so the chord variance is  s_i^2 + sum_{v in tree_i} tau_v^2  --
the same formula as the chain with the sum over *internal nodes* (n_i - 1 terms).  Compensation
s_i^2 = max(sigma_i^2 - sum_v tau_v^2, comp_floor sigma_i^2) therefore makes the marginal exactly N(b_i; sum m, sigma_i^2).

Graph and colouring.  A term couples z_v with its two children (internal or pixel), and the two children with each
other (the square contains the product c_l c_r), so every node has degree <= 3 variables (parent + 2 children + sibling)
and the variable graph is a union of triangles {v, l, r} plus pixel-Laplacian edges.  Hence internal nodes need
3 classes (a proper 3-colouring of the triangles: a node of colour c gives its children colours c+1, c+2 mod 3).  Sibling *pixel* leaves (and pixel leaf + z-leaf siblings) are mutually coupled, so pixels are not simply
checkerboard-colourable any more (chord-adjacent pixels are often diagonal neighbours of the same parity); the pixel
classes are obtained by greedy colouring of the pixel-pixel variable graph (Laplacian + sibling pixel pairs; 3 classes on the default problem).
Bit colour = (variable class, bit index).  ``check_coloring`` is run on the real graph; if the structured colouring is
not proper the builder falls back to greedy colouring of the whole variable graph (``meta['colouring_used']``).

Mixing.  Changing a pixel moves the partial sums of its <= ceil(log2 n) ancestors; a collective z move needs
O(log n) sequential single-level steps instead of O(n) for the chain, so the relaxation time should scale like
(log n)^2 rather than n^2 (``experiments`` / the report measure this).

Public API: ``build_ising_tree``, ``sample_tree``, ``tree_energy``, ``decode_tree``, ``tree_layout``, ``balanced_tree``.
"""
from __future__ import annotations

import time

import jax
import numpy as np
import scipy.sparse as sp

from .config import load_config
from .ebm_chain import (_bit_qubo, _choose_A, chain_layout, decode_chain, effective_noise_variance)
from .ebm_common import dw_encode, invalid_fraction, make_result, prepare
from .metrics import split_rhat, sweeps_to_converge
from .sampling import (IsingSampler, anneal_ising, bits_to_spins_qubo, check_coloring, geometric_betas,
                       greedy_coloring, parallel_tempering, run_ising)

__all__ = ["build_ising_tree", "sample_tree", "tree_energy", "decode_tree", "tree_layout", "balanced_tree",
           "effective_noise_variance"]

decode_tree = decode_chain          # identical bit layout: [pixel bits | aux-node bits]


# ----------------------------------------------------------------------------------------
# topology
# ----------------------------------------------------------------------------------------
def balanced_tree(n: int) -> list:
    """Balanced binary tree over n >= 1 ordered leaves (contiguous subtrees, depth ceil(log2 n)).

    Returns the internal nodes in pre-order (root first) as dicts ``lo, hi`` (leaf range), ``kids`` (list of
    ``('p', leaf)`` / ``('n', node index)``), ``depth`` (root 0), ``is_right``, ``col`` (proper 3-colouring of the {parent, left, right} triangles: a node of colour c
    gives its left child (c+1) % 3 and its right child (c+2) % 3) and ``parent`` (-1 for the root).
    n == 1 gives a single unary node (its only child is the pixel), n >= 2 gives n - 1 binary nodes.
    """
    nodes: list = []

    def rec(lo, hi, depth, is_right, parent, col):
        if hi - lo == 1:
            return ("p", lo)
        idx = len(nodes)
        nodes.append(dict(lo=lo, hi=hi, depth=depth, is_right=is_right, parent=parent, col=col, kids=None))
        mid = lo + (hi - lo + 1) // 2
        nodes[idx]["kids"] = [rec(lo, mid, depth + 1, 0, idx, (col + 1) % 3), rec(mid, hi, depth + 1, 1, idx, (col + 2) % 3)]
        return ("n", idx)

    if n == 1:
        return [dict(lo=0, hi=1, depth=0, is_right=0, parent=-1, col=0, kids=[("p", 0)])]
    rec(0, n, 0, 0, -1, 0)
    return nodes


def tree_layout(problem, t_min: float = 0.0) -> dict:
    """Tree structure of all chords (global arrays over aux nodes, chord-major, pre-order within a chord).

    Keys: ``chord`` (V,), ``depth``, ``is_right``, ``root`` (V,) bool, ``kids_node`` / ``kids_pix`` : lists of
    (child node index) / (link index) per node, ``Sub`` sparse (V, N) with T_ij (unit Delta) summed over the subtree
    leaves (so the subtree sum is Delta * Sub @ x), ``root_of`` (M,) node index of each chord's root (-1 = no pixel),
    ``lay`` (the chain layout supplying chord / pixel order), ``V``.
    """
    lay = chain_layout(problem, group=1, t_min=t_min)
    M = len(lay["off"]) - 1
    N = np.asarray(problem.T).shape[1]
    chord, depth, isr, root, colz = [], [], [], [], []
    kn, kp = [], []
    subr, subc, subv = [], [], []
    root_of = np.full(M, -1, np.int64)
    V = 0
    for i in range(M):
        lo, hi = lay["off"][i], lay["off"][i + 1]
        n = hi - lo
        if n == 0:
            continue
        nodes = balanced_tree(n)
        for q, nd in enumerate(nodes):
            v = V + q
            chord.append(i)
            depth.append(nd["depth"])
            isr.append(nd["is_right"])
            colz.append(nd["col"])
            root.append(nd["parent"] < 0)
            kn.append([V + c for t, c in nd["kids"] if t == "n"])
            kp.append([lo + c for t, c in nd["kids"] if t == "p"])
            for p in range(lo + nd["lo"], lo + nd["hi"]):
                subr.append(v)
                subc.append(lay["link_pix"][p])
                subv.append(lay["link_t"][p])
        root_of[i] = V
        V += len(nodes)
    Sub = sp.csr_matrix((subv, (subr, subc)), shape=(V, N))
    return dict(chord=np.asarray(chord, np.int64), depth=np.asarray(depth, np.int64), is_right=np.asarray(isr, np.int64), col=np.asarray(colz, np.int64),
                root=np.asarray(root, bool), kids_node=kn, kids_pix=kp, Sub=Sub, root_of=root_of, lay=lay, V=V)


def _node_scales(setup, tl, Kz, tau, tau_mode, window, std_mult, sigma_mult):
    """Per-node z window (z0, dz), tau; analogue of ``ebm_chain._element_scales`` with subtree sums."""
    p = setup.problem
    Delta = setup.Delta
    b, sig = np.asarray(p.b, float), np.asarray(p.sigma, float)
    T = np.asarray(p.T, float)
    ch, Sub = tl["chord"], tl["Sub"]
    S = np.maximum(T @ (Delta * np.clip(setup.x0, 0, None)), T @ np.clip(setup.eps_tik, 0, None))
    s = np.maximum.reduce([b, S, sig])
    x0 = np.asarray(setup.x0, float)
    S0 = Delta * (Sub @ x0)
    if window == "local":
        A = Delta * Sub
        Cov = np.linalg.inv(setup.Q)
        var = np.asarray(A.multiply(A @ Cov).sum(1)).ravel() + np.asarray(A.multiply(A).sum(1)).ravel() / 12.0
        H = np.maximum(std_mult * np.sqrt(np.maximum(var, 0)), sigma_mult * sig[ch])
        z0 = np.maximum(S0 - H, 0.0)
        dz = 2 * H / (Kz - 1)
    elif window == "chord":
        zmax = np.maximum(1.2 * s, s + 4 * sig)
        z0, dz = np.zeros(tl["V"]), (zmax / (Kz - 1))[ch]
    else:
        raise ValueError(f"unknown window {window!r}")
    if tau_mode == "dz":
        tau_v = tau * dz
    elif tau_mode == "sigma":
        tau_v = tau * sig[ch]
    elif tau_mode == "signal":
        tau_v = tau * s[ch]
    else:
        raise ValueError(f"unknown tau_mode {tau_mode!r}")
    return dict(s=s, z0=z0, dz=dz, tau=tau_v, S0=S0)


def _subtree_levels(x, tl, Delta, sc, Kz):
    """Aux levels w (..., V) implied by pixel levels x (..., N): subtree sums rounded to each node's grid."""
    x = np.asarray(x, float)
    lead = x.shape[:-1]
    xf = x.reshape(-1, x.shape[-1])
    z = Delta * np.asarray((tl["Sub"] @ xf.T).T)
    w = np.clip(np.rint((z - sc["z0"]) / sc["dz"]), 0, Kz - 1).astype(np.int64)
    return w.reshape(lead + (tl["V"],))


# ----------------------------------------------------------------------------------------
# build
# ----------------------------------------------------------------------------------------
def build_ising_tree(problem, K, lam=None, A=None, tau=None, Kz=16, *, tau_mode="dz", A_z=None, A_margin=1.05,
                     setup=None, eps_max_factor=1.2, colouring="structured", t_min=0.0, A_floor=12.0, window="local",
                     std_mult=5.0, sigma_mult=3.0, compensate=False, comp_floor=0.1):
    """Build the I-tree Ising model. Returns ``(IsingProblem, meta)`` (meta keys as ``build_ising_chain`` plus ``tl``,
    ``depth_max``, ``n_pix_colours``, ``colouring_used``; ``L`` = number of aux nodes, ``lay`` = chain layout)."""
    tau = 0.5 if tau is None else float(tau)
    if setup is None or setup.K != K or (lam is not None and lam != setup.lam):
        setup = prepare(problem, K, lam=lam, eps_max_factor=eps_max_factor)
    N, Delta = setup.N, setup.Delta
    tl = tree_layout(problem, t_min=t_min)
    V = tl["V"]
    sc = _node_scales(setup, tl, Kz, tau, tau_mode, window, std_mult, sigma_mult)
    ch = tl["chord"]
    dz_v, tau_v, z0_v = sc["dz"], sc["tau"], sc["z0"]
    sig = np.asarray(problem.sigma, float)
    b = np.asarray(problem.b, float)
    lay = tl["lay"]
    nv = N + V

    # residual rows  r_v = z0_v - sum_{node kids} z0_c + dz_v w_v - sum_{node kids} dz_c w_c - sum_{pixel kids} t Delta x
    rows, cols, vals = [np.arange(V)], [N + np.arange(V)], [dz_v]
    r0 = z0_v.copy()
    for v in range(V):
        for c in tl["kids_node"][v]:
            rows.append(np.array([v]))
            cols.append(np.array([N + c]))
            vals.append(np.array([-dz_v[c]]))
            r0[v] -= z0_v[c]
        for p in tl["kids_pix"][v]:
            rows.append(np.array([v]))
            cols.append(np.array([lay["link_pix"][p]]))
            vals.append(np.array([-lay["link_t"][p] * Delta]))
    Alk = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(V, nv))
    wl = 1.0 / tau_v ** 2
    Qy = (Alk.T @ sp.diags(wl) @ Alk).tocsr()
    cy = -(Alk.T @ (wl * r0))
    const = 0.5 * float(np.sum(wl * r0 ** 2))
    # data term on every chord root
    roots = np.flatnonzero(tl["root"])
    chords_with = ch[roots]
    dzc, sgc, bc, z0c = dz_v[roots], sig[chords_with], b[chords_with], z0_v[roots]
    var_sum = np.bincount(ch, weights=tau_v ** 2, minlength=len(sig))
    s2_all = sig ** 2
    if compensate:
        s2_all = np.maximum(sig ** 2 - var_sum, comp_floor * sig ** 2)
    s2c = s2_all[chords_with]
    dq = np.zeros(nv)
    dq[N + roots] = dzc ** 2 / s2c
    cy[N + roots] += dzc * (bc - z0c) / s2c
    const += 0.5 * float(np.sum((bc - z0c) ** 2 / s2c))
    Lap = sp.csr_matrix(problem.L) if sp.issparse(problem.L) else sp.csr_matrix(np.asarray(problem.L, float))
    QL = sp.bmat([[setup.lam * Delta ** 2 * Lap, None], [None, sp.csr_matrix((V, V))]], format="csr")
    Qy = (Qy + sp.diags(dq) + QL).tocsr()
    Qy = ((Qy + Qy.T) * 0.5).tocsr()

    nbits = np.concatenate([np.full(N, K - 1), np.full(V, Kz - 1)]).astype(np.int64)
    Lv = nbits.astype(float)
    x0 = np.asarray(setup.x0)
    w0 = _subtree_levels(x0, tl, Delta, sc, Kz)
    y0 = np.concatenate([x0, w0]).astype(float)
    A_pix, A_chn = _choose_A(Qy, cy, Lv, y0, N, A, A_z, A_margin, A_floor)
    Av = np.concatenate([np.full(N, A_pix), np.full(V, A_chn)])
    Qb, cb, constb, rowvar, off = _bit_qubo(Qy, cy, const, nbits, Av)
    prob = bits_to_spins_qubo(Qb, cb, constb)

    # colouring
    bitidx = np.arange(Qb.shape[0]) - off[rowvar]
    Qv = (Qy - sp.diags(Qy.diagonal())).tocoo()
    ev = np.stack([Qv.row, Qv.col], 1)
    ev = ev[ev[:, 0] < ev[:, 1]]

    def expand(cls):
        return cls[rowvar] * (max(K, Kz) + 1) + bitidx

    colouring_used, n_pc = "greedy", 0
    colors = None
    if colouring == "structured":
        ep = ev[(ev[:, 0] < N) & (ev[:, 1] < N)]
        cp = greedy_coloring(N, ep)
        n_pc = int(cp.max() + 1)
        cls = np.concatenate([cp, n_pc + tl["col"]])
        cand = expand(cls)
        if check_coloring(prob.n, prob.edges, cand):
            colors, colouring_used = cand, "structured"
    if colors is None:
        cls = greedy_coloring(nv, ev)
        colors = expand(cls)
        if not check_coloring(prob.n, prob.edges, colors):
            raise AssertionError("colouring is not proper on the Ising graph")
    _, colors = np.unique(colors, return_inverse=True)
    n_blocks = int(colors.max() + 1)
    deg = prob.degrees
    init = np.concatenate([dw_encode(x0, K).ravel(), dw_encode(w0, Kz).ravel()]).astype(bool)

    meta = dict(setup=setup, N=N, K=K, Kz=Kz, L=V, group=1, lay=lay, tl=tl, dz=dz_v, z0=z0_v, tau_i=tau_v, window=window,
                s=sc["s"], tau=tau, tau_mode=tau_mode, off=off, nbits=nbits, n_pix_bits=N * (K - 1), colors=colors,
                n_blocks=n_blocks, max_degree=int(deg.max()), degrees=deg, A=A_pix, A_z=A_chn, init=init, Delta=Delta,
                lam=setup.lam, rowvar=rowvar, Qb=Qb, cb=cb, constb=constb, Qy=Qy, cy=cy, const_y=const,
                chords_with=chords_with, roots=roots, n_aux=V, n_spins=prob.n, compensate=bool(compensate), s2=s2_all,
                depth_max=int(tl["depth"].max()), n_pix_colours=n_pc, colouring_used=colouring_used, S0=sc["S0"])
    n_el = np.bincount(ch, minlength=len(sig))
    sel = n_el > 0
    infl = (s2_all + var_sum) / sig ** 2
    meta["comp_inflation"] = infl
    meta["comp_clamped_frac"] = float(np.mean((sig ** 2 - var_sum < comp_floor * sig ** 2)[sel])) if compensate else 0.0
    meta["eff_noise_ratio"] = float(np.mean(infl[sel] - 1.0))
    meta["eff_noise_ratio_max"] = float(np.max(infl[sel] - 1.0))
    meta["tau2_over_sigma2"] = float(np.mean(var_sum[sel] / sig[sel] ** 2))
    meta["tau_over_dz"] = float(np.mean(tau_v / dz_v))
    meta["dz_over_sigma"] = float(np.mean(dz_v / sig[ch]))
    return prob, meta


# ----------------------------------------------------------------------------------------
# direct energy (for tests/diagnostics), written from T, b, sigma only
# ----------------------------------------------------------------------------------------
def tree_energy(x, w, meta, problem=None):
    """Energy (1) *without* DW penalties for integer levels x (N,), w (V,), evaluated node by node from ``problem.T``."""
    problem = problem or meta["setup"].problem
    tl, lay, Delta = meta["tl"], meta["lay"], meta["Delta"]
    T = np.asarray(problem.T, float)
    x, w = np.asarray(x, float), np.asarray(w, float)
    z = meta["z0"] + meta["dz"] * w
    e = 0.0
    for v in range(tl["V"]):
        i = tl["chord"][v]
        s = sum(z[c] for c in tl["kids_node"][v]) + sum(T[i, lay["link_pix"][p]] * Delta * x[lay["link_pix"][p]]
                                                         for p in tl["kids_pix"][v])
        e += (z[v] - s) ** 2 / (2 * meta["tau_i"][v] ** 2)
    for v in np.flatnonzero(tl["root"]):
        i = tl["chord"][v]
        e += (problem.b[i] - z[v]) ** 2 / (2 * meta["s2"][i])
    ed = problem.edges
    e += 0.5 * meta["lam"] * Delta ** 2 * np.sum((x[ed[:, 0]] - x[ed[:, 1]]) ** 2)
    return float(e)


# ----------------------------------------------------------------------------------------
# sampling
# ----------------------------------------------------------------------------------------
def sample_tree(problem, cfg=None, key=None, backend: str = "jax", tau=None, *, tau_mode=None, Kz=None,
                n_chains=None, schedule=None, anneal=True, tempering=False, init_jitter: float = 1.0,
                init: str = "tikhonov", lam=None, A=None, A_z=None, built=None, return_bits=False, compensate=None,
                **build_kw) -> dict:
    """Sample / optimise the I-tree posterior; same arguments, result dict and diagnostics as ``sample_chain``
    (``init`` in {'tikhonov', 'random', 'half'}), plus ``depth_max``, ``colouring_used``, ``n_pix_colours``."""
    cfg = load_config(cfg) if not isinstance(cfg, dict) or "model" not in cfg else cfg
    key = jax.random.PRNGKey(0) if key is None else key
    m, sch = cfg["model"], cfg["schedule"]
    K = int(m["K"])
    Kz = int(m.get("Kz", 16) if Kz is None else Kz)
    tau = m.get("tau", 0.5) if tau is None else tau
    tau_mode = (m.get("tau_mode", "dz") if tau_mode is None else tau_mode)
    compensate = bool(m.get("chain_compensate", False) if compensate is None else compensate)
    lam = m.get("lam") if lam is None else lam
    A = m.get("A") if A is None else A
    C = int(sch["n_chains"] if n_chains is None else n_chains)
    schedule = sch["posterior"] if schedule is None else schedule
    t_all = time.perf_counter()
    if built is None:
        prob, meta = build_ising_tree(problem, K, lam, A, tau, Kz, tau_mode=tau_mode, A_z=A_z, compensate=compensate,
                                      **({"comp_floor": float(m["chain_comp_floor"])}
                                         if "chain_comp_floor" in m and "comp_floor" not in build_kw else {}),
                                      eps_max_factor=m.get("eps_max_factor", 1.2), **build_kw)
    else:
        prob, meta = built
    setup = meta["setup"]
    L, Delta = meta["L"], meta["Delta"]
    sm = IsingSampler(prob, meta["colors"], backend=backend)
    k_pt, k_an, k_j = jax.random.split(key, 3)

    init_mode = init
    rng = np.random.default_rng(int(jax.random.randint(k_j, (), 0, 2 ** 30)))
    x0 = np.tile(setup.x0, (C, 1)).astype(np.int64)
    if init == "random":
        x0 = rng.integers(0, K, x0.shape).astype(np.int64)
    elif init == "half":
        x0 = np.clip(x0 + np.rint(rng.normal(0, init_jitter, x0.shape)).astype(np.int64), 0, K - 1)
        x0[1::2] = rng.integers(0, K, x0[1::2].shape)
    elif init_jitter > 0:
        x0 = np.clip(x0 + np.rint(rng.normal(0, init_jitter, x0.shape)).astype(np.int64), 0, K - 1)
    w0 = _subtree_levels(x0, meta["tl"], Delta, dict(dz=meta["dz"], z0=meta["z0"]), Kz)
    init = np.concatenate([dw_encode(x0, K).reshape(C, -1), dw_encode(w0, Kz).reshape(C, -1)], axis=1).astype(bool)

    if isinstance(schedule, dict):
        n_w, n_s, n_t = schedule["n_warmup"], schedule["n_samples"], schedule["steps_per_sample"]
    else:
        n_w, n_s, n_t = schedule.n_warmup, schedule.n_samples, schedule.steps_per_sample
    if tempering:
        tp = sch["tempering"]
        betas = geometric_betas(tp["beta_min"], 1.0, int(tp["n_replicas"]))
        r = parallel_tempering(prob, meta["colors"], betas, n_w + n_s * n_t, max(n_t, 1), k_pt, n_chains=C,
                               init=init, backend=backend, sampler=sm)
        samples = r["samples"][:, -n_s:]
        energy_trace = r["energy_trace"][:, -n_s:, r["target"]]
        extra_pt = dict(swap_rate=r["swap_rate"])
        t_samp = r["time"]
    else:
        r = run_ising(prob, meta["colors"], dict(n_warmup=n_w, n_samples=n_s, steps_per_sample=n_t), C, k_pt,
                      beta=1.0, init=init, backend=backend, sampler=sm)
        samples, energy_trace, extra_pt, t_samp = r["samples"], r["energy_trace"], {}, r["time"]

    dec = decode_tree(samples, meta) if samples.nbytes < 4e8 else None
    if dec is None:
        parts = [decode_tree(samples[c], meta) for c in range(C)]
        dec = {k: np.stack([p[k] for p in parts]) for k in parts[0]}
    levels = dec["x"]
    inv_pix = invalid_fraction(dec["valid_pix"])
    inv_z = invalid_fraction(dec["valid_z"])
    wlev = dec["w"]
    z_edge = float(np.mean((wlev == Kz - 1) | ((wlev == 0) & (meta["z0"] > 0))))

    e_stride = samples.shape[1] // max(energy_trace.shape[1], 1) if energy_trace.shape[1] else 1
    ci, si = np.unravel_index(np.argmin(energy_trace), energy_trace.shape)
    best_bits, best_e = samples[ci, min(si * max(e_stride, 1), samples.shape[1] - 1)], float(energy_trace.min())
    t_an = 0.0
    if anneal:
        an = sch["anneal"]
        betas = geometric_betas(an["beta_min"], an["beta_max"], int(an["n_betas"]))
        ra = anneal_ising(prob, meta["colors"], betas, int(an["sweeps_per_beta"]), C, k_an, init=init,
                          backend=backend, sampler=sm)
        t_an = ra["time"]
        k = int(np.argmin(ra["best_energy"]))
        if float(ra["best_energy"][k]) < best_e:
            best_bits, best_e = ra["best"][k], float(ra["best_energy"][k])
    map_levels = decode_tree(best_bits, meta)["x"]

    s2c = sweeps_to_converge(levels, n_t)
    lv = np.asarray(levels, float)
    frozen = float(np.mean(np.all(lv.max(1) == lv.min(1), axis=0)))
    rh = np.asarray(split_rhat(lv), float)
    rfin = rh[np.isfinite(rh)]
    rhat_med = float(np.median(rfin)) if len(rfin) else float("inf")
    rhat_q95 = float(np.quantile(rh, 0.95)) if np.all(np.isfinite(rh)) else float("inf")
    converged = bool(np.all(np.isfinite(rh)) and rh.max() < 1.05)
    invalid_ok = bool(inv_pix < 1e-3 and inv_z < 1e-3)
    tot = time.perf_counter() - t_all
    out = make_result(levels, Delta, map_levels, energy_trace=energy_trace, n_blocks=sm.n_blocks,
                      n_spins=prob.n, max_degree=sm.max_degree, time=tot, invalid_frac=inv_pix,
                      tau=float(tau), tau_mode=tau_mode, tau_over_dz=meta["tau_over_dz"],
                      eff_noise_ratio=meta["eff_noise_ratio"], tau2_over_sigma2=meta["tau2_over_sigma2"],
                      compensate=compensate, comp_clamped_frac=meta["comp_clamped_frac"], n_aux=L,
                      chain_invalid_frac=inv_z, z_edge_frac=z_edge, dz_over_sigma=meta["dz_over_sigma"], Kz=Kz,
                      A=meta["A"], A_z=meta["A_z"], sweeps_to_converge=s2c, frozen_frac=frozen, rhat_med=rhat_med,
                      rhat_q95=rhat_q95, converged=converged, invalid_ok=invalid_ok, init=init_mode,
                      rhat_nonfinite_frac=float(np.mean(~np.isfinite(rh))), sweeps_total=n_w + n_s * n_t,
                      sample_time=t_samp, anneal_time=t_an, map_energy=best_e, build_time=sm.build_time,
                      backend=backend, lam=meta["lam"], eps_max=setup.eps_max, depth_max=meta["depth_max"],
                      colouring_used=meta["colouring_used"], n_pix_colours=meta["n_pix_colours"], **extra_pt)
    if return_bits:
        out["bits"] = samples
    return out
