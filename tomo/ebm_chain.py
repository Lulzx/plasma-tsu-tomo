r"""I-chain: domain-wall Ising tomography with a *local* chain of running partial sums per chord.

This is the project's main contribution (spec "Variant I-chain").  The dense Ising model has the
quadratic form  1/2 x^T (Delta^2 T^T W T) x  whose coupling graph joins every pair of pixels that share
a chord (degree ~ 500-2500 bits per spin).  I-chain removes ``T^T W T`` altogether.

Model
-----
Pixels  x_j in {0..K-1}  (eps_j = Delta x_j).  For chord i let j_1..j_n be the pixels with
T_ij > 0, ordered by distance along the chord (order is irrelevant for exactness, it only keeps the
chain physically local).  Add auxiliary levels  w_{i,k} in {0..Kz-1}, k = 1..n, with
z_{i,k} = dz_i * w_{i,k},  z_{i,0} = 0 (not a variable).  The energy is

    E = sum_i [ sum_k (z_{i,k} - z_{i,k-1} - T_{i j_k} Delta x_{j_k})^2 / (2 tau_i^2)
                + (b_i - z_{i,n})^2 / (2 sigma_i^2) ]
        + lam Delta^2 / 2 sum_<jk> (x_j - x_k)^2  +  A_pix * (#pixel DW violations)
        + A_z * (#chain DW violations)                                              (1)

Every term is a square of a *linear* function of the integer levels (x, w), so E is an exact
quadratic  1/2 y^T Q y - c^T y + const  in y = (x, w).  Each integer level is a thermometer
(domain-wall) code  y_v = sum_m u_{v,m}  (K-1 resp. Kz-1 bits), hence E is quadratic in bits = Ising.
Penalised violations (u_{m+1}=1, u_m=0) cost A per violation; any A > 0 makes every ground state
valid, and A only controls the invalid fraction at beta = 1.

Coupling graph (variable level; bits of two coupled variables are completely coupled, and the bits
of one variable form a clique, because 1/2 Q_vv (sum u)^2 contains all pair products):
    pixel j  --  its 4 grid neighbours   (Laplacian)
    pixel j_k -- z_{i,k}, z_{i,k-1}     (link k of chord i)
    z_{i,k}  --  z_{i,k-1}, z_{i,k+1}   (chain)
No pixel-pixel T^T T edges remain.  The degree of a pixel bit is bounded by
(K-2) + 4(K-1) + 2 c_j (Kz-1) (c_j = number of chords through pixel j: set by the camera geometry,
*not* by the grid size; this is an upper bound: border pixels have <4 Laplacian neighbours and a pixel at a chord
end has no z_{k-1}/z_{k+1} partner for its first/last link, so the measured max is 364 vs bound 387 on the default problem;
the pixel degree is therefore NOT independent of the grid in general, only of the grid for fixed chords per pixel) and a chain bit has degree (Kz-2) + 2 (Kz-1) + 2 (K-1)  (= 58 at K=8, Kz=16).

Structured colouring (proper by construction, verified by ``check_coloring`` on the real graph):
    variable class: pixel -> checkerboard parity (2 classes), z_{i,k} -> 2 + (k mod 2) (2 classes);
    bit colour = (variable class, bit index)  =>  2(K-1) + 2(Kz-1) blocks (= 44 at K=8, Kz=16).
Bits of one variable are a clique and bits of coupled variables are completely coupled, so a
pixel + z_k + z_{k-1} triangle is a clique of 2(Kz-1)+(K-1) = 37 bits: chi >= 37, i.e. 44 is within
7 colours of optimal.  The spec's 44 is therefore correct for I-chain (unlike the 2 blocks of Potts).

Noise scale tau and effective noise
-----------------------------------
If z were continuous, integrating out z_{i,1..n-1} (Gaussian) gives, for the data b_i given pixels,

    b_i = sum_k T_ij_k Delta x_jk + N(0, sigma_i^2 + sum_k tau_ik^2)      (= sigma_i^2 + n_i tau_i^2),   (2)

because the link residuals e_k = z_k - z_{k-1} - T Delta x are i.i.d. N(0, tau_k^2) and add up in z_n.  So tau is
not a "tolerance": it inflates the chord variance by n tau^2 and an unbiased posterior needs
sum_k tau_k^2 << sigma_i^2 (``eff_noise_ratio`` = mean_i sum_k tau_ik^2 / sigma_i^2 is reported).  This is
verified numerically (tests: exact DP over a fine z grid on a toy chord) and at sampler level (on tiny_problem the
chain posterior mean is closer to the *dense* model with sigma_eff than with sigma).

With *discrete* z there is a competing lower bound: z lives on a grid of spacing dz_k, and each link residual
carries a rounding error ~ dz_k/sqrt(12); if tau_k << dz_k the chain is rigid (moving a pixel one level changes a
partial sum by T Delta ~ 0.3 dz but z can only move in steps of dz): the sampler freezes at its warm start
(R-hat >> 1, "glassy").  Hence the usable window is  dz/2 <~ tau <~ sigma/sqrt(n).  It is only non-empty if
dz_k << sigma_i/sqrt(n): that is what the *local z window* below buys.  Parameterisation (``tau_mode``):
'dz' (default) tau_k = tau * dz_k (so tau is "rounding units"; tau ~ 0.5 is the smallest value that still moves),
'sigma' tau_k = tau * sigma_i, 'signal' tau_k = tau * s_i with s_i = max(b_i, cumulative Tikhonov signal, sigma_i)
(the config's old "relative to the chord signal scale"; it makes eff_noise_ratio ~ 10-30 and is NOT recommended).

z window.  A single range [0, zmax_i] with zmax_i ~ 1.2 b_i gives dz_i ~ 0.5-2 sigma_i at Kz = 16, so the chain
cannot be both rigid enough and unbiased (n tau^2 ~ 10-30 sigma^2).  The default ``window='local'`` instead puts,
per element, z_{i,k} in [z0_k, z0_k + 2H_k], centred on the warm-start partial sum S0_k (rounded Tikhonov levels),
with H_k = max(5 sd_k, 3 sigma_i) (``std_mult``, ``sigma_mult``) where sd_k is the std of S_k under the Gaussian
relaxation of the pixel posterior (Q^{-1}, plus rounding variance), clipped at z >= 0.  z0_k and dz_k are node-local
biases/scales, so hardware cost is unchanged; only data and the Tikhonov solution are used (never truth).  This gives
dz/sigma ~ 0.55 at Kz = 16 and ~0.25 at Kz = 32 (``dz_over_sigma``) and the fraction of samples sitting on a truncating
window edge (``z_edge_frac``) is <= 1.5% (typically < 0.5%).  ``window='chord'`` keeps the single-range variant.

Domain-wall penalty A.  The worst-case single-level bound of ``ebm_common.auto_A`` is 1e4-1e6 for the stiff chain and
destroys float32 local fields, so the default ('typical') is A_class = 12 + 1.05 max_v(|force at the warm start| + Q_vv)
per class (pixel: 100-8000, chain: 17-100 depending on tau).  Measured invalid fractions are < 1e-5 (target 1e-3).

Optional ``group=g``: one chain element per g consecutive pixels (n_i/g auxiliaries, g x less accumulated
rounding/noise: eff_noise_ratio / g at equal tau/dz).  The g pixels of a group become mutually coupled
(T T / tau^2 cross terms) so the structured colouring no longer applies (greedy colouring of the variable graph:
90 / 120 blocks at g = 2 / 4) and pixel degree grows.  Accuracy improves (peaked rel-L2 0.340 at g=4 vs 0.415 at g=1, both
tau/dz = 1) but it moves toward the dense model; g = 1 is the spec's chain.

Mixing.  Information about a pixel change reaches the data only by diffusing along the chain (z_k flips one level at a
time), ~ n_i^2 sweeps per chord, so single-chain Gibbs mixes slowly for small tau (R-hat med 1.1-2 and max = inf where
chains froze in different states).  Parallel tempering with the spec's 8 replicas (beta 0.2..1) has swap rates of
0-5% at 9000 spins, i.e. does not help; ~sqrt(n_dof) replicas would be needed.

``t_min > 0`` drops weak (pixel, chord) links, which makes the model an approximation of E_chain (default 0 = exact).

Diagnostics of ``sample_chain`` that qualify accuracy numbers: ``frozen_frac`` (fraction of pixels whose posterior
samples are constant within every chain: such runs are warm-start values, not sampler results), ``rhat_med`` / ``rhat_q95``,
``converged`` (finite max R-hat < 1.05), ``invalid_ok`` (pixel and chain invalid fractions < 1e-3).  All chains are jittered
(``init_jitter``) and ``init='random'`` starts from uniformly random pixel levels instead of the Tikhonov warm start; ``init='half'`` makes every odd chain random (overdispersed, honest R-hat).

Variance compensation (``compensate=True``, config ``model.chain_compensate``)
----------------------------------------------------------------------------
Uncompensated, the continuous marginal (2) has chord variance sigma_i^2 + S_i, S_i = sum_k tau_ik^2, i.e. the
posterior is over-dispersed/over-smoothed by S_i/sigma_i^2 (the bias that forced tau small).  Derivation: with
z_0 = 0 and link residuals e_k = z_k - z_{k-1} - m_k (m_k = T_ik Delta x_jk), z_n = sum_k m_k + sum_k e_k, the e_k
being independent N(0, tau_k^2) under the link terms of (1), so  int dz exp(-E_links) exp(-(b - z_n)^2 / (2 s^2))
is a Gaussian convolution:  prop. to  exp(-(b - sum m)^2 / (2 (s^2 + S))).  Choosing the endpoint variance

    s_i^2 = sigma_i^2 - S_i     (requires S_i < sigma_i^2)

makes the marginal exactly N(b_i; sum m, sigma_i^2) for ANY admissible tau, so tau (mixing) is decoupled from
bias.  Where S_i >= sigma_i^2 (n tau^2 too big) s_i^2 is clamped to ``comp_floor`` sigma_i^2 (0.1) and the
chord keeps an inflation (s_i^2 + S_i)/sigma_i^2 - 1 > 0 (meta['comp_inflation'] per chord, ``comp_clamped_frac``).
``eff_noise_ratio`` is the mean excess variance (total/sigma^2 - 1) *after* compensation; ``tau2_over_sigma2`` is
the raw S_i/sigma_i^2.  Cost: the endpoint precision dz^2/s_i^2 rises (a harder, stiffer last link), nothing else changes.
With *discrete* z the Gaussian convolution becomes a lattice sum, whose residual error is the periodic (Poisson-
summation) ripple of a discrete Gaussian, ~ 2 exp(-2 pi^2 tau^2/dz^2).  Measured by exact DP on a toy chord (n = 5,
tau = 0.03, sigma = 0.1, compensated; max deviation of -log marginal from (b - sum m)^2/(2 sigma^2) up to a constant,
and fitted effective variance / sigma^2): tau/dz = 0.5: 4e-2, 1.028; 0.75: 2e-4, 1.00005; 1.0: 2e-6, 1.0000001;
fine grid: <3e-5.  Uncompensated is off by 0.25 nats (variance ratio 1.45 = 1 + n tau^2/sigma^2) at every tau/dz.  So
rounding error is negligible for tau/dz >= 0.75 and ~3% in variance at 0.5.  ``test_compensation_toy_chord`` checks this.

Public API: ``build_ising_chain``, ``sample_chain``, ``chain_energy``, ``decode_chain``,
``chain_layout``, ``effective_noise_variance``.
"""
from __future__ import annotations

import time

import jax
import numpy as np
import scipy.sparse as sp

from .config import load_config
from .ebm_common import (dw_decode, dw_encode, invalid_fraction, make_result, prepare)
from .geometry import checkerboard
from .metrics import split_rhat, sweeps_to_converge
from .sampling import (IsingSampler, anneal_ising, bits_to_spins_qubo, check_coloring, geometric_betas,
                       greedy_coloring, parallel_tempering, run_ising)

__all__ = ["build_ising_chain", "sample_chain", "chain_energy", "decode_chain", "chain_layout",
           "effective_noise_variance"]


def effective_noise_variance(sigma, tau, n):
    """Variance of the chord data after marginalising a *continuous* chain: sigma^2 + n tau^2."""
    return np.asarray(sigma, float) ** 2 + np.asarray(n, float) * np.asarray(tau, float) ** 2


# ----------------------------------------------------------------------------------------
# chain layout
# ----------------------------------------------------------------------------------------
def chain_layout(problem, group: int = 1, t_min: float = 0.0) -> dict:
    """Chord-ordered chain structure.

    Returns dict with, over *elements* (links), chord-major: ``chord`` (L,), ``pos`` (L,) position in the
    chord, ``off`` (M+1,) element offsets, ``pix`` / ``t`` lists per link (``link_pix`` (P,), ``link_t``
    (P,), ``link_of`` (P,) element index of every (link, pixel) membership).  Pixels with T <= t_min are dropped.
    Pixel order along chord = projection of the pixel centre on the chord direction.
    """
    T = np.asarray(problem.T, float)
    M, N = T.shape
    g = problem.grid
    idx = np.flatnonzero(np.asarray(problem.mask).ravel())
    px = np.asarray(g.xc)[idx % g.n]
    py = np.asarray(g.yc)[idx // g.n]
    p0, p1 = np.asarray(problem.chords.p0, float), np.asarray(problem.chords.p1, float)
    chord, pos, off = [], [], [0]
    link_pix, link_t, link_of = [], [], []
    L = 0
    for i in range(M):
        nz = np.flatnonzero(T[i] > t_min)
        if len(nz):
            d = p1[i] - p0[i]
            s = (px[nz] - p0[i, 0]) * d[0] + (py[nz] - p0[i, 1]) * d[1]
            nz = nz[np.argsort(s, kind="stable")]
            n_el = -(-len(nz) // group)
            for k in range(n_el):
                chord.append(i)
                pos.append(k)
            for q, j in enumerate(nz):
                link_pix.append(j)
                link_t.append(T[i, j])
                link_of.append(L + q // group)
            L += n_el
        off.append(L)
    return dict(chord=np.asarray(chord, np.int64), pos=np.asarray(pos, np.int64), off=np.asarray(off, np.int64),
                link_pix=np.asarray(link_pix, np.int64), link_t=np.asarray(link_t, float),
                link_of=np.asarray(link_of, np.int64), L=L, group=group)


def _element_scales(setup, lay, Kz, tau, tau_mode, window, std_mult, sigma_mult):
    """Per-element z window (z0, dz), tau, and diagnostics; per-chord signal scale s.

    window='local' (default): z_{i,k} in [z0, z0 + 2H] centred on the warm-start (rounded Tikhonov)
    partial sum S0_k with half width H = max(std_mult * sd_k, sigma_mult * sigma_i), sd_k the std of the
    partial sum under the Gaussian relaxation of the *pixel* posterior (Q^{-1}, rounding variance added),
    clipped at z >= 0.  window='chord': one range [0, zmax_i], zmax_i = max(1.2 s_i, s_i + 4 sigma_i),
    s_i = max(b_i, Tikhonov chord total, sigma_i).  Only data / Tikhonov information is used.
    """
    p = setup.problem
    T = np.asarray(p.T, float)
    Delta, L = setup.Delta, lay["L"]
    b, sig = np.asarray(p.b, float), np.asarray(p.sigma, float)
    M = len(b)
    S = np.maximum(T @ (Delta * np.clip(setup.x0, 0, None)), T @ np.clip(setup.eps_tik, 0, None))
    s = np.maximum.reduce([b, S, sig])
    ch = lay["chord"]
    ends = np.searchsorted(lay["link_of"], np.arange(L), side="right") - 1      # last pixel of every element
    # cumulative partial sums / variances at pixel level along each chord
    P = len(lay["link_pix"])
    a = lay["link_t"] * Delta
    cp = ch[lay["link_of"]]
    poff = np.searchsorted(cp, np.arange(M + 1))
    S0 = np.zeros(P)
    var = np.zeros(P)
    Cov = np.linalg.inv(setup.Q) if window == "local" else None
    x0 = np.asarray(setup.x0, float)
    for i in range(M):
        lo, hi = poff[i], poff[i + 1]
        if hi == lo:
            continue
        pix, ai = lay["link_pix"][lo:hi], a[lo:hi]
        S0[lo:hi] = np.cumsum(ai * x0[pix])
        if Cov is not None:
            B = np.outer(ai, ai) * Cov[np.ix_(pix, pix)]
            var[lo:hi] = np.diag(B.cumsum(0).cumsum(1)) + np.cumsum(ai ** 2) / 12.0
    if window == "local":
        Sk, sd = S0[ends], np.sqrt(var[ends])
        H = np.maximum(std_mult * sd, sigma_mult * sig[ch])
        z0 = np.maximum(Sk - H, 0.0)
        dz = 2 * H / (Kz - 1)
    elif window == "chord":
        zmax = np.maximum(1.2 * s, s + 4 * sig)
        z0, dz = np.zeros(L), (zmax / (Kz - 1))[ch]
    else:
        raise ValueError(f"unknown window {window!r}")
    if tau_mode == "dz":
        tau_el = tau * dz
    elif tau_mode == "sigma":
        tau_el = tau * sig[ch]
    elif tau_mode == "signal":
        tau_el = tau * s[ch]
    else:
        raise ValueError(f"unknown tau_mode {tau_mode!r}")
    return dict(s=s, z0=z0, dz=dz, tau=tau_el, S0=S0[ends])


# ----------------------------------------------------------------------------------------
# build
# ----------------------------------------------------------------------------------------
def _bit_qubo(Qy, cy, const, nbits, A_per_var):
    """Expand integer-level quadratic 1/2 y^T Qy y - cy.y + const into a bit QUBO with DW penalties.

    Returns (Qb csr symmetric zero-diag, cb, const, rowvar (bits->variable), offsets).
    """
    nv = len(nbits)
    off = np.concatenate([[0], np.cumsum(nbits)])
    nb_tot = int(off[-1])
    rowvar = np.repeat(np.arange(nv), nbits)
    R = sp.csr_matrix((np.ones(nb_tot), (np.arange(nb_tot), rowvar)), shape=(nb_tot, nv))
    Qfull = (0.5 * (R @ Qy @ R.T)).tocsr()          # u^T Qfull u = 1/2 y^T Qy y
    d = Qfull.diagonal()
    Qb = (Qfull - sp.diags(d)).tocsr()
    cb = -np.asarray(cy)[rowvar] + d
    # domain-wall penalty A_v u_{m+1} (1 - u_m)
    a = np.flatnonzero(rowvar[:-1] == rowvar[1:])
    Av = np.asarray(A_per_var)[rowvar[a]]
    P = sp.csr_matrix((np.concatenate([-0.5 * Av, -0.5 * Av]), (np.concatenate([a, a + 1]), np.concatenate([a + 1, a]))),
                      shape=(nb_tot, nb_tot))
    Qb = (Qb + P).tocsr()
    Qb.eliminate_zeros()
    np.add.at(cb, a + 1, Av)
    return Qb, cb, float(const), rowvar, off


def _auto_A_levels(Qy, cy, Lv, margin):
    """Per-variable bound on the single-level energy change (generalises ebm_common.auto_A)."""
    d = Qy.diagonal()
    off = (Qy - sp.diags(d)).tocsr()
    pos = off.maximum(0) @ Lv
    neg = off.minimum(0) @ Lv
    hi = d * (Lv - 0.5) - cy + pos
    lo = 0.5 * d - cy + neg
    return margin * np.maximum(np.abs(hi), np.abs(lo))


def _choose_A(Qy, cy, Lv, y0, N, A, A_z, margin, floor):
    """Domain-wall penalties (pixel class, chain class).

    ``'bound'`` / numbers / ``'typical'`` (default for None).  The worst-case bound of
    ``ebm_common.auto_A`` ((K-1) sum |Q|) is 1e4-1e6 for the stiff chain (Q_zz ~ 2 dz^2/tau^2) and ruins
    float32 local fields, so the default is *typical*:  A = floor + margin * max_v max(|g_up|, |g_down|)
    where g are the single-level energy changes at the warm start y0 (this is the actual force that can
    pay for a violation) plus the stiffness Q_vv of the variable (one-level excursion).  ``floor`` = 12
    nats makes a single violation cost >= e^-12 relative weight.
    """
    d = Qy.diagonal()
    Qy_y = Qy @ y0
    g_up = d * (y0 + 0.5) - cy + (Qy_y - d * y0)
    g_dn = -(d * (y0 - 0.5) - cy + (Qy_y - d * y0))
    typ = floor + margin * (np.maximum(np.abs(g_up), np.abs(g_dn)) + d)
    bound = _auto_A_levels(Qy, cy, Lv, margin)
    out = []
    for given, sl in ((A, slice(0, N)), (A_z, slice(N, None))):
        if given is None or given == "typical":
            a = typ[sl]
        elif given == "bound":
            a = bound[sl]
        else:
            out.append(float(given))
            continue
        out.append(float(a.max()) if len(a) else 0.0)
    return out[0], out[1]


def build_ising_chain(problem, K, lam=None, A=None, tau=None, Kz=16, *, tau_mode="dz", group=1,
                      A_z=None, A_margin=1.05, setup=None, eps_max_factor=1.2, colouring="structured",
                      t_min=0.0, A_floor=12.0, window="local", std_mult=5.0, sigma_mult=3.0,
                      compensate=False, comp_floor=0.1):
    """Build the I-chain Ising model. Returns ``(IsingProblem, meta)``.

    ``lam=None`` -> discrepancy-principle Tikhonov lambda (via ``ebm_common.prepare``); ``A`` /
    ``A_z`` are the pixel / chain domain-wall penalties (None -> 'typical' warm-start estimate
    (see module docstring; the worst-case bound is only used if explicitly requested via ``ebm_common.auto_A``); numbers are also accepted); ``tau`` is
    in units set by ``tau_mode`` (see module docstring; default ``'dz'``: tau_k = tau * dz_k,
    ``tau=None`` -> 0.5).
    ``compensate=True`` replaces the endpoint variance sigma_i^2 by s_i^2 = max(sigma_i^2 - sum_k tau_ik^2,
    comp_floor sigma_i^2) (module docstring, "Variance compensation"); meta then has ``s2`` (per chord),
    ``comp_inflation`` (per chord total marginal variance / sigma_i^2 after clamping, >= 1) and ``comp_clamped_frac``.
    ``meta`` holds the decoding maps, scales and counts: ``setup, N, K, Kz, L, group, lay, dz, tau_i, zmax,
    bit offsets ('off', 'nbits'), 'n_pix_bits', 'colors', 'n_blocks', 'max_degree', 'degrees', 'A', 'A_z',
    'init' (bool warm start spins)``.
    """
    tau = 0.5 if tau is None else float(tau)
    if setup is None or setup.K != K:
        setup = prepare(problem, K, lam=lam, eps_max_factor=eps_max_factor)
    elif lam is not None and lam != setup.lam:
        setup = prepare(problem, K, lam=lam, eps_max_factor=eps_max_factor)
    N, Delta = setup.N, setup.Delta
    lay = chain_layout(problem, group=group, t_min=t_min)
    L = lay["L"]
    sc = _element_scales(setup, lay, Kz, tau, tau_mode, window, std_mult, sigma_mult)
    ch = lay["chord"]
    dz_l, tau_l, z0_l = sc["dz"], sc["tau"], sc["z0"]              # per element
    sig = np.asarray(problem.sigma, float)
    b = np.asarray(problem.b, float)
    nv = N + L

    # link residual  r_a = z0_a - z0_{a-1}[pos>0] + dz_a w_a - dz_{a-1} w_{a-1}[pos>0] - sum_{pix in link} t Delta x_j
    # (z_{i,0} = 0).  Per-element dz_a, z0_a are just node-local biases/scales.
    ar = np.arange(L)
    prev = lay["pos"] > 0
    rows = [ar, ar[prev], lay["link_of"]]
    cols = [N + ar, N + ar[prev] - 1, lay["link_pix"]]
    vals = [dz_l, -dz_l[ar[prev] - 1], -lay["link_t"] * Delta]
    Alk = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(L, nv))
    r0 = z0_l - np.where(prev, z0_l[np.maximum(ar - 1, 0)], 0.0)
    wl = 1.0 / tau_l ** 2
    Qy = (Alk.T @ sp.diags(wl) @ Alk).tocsr()
    cy = -(Alk.T @ (wl * r0))
    const = 0.5 * float(np.sum(wl * r0 ** 2))
    # data term (b_i - z_{i,n})^2/(2 sigma^2) on the last element of every chord
    last = lay["off"][1:] - 1
    has = lay["off"][1:] > lay["off"][:-1]
    chords_with = np.flatnonzero(has)
    lastel = last[has]
    dzc, sgc, bc, z0c = dz_l[lastel], sig[chords_with], b[chords_with], z0_l[lastel]
    var_sum = np.bincount(ch, weights=sc["tau"] ** 2, minlength=len(sig))      # sum_k tau_ik^2 per chord
    s2_all = sig ** 2
    if compensate:
        s2_all = np.maximum(sig ** 2 - var_sum, comp_floor * sig ** 2)
    s2c = s2_all[chords_with]
    dq = np.zeros(nv)
    dq[N + lastel] = dzc ** 2 / s2c
    cy[N + lastel] += dzc * (bc - z0c) / s2c
    const += 0.5 * float(np.sum((bc - z0c) ** 2 / s2c))
    Lap = sp.csr_matrix(problem.L) if sp.issparse(problem.L) else sp.csr_matrix(np.asarray(problem.L, float))
    QL = sp.bmat([[setup.lam * Delta ** 2 * Lap, None], [None, sp.csr_matrix((L, L))]], format="csr")
    Qy = (Qy + sp.diags(dq) + QL).tocsr()
    Qy = ((Qy + Qy.T) * 0.5).tocsr()

    nbits = np.concatenate([np.full(N, K - 1), np.full(L, Kz - 1)]).astype(np.int64)
    Lv = nbits.astype(float)
    x0 = np.asarray(setup.x0)
    w0 = _partial_sum_levels(x0, lay, Delta, sc, Kz, problem)
    y0 = np.concatenate([x0, w0]).astype(float)
    A_pix, A_chn = _choose_A(Qy, cy, Lv, y0, N, A, A_z, A_margin, A_floor)
    Av = np.concatenate([np.full(N, A_pix), np.full(L, A_chn)])
    Qb, cb, constb, rowvar, off = _bit_qubo(Qy, cy, const, nbits, Av)
    prob = bits_to_spins_qubo(Qb, cb, constb)

    # colouring
    n_bits = Qb.shape[0]
    bitidx = np.arange(n_bits) - off[rowvar]
    if group == 1 and colouring == "structured":
        cls = np.concatenate([checkerboard(problem.mask).astype(np.int64), 2 + (lay["pos"] % 2)])
        colors = cls[rowvar] * (max(K, Kz) + 1) + bitidx
    else:  # greedy on the variable graph then expand (bit index x variable class)
        Qv = (Qy - sp.diags(Qy.diagonal())).tocoo()
        e = np.stack([Qv.row, Qv.col], 1)
        e = e[e[:, 0] < e[:, 1]]
        cls = greedy_coloring(nv, e)
        colors = cls[rowvar] * (max(K, Kz) + 1) + bitidx
    if not check_coloring(prob.n, prob.edges, colors):
        raise AssertionError("colouring is not proper on the Ising graph")
    _, colors = np.unique(colors, return_inverse=True)
    n_blocks = int(colors.max() + 1)
    deg = prob.degrees

    # warm start (Tikhonov levels, consistent partial sums)
    init = np.concatenate([dw_encode(x0, K).ravel(), dw_encode(w0, Kz).ravel()]).astype(bool)

    meta = dict(setup=setup, N=N, K=K, Kz=Kz, L=L, group=group, lay=lay, dz=sc["dz"], z0=sc["z0"], tau_i=sc["tau"], window=window,
                s=sc["s"], tau=tau, tau_mode=tau_mode, off=off, nbits=nbits,
                n_pix_bits=N * (K - 1), colors=colors, n_blocks=n_blocks, max_degree=int(deg.max()),
                degrees=deg, A=A_pix, A_z=A_chn, init=init, Delta=Delta, lam=setup.lam, rowvar=rowvar,
                Qb=Qb, cb=cb, constb=constb, Qy=Qy, cy=cy, const_y=const, chords_with=chords_with,
                n_aux=L, n_spins=prob.n, compensate=bool(compensate), s2=s2_all)
    n_el = np.diff(lay["off"])
    sel = n_el > 0
    # eff_noise_ratio: total marginal chord variance (s_i^2 + sum_k tau_ik^2) / sigma_i^2 - 1 = excess variance
    # over the data noise.  Uncompensated: mean sum tau^2/sigma^2; compensated: ~0 unless clamped.
    infl = (s2_all + var_sum) / sig ** 2
    meta["comp_inflation"] = infl
    meta["comp_clamped_frac"] = float(np.mean((sig ** 2 - var_sum < comp_floor * sig ** 2)[sel])) if compensate else 0.0
    meta["eff_noise_ratio"] = float(np.mean(infl[sel] - 1.0))
    meta["eff_noise_ratio_max"] = float(np.max(infl[sel] - 1.0))
    meta["tau2_over_sigma2"] = float(np.mean(var_sum[sel] / sig[sel] ** 2))
    meta["tau_over_dz"] = float(np.mean(sc["tau"] / sc["dz"]))
    meta["dz_over_sigma"] = float(np.mean(sc["dz"] / sig[ch]))
    return prob, meta


def _link_matrix(lay, N, scale):
    """Sparse (L, N): row a sums scale * t_j x_j over the pixels of link a."""
    return sp.csr_matrix((lay["link_t"] * scale, (lay["link_of"], lay["link_pix"])), shape=(lay["L"], N))


def _partial_sum_levels(x, lay, Delta, sc, Kz, problem=None):
    """Partial-sum levels w (..., L) implied by pixel levels x (..., N) (rounded to the element's grid, clipped)."""
    x = np.asarray(x, float)
    lead = x.shape[:-1]
    xf = x.reshape(-1, x.shape[-1])
    inc = np.asarray((_link_matrix(lay, xf.shape[1], Delta) @ xf.T).T)       # (B, L)
    cs = np.cumsum(inc, axis=1)
    first = lay["off"][lay["chord"]]
    start = np.concatenate([np.zeros((cs.shape[0], 1)), cs], axis=1)[:, first]   # cs before the chord's first element
    z = cs - start
    w = np.clip(np.rint((z - sc["z0"]) / sc["dz"]), 0, Kz - 1).astype(np.int64)
    return w.reshape(lead + (lay["L"],))


# ----------------------------------------------------------------------------------------
# decode + direct energy (for tests/diagnostics)
# ----------------------------------------------------------------------------------------
def decode_chain(bits, meta):
    """bits (..., n) -> dict(x (...,N) levels, w (...,L) levels, valid_pix (...,N), valid_z (...,L))."""
    bits = np.asarray(bits)
    npb = meta["n_pix_bits"]
    N, K, Kz, L = meta["N"], meta["K"], meta["Kz"], meta["L"]
    x, vp = dw_decode(bits[..., :npb].reshape(bits.shape[:-1] + (N, K - 1)))
    w, vz = dw_decode(bits[..., npb:].reshape(bits.shape[:-1] + (L, Kz - 1)))
    return dict(x=x, w=w, valid_pix=vp, valid_z=vz)


def chain_energy(x, w, meta, problem=None):
    """Direct evaluation of the I-chain energy (1) *without* DW penalties for integer levels x (N,), w (L,)."""
    problem = problem or meta["setup"].problem
    lay, Delta = meta["lay"], meta["Delta"]
    x, w = np.asarray(x, float), np.asarray(w, float)
    ch = lay["chord"]
    dz, tau = meta["dz"], meta["tau_i"]
    z = meta["z0"] + dz * w
    zprev = np.where(lay["pos"] > 0, np.concatenate([[0.0], z[:-1]]), 0.0)
    inc = np.zeros(lay["L"])
    np.add.at(inc, lay["link_of"], lay["link_t"] * Delta * x[lay["link_pix"]])
    e = np.sum((z - zprev - inc) ** 2 / (2 * tau ** 2))
    cw = meta["chords_with"]
    zl = z[lay["off"][1:][cw] - 1]
    e += np.sum((np.asarray(problem.b)[cw] - zl) ** 2 / (2 * meta["s2"][cw]))
    ed = problem.edges
    e += 0.5 * meta["lam"] * Delta ** 2 * np.sum((x[ed[:, 0]] - x[ed[:, 1]]) ** 2)
    return float(e)


# ----------------------------------------------------------------------------------------
# sampling
# ----------------------------------------------------------------------------------------
def sample_chain(problem, cfg=None, key=None, backend: str = "jax", tau=None, *, tau_mode=None, Kz=None,
                 n_chains=None, schedule=None, anneal=True, tempering=False, init_jitter: float = 1.0, init: str = "tikhonov",
                 lam=None, A=None, A_z=None, group=1, built=None, return_bits=False, compensate=None, **build_kw) -> dict:
    """Sample / optimise the I-chain posterior; returns the standard result dict plus
    ``tau, tau_mode, tau_over_dz, eff_noise_ratio, n_aux, chain_invalid_frac, Kz, A, A_z, sweeps_to_converge,
    sweeps_total, build_time``.

    Warm start: Tikhonov levels for pixels + consistent partial sums for z; chain c>0 is jittered by
    round(N(0, init_jitter^2)) levels on pixels (z recomputed) so R-hat is meaningful.
    Posterior: beta=1, ``schedule`` (default cfg['schedule']['posterior']); ``tempering=True`` uses
    ``parallel_tempering`` (cfg['schedule']['tempering']) instead.  MAP: lowest-energy state found by
    geometric annealing (cfg anneal) or among the posterior samples (whichever has lower energy).
    ``built=(prob, meta)`` reuses a prebuilt model.
    """
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
        prob, meta = build_ising_chain(problem, K, lam, A, tau, Kz, tau_mode=tau_mode, group=group, A_z=A_z, compensate=compensate,
                                       **({"comp_floor": float(m["chain_comp_floor"])} if "chain_comp_floor" in m and "comp_floor" not in build_kw else {}),
                                       eps_max_factor=m.get("eps_max_factor", 1.2), **build_kw)
    else:
        prob, meta = built
    setup = meta["setup"]
    N, L, Delta = meta["N"], meta["L"], meta["Delta"]
    sm = IsingSampler(prob, meta["colors"], backend=backend)
    k_pt, k_an, k_j = jax.random.split(key, 3)

    init_mode = init
    # warm starts (C, n)
    rng = np.random.default_rng(int(jax.random.randint(k_j, (), 0, 2 ** 30)))
    x0 = np.tile(setup.x0, (C, 1)).astype(np.int64)
    if init == "random":
        x0 = rng.integers(0, K, x0.shape).astype(np.int64)
    elif init == "half":      # overdispersed starts: odd chains uniformly random, even chains jittered Tikhonov
        x0 = np.clip(x0 + np.rint(rng.normal(0, init_jitter, x0.shape)).astype(np.int64), 0, K - 1)
        x0[1::2] = rng.integers(0, K, x0[1::2].shape)
    elif init_jitter > 0:      # all chains jittered (a frozen un-jittered chain would give W=0 -> R-hat inf)
        x0 = np.clip(x0 + np.rint(rng.normal(0, init_jitter, x0.shape)).astype(np.int64), 0, K - 1)
    sc = dict(dz=meta["dz"], z0=meta["z0"])
    w0 = _partial_sum_levels(x0, meta["lay"], Delta, sc, Kz, problem)
    init = np.concatenate([dw_encode(x0, K).reshape(C, -1), dw_encode(w0, Kz).reshape(C, -1)], axis=1).astype(bool)

    npb = meta["n_pix_bits"]
    if isinstance(schedule, dict):
        n_w, n_s, n_t = schedule["n_warmup"], schedule["n_samples"], schedule["steps_per_sample"]
    else:
        n_w, n_s, n_t = schedule.n_warmup, schedule.n_samples, schedule.steps_per_sample
    if tempering:
        tp = sch["tempering"]
        betas = geometric_betas(tp["beta_min"], 1.0, int(tp["n_replicas"]))
        r = parallel_tempering(prob, meta["colors"], betas, n_w + n_s * n_t, max(n_t, 1), k_pt, n_chains=C,
                               init=init, backend=backend, sampler=sm)
        samples = r["samples"][:, -n_s:]  # (C, n_s, n) target replica after each swap round
        energy_trace = r["energy_trace"][:, -n_s:, r["target"]]
        extra_pt = dict(swap_rate=r["swap_rate"])
        t_samp = r["time"]
    else:
        r = run_ising(prob, meta["colors"], dict(n_warmup=n_w, n_samples=n_s, steps_per_sample=n_t), C, k_pt,
                      beta=1.0, init=init, backend=backend, sampler=sm)
        samples, energy_trace, extra_pt, t_samp = r["samples"], r["energy_trace"], {}, r["time"]

    dec = decode_chain(samples[..., :], meta) if samples.nbytes < 4e8 else None
    if dec is None:  # chunk over chains to bound memory
        parts = [decode_chain(samples[c], meta) for c in range(C)]
        dec = {k: np.stack([p[k] for p in parts]) for k in parts[0]}
    levels = dec["x"]                                           # (C,S,N)
    inv_pix = invalid_fraction(dec["valid_pix"])
    inv_z = invalid_fraction(dec["valid_z"])
    # fraction of samples where a z element sits on a *truncating* window edge (upper edge, or lower edge with z0 > 0)
    wlev = dec["w"]
    z_edge = float(np.mean((wlev == Kz - 1) | ((wlev == 0) & (meta["z0"] > 0))))

    # MAP candidates
    best_bits, best_e = None, np.inf
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
    map_levels = decode_chain(best_bits, meta)["x"]

    sps = n_t
    s2c = sweeps_to_converge(levels, sps)
    lv = np.asarray(levels, float)
    frozen = float(np.mean(np.all(lv.max(1) == lv.min(1), axis=0)))     # constant in every chain
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
                      compensate=compensate, comp_clamped_frac=meta["comp_clamped_frac"], n_aux=L, chain_invalid_frac=inv_z, z_edge_frac=z_edge, dz_over_sigma=meta["dz_over_sigma"], Kz=Kz,
                      A=meta["A"], A_z=meta["A_z"], sweeps_to_converge=s2c, frozen_frac=frozen, rhat_med=rhat_med, rhat_q95=rhat_q95,
                      converged=converged, invalid_ok=invalid_ok, init=init_mode, rhat_nonfinite_frac=float(np.mean(~np.isfinite(rh))),
                      sweeps_total=n_w + n_s * n_t, sample_time=t_samp, anneal_time=t_an,
                      map_energy=best_e, build_time=sm.build_time, backend=backend, lam=meta["lam"],
                      eps_max=setup.eps_max, **extra_pt)
    if return_bits:
        out["bits"] = samples
    return out
