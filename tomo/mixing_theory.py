r"""Linear-Gaussian theory of Gibbs mixing for the dense / I-chain / I-tree constructions.

Continuous relaxation: pixels x (emissivity units), auxiliary partial sums z unconstrained and continuous, same
energy as the Ising models (see ``ebm_chain`` / ``ebm_tree`` docstrings).  The joint law is N(0, P^{-1}) (after
centring) with sparse precision P over (x, z).  A deterministic-scan colour-block Gibbs sampler is then a linear
iteration  x_{t+1} = B x_t + noise  with

    P = D + L + U (variables ordered by colour class),   B = -(D + L)^{-1} U     (Gauss-Seidel),

(Amit 1991; Roberts & Sahu 1997; Goodman & Sokal 1989).  Because variables of one class share no edge, block GS
with the sampler's colour classes equals scalar GS in class order.  Consequences implemented here:

* ``gs_rho``            spectral radius rho(B); relaxation time 1/(1-rho) sweeps.
* ``iat_functional``    exact integrated autocorrelation time of a linear functional f^T x of the stationary chain:
                        tau_int(f) = g^T D g / (f^T P^{-1} f),  g = P^{-1} f   (independent of the scan order, because
                        only the symmetric part of D+L enters  sum_t f^T B^t S f = f^T P^{-1}(D+L)P^{-1} f).
* ``iat_sup``           sup_f tau_int(f) = 1 / lambda_min(D^{-1/2} P D^{-1/2}).
* ``da_rate``           two-block (all x | all z) rate = squared maximal canonical correlation between x and z
                        = lambda_max(P_xx^{-1} P_xz P_zz^{-1} P_zx) (Liu, Wong & Kong 1994) = 1 - 1/slowdown with
                        slowdown = max_v v^T P_xx v / v^T S_xx^{-1} v, S_xx^{-1} the marginal precision of x.
* ``simulate_gs``       the exact linear Gibbs chain (for validation).
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .baselines import discrepancy_lambda
from .ebm_chain import _element_scales, chain_layout
from .ebm_common import prepare
from .ebm_tree import _node_scales, tree_layout
from .geometry import checkerboard
from .sampling import greedy_coloring

__all__ = ["build_joint", "gs_split", "gs_rho", "sor_rho", "iat_functional", "iat_sup", "iat_pixel_sup", "da_rate",
           "da_slowdown_closed", "chord_sum_rayleigh", "simulate_gs", "iat_series", "relaxation_time",
           "iat_from_rho", "standard_functionals", "stiff_lambda"]


def stiff_lambda(problem) -> float:
    """Discrepancy-principle lambda (the stiffer prior used before the evidence lambda)."""
    return float(discrepancy_lambda(problem.T, problem.b, problem.sigma, problem.L))


# ----------------------------------------------------------------------------------------
# joint precision
# ----------------------------------------------------------------------------------------
def _sp(L):
    return sp.csr_matrix(L) if sp.issparse(L) else sp.csr_matrix(np.asarray(L, float))


def build_joint(problem, kind: str, *, K: int = 8, lam=None, tau: float = 0.75, tau_mode: str = "dz", Kz: int = 32,
                compensate: bool = True, comp_floor: float = 0.1, group: int = 1, setup=None) -> dict:
    """Joint precision of the continuous relaxation.

    kind : 'dense' (pixels only: P = T^T W T + lam L), 'chain', 'tree'.
    tau, tau_mode : 'dz' (tau_k = tau dz_k as in the samplers; dz from the local z windows at ``Kz``),
        'sigma' (tau_k = tau sigma_i), 'frac' (sum_k tau_ik^2 = tau * sigma_i^2: ``tau`` is the *admissibility
        fraction* f = S_i / sigma_i^2), 'abs' (tau_k = tau).
    lam : emissivity-unit smoothing (None = evidence lambda from ``prepare``; ``stiff_lambda`` gives the stiff one).
    group : chain only; g pixels per auxiliary (couples the pixels of a group).

    Returns dict(P csr (n,n), nx, nz, cls (n,) colour class in the sampler's sweep order, kind, mean/b-vector ``c``
    (so that the target mean is P^{-1} c), tau_el, s2 (per chord), n_el (per chord), lam, chord (per z), setup).
    """
    setup = prepare(problem, K, lam=lam) if (setup is None or (lam is not None and lam != setup.lam)) else setup
    T = np.asarray(problem.T, float)
    sig = np.asarray(problem.sigma, float)
    b = np.asarray(problem.b, float)
    M, N = T.shape
    Lap = _sp(problem.L)
    lam_ = setup.lam
    if kind == "dense":
        P = (sp.csr_matrix(T.T * (1 / sig ** 2) @ T) + lam_ * Lap).tocsr()
        Q = P.toarray()
        off = sp.csr_matrix(np.where(np.abs(Q - np.diag(np.diag(Q))) > 0, 1.0, 0.0))
        e = np.stack(sp.triu(off, 1).nonzero(), 1)
        cls = greedy_coloring(N, e)
        return dict(P=P, nx=N, nz=0, cls=cls, kind=kind, c=T.T @ (b / sig ** 2), lam=lam_, setup=setup,
                    chord=np.zeros(0, int), tau_el=np.zeros(0), s2=sig ** 2, n_el=np.zeros(M, int))
    if kind == "chain":
        lay = chain_layout(problem, group=group)
        sc = _element_scales(setup, lay, Kz, tau if tau_mode in ("dz", "sigma") else 1.0,
                             tau_mode if tau_mode in ("dz", "sigma") else "sigma", "local", 5.0, 3.0)
        V, ch = lay["L"], lay["chord"]
        ar = np.arange(V)
        prev = lay["pos"] > 0
        rows = np.concatenate([ar, ar[prev], lay["link_of"]])
        cols = np.concatenate([N + ar, N + ar[prev] - 1, lay["link_pix"]])
        vals = np.concatenate([np.ones(V), -np.ones(prev.sum()), -lay["link_t"]])
        A = sp.csr_matrix((vals, (rows, cols)), shape=(V, N + V))
        roots = (lay["off"][1:] - 1)[lay["off"][1:] > lay["off"][:-1]]
        pos = lay["pos"]
        cls = None if group != 1 else np.concatenate([checkerboard(problem.mask).astype(np.int64), 2 + (pos % 2)])
        tau_el = sc["tau"]
    elif kind == "tree":
        tl = tree_layout(problem)
        lay = tl["lay"]
        sc = _node_scales(setup, tl, Kz, tau if tau_mode in ("dz", "sigma") else 1.0,
                          tau_mode if tau_mode in ("dz", "sigma") else "sigma", "local", 5.0, 3.0)
        V, ch = tl["V"], tl["chord"]
        rows, cols, vals = [np.arange(V)], [N + np.arange(V)], [np.ones(V)]
        for v in range(V):
            for c in tl["kids_node"][v]:
                rows.append([v]); cols.append([N + c]); vals.append([-1.0])
            for p in tl["kids_pix"][v]:
                rows.append([v]); cols.append([lay["link_pix"][p]]); vals.append([-lay["link_t"][p]])
        A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(V, N + V))
        roots = np.flatnonzero(tl["root"])
        cls = None                                        # decided after P is assembled (needs the Laplacian edges)
        tau_el = sc["tau"]
    else:
        raise ValueError(kind)
    n_el = np.bincount(ch, minlength=M)
    if tau_mode == "frac":
        tau_el = np.sqrt(tau * sig[ch] ** 2 / np.maximum(n_el[ch], 1))
    elif tau_mode == "abs":
        tau_el = np.full(V, float(tau))
    wl = 1.0 / tau_el ** 2
    S = np.bincount(ch, weights=tau_el ** 2, minlength=M)
    s2 = np.maximum(sig ** 2 - S, comp_floor * sig ** 2) if compensate else sig ** 2
    Pl = (A.T @ sp.diags(wl) @ A).tocsr()
    dq = np.zeros(N + V)
    dq[N + roots] = 1.0 / s2[ch[roots]]
    PL = sp.bmat([[lam_ * Lap, None], [None, sp.csr_matrix((V, V))]], format="csr")
    P = (Pl + sp.diags(dq) + PL).tocsr()
    P = ((P + P.T) * 0.5).tocsr()
    if kind == "tree" or group != 1:
        Qv = (P - sp.diags(P.diagonal())).tocoo()
        e = np.stack([Qv.row, Qv.col], 1)
        e = e[e[:, 0] < e[:, 1]]
        if kind == "tree":
            ep = e[(e[:, 0] < N) & (e[:, 1] < N)]
            cp = greedy_coloring(N, ep)
            cls = np.concatenate([cp, int(cp.max() + 1) + tl["col"]])
            if np.any(cls[e[:, 0]] == cls[e[:, 1]]):      # structured colouring improper -> builder falls back to greedy
                cls = greedy_coloring(N + V, e)
        else:
            cls = greedy_coloring(N + V, e)
    c = np.zeros(N + V)
    c[N + roots] = b[ch[roots]] / s2[ch[roots]]
    return dict(P=P, nx=N, nz=V, cls=cls, kind=kind, c=c, lam=lam_, setup=setup, chord=ch, tau_el=tau_el, s2=s2,
                n_el=n_el, S=S, roots=roots, A=A, group=group)


# ----------------------------------------------------------------------------------------
# Gauss-Seidel structure and spectra
# ----------------------------------------------------------------------------------------
def gs_split(P, cls):
    """Permute to class order; returns (perm, Pp, D (diag vector), DL csc lower incl. diagonal, U csr strict upper)."""
    perm = np.argsort(cls, kind="stable")
    Pp = sp.csr_matrix(P)[perm][:, perm].tocsr()
    return perm, Pp, Pp.diagonal(), sp.tril(Pp, 0).tocsc(), sp.triu(Pp, 1).tocsr()


def gs_rho(P, cls, dense_max: int = 2500, tol: float = 1e-8) -> float:
    """Spectral radius of the block Gibbs (= GS in class order) mean iteration B = -(D+L)^{-1} U."""
    _, Pp, D, DL, U = gs_split(P, cls)
    n = Pp.shape[0]
    if n <= dense_max:
        B = -sla.solve_triangular(DL.toarray(), U.toarray(), lower=True)
        return float(np.max(np.abs(np.linalg.eigvals(B))))
    lu = spla.splu(DL, permc_spec="NATURAL")
    op = spla.LinearOperator((n, n), matvec=lambda v: -lu.solve(U @ v), dtype=float)
    w = spla.eigs(op, k=1, which="LM", tol=tol, return_eigenvectors=False, maxiter=20000, ncv=40)
    return float(np.abs(w[0]))


def sor_rho(P, cls, omega: float, dense_max: int = 2500) -> float:
    """Spectral radius of the over-relaxed (SOR) Gibbs mean iteration; omega=1 is plain Gibbs.

    Update x_c <- (1-w) x_c + w mu_c + sqrt(w(2-w)) sd_c xi leaves N(0,P^{-1}) invariant for w in (0,2)
    (Adler 1981; Barone & Frigessi 1990); B_w = (D + w L)^{-1} ((1-w) D - w U).
    """
    _, Pp, D, DL, U = gs_split(P, cls)
    Dm = sp.diags(D)
    Lw = (DL - Dm)
    A = (Dm + omega * Lw).toarray() if Pp.shape[0] <= dense_max else None
    if A is not None:
        B = sla.solve_triangular(A, ((1 - omega) * Dm - omega * U).toarray(), lower=True)
        return float(np.max(np.abs(np.linalg.eigvals(B))))
    lu = spla.splu((Dm + omega * Lw).tocsc(), permc_spec="NATURAL")
    n = Pp.shape[0]
    op = spla.LinearOperator((n, n), matvec=lambda v: lu.solve((1 - omega) * D * v - omega * (U @ v)), dtype=float)
    return float(np.abs(spla.eigs(op, k=1, which="LM", return_eigenvectors=False, maxiter=20000, ncv=40)[0]))


def relaxation_time(rho: float) -> float:
    return 1.0 / (1.0 - rho)


def iat_from_rho(rho: float) -> float:
    """IAT of the slowest (single-exponential) mode, (1+rho)/(1-rho)."""
    return (1.0 + rho) / (1.0 - rho)


def _solve(P, F):
    P = sp.csc_matrix(P)
    return spla.splu(P).solve(np.asarray(F, float))


def iat_functional(P, f) -> np.ndarray:
    """Exact IAT (sweeps) of linear functionals f (n,) or (n,k): g^T D g / f^T P^{-1} f, g = P^{-1} f."""
    f = np.asarray(f, float)
    F = f if f.ndim == 2 else f[:, None]
    G = _solve(P, F)
    D = sp.csr_matrix(P).diagonal()
    out = (G * G * D[:, None]).sum(0) / (F * G).sum(0)
    return out if f.ndim == 2 else float(out[0])


def iat_sup(P, dense_max: int = 2500) -> float:
    """sup over all linear functionals of the IAT = 1/lambda_min(D^{-1/2} P D^{-1/2})."""
    P = sp.csr_matrix(P)
    d = 1 / np.sqrt(P.diagonal())
    S = sp.diags(d) @ P @ sp.diags(d)
    if S.shape[0] <= dense_max:
        return float(1.0 / np.linalg.eigvalsh(S.toarray())[0])
    w = spla.eigsh(S.tocsc(), k=1, sigma=0, which="LM", return_eigenvectors=False)
    return float(1.0 / w[0])


def iat_pixel_sup(P, nx: int):
    """Worst IAT over functionals of the pixels only; returns (value, f (nx,)). Dense, for n up to a few thousand."""
    Pd = sp.csr_matrix(P).toarray()
    S = np.linalg.inv(Pd)
    D = np.diag(Pd)
    Sx = S[:, :nx]
    Mx = Sx.T @ (D[:, None] * Sx)
    w, V = sla.eigh(Mx, S[:nx, :nx])
    return float(w[-1]), V[:, -1]


def da_rate(P, nx: int, dense_max: int = 2500) -> float:
    """Two-block (x | z) data-augmentation rate = squared max canonical correlation between x and z."""
    P = sp.csr_matrix(P)
    if nx == P.shape[0]:
        return 0.0
    Pxx, Pxz, Pzz = P[:nx, :nx].tocsc(), P[:nx, nx:].tocsc(), P[nx:, nx:].tocsc()
    if P.shape[0] <= dense_max:
        X = np.linalg.solve(Pxx.toarray(), Pxz.toarray() @ np.linalg.solve(Pzz.toarray(), Pxz.toarray().T))
        return float(np.max(np.linalg.eigvals(X).real))
    lx, lz = spla.splu(Pxx), spla.splu(Pzz)
    op = spla.LinearOperator((nx, nx), matvec=lambda v: lx.solve(Pxz @ lz.solve(Pxz.T @ v)), dtype=float)
    return float(spla.eigs(op, k=1, which="LM", return_eigenvectors=False, maxiter=20000)[0].real)


def da_slowdown_closed(J: dict, marginal_precision) -> float:
    """1/(1-rho_DA) = lambda_max(P_xx S_xx^{-1}...) evaluated as max_v v^T P_xx v / v^T S^{-1} v (dense, small n).

    ``marginal_precision`` is the marginal precision of x implied by the model (lam L + T^T W_eff T); for a
    compensated continuous chain/tree this is the exact Gaussian posterior precision.
    """
    nx = J["nx"]
    Pxx = J["P"][:nx, :nx].toarray()
    w = sla.eigh(Pxx, np.asarray(marginal_precision), eigvals_only=True)
    return float(w[-1])


def chord_sum_rayleigh(J: dict, problem) -> np.ndarray:
    """Per-chord Rayleigh lower bound on the DA slowdown, in the direction v = S_xx t_i (chord-sum functional).

    R_i = v^T P_xx v / v^T S_xx^{-1} v = (t_i^T S P_xx S t_i) / (t_i^T S t_i), S = marginal covariance of x.
    For a chord of free pixels this is ~ Var_marg(sum) / sum_k tau_k^2: the heuristic 1/(1-F) with
    F = 1 - (tau-induced variance)/(posterior variance of the partial sum).  Returns (M,).
    """
    nx = J["nx"]
    P = J["P"].toarray()
    Sxx = np.linalg.inv(P)[:nx, :nx]
    Pxx = P[:nx, :nx]
    T = np.asarray(problem.T, float)
    G = Sxx @ T.T                                    # (nx, M)
    num = np.einsum("ij,ik,kj->j", G, Pxx, G)
    den = np.einsum("ij,ij->j", T.T, G)
    return num / den


# ----------------------------------------------------------------------------------------
# simulation of the linear Gibbs chain (validation)
# ----------------------------------------------------------------------------------------
def simulate_gs(P, cls, F, n_chains: int = 64, n_sweeps: int = 4000, seed: int = 0, omega: float = 1.0, x0=None) -> np.ndarray:
    """Exact (zero-mean) colour-block Gibbs on N(0, P^{-1}); returns functional traces (n_sweeps, n_chains, k).

    Chains start from the exact stationary law (dense Cholesky of P), so no burn-in is needed.  F is (n,k).
    One sweep updates the classes in increasing order (the sampler's colour order).  ``x0`` starts every chain at x0.
    """
    rng = np.random.default_rng(seed)
    P = sp.csr_matrix(P)
    n = P.shape[0]
    F = np.asarray(F, float)
    F = F if F.ndim == 2 else F[:, None]
    if x0 is None:
        Lc = np.linalg.cholesky(P.toarray())
        x = sla.solve_triangular(Lc.T, rng.standard_normal((n, n_chains)), lower=False)
    else:                                   # deterministic start (for mean-decay checks)
        x = np.tile(np.asarray(x0, float)[:, None], (1, n_chains))
    classes = [np.flatnonzero(cls == c) for c in np.unique(cls)]
    blocks = [(idx, P[idx].tocsr(), P.diagonal()[idx]) for idx in classes]
    out = np.empty((n_sweeps, n_chains, F.shape[1]))
    a = np.sqrt(omega * (2 - omega))
    for t in range(n_sweeps):
        for idx, Pc, d in blocks:
            r = Pc @ x
            x[idx] = x[idx] - omega * r / d[:, None] + a * rng.standard_normal((len(idx), n_chains)) / np.sqrt(d)[:, None]
        out[t] = (F.T @ x).T
    return out


def iat_series(y, c: float = 6.0, known_mean: float | None = None) -> float:
    """Integrated autocorrelation time of y (T,) or (T, C) (chains pooled in the autocovariance) with Sokal's
    automatic window (smallest M with M >= c * tau(M))."""
    y = np.asarray(y, float)
    if y.ndim == 1:
        y = y[:, None]
    mu = y.mean() if known_mean is None else known_mean
    y = y - mu
    Tn, C = y.shape
    nfft = 1 << (2 * Tn - 1).bit_length()
    f = np.fft.rfft(y, nfft, axis=0)
    ac = np.fft.irfft(f * np.conj(f), nfft, axis=0)[:Tn] / (Tn - np.arange(Tn))[:, None]
    ac = ac.mean(1)
    rho = ac / ac[0]
    tau = 1 + 2 * np.cumsum(rho[1:])
    tau = np.concatenate([[1.0], tau])
    M = np.arange(Tn)
    ok = np.flatnonzero(M >= c * tau)
    m = ok[0] if len(ok) else Tn - 1
    return float(tau[m])


def standard_functionals(problem, J: dict) -> dict:
    """Pixel functionals (columns over the joint vector, zero on z): 'centre' pixel, 'total' (sum), 'pc1' (top
    posterior principal component), 'worst' (worst pixel functional: exact sup of IAT over pixel functionals)."""
    nx, n = J["nx"], J["P"].shape[0]
    Sxx = np.linalg.inv(J["P"].toarray())[:nx, :nx]
    g = problem.grid
    idx = np.flatnonzero(np.asarray(problem.mask).ravel())
    cx, cy = np.asarray(g.xc)[idx % g.n], np.asarray(g.yc)[idx // g.n]
    centre = int(np.argmin((cx - cx.mean()) ** 2 + (cy - cy.mean()) ** 2))
    out = {}
    e = np.zeros(n); e[centre] = 1
    out["centre"] = e
    t = np.zeros(n); t[:nx] = 1.0 / np.sqrt(nx)
    out["total"] = t
    w, V = np.linalg.eigh(Sxx)
    p = np.zeros(n); p[:nx] = V[:, -1]
    out["pc1"] = p
    _, fw = iat_pixel_sup(J["P"], nx)
    q = np.zeros(n); q[:nx] = fw / np.linalg.norm(fw)
    out["worst"] = q
    return out


def null_space_bound(J: dict, problem) -> float:
    """Two-block slowdown restricted to null(T): 1 + max_{v in null T} v^T Lambda v / (lam v^T L v).

    Lambda = P_xx - lam L is the precision the z-couplings add to x (diagonal for the chain).  In a direction the
    data do not see, the marginal precision is the prior's lam v^T L v alone, so this is a *lower bound* on
    1/(1-rho_DA) = max_v v^T P_xx v / v^T S_xx^{-1} v, and it is the mechanism that dominates in practice.
    """
    nx = J["nx"]
    T = np.asarray(problem.T, float)
    Pxx = J["P"][:nx, :nx].toarray()
    Lap = _sp(problem.L).toarray()
    Lam = Pxx - J["lam"] * Lap
    _, s, Vt = np.linalg.svd(T, full_matrices=True)
    Nb = Vt[np.sum(s > 1e-9 * s[0]):].T
    A = Nb.T @ Lam @ Nb
    B = J["lam"] * (Nb.T @ Lap @ Nb)
    return float(1.0 + sla.eigh(0.5 * (A + A.T), 0.5 * (B + B.T), eigvals_only=True)[-1])
