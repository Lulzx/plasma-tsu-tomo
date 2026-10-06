"""Continuous positivity / bound comparators for the discrete Potts posterior (ablation of its gain).

Reviewer question: is the Potts gain over the Gaussian posterior due to positivity, to the upper bound
eps_max, or to discreteness?  With the SAME lambda and smoothness prior the posteriors are

    Gaussian            p(e) ~ N(P^-1 h, P^-1),    P = A^T A + lam L,  h = A^T y   (A = W^{1/2} T, y = W^{1/2} b)
    truncated >= 0      the same density restricted to e >= 0
    truncated [0, ub]   restricted to 0 <= e <= ub  (ub = eps_max: continuous limit K -> inf of the Potts model)
    Potts K             the same density restricted to the K-point grid {0, Delta, ..., ub}

Contents
--------
``truncated_gaussian_sample(P, h, lower, upper, ...)``  generic exact sampler (dense precision).
``truncated_gaussian_posterior(problem, lam, lower, upper, ...)``  tomography wrapper -> result dict.
``gaussian_posterior_eps(problem, lam)``  exact unconstrained posterior (mean/std/cov).
``log_laplace(problem, lam_f=None)``  log-Gaussian + Laplace positivity baseline (OUR RE-IMPLEMENTATION
    of the idea of Ueda & Nishiura, arXiv:2410.11454; not their code and not their exact model).

Sampler
-------
Exact sweeps, all keeping the truncated Gaussian invariant:
* ``'gibbs'``  systematic-scan coordinate Gibbs.  eps_j | rest ~ N(m_j, 1/P_jj) truncated to [lo, hi],
  m_j = (h_j - sum_{k!=j} P_jk eps_k)/P_jj.  P is dense (every pair sharing a chord is coupled, so a colouring
  of its non-zero pattern has ~N colours and gives no parallelism); one sweep is a jitted lax.fori_loop over
  coordinates, vectorised over chains.
* ``'ess'``    elliptical slice sampling (Murray, Adams, MacKay 2010) with the unconstrained Gaussian as
  the "prior" and the box indicator as the "likelihood": exact, but on the tomography problem (many
  active bounds in N=806 dimensions) the slice shrinks to tiny angles and it does not mix (R-hat >> 1
  in our tests) -- kept for small problems / tests only.
* ``'mixed'``  one Gibbs sweep + one ESS update per iteration (no gain over 'gibbs' measured).
Default ``'gibbs'``: on the default problem 300 sweeps give R-hat ~1.03 even from over-dispersed starts.
All are exact for any schedule, so mixing only affects R-hat/ESS, never the target.
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from scipy.optimize import minimize  # noqa: F401  (kept for users extending the Laplace fit)

from .metrics import split_rhat

__all__ = ["truncated_gaussian_sample", "truncated_gaussian_posterior", "gaussian_posterior_eps",
           "precision_system", "log_laplace"]


# ----------------------------------------------------------------------------------------
# problem -> (P, h)
# ----------------------------------------------------------------------------------------
def _dense_L(problem):
    L = problem.L
    return L.toarray() if sp.issparse(L) else np.asarray(L, float)


def precision_system(problem, lam):
    """(P, h) with P = T^T W T + lam L (emissivity units, same as Potts/Tikhonov), h = T^T W b."""
    T = np.asarray(problem.T, float)
    w = 1.0 / np.asarray(problem.sigma, float) ** 2
    TtW = T.T * w
    P = TtW @ T + lam * _dense_L(problem)
    return 0.5 * (P + P.T), TtW @ np.asarray(problem.b, float)


def gaussian_posterior_eps(problem, lam) -> dict:
    """Exact unconstrained Gaussian posterior N(P^-1 h, P^-1): 'mean', 'std', 'cov'."""
    t0 = time.perf_counter()
    P, h = precision_system(problem, lam)
    cf = sla.cho_factor(P, lower=True)
    mean = sla.cho_solve(cf, h)
    cov = sla.cho_solve(cf, np.eye(len(h)))
    cov = 0.5 * (cov + cov.T)
    return {"mean": mean, "cov": cov, "std": np.sqrt(np.clip(np.diag(cov), 0, None)),
            "lam": float(lam), "time": time.perf_counter() - t0}


# ----------------------------------------------------------------------------------------
# generic truncated-Gaussian sampler
# ----------------------------------------------------------------------------------------
def _trunc_std_normal(key, a, b):
    """Draw z ~ N(0,1) truncated to [a, b] elementwise (float32, tail-safe).

    jax.random.truncated_normal inverts the CDF directly and returns garbage once ndtr(a) rounds to 1 (a > ~5),
    which is exactly the regime of pixels whose conditional mean is far below the lower bound.  Here the
    interval is reflected so that its nearer end is >= 0 away from the origin, then sampled by inverting
    the survival function (a <= 8) or by the exponential approximation z = a + Exp(1)/a (a > 8).
    """
    from jax.scipy.special import ndtr, ndtri
    flip = b < 0
    a2 = jnp.where(flip, -b, a)
    b2 = jnp.where(flip, -a, b)
    ku, ke = jax.random.split(key)
    u = jax.random.uniform(ku, a.shape, jnp.float32, 1e-7, 1 - 1e-7)
    # tail branch (a2 > 0)
    pa, pb = ndtr(-jnp.minimum(a2, 8.0)), ndtr(-jnp.minimum(b2, 40.0))
    z_t = -ndtri(jnp.clip(pa - u * (pa - pb), 1e-37, 1.0))
    e = jax.random.exponential(ke, a.shape, jnp.float32)
    z_e = a2 + e / jnp.maximum(a2, 1.0)
    z_tail = jnp.where(a2 > 8.0, z_e, z_t)
    # body branch (a2 <= 0)
    qa, qb = ndtr(a2), ndtr(jnp.minimum(b2, 40.0))
    z_b = ndtri(jnp.clip(qa + u * (qb - qa), 1e-37, 1 - 1e-7))
    z = jnp.where(a2 > 0, z_tail, z_b)
    z = jnp.clip(z, a2, b2)
    return jnp.where(flip, -z, z)


def _make_runner(P, h, lo, hi, method, n_warmup, n_samples, thin, n_ess):
    """Compile the (key, x0 (C,N)) -> samples (C,S,N) function.  P, h, lo, hi are already scaled."""
    N = len(h)
    Pj = jnp.asarray(P, jnp.float32)
    hj = jnp.asarray(h, jnp.float32)
    dj = jnp.diag(Pj)
    sd = 1.0 / jnp.sqrt(dj)
    lo_j = jnp.asarray(lo, jnp.float32)
    hi_j = jnp.asarray(hi, jnp.float32)
    cov = np.linalg.inv(P)
    cov = 0.5 * (cov + cov.T)
    Lc = jnp.asarray(np.linalg.cholesky(cov), jnp.float32)
    mu = jnp.asarray(np.linalg.solve(P, h), jnp.float32)

    def gibbs(key, x):  # x (C, N)
        C = x.shape[0]

        def body(j, x):
            pj = Pj[j]
            r = x @ pj - pj[j] * x[:, j]
            m = (hj[j] - r) / dj[j]
            a = (lo_j[j] - m) / sd[j]
            b = (hi_j[j] - m) / sd[j]
            z = _trunc_std_normal(jax.random.fold_in(key, j), a, b)
            return x.at[:, j].set(jnp.clip(m + sd[j] * z, lo_j[j], hi_j[j]))
        return jax.lax.fori_loop(0, N, body, x)

    def ess_one(key, x):  # x (N,)
        k1, k2, k3 = jax.random.split(key, 3)
        nu = Lc @ jax.random.normal(k1, (N,), jnp.float32)
        y = x - mu
        th0 = jax.random.uniform(k2, (), jnp.float32, 0.0, 2 * jnp.pi)

        def inside(th):
            xp = mu + y * jnp.cos(th) + nu * jnp.sin(th)
            return jnp.all((xp >= lo_j) & (xp <= hi_j)), xp

        def cond(s):
            ok, _ = inside(s[2])
            return (~ok) & (s[4] < 60)

        def step(s):
            tmin, tmax, th, k, it = s
            tmin = jnp.where(th < 0, th, tmin)
            tmax = jnp.where(th >= 0, th, tmax)
            k, kk = jax.random.split(k)
            th = jax.random.uniform(kk, (), jnp.float32, tmin, tmax)
            return (tmin, tmax, th, k, it + 1)
        # first proposal at theta = th0, bracket [th0 - 2pi, th0] (Murray et al. 2010)
        s = jax.lax.while_loop(cond, step, (th0 - 2 * jnp.pi, th0, th0, k3, 0))
        ok, xp = inside(s[2])
        return jnp.where(ok, xp, x)  # if the cap was hit (never observed) stay put: still invariant

    def ess(key, x):
        keys = jax.random.split(key, x.shape[0])
        return jax.vmap(ess_one)(keys, x)

    def update(key, x):
        k1, k2 = jax.random.split(key)
        if method in ("gibbs", "mixed"):
            x = gibbs(k1, x)
        if method in ("ess", "mixed"):
            for i in range(n_ess):
                x = ess(jax.random.fold_in(k2, i), x)
        return x

    def run(key, x0):
        kw, ks = jax.random.split(key)
        x = jax.lax.fori_loop(0, n_warmup, lambda i, c: update(jax.random.fold_in(kw, i), c), x0)

        def outer(c, k):
            c = jax.lax.fori_loop(0, thin, lambda i, cc: update(jax.random.fold_in(k, i), cc), c)
            return c, c
        _, S = jax.lax.scan(outer, x, jax.random.split(ks, n_samples))  # (S, C, N)
        return jnp.swapaxes(S, 0, 1)

    return jax.jit(run)


def truncated_gaussian_sample(P, h, lower=0.0, upper=None, n_samples=500, n_warmup=200, n_chains=4,
                              thin=1, key=0, x0=None, method="gibbs", n_ess=1, frac_random=0.5) -> dict:
    """Exact samples of N(P^-1 h, P^-1) truncated to the box [lower, upper]^N (dense precision P).

    ``lower``/``upper`` are scalars or (N,) arrays (``None`` = unbounded above).  Returns dict with
    'samples' (C, S, N), 'mean', 'std', 'rhat' (split R-hat per coordinate), 'time' (sampling seconds,
    excluding compilation), 'compile_time'.  Chains start from ``x0`` (clipped into the box, default the
    unconstrained mean); ``frac_random`` of the chains start from uniform draws over the box (or over
    [0, 2*max mean] if unbounded above) for an honest R-hat.
    """
    P = np.asarray(P, float)
    h = np.asarray(h, float)
    N = len(h)
    lo = np.broadcast_to(np.asarray(-np.inf if lower is None else lower, float), (N,)).copy()
    hi = np.broadcast_to(np.asarray(np.inf if upper is None else upper, float), (N,)).copy()
    mean0 = np.linalg.solve(P, h)
    # rescale x = s * z (z O(1)) so float32 is comfortable
    m0 = float(np.max(np.abs(mean0)))
    fin = np.concatenate([np.abs(b[np.isfinite(b) & (np.abs(b) < 10 * m0)]) for b in (lo, hi)] + [[m0]])
    s = float(max(np.max(fin), 1e-30))
    Ps, hs = P * s * s, h * s
    los, his = np.clip(lo / s, -1e4, None), np.clip(hi / s, None, 1e4)  # |bound| > 5e3 scales = unbounded
    if x0 is None:
        x0 = mean0
    x0 = np.clip(np.broadcast_to(np.asarray(x0, float), (N,)), lo, hi) / s
    rng = np.random.default_rng(int(key) if np.isscalar(key) else 0)
    X0 = np.tile(x0, (n_chains, 1))
    n_rand = int(np.ceil(frac_random * n_chains)) if n_chains > 1 else 0
    if n_rand:
        ref = 2 * max(np.max(mean0), 1e-30) / s
        top = np.where(his < 5e3, his, ref)
        bot = np.where(los > -5e3, los, -ref)
        X0[:n_rand] = rng.uniform(bot, top, size=(n_rand, N))
    k = jax.random.PRNGKey(int(key)) if np.isscalar(key) else key
    t0 = time.perf_counter()
    run = _make_runner(Ps, hs, los, his, method, int(n_warmup), int(n_samples), int(thin), int(n_ess))
    lowered = run.lower(k, jnp.asarray(X0, jnp.float32)).compile()
    ct = time.perf_counter() - t0
    t0 = time.perf_counter()
    S = np.asarray(lowered(k, jnp.asarray(X0, jnp.float32)), np.float64) * s
    t = time.perf_counter() - t0
    flat = S.reshape(-1, N)
    return {"samples": S, "mean": flat.mean(0), "std": flat.std(0), "rhat": split_rhat(S), "time": t,
            "compile_time": ct, "method": method}


def truncated_gaussian_posterior(problem, lam, lower=0.0, upper=None, n_samples=500, n_warmup=200,
                                 n_chains=4, key=0, seed=None, thin=1, method="gibbs", n_ess=1,
                                 frac_random=0.5) -> dict:
    """Exact continuous posterior N(P^-1 h, P^-1), P = A^T A + lam L, truncated to [lower, upper]^N.

    ``lower=0, upper=None``: positivity only.  ``lower=0, upper=eps_max``: positivity + the Potts bound
    (continuous limit of the Potts model).  ``lam`` is in emissivity units (same as Potts/Tikhonov).
    Returns 'mean', 'std', 'samples' (C, S, N) emissivity, 'rhat', 'time', 'lam', 'lower', 'upper'.
    """
    P, h = precision_system(problem, lam)
    r = truncated_gaussian_sample(P, h, lower, upper, n_samples=n_samples, n_warmup=n_warmup,
                                  n_chains=n_chains, thin=thin, key=seed if seed is not None else key,
                                  method=method, n_ess=n_ess, frac_random=frac_random)
    r.update(lam=float(lam), lower=lower, upper=upper)
    return r


# ----------------------------------------------------------------------------------------
# log-Gaussian + Laplace  (our re-implementation of the idea of Ueda & Nishiura, arXiv:2410.11454)
# ----------------------------------------------------------------------------------------
def _laplace_fit(A, y, Lr, lam_f, mu, f0, n_iter=60):
    """MAP of  1/2||y - A e^f||^2 + lam_f/2 (f-mu)^T Lr (f-mu)  by damped Newton (exact Hessian, fall back to
    Gauss-Newton when indefinite) with backtracking.  Returns f, Hessian-GN, objective."""
    def obj(f):
        e = np.exp(f)
        r = A @ e - y
        d = f - mu
        return 0.5 * r @ r + 0.5 * lam_f * d @ Lr @ d

    f = f0.copy()
    fo = obj(f)
    AtA = A.T @ A
    for _ in range(n_iter):
        e = np.exp(f)
        r = A @ e - y
        g = e * (A.T @ r) + lam_f * Lr @ (f - mu)
        GN = (e[:, None] * AtA) * e[None, :] + lam_f * Lr
        Hn = GN + np.diag(e * (A.T @ r))
        try:
            np.linalg.cholesky(Hn)
            Hm = Hn
        except np.linalg.LinAlgError:
            Hm = GN
        step = np.linalg.solve(Hm, g)
        t = 1.0
        while t > 1e-6:
            fn = f - t * step
            fo_new = obj(fn)
            if fo_new <= fo - 1e-4 * t * (g @ step):
                break
            t *= 0.5
        else:
            break
        done = abs(fo - fo_new) < 1e-10 * max(abs(fo), 1.0) and np.max(np.abs(t * step)) < 1e-6
        f, fo = fn, fo_new
        if done:
            break
    e = np.exp(f)
    GN = (e[:, None] * AtA) * e[None, :] + lam_f * Lr
    return f, GN, fo


def log_laplace(problem, lam_f=None, lam_grid=None, n_samples=400, seed=0, floor_rel=1e-3, ridge=1e-3,
                lam_ref=None) -> dict:
    """Log-Gaussian positivity baseline: eps = exp(f), smoothness prior on f, MAP + Laplace.

    OUR RE-IMPLEMENTATION OF THE IDEA of Ueda & Nishiura (arXiv:2410.11454, log-GP with Laplace
    approximation); it is not their code and uses a graph-Laplacian (smoothness) prior on f instead of
    their GP kernel.  Model: y = A exp(f) + noise (whitened), f ~ N(mu 1, (lam_f (L + ridge I))^-1), with mu the log
    of a data-driven emissivity scale.  The posterior is approximated by N(f_MAP, H^-1), H the Gauss-Newton
    Hessian at the MAP; 'mean' is the posterior median/MAP exp(f_MAP) (the Laplace posterior mean exp(f+v/2) is inflated by the
    heavy lognormal tail in poorly constrained pixels and is returned as 'mean_sample'); 'std' and 'samples' come from
    exp(f) draws (strictly positive).

    ``lam_f=None`` picks lam_f on ``lam_grid`` by the Laplace marginal likelihood (no ground truth; the
    penalty acts on log-emissivity so the Gaussian lambda does not transfer).  Returns dict with 'mean',
    'std', 'map', 'samples' (1, S, N) (iid Laplace draws -- NOT MCMC, so no R-hat), 'lam_f', 'time'.
    """
    t0 = time.perf_counter()
    T = np.asarray(problem.T, float)
    w = 1.0 / np.asarray(problem.sigma, float)
    A, y = T * w[:, None], np.asarray(problem.b, float) * w
    L = _dense_L(problem)
    N = L.shape[0]
    Lr = L + ridge * (np.trace(L) / N) * np.eye(N)
    P0, h0 = A.T @ A + (lam_ref if lam_ref is not None else 1e-3 * np.linalg.norm(A, 2) ** 2) * L, A.T @ y
    tik = np.linalg.solve(P0 + 1e-8 * np.trace(P0) / N * np.eye(N), h0)
    scale = max(float(np.max(tik)), 1e-30)
    f0 = np.log(np.maximum(tik, floor_rel * scale))
    mu = float(np.log(max(float(np.median(np.maximum(tik, floor_rel * scale))), 1e-30)))
    s2 = np.linalg.norm(A, 2) ** 2 * scale ** 2
    grid = [lam_f] if lam_f is not None else (np.asarray(lam_grid, float) if lam_grid is not None
                                              else s2 * np.logspace(-8, 0, 9))
    ld_Lr = 2 * np.log(np.diag(np.linalg.cholesky(Lr))).sum()
    best = None
    for lf in grid:
        f, GN, fo = _laplace_fit(A, y, Lr, lf, mu, f0)
        # Laplace evidence: -log p ~ fo + 1/2 logdet(H) - 1/2 logdet(lf Lr)
        try:
            ld_h = 2 * np.log(np.diag(np.linalg.cholesky(GN))).sum()
        except np.linalg.LinAlgError:
            continue
        ld_p = N * np.log(lf) + ld_Lr
        nlp = fo + 0.5 * ld_h - 0.5 * ld_p
        if not np.isfinite(nlp):
            continue
        if best is None or nlp < best[0]:
            best = (nlp, lf, f, GN)
    nlp, lf, f, GN = best
    cov = np.linalg.inv(GN)
    cov = 0.5 * (cov + cov.T)
    Lc = np.linalg.cholesky(cov + 1e-12 * np.eye(N))
    rng = np.random.default_rng(seed)
    F = f[None] + rng.standard_normal((int(n_samples), N)) @ Lc.T
    E = np.exp(F)
    return {"mean": np.exp(f), "std": E.std(0), "map": np.exp(f), "mean_sample": E.mean(0), "samples": E[None], "lam_f": float(lf),
            "f_map": f, "f_std": np.sqrt(np.diag(cov)), "neg_log_evidence": float(nlp),
            "time": time.perf_counter() - t0, "note": "re-implementation of the idea of arXiv:2410.11454"}
