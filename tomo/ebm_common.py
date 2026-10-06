"""Shared pieces of the energy-based tomography models (Potts / domain-wall Ising variants).

Pixel levels x_j in {0..K-1}, eps_j = Delta * x_j, Delta = eps_max/(K-1).  The spec energy

    E(x) = 1/2 ||(b - Delta T x)/sigma||^2 + lam Delta^2 / 2 sum_<jk> (x_j - x_k)^2

(per-chord sigma) is written exactly as  E(x) = 1/2 x^T Q x - c^T x + const  with
Q = Delta^2 (T^T W T + lam L), c = Delta T^T W b, const = 1/2 b^T W b, W = diag(1/sigma^2).
``lam`` is in emissivity units and has the same meaning as in ``baselines.tikhonov``, so the
Tikhonov solution is exactly the continuous minimiser of this energy (in level units Delta*x).

Bit layout: pixel j owns bits j*(K-1) .. j*(K-1)+K-2 (``u.reshape(N, K-1)``); bit m (0-based)
is the thermometer bit "x_j > m".  Bit energies use ``E(u) = u^T Qb u + cb.u + const`` with Qb
symmetric and zero diagonal, which is what ``sampling.bits_to_spins_qubo`` expects.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from .baselines import tuned_tikhonov
from .forward import eps_max_from_estimate
from .metrics import split_rhat

__all__ = ["EBMSetup", "prepare", "quadratic_form", "pixel_energy", "levels_from_eps", "eps_from_levels",
           "dw_encode", "dw_decode", "dw_penalty_qubo", "pixel_quadratic_to_bit_qubo", "bit_energy",
           "auto_A", "make_result", "gaussian_posterior", "invalid_fraction"]


# ----------------------------------------------------------------------------------------
# levels
# ----------------------------------------------------------------------------------------
def levels_from_eps(eps, Delta, K):
    """Round emissivity to the nearest level and clip to {0..K-1} (int64)."""
    return np.clip(np.rint(np.asarray(eps, float) / Delta), 0, K - 1).astype(np.int64)


def eps_from_levels(x, Delta):
    return Delta * np.asarray(x, float)


# ----------------------------------------------------------------------------------------
# quadratic form
# ----------------------------------------------------------------------------------------
def quadratic_form(problem, K, lam, Delta):
    """(Q, c, const) with E(x) = 1/2 x^T Q x - c^T x + const equal to the spec energy exactly.

    Q (N,N) dense float64 = Delta^2 (T^T W T + lam L); c = Delta T^T W b; const = 1/2 b^T W b.
    ``K`` is accepted for interface symmetry (the form itself does not depend on it).
    """
    T = np.asarray(problem.T, float)
    w = 1.0 / np.asarray(problem.sigma, float) ** 2
    b = np.asarray(problem.b, float)
    L = problem.L.toarray() if sp.issparse(problem.L) else np.asarray(problem.L, float)
    TtW = T.T * w
    Q = Delta ** 2 * (TtW @ T + lam * L)
    Q = 0.5 * (Q + Q.T)
    c = Delta * (TtW @ b)
    const = 0.5 * float(b @ (w * b))
    return Q, c, const


def pixel_energy(x, Q, c, const=0.0):
    """E(x) = 1/2 x^T Q x - c^T x + const for x (..., N) (levels)."""
    x = np.asarray(x, float)
    return 0.5 * np.einsum("...i,ij,...j->...", x, Q, x) - x @ c + const


# ----------------------------------------------------------------------------------------
# setup
# ----------------------------------------------------------------------------------------
@dataclass
class EBMSetup:
    problem: object
    K: int
    eps_max: float
    Delta: float
    lam: float                 # emissivity units (tikhonov convention)
    w: np.ndarray              # 1/sigma^2 (M,)
    x0: np.ndarray             # Tikhonov warm start rounded to levels (N,) int
    eps_tik: np.ndarray        # Tikhonov estimate (N,)
    tik: dict = field(default_factory=dict)
    Q: np.ndarray = None
    c: np.ndarray = None
    const: float = 0.0

    @property
    def W(self):
        return sp.diags(self.w)

    @property
    def N(self):
        return len(self.x0)


def prepare(problem, K, lam=None, eps_max_factor=1.2) -> EBMSetup:
    """Tuned-Tikhonov warm start, eps_max (data only), Delta, lam and the quadratic form.

    eps_max = factor * max(clip(Tikhonov, 0)) and is stored on ``problem.eps_max``.
    lam=None uses the Tikhonov (discrepancy-principle) lambda; a given lam is used as is, but the
    warm start is always the tuned Tikhonov solution. No ground truth is used.
    """
    tik = tuned_tikhonov(problem.T, problem.b, problem.sigma, problem.L)
    eps_max = eps_max_from_estimate(tik["mean"], eps_max_factor)
    problem.eps_max = eps_max
    Delta = eps_max / (K - 1)
    lam = float(tik["lam"] if lam is None else lam)
    Q, c, const = quadratic_form(problem, K, lam, Delta)
    return EBMSetup(problem=problem, K=int(K), eps_max=eps_max, Delta=Delta, lam=lam,
                    w=1.0 / np.asarray(problem.sigma, float) ** 2,
                    x0=levels_from_eps(tik["mean"], Delta, K), eps_tik=tik["mean"], tik=tik,
                    Q=Q, c=c, const=const)


# ----------------------------------------------------------------------------------------
# domain-wall (thermometer) encoding
# ----------------------------------------------------------------------------------------
def dw_encode(x, K):
    """Levels x (..., N) -> thermometer bits u (..., N, K-1) with u_m = [x > m]."""
    x = np.asarray(x)
    return (x[..., None] > np.arange(K - 1)).astype(np.int8)


def dw_decode(u, K=None):
    """Bits u (..., N, K-1) -> (x (...,N) int, valid (...,N) bool); x = sum_m u_m.

    valid iff the pixel's bits have the form 1..10..0 (nonincreasing in m).
    """
    u = np.asarray(u)
    x = u.sum(-1).astype(np.int64)
    valid = np.all(u[..., 1:] <= u[..., :-1], axis=-1)
    return x, valid


def invalid_fraction(valid) -> float:
    """Fraction of (pixel, sample) entries that violate the domain-wall ordering."""
    return float(1.0 - np.mean(valid))


def dw_penalty_qubo(N, K, A):
    """(Qb, cb) sparse/vector with u^T Qb u + cb.u = A sum_j sum_{m=0}^{K-3} u_{j,m+1}(1 - u_{j,m})."""
    nb = K - 1
    if nb < 2:
        return sp.csr_matrix((N * nb, N * nb)), np.zeros(N * nb)
    j = np.repeat(np.arange(N), nb - 1) * nb + np.tile(np.arange(nb - 1), N)  # index of bit m
    rows = np.concatenate([j, j + 1])
    cols = np.concatenate([j + 1, j])
    Qb = sp.csr_matrix((np.full(2 * len(j), -0.5 * A), (rows, cols)), shape=(N * nb, N * nb))
    cb = np.zeros(N * nb)
    cb[j + 1] += A  # +A u_{m+1} ; the -A u_m u_{m+1} part is in Qb
    return Qb, cb


def _pair_mask_matrix(mask_pairs, N):
    if sp.issparse(mask_pairs):
        M = sp.csr_matrix(mask_pairs, dtype=bool)
    else:
        a = np.asarray(mask_pairs)
        if a.ndim == 2 and a.shape == (N, N):
            M = sp.csr_matrix(a.astype(bool))
        else:  # (P,2) list of pixel pairs
            a = a.reshape(-1, 2)
            M = sp.coo_matrix((np.ones(len(a), bool), (a[:, 0], a[:, 1])), shape=(N, N)).tocsr()
    return (M + M.T).astype(bool)


def pixel_quadratic_to_bit_qubo(Q, c, const, K, A, mask_pairs=None):
    """Map E(x)=1/2 x^T Q x - c^T x + const with x_j = sum_m u_{j,m} to a bit QUBO + domain-wall penalty.

    Returns (Qb sparse CSR symmetric zero-diagonal, cb (N(K-1),), constb) with
    E(u) = u^T Qb u + cb.u + constb  ==  E_pixel(sum_m u) + A * #violations (violation = u_{m+1}=1, u_m=0).
    Within-pixel couplings come from 1/2 Q_jj (sum_m u_m)^2. ``mask_pairs`` ((N,N) bool/sparse or (P,2)
    pixel pairs) keeps only the listed (symmetrised) j != k couplings -- used by I-sparse; the diagonal
    (within-pixel) block is always kept. Dropping pairs makes the model approximate.
    """
    Q = Q.toarray() if sp.issparse(Q) else np.asarray(Q, float)
    N = Q.shape[0]
    nb = K - 1
    ones = np.ones((nb, nb))
    Qoff = Q - np.diag(np.diag(Q))
    if mask_pairs is not None:
        Qoff = sp.csr_matrix(Qoff).multiply(_pair_mask_matrix(mask_pairs, N)).tocsr()
    else:
        Qoff = sp.csr_matrix(Qoff)
    Qoff.eliminate_zeros()
    # 1/2 Q_jk x_j x_k (both orders) -> symmetric block 1/2 Q_jk * ones(nb, nb)
    Qb = sp.kron(0.5 * Qoff, ones, format="csr")
    d = np.diag(Q)
    # within pixel: 1/2 Q_jj (sum u)^2 = 1/2 Q_jj sum u + sum_{m<m'} Q_jj u u'  -> off-diag entries 1/2 Q_jj
    blk = sp.kron(sp.diags(0.5 * d), ones - np.eye(nb), format="csr")
    Qp, cp = dw_penalty_qubo(N, K, A)
    Qb = (Qb + blk + Qp).tocsr()
    Qb.eliminate_zeros()
    cb = np.repeat(-np.asarray(c, float) + 0.5 * d, nb) + cp
    return Qb, cb, float(const)


def bit_energy(u, Qb, cb, constb=0.0):
    """E(u) = u^T Qb u + cb.u + constb for u (..., n) flat bits (batched over leading dims)."""
    u = np.asarray(u, float)
    flat = u.reshape(-1, u.shape[-1])
    e = np.einsum("si,si->s", flat @ Qb.T if sp.issparse(Qb) else flat @ Qb.T, flat) + flat @ cb + constb
    return e.reshape(u.shape[:-1])


def auto_A(Q, c, K, margin: float = 1.05):
    """Domain-wall penalty A exceeding the largest possible single-level energy change.

    Moving pixel j by one level changes E by  dE_j = Q_jj (x_j + 1/2) - c_j + sum_{k!=j} Q_jk x_k  (up move;
    the down move gives -dE at x_j - 1), with x in [0, K-1]. Bound with the sign-split sums
        hi_j = Q_jj (K - 3/2) - c_j + (K-1) sum_{k!=j} max(Q_jk, 0)
        lo_j = Q_jj / 2       - c_j + (K-1) sum_{k!=j} min(Q_jk, 0)
    |dE_j| <= max(|hi_j|, |lo_j|); A = margin * max_j of that (tighter than the crude
    |c_j| + (K-1) sum|Q_jk| + Q_jj/2 since T^T W T >= 0 off-diagonal and only the lambda L couplings are
    negative). Note any A > 0 already makes every ground state/MAP valid (an invalid bit pattern with
    count x is beaten by the sorted pattern with the same x by A per violation); the bound makes a
    single-bit flip out of a valid state lose against the data energy, keeping the invalid
    fraction at beta = 1 small.
    """
    Q = Q.toarray() if sp.issparse(Q) else np.asarray(Q, float)
    d = np.diag(Q)
    off = Q - np.diag(d)
    hi = d * (K - 1.5) - c + (K - 1) * np.clip(off, 0, None).sum(1)
    lo = 0.5 * d - c + (K - 1) * np.clip(off, None, 0).sum(1)
    return float(margin * np.max(np.maximum(np.abs(hi), np.abs(lo))))


# ----------------------------------------------------------------------------------------
# results & sanity
# ----------------------------------------------------------------------------------------
def make_result(samples_levels, Delta, map_levels, rhat=None, energy_trace=None, n_blocks=None,
                n_spins=None, max_degree=None, time=None, invalid_frac=0.0, **extra) -> dict:
    """Assemble the standard result dict (INTERFACES.md). samples_levels is (C,S,N) in level units.

    mean/std are over all chains and samples (emissivity), 'samples' is (C,S,N) emissivity, 'map' is
    Delta*map_levels; rhat defaults to split R-hat on the samples.
    """
    s = Delta * np.asarray(samples_levels, float)
    flat = s.reshape(-1, s.shape[-1])
    if rhat is None:
        rhat = split_rhat(s)
    out = {"mean": flat.mean(0), "std": flat.std(0), "map": None if map_levels is None else Delta * np.asarray(map_levels, float),
           "samples": s, "rhat": rhat, "energy_trace": energy_trace, "n_blocks": n_blocks,
           "n_spins": n_spins, "max_degree": max_degree, "time": time, "invalid_frac": invalid_frac}
    out.update(extra)
    return out


def gaussian_posterior(Q, c, beta: float = 1.0) -> dict:
    """Continuous relaxation of p(x) ~ exp(-beta E): Gaussian with mean Q^{-1} c (levels), cov Q^{-1}/beta.

    The discrete (K-level, clipped) posterior is close to this when the posterior std is >> 1 level
    and the mean is away from the [0, K-1] boundaries; use it to sanity-check samplers.
    """
    Q = Q.toarray() if sp.issparse(Q) else np.asarray(Q, float)
    cov = np.linalg.inv(Q) / beta
    cov = 0.5 * (cov + cov.T)
    return {"mean": cov @ c * beta, "cov": cov, "std": np.sqrt(np.clip(np.diag(cov), 0, None))}
