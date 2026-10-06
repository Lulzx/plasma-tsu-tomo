"""Tests of the linear-Gaussian mixing theory (tomo/mixing_theory.py)."""
import numpy as np
import pytest
import scipy.linalg as sla
import scipy.sparse as sp
import jax

from tomo import mixing_theory as mt
from tomo.ebm_chain import build_ising_chain
from tomo.ebm_tree import build_ising_tree
from tomo.sampling import greedy_coloring


def _random_gaussian(n=30, seed=0):
    rng = np.random.default_rng(seed)
    A = sp.random(n, n, density=0.12, random_state=seed, data_rvs=lambda k: rng.normal(size=k))
    A = (A + A.T).toarray()
    np.fill_diagonal(A, 0)
    P = A + (np.abs(A).sum(1).max() * 0.9 + 0.3) * np.eye(n)       # SPD, moderately correlated
    e = np.stack(np.nonzero(np.triu(A != 0, 1)), 1)
    return sp.csr_matrix(P), greedy_coloring(n, e)


def test_rho_matches_simulated_decay():
    """rho(B) is the decay rate of the mean of the simulated colour-block Gibbs chain."""
    P, cls = _random_gaussian()
    n = P.shape[0]
    rho = mt.gs_rho(P, cls, dense_max=10_000)
    perm, Pp, D, DL, U = mt.gs_split(P, cls)
    B = -sla.solve_triangular(DL.toarray(), U.toarray(), lower=True)
    assert abs(rho - np.max(np.abs(np.linalg.eigvals(B)))) < 1e-12
    assert 0.2 < rho < 0.99
    # start at the dominant eigenvector of B (real part) -> E[x_t] = B^t x0 ; compare against simulation in original ordering
    w, V = np.linalg.eig(B)
    k = np.argmax(np.abs(w))
    v = np.real(V[:, k])
    x0 = np.empty(n)
    x0[perm] = v / np.linalg.norm(v)
    if abs(w[k].imag) > 1e-9:
        pytest.skip("complex dominant eigenvalue")
    tr = mt.simulate_gs(P, cls, np.eye(n), n_chains=4000, n_sweeps=12, seed=1, x0=x0)   # (T, C, n)
    m = np.linalg.norm(tr.mean(1), axis=1)
    pred = np.linalg.norm(x0) * abs(w[k]) ** np.arange(1, 13)
    assert np.allclose(m[:3], pred[:3], rtol=0.15, atol=0.02)


def test_iat_formula_matches_simulation():
    """Exact IAT g^T D g / f^T S f of a linear functional vs. the simulated stationary chain, and sup formula."""
    P, cls = _random_gaussian(seed=3)
    n = P.shape[0]
    rng = np.random.default_rng(0)
    F = rng.normal(size=(n, 3))
    th = mt.iat_functional(P, F)
    y = mt.simulate_gs(P, cls, F, n_chains=200, n_sweeps=1500, seed=2)
    for k in range(3):
        assert mt.iat_series(y[:, :, k], known_mean=0.0) == pytest.approx(th[k], rel=0.12)
    # sup over all functionals is attained by the worst eigen-direction and bounds every functional
    assert mt.iat_sup(P, dense_max=10_000) >= th.max() - 1e-9
    d = 1 / np.sqrt(P.diagonal())
    S = np.diag(d) @ P.toarray() @ np.diag(d)
    w, V = np.linalg.eigh(S)
    f = d * V[:, 0]
    assert mt.iat_functional(P, f) == pytest.approx(mt.iat_sup(P, dense_max=10_000), rel=1e-8)


def test_two_block_rate_is_squared_canonical_correlation():
    rng = np.random.default_rng(5)
    nx, nz = 6, 4
    A = rng.normal(size=(nx + nz, nx + nz))
    P = A @ A.T + 1.5 * np.eye(nx + nz)
    S = np.linalg.inv(P)
    Lx = np.linalg.cholesky(S[:nx, :nx])
    Lz = np.linalg.cholesky(S[nx:, nx:])
    cc = np.linalg.svd(np.linalg.solve(Lx, S[:nx, nx:]) @ np.linalg.inv(Lz).T, compute_uv=False)
    assert mt.da_rate(sp.csr_matrix(P), nx, dense_max=100) == pytest.approx(cc[0] ** 2, rel=1e-9)
    assert mt.da_rate(sp.csr_matrix(P), nx, dense_max=1) == pytest.approx(cc[0] ** 2, rel=1e-6)  # sparse path
    # equals the spectral radius of the two-block Gauss-Seidel iteration (x | z)
    cls = np.r_[np.zeros(nx, int), np.ones(nz, int)]
    _, Pp, D, DL, U = mt.gs_split(sp.csr_matrix(P), cls)
    Dblk = np.zeros_like(P); Dblk[:nx, :nx] = P[:nx, :nx]; Dblk[nx:, nx:] = P[nx:, nx:]
    Lblk = np.tril(P, 0) * 0; Lblk[nx:, :nx] = P[nx:, :nx]
    Ublk = np.zeros_like(P); Ublk[:nx, nx:] = P[:nx, nx:]
    B = -np.linalg.solve(Dblk + Lblk, Ublk)
    assert np.max(np.abs(np.linalg.eigvals(B))) == pytest.approx(cc[0] ** 2, rel=1e-9)


def test_dense_vs_sparse_rho(tiny_problem):
    J = mt.build_joint(tiny_problem, "chain", tau=0.75, Kz=16)
    r1 = mt.gs_rho(J["P"], J["cls"], dense_max=10_000)
    r2 = mt.gs_rho(J["P"], J["cls"], dense_max=10)
    assert abs(r1 - r2) < 2e-5


def test_theory_orders_models(tiny_problem):
    """Dense GS mixes far faster than chain/tree; the null-space bound explains the DA slowdown."""
    Jd = mt.build_joint(tiny_problem, "dense")
    rd = mt.gs_rho(Jd["P"], Jd["cls"])
    for kind in ("chain", "tree"):
        J = mt.build_joint(tiny_problem, kind, tau=0.5, tau_mode="frac")
        assert mt.gs_rho(J["P"], J["cls"]) > rd
        da = mt.da_rate(J["P"], J["nx"])
        nb = mt.null_space_bound(J, tiny_problem)
        assert nb <= 1 / (1 - da) * (1 + 1e-6)             # lower bound
        assert nb >= 0.8 / (1 - da)                        # and a good one
        # exact chord marginal: the compensated model's x-marginal is the Gaussian posterior
        Sxx = np.linalg.inv(J["P"].toarray())[:J["nx"], :J["nx"]]
        T = np.asarray(tiny_problem.T, float)
        Q = T.T @ (T / np.asarray(tiny_problem.sigma)[:, None] ** 2) + J["lam"] * np.asarray(tiny_problem.L)
        if np.all(J["S"] < 0.9 * np.asarray(tiny_problem.sigma) ** 2):
            assert np.allclose(np.linalg.inv(Sxx), Q, rtol=1e-6, atol=1e-6 * np.abs(Q).max())


@pytest.mark.parametrize("kind,builder", [("chain", build_ising_chain), ("tree", build_ising_tree)])
def test_block_structure_matches_sampler(tiny_problem, kind, builder):
    """Variable classes / sweep order equal the Ising sampler's colour blocks; P equals the builder's Qy up to scaling."""
    Kz, tau = 16, 0.75
    prob, meta = builder(tiny_problem, 8, tau=tau, Kz=Kz, compensate=True)
    J = mt.build_joint(tiny_problem, kind, tau=tau, Kz=Kz, compensate=True, setup=meta["setup"])
    N, nz = meta["N"], meta["L"]
    assert (J["nx"], J["nz"]) == (N, nz)
    cls, colors, rowvar = J["cls"], meta["colors"], meta["rowvar"]
    # every sampler colour block contains bits of variables of a single class
    cls_of_bit = cls[rowvar]
    for c in np.unique(colors):
        assert len(np.unique(cls_of_bit[colors == c])) == 1
    # and the order of the class sweep matches the order of the colour blocks
    first = {k: colors[cls_of_bit == k].min() for k in np.unique(cls)}
    last = {k: colors[cls_of_bit == k].max() for k in np.unique(cls)}
    ks = sorted(first)
    assert all(last[a] < first[b] for a, b in zip(ks[:-1], ks[1:]))
    # precision equals the builder's integer-level Qy after undoing the level scalings (x = Delta x_lev, z = z0 + dz w)
    sc = np.concatenate([np.full(N, meta["Delta"]), meta["dz"]])
    Pm = sp.diags(sc) @ J["P"] @ sp.diags(sc)
    assert abs(Pm - meta["Qy"]).max() <= 1e-6 * abs(meta["Qy"]).max()


def test_sor_improves_rho():
    """SOR / over-relaxed Gibbs on a consistently ordered (tridiagonal, red-black) Gaussian: w_opt beats w=1."""
    n = 40
    P = sp.diags([-np.ones(n - 1), 2 * np.ones(n), -np.ones(n - 1)], [-1, 0, 1], format="csr")
    cls = np.arange(n) % 2
    r1 = mt.sor_rho(P, cls, 1.0)
    assert r1 == pytest.approx(mt.gs_rho(P, cls), abs=1e-9)
    assert r1 == pytest.approx(np.cos(np.pi / (n + 1)) ** 2, abs=1e-6)       # classical red-black result
    mu = np.cos(np.pi / (n + 1))
    w_opt = 2 / (1 + np.sqrt(1 - mu ** 2))
    assert mt.sor_rho(P, cls, w_opt) == pytest.approx(w_opt - 1, abs=1e-4)
    assert mt.sor_rho(P, cls, w_opt) < r1
