import numpy as np
import pytest

from tomo.baselines import discrepancy_lambda, lcurve_lambda, tuned_tikhonov, default_lams, gcv_lambda, gp_tomography, mfi, tikhonov
from tomo.forward import make_problem
from tomo.metrics import coverage, rel_l2


@pytest.fixture(scope="module")
def hollow(cfg):
    return make_problem(cfg, "hollow", 0)


def _rel(est, p):
    return float(np.linalg.norm(est - p.eps_true) / np.linalg.norm(p.eps_true))


def test_tikhonov_peaked(problem):
    p = problem
    r = tikhonov(p.T, p.b, p.sigma, p.L)
    assert r["mean"].shape == p.eps_true.shape and np.isfinite(r["lam"])
    assert _rel(r["mean"], p) < 0.35


def test_gcv_interior(problem):
    p = problem
    lam = gcv_lambda(p.T, p.b, p.sigma, p.L)
    A = p.T / p.sigma[:, None]
    grid = default_lams(np.linalg.norm(A, 2) ** 2)
    assert np.isfinite(lam) and grid[0] < lam < grid[-1]  # strictly interior to the grid


def test_mfi(problem, hollow):
    for p in (problem, hollow):
        r = mfi(p.T, p.b, p.sigma, p.mask, p.grid)
        assert r["mean"].min() >= 0 and r["n_iter"] >= 1
    assert _rel(mfi(problem.T, problem.b, problem.sigma, problem.mask, problem.grid)["mean"], problem) < 0.35


def test_gp(problem):
    p = problem
    r = gp_tomography(p.T, p.b, p.sigma, p.grid, p.mask)
    assert _rel(r["mean"], p) < 0.35
    assert r["cov"].shape == (len(r["mean"]),) * 2 and np.all(r["std"] >= 0)
    cov95 = coverage((r["mean"], r["std"]), p.eps_true, 0.95)
    print("GP 95% coverage (peaked):", cov95, "hyper:", r["hyper"])
    assert 0.0 <= cov95 <= 1.0


def test_hollow_reasonable(hollow):
    p = hollow
    assert _rel(tikhonov(p.T, p.b, p.sigma, p.L)["mean"], p) < 0.8


def test_lambda_methods(problem):
    p = problem
    from tomo.metrics import reduced_chi2
    rg = tikhonov(p.T, p.b, p.sigma, p.L)  # default unchanged: gcv
    assert rg["lam"] == gcv_lambda(p.T, p.b, p.sigma, p.L)
    rd = tikhonov(p.T, p.b, p.sigma, p.L, method="discrepancy")
    assert reduced_chi2(p.T, p.b, p.sigma, rd["mean"]) == pytest.approx(1.0, abs=0.02)
    assert rd["lam"] > rg["lam"] and rd["lam"] == discrepancy_lambda(p.T, p.b, p.sigma, p.L)
    assert reduced_chi2(p.T, p.b, p.sigma, rg["mean"]) < 0.1  # GCV overfits (M << N)
    rl = tikhonov(p.T, p.b, p.sigma, p.L, method="lcurve")
    assert np.isfinite(rl["lam"]) and rl["lam"] == lcurve_lambda(p.T, p.b, p.sigma, p.L)
    t = tuned_tikhonov(p.T, p.b, p.sigma, p.L)
    assert t["method"] == "discrepancy" and abs(t["chi2_red"] - 1) < 0.02 and _rel(t["mean"], p) < 0.35
