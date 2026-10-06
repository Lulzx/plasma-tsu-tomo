import numpy as np
import pytest

from tomo import baselines as B
from tomo.ebm_common import (auto_A, bit_energy, dw_decode, dw_encode, dw_penalty_qubo, eps_from_levels,
                             gaussian_posterior, levels_from_eps, make_result, pixel_energy, prepare,
                             pixel_quadratic_to_bit_qubo, quadratic_form)
from tomo.metrics import reduced_chi2
from tomo.sampling import bits_to_spins_qubo

K = 4


@pytest.fixture(scope="module")
def setup(tiny_problem):
    return prepare(tiny_problem, K)


def _direct(p, x, lam, Delta):
    r = (p.b - Delta * p.T.astype(float) @ x) / p.sigma
    d = x[p.edges[:, 0]] - x[p.edges[:, 1]]
    return 0.5 * r @ r + 0.5 * lam * Delta ** 2 * d @ d


def test_prepare(setup, tiny_problem):
    s = setup
    assert tiny_problem.eps_max == s.eps_max and s.Delta == pytest.approx(s.eps_max / (K - 1))
    assert s.lam == s.tik["lam"] > 0 and s.x0.min() >= 0 and s.x0.max() <= K - 1
    assert prepare(tiny_problem, K, lam=3.0).lam == 3.0
    assert reduced_chi2(tiny_problem.T, tiny_problem.b, tiny_problem.sigma, s.eps_tik) < 1.05  # evidence lambda <= discrepancy lambda


def test_quadratic_form_exact(setup, tiny_problem):
    s, p = setup, tiny_problem
    Q, c, const = quadratic_form(p, K, s.lam, s.Delta)
    rng = np.random.default_rng(0)
    for _ in range(5):
        x = rng.integers(0, K, p.eps_true.size).astype(float)
        assert pixel_energy(x, Q, c, const) == pytest.approx(_direct(p, x, s.lam, s.Delta), rel=1e-9)
    # continuous minimiser = Tikhonov (level units)
    # (baselines add a 1e-6 relative ridge to L, hence the loose tolerance)
    assert np.linalg.norm(np.linalg.solve(Q, c) * s.Delta - s.eps_tik) < 1e-3 * np.linalg.norm(s.eps_tik)


def test_levels():
    assert list(levels_from_eps([-1, 0.4, 0.6, 9], 1.0, 4)) == [0, 0, 1, 3]
    assert np.allclose(eps_from_levels([0, 2], 0.5), [0, 1.0])


def test_dw_roundtrip():
    x = np.array([[0, 1, 3], [2, 3, 0]])
    u = dw_encode(x, K)
    xd, valid = dw_decode(u, K)
    assert (xd == x).all() and valid.all()
    bad = u.copy()
    bad[0, 1] = [0, 1, 0]  # 010
    xd, valid = dw_decode(bad, K)
    assert not valid[0, 1] and valid.sum() == 5 and xd[0, 1] == 1


def test_bit_qubo(setup, tiny_problem):
    s = setup
    N = s.N
    A = auto_A(s.Q, s.c, K)
    assert A > 0
    Qb, cb, cst = pixel_quadratic_to_bit_qubo(s.Q, s.c, s.const, K, A)
    assert (abs(Qb.diagonal()) == 0).all() and abs(Qb - Qb.T).max() == 0
    rng = np.random.default_rng(1)
    for _ in range(5):
        x = rng.integers(0, K, N)
        u = dw_encode(x, K).reshape(-1)
        assert bit_energy(u, Qb, cb, cst) == pytest.approx(pixel_energy(x.astype(float), s.Q, s.c, s.const), rel=1e-9)
    # invalid: +A per violation
    for _ in range(5):
        u = rng.integers(0, 2, N * (K - 1))
        x, valid = dw_decode(u.reshape(N, K - 1), K)
        nviol = int(np.sum((u.reshape(N, K - 1)[:, 1:] == 1) & (u.reshape(N, K - 1)[:, :-1] == 0)))
        e0 = pixel_energy(x.astype(float), s.Q, s.c, s.const)
        assert bit_energy(u, Qb, cb, cst) == pytest.approx(e0 + A * nviol, rel=1e-9)
    # penalty alone
    Qp, cp = dw_penalty_qubo(N, K, 2.0)
    u = np.zeros((N, K - 1), int); u[0, 1] = 1
    assert bit_energy(u.reshape(-1), Qp, cp) == pytest.approx(2.0)
    # ising conversion preserves energy
    prob = bits_to_spins_qubo(Qb, cb, cst)
    u = rng.integers(0, 2, N * (K - 1)); sp_ = 2 * u - 1
    e = -prob.h @ sp_ - np.sum(prob.vals * sp_[prob.rows] * sp_[prob.cols]) + prob.offset
    assert e == pytest.approx(bit_energy(u, Qb, cb, cst), rel=1e-8)


def test_pair_mask(setup):
    s = setup
    N = s.N
    Qb_full, _, _ = pixel_quadratic_to_bit_qubo(s.Q, s.c, s.const, K, 1.0)
    keep = np.abs(s.Q) > 0.5 * np.abs(s.Q - np.diag(np.diag(s.Q))).max()
    Qb, cb, _ = pixel_quadratic_to_bit_qubo(s.Q, s.c, s.const, K, 1.0, mask_pairs=keep)
    assert Qb.nnz < Qb_full.nnz
    Qb0, _, _ = pixel_quadratic_to_bit_qubo(s.Q, s.c, s.const, K, 1.0, mask_pairs=np.zeros((0, 2), int))
    # only within-pixel + penalty remain: block diagonal
    r, cc = Qb0.nonzero()
    assert (r // (K - 1) == cc // (K - 1)).all()


def test_auto_A_bound(setup):
    s = setup
    A = auto_A(s.Q, s.c, K, margin=1.0)
    rng = np.random.default_rng(2)
    worst = 0.0
    for _ in range(200):
        x = rng.integers(0, K, s.N).astype(float)
        j = rng.integers(s.N)
        for dlt in (1, -1):
            if 0 <= x[j] + dlt < K:
                y = x.copy(); y[j] += dlt
                worst = max(worst, abs(pixel_energy(y, s.Q, s.c) - pixel_energy(x, s.Q, s.c)))
    assert worst <= A


def test_make_result_and_gaussian(setup):
    s = setup
    rng = np.random.default_rng(3)
    lv = rng.integers(0, K, (3, 40, s.N))
    r = make_result(lv, s.Delta, lv[0, 0], energy_trace=np.zeros((3, 4)), n_blocks=5, n_spins=7, max_degree=9,
                    time=0.1, invalid_frac=0.0, variant="x")
    for k in ("mean", "std", "map", "samples", "rhat", "energy_trace", "n_blocks", "n_spins", "max_degree", "time", "invalid_frac"):
        assert k in r
    assert r["samples"].shape == lv.shape and np.allclose(r["mean"], s.Delta * lv.mean((0, 1))) and r["variant"] == "x"
    g = gaussian_posterior(s.Q, s.c)
    assert g["mean"].shape == (s.N,) and (g["std"] > 0).all()
    assert np.linalg.norm(g["mean"] * s.Delta - s.eps_tik) < 1e-3 * np.linalg.norm(s.eps_tik)
