"""Tests for tomo.ebm_ising (domain-wall Ising variants), tiny problem, reduced schedules."""
import jax
import numpy as np
import pytest

from tomo import ebm_common as ec
from tomo.config import load_config
from tomo.ebm_ising import (bits_to_levels, build_ising_dense, build_ising_sparse, posterior_bias,
                            sample_ising_variant)
from tomo.metrics import rel_l2
from tomo.sampling import check_coloring

K = 8
SCHED = dict(n_warmup=200, n_samples=60, steps_per_sample=4)
ANNEAL = dict(beta_min=0.1, beta_max=50.0, n_betas=6, sweeps_per_beta=4)


@pytest.fixture(scope="module")
def tcfg(tiny_cfg):
    return load_config({**tiny_cfg, "model": {**tiny_cfg["model"], "K": K}})


@pytest.fixture(scope="module")
def dense(tiny_problem):
    return build_ising_dense(tiny_problem, K)


@pytest.fixture(scope="module")
def dense_run(tiny_problem, tcfg):
    return sample_ising_variant(tiny_problem, "dense", tcfg, jax.random.PRNGKey(0), schedule=SCHED,
                                anneal=ANNEAL, n_chains=6)


def test_energy_matches_pixel_energy(dense):
    prob, meta = dense
    s = meta["setup"]
    rng = np.random.default_rng(0)
    x = np.vstack([rng.integers(0, K, (20, s.N)), s.x0[None], np.zeros((1, s.N), int)])
    u = ec.dw_encode(x, K).reshape(len(x), -1)
    np.testing.assert_allclose(prob.energy(u.astype(bool)), ec.pixel_energy(x, s.Q, s.c, s.const),
                               rtol=1e-8, atol=1e-6)


def test_invalid_costs_A(dense):
    prob, meta = dense
    s = meta["setup"]
    x = s.x0.copy()
    u = ec.dw_encode(x, K).reshape(s.N, K - 1).copy()
    j = int(np.argmax(x < K - 3))
    u[j, x[j] + 1] = 1  # skip a level: one violation, level count +1
    xe = x.copy(); xe[j] += 1
    e = prob.energy(u.reshape(-1).astype(bool))
    assert abs(e - (ec.pixel_energy(xe, s.Q, s.c, s.const) + meta["A"])) < 1e-6 * max(1, abs(e))


def test_coloring_valid_and_blocks(dense):
    prob, meta = dense
    assert check_coloring(prob.n, prob.edges, meta["colors"])
    assert meta["n_blocks"] >= K - 1 and meta["n_blocks"] == len(np.unique(meta["colors"]))
    assert prob.n == meta["N"] * (K - 1)


def test_sparse_threshold_zero_equals_dense(tiny_problem, dense):
    pd, _ = dense
    ps, meta = build_ising_sparse(tiny_problem, K, threshold=0.0)
    assert meta["kept_fraction"] == 1.0
    np.testing.assert_allclose(ps.h, pd.h)
    assert np.array_equal(ps.rows, pd.rows) and np.array_equal(ps.cols, pd.cols)
    np.testing.assert_allclose(ps.vals, pd.vals)
    assert ps.offset == pytest.approx(pd.offset)


def test_sparse_is_sparser_and_keeps_laplacian(tiny_problem, dense):
    pd, _ = dense
    ps, meta = build_ising_sparse(tiny_problem, K, threshold=0.2)
    assert ps.n_edges < pd.n_edges and 0 < meta["kept_fraction"] < 1
    assert meta["n_blocks"] <= dense[1]["n_blocks"]
    assert check_coloring(ps.n, ps.edges, meta["colors"])
    pr, _ = build_ising_sparse(tiny_problem, K, radius=0.15)
    assert pr.n_edges < pd.n_edges
    assert np.all(np.isfinite(ps.h))


def test_dense_run_valid_and_close_to_gaussian(dense_run, tiny_problem):
    r = dense_run
    for k in ("mean", "std", "map", "samples", "rhat", "energy_trace", "n_blocks", "n_spins", "max_degree",
              "time", "invalid_frac"):
        assert k in r
    assert r["invalid_frac"] < 1e-3
    s = r["setup"]
    g = ec.gaussian_posterior(s.Q, s.c)
    gm = np.clip(g["mean"], 0, K - 1) * s.Delta
    assert rel_l2(r["mean"], gm) < 0.2          # loose: discrete posterior vs relaxed Gaussian
    assert rel_l2(r["mean"], tiny_problem.eps_true) < 1.0
    assert np.all(np.isfinite(r["map"])) and r["rhat_max"] < 1.3
    # decode round trip: samples are Delta * integer levels
    lv = r["samples"] / s.Delta
    np.testing.assert_allclose(lv, np.rint(lv), atol=1e-6)


def test_sparse_run_and_bias(tiny_problem, tcfg, dense_run):
    r = sample_ising_variant(tiny_problem, "sparse", tcfg, jax.random.PRNGKey(1), schedule=SCHED,
                             anneal=ANNEAL, n_chains=6, threshold=0.05)
    assert r["invalid_frac"] < 1e-3 and r["n_blocks"] <= dense_run["n_blocks"]
    b = posterior_bias(r, dense_run, tiny_problem.eps_true)
    assert b["rel_l2_vs_ref"] < 0.5


def test_bits_to_levels_decoding():
    u = np.array([[1, 1, 0, 0], [0, 1, 0, 0]])  # pixel 0 valid (2), pixel 1 invalid (sum 1)
    x, valid = bits_to_levels(u.reshape(-1), 5)
    assert x.tolist() == [2, 1] and valid.tolist() == [True, False]


def test_unknown_variant(tiny_problem, tcfg):
    with pytest.raises(ValueError):
        sample_ising_variant(tiny_problem, "bogus", tcfg, jax.random.PRNGKey(0))


def test_sparse_keeps_laplacian_pairs(tiny_problem, dense):
    from tomo.ebm_ising import pixel_pair_mask
    s = dense[1]["setup"]
    pairs, _, n_lap = pixel_pair_mask(tiny_problem, s.Q, threshold=0.5)
    kept = set(map(tuple, pairs.tolist()))
    e = np.sort(np.asarray(tiny_problem.edges).reshape(-1, 2), axis=1)
    assert n_lap == len(e) and all(tuple(x) in kept for x in e.tolist())


def test_linear_compensation_gradient_exact_and_differs_from_none(tiny_problem, dense):
    s = dense[1]["setup"]
    pl, ml = build_ising_sparse(tiny_problem, K, threshold=0.3, compensate="linear")
    pn, mn = build_ising_sparse(tiny_problem, K, threshold=0.3, compensate="none")
    assert not np.allclose(pl.h, pn.h)
    # linearised model: E_sparse(x) - E_dense(x) = -1/2 (x-m)^T Qd (x-m)  (zero value and gradient at m)
    from tomo.ebm_ising import pixel_pair_mask
    pairs, _, _ = pixel_pair_mask(tiny_problem, s.Q, threshold=0.3)
    Qo = s.Q - np.diag(np.diag(s.Q))
    keep = np.zeros_like(Qo, bool)
    keep[pairs[:, 0], pairs[:, 1]] = keep[pairs[:, 1], pairs[:, 0]] = True
    Qd = np.where(keep, 0.0, Qo)
    m = np.clip(s.eps_tik / s.Delta, 0, K - 1)
    rng = np.random.default_rng(0)
    for x in rng.integers(0, K, (5, s.N)):
        u = ec.dw_encode(x, K).reshape(-1).astype(bool)
        want = -0.5 * (x - m) @ Qd @ (x - m)
        assert abs((pl.energy(u) - pd_energy(dense, u)) - want) < 1e-6 * max(1, abs(want))


def pd_energy(dense, u):
    return dense[0].energy(u)


def test_overdispersed_rhat_inf_counts(tiny_problem, tcfg):
    r = sample_ising_variant(tiny_problem, "sparse", tcfg, jax.random.PRNGKey(3), schedule=SCHED,
                             anneal=ANNEAL, n_chains=4, do_map=False, overdispersed=True)
    assert "rhat_n_nonfinite" in r and (np.isnan(r["rhat_max"]) or r["rhat_max"] >= 1.0 - 1e-9)
