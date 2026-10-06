"""Tests for tomo.ebm_potts (thrml reference and pure-JAX backends)."""
import itertools

import jax
import numpy as np
import pytest

from tomo.ebm_common import prepare
from tomo.ebm_potts import PottsSampler, build_potts, build_potts_from_Q, coupling_split, sample_potts
from tomo.metrics import rel_l2
from tomo.sampling import check_coloring

K_TINY = 4


@pytest.fixture(scope="module")
def tiny_model(tiny_problem):
    return build_potts(tiny_problem, K_TINY)


def test_thrml_energy_matches_quadratic_form(tiny_model):
    m = tiny_model
    rng = np.random.default_rng(0)
    assert check_coloring(m.N, m.edges, m.colors)
    assert m.n_blocks > 2  # TtT couples far more than a checkerboard allows
    xs = [rng.integers(0, K_TINY, m.N) for _ in range(5)]
    e_t = np.array([m.thrml_energy(x) for x in xs])
    e_q = np.array([m.energy(x) - m.const for x in xs])
    # float32 weights: compare differences (constant dropped) to a loose absolute tolerance
    assert np.allclose(e_t - e_t[0], e_q - e_q[0], atol=2e-2 + 1e-4 * np.abs(e_q).max())
    assert np.allclose(e_t, e_q, atol=2e-2 + 1e-4 * np.abs(e_q).max())


def toy_model(seed=0, N=6, K=3):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(N, N)) * 0.5
    Q = A @ A.T + 0.8 * np.eye(N)
    mask = np.triu(rng.random((N, N)) < 0.6, 1)
    mask[0, 1] = mask[1, 2] = mask[2, 3] = mask[3, 4] = mask[4, 5] = True
    mask = mask | mask.T
    Q = np.where(mask | np.eye(N, dtype=bool), Q, 0.0)
    Q = Q + (abs(np.linalg.eigvalsh(Q).min()) + 0.3) * np.eye(N) if np.linalg.eigvalsh(Q).min() < 0.2 else Q
    c = rng.normal(size=N) * 1.2 + 1.0
    return build_potts_from_Q(Q, c, 0.0, K)


@pytest.mark.parametrize("backend", ["jax", "thrml"])
def test_toy_marginals_exact(backend):
    m = toy_model()
    K, N = m.K, m.N
    X = np.array(list(itertools.product(range(K), repeat=N)))
    E = m.energy(X)
    p = np.exp(-(E - E.min()))
    p /= p.sum()
    exact = np.stack([(p[:, None] * (X == a)).sum(0) for a in range(K)], axis=1)  # (N, K)
    sm = PottsSampler(m, backend)
    C = 64
    key = jax.random.PRNGKey(1)
    init = sm.init_state("random", C, key)
    from thrml import SamplingSchedule
    sched = SamplingSchedule(20, 600, 2)
    samples, _ = sm.run(jax.random.split(key, C), init, 1.0, sched)
    s = np.asarray(samples).reshape(-1, N)
    emp = np.stack([(s == a).mean(0) for a in range(K)], axis=1)
    assert np.abs(emp - exact).max() < 0.02, (emp, exact)
    # and at beta=2 (weights scaling is traced)
    p2 = np.exp(-2 * (E - E.min()))
    p2 /= p2.sum()
    exact2 = np.stack([(p2[:, None] * (X == a)).sum(0) for a in range(K)], axis=1)
    samples, _ = sm.run(jax.random.split(key, C), init, 2.0, sched)
    s = np.asarray(samples).reshape(-1, N)
    emp2 = np.stack([(s == a).mean(0) for a in range(K)], axis=1)
    assert np.abs(emp2 - exact2).max() < 0.02


def test_toy_anneal_finds_ground_state():
    m = toy_model(3)
    X = np.array(list(itertools.product(range(m.K), repeat=m.N)))
    gs = X[np.argmin(m.energy(X))]
    sm = PottsSampler(m, "jax")
    init = sm.init_state("random", 8, jax.random.PRNGKey(0))
    r = sm.anneal(jax.random.PRNGKey(2), init, np.geomspace(0.1, 50, 20), 10)
    assert np.isclose(r["best_energy"].min(), m.energy(gs) - 0.0, atol=1e-3)
    assert r["energy_trace"].shape == (8, 20)


def test_backends_agree_on_tiny(tiny_problem, tiny_model):
    from tomo.config import load_config
    cfg = load_config({"grid": {"n": 12}})
    kw = dict(n_chains=8, posterior=dict(n_warmup=100, n_samples=100, steps_per_sample=3),
              anneal=dict(n_betas=8, sweeps_per_beta=5), model=tiny_model)
    rj = sample_potts(tiny_problem, K_TINY, None, cfg, jax.random.PRNGKey(0), backend="jax", **kw)
    rt = sample_potts(tiny_problem, K_TINY, None, cfg, jax.random.PRNGKey(1), backend="thrml", **kw)
    d = np.abs(rj["mean"] - rt["mean"]) / tiny_model.setup.Delta
    assert d.mean() < 0.12 and d.max() < 0.6
    assert abs(rj["energy_trace"].mean() - rt["energy_trace"].mean()) < 0.05 * abs(rj["energy_trace"].mean()) + 3


def test_tiny_end_to_end(tiny_problem, tiny_model):
    from tomo.config import load_config
    cfg = load_config({"grid": {"n": 12}})
    p = tiny_problem
    res = sample_potts(p, K_TINY, None, cfg, jax.random.PRNGKey(0), model=tiny_model, n_chains=8,
                       posterior=dict(n_warmup=200, n_samples=100, steps_per_sample=2),
                       anneal=dict(n_betas=15, sweeps_per_beta=10))
    for k in ("mean", "std", "map", "samples", "rhat", "energy_trace", "n_blocks", "n_spins", "max_degree",
              "time", "invalid_frac"):
        assert k in res
    S = tiny_model.setup
    assert res["samples"].shape == (8, 100, tiny_model.N)
    assert np.all(res["samples"] >= -1e-12) and np.all(res["samples"] <= S.eps_max + 1e-9)
    assert res["invalid_frac"] == 0.0
    tik = rel_l2(S.eps_tik, p.eps_true)
    assert rel_l2(res["mean"], p.eps_true) <= 1.35 * tik
    assert rel_l2(res["map"], p.eps_true) <= 1.6 * tik
    # MAP energy is below typical posterior energies and not above the Tikhonov start
    assert res["map_energy"] < res["energy_trace"].mean()
    assert res["map_energy"] <= tiny_model.energy(S.x0) + 1e-6
    assert res["n_random_chains"] == 2 and 0.0 <= res["frac_top"] <= 1.0
    assert abs(res["map_energy"] - tiny_model.energy(res["map_lev"])) < 1e-9
    split = coupling_split(S)
    assert 0 < split["frac_laplacian"] < 1 and abs(split["frac_laplacian"] + split["frac_TtT"] - 1) < 1e-12


def test_Q_matches_spec_energy(tiny_model, tiny_problem):
    """Independent check of Q, c, const against the spec energy written out directly."""
    S, p = tiny_model.setup, tiny_problem
    rng = np.random.default_rng(3)
    x = rng.integers(0, K_TINY, tiny_model.N)
    T = np.asarray(p.T, float)
    r = (p.b - S.Delta * T @ x) / p.sigma
    xi = S.Delta * x
    E_spec = 0.5 * r @ r + 0.5 * S.lam * xi @ (p.L @ xi)
    assert abs(tiny_model.energy(x) - E_spec) < 1e-8 * abs(E_spec)


def test_blocks_partition_and_padding(tiny_model):
    m = tiny_model
    allidx = np.concatenate(m.blocks_idx)
    assert np.array_equal(np.sort(allidx), np.arange(m.N))
    sizes = {len(b) for b in m.blocks_idx}
    assert len(sizes) > 1  # uneven blocks exercise the padded scan
    assert check_coloring(m.N, m.edges, m.colors)
    # every nonzero off-diagonal coupling is an edge (jax matvec and colouring see the same graph)
    off = np.abs(m.Q - np.diag(np.diag(m.Q)))
    assert len(m.edges) == int((off > 0).sum() // 2)
