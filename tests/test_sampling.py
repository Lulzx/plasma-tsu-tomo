"""Tests for tomo.sampling (thrml and pure-JAX backends)."""
import itertools

import jax
import numpy as np
import pytest
import scipy.sparse as sp

from tomo.sampling import (IsingProblem, IsingSampler, anneal_ising, bits_to_spins_qubo, check_coloring,
                           geometric_betas, greedy_coloring, ising_energy, parallel_tempering, run_ising)


def all_states(n):
    return np.array(list(itertools.product([0, 1], repeat=n)), dtype=bool)  # (2^n, n)


def random_ising(n, seed, density=0.7, hs=0.5, js=0.5):
    rng = np.random.default_rng(seed)
    Jd = np.triu(rng.normal(0, js, (n, n)) * (rng.random((n, n)) < density), 1)
    return IsingProblem.from_dense(rng.normal(0, hs, n), Jd + Jd.T, offset=0.3)


def exact_stats(prob, beta=1.0):
    S = all_states(prob.n)
    E = prob.energy(S)
    w = np.exp(-beta * (E - E.min()))
    w /= w.sum()
    s = np.where(S, 1.0, -1.0)
    m = w @ s
    C = (s * w[:, None]).T @ s
    return m, C, E


# ------------------------------------------------------------------ conversion / colouring
@pytest.mark.parametrize("sparse", [False, True])
def test_bits_to_spins_exact(sparse):
    rng = np.random.default_rng(0)
    for _ in range(5):
        n = 8
        A = rng.normal(size=(n, n)) * (rng.random((n, n)) < 0.6)
        Q = (A + A.T) / 2
        c = rng.normal(size=n)
        const = rng.normal()
        prob = bits_to_spins_qubo(sp.csr_matrix(Q) if sparse else Q, c, const)
        U = all_states(n)
        u = U.astype(float)
        Eb = np.einsum("bi,ij,bj->b", u, Q, u) + u @ c + const
        assert np.allclose(prob.energy(U), Eb, atol=1e-10)
        assert np.all(prob.rows < prob.cols)


def test_bits_to_spins_nonsymmetric_input():
    rng = np.random.default_rng(1)
    Q = rng.normal(size=(6, 6))
    c = rng.normal(size=6)
    prob = bits_to_spins_qubo(Q, c, 0.5)
    U = all_states(6)
    u = U.astype(float)
    assert np.allclose(prob.energy(U), np.einsum("bi,ij,bj->b", u, Q, u) + u @ c + 0.5)


def test_greedy_coloring_proper():
    rng = np.random.default_rng(2)
    n = 200
    A = np.triu(rng.random((n, n)) < 0.05, 1)
    edges = np.argwhere(A)
    col = greedy_coloring(n, edges)
    assert check_coloring(n, edges, col)
    assert col.min() == 0
    # bipartite grid -> 2 colours
    k = 6
    ed = [(i * k + j, i * k + j + 1) for i in range(k) for j in range(k - 1)] + \
         [(i * k + j, (i + 1) * k + j) for i in range(k - 1) for j in range(k)]
    assert greedy_coloring(k * k, np.array(ed)).max() + 1 == 2
    assert not check_coloring(3, np.array([[0, 1]]), np.array([0, 0, 1]))


def test_energy_jax_matches_numpy():
    prob = random_ising(10, 3)
    S = all_states(10)
    assert np.allclose(np.asarray(ising_energy(prob, S)), prob.energy(S), atol=1e-4)
    assert ising_energy(prob, S.reshape(4, 256, 10)).shape == (4, 256)


# ------------------------------------------------------------------ sampling correctness
@pytest.mark.parametrize("backend", ["thrml", "jax"])
def test_marginals_and_correlations(backend):
    n = 10
    prob = random_ising(n, 4)
    colors = greedy_coloring(n, prob.edges)
    m_ex, C_ex, _ = exact_stats(prob)
    C, S = 64, 400
    res = run_ising(prob, colors, dict(n_warmup=100, n_samples=S, steps_per_sample=2), C,
                    jax.random.key(0), backend=backend)
    smp = res["samples"]
    assert smp.shape == (C, S, n) and smp.dtype == bool
    assert res["energy_trace"].shape == (C, S)
    s = np.where(smp, 1.0, -1.0)
    # standard error: chains are independent; use per-chain means (autocorrelation-safe)
    cm = s.mean(1)  # (C, n)
    m_hat, se = cm.mean(0), cm.std(0, ddof=1) / np.sqrt(C)
    assert np.all(np.abs(m_hat - m_ex) < 4.5 * se + 1e-3), (m_hat, m_ex, se)
    cc = np.einsum("csi,csj->cij", s, s) / S
    C_hat, se2 = cc.mean(0), cc.std(0, ddof=1) / np.sqrt(C)
    off = ~np.eye(n, dtype=bool)
    assert np.all((np.abs(C_hat - C_ex) < 4.5 * se2 + 1e-3)[off])
    # energy trace consistent with numpy energy
    assert np.allclose(res["energy_trace"], prob.energy(smp), atol=1e-3)


def test_beta_scaling():
    n = 8
    prob = random_ising(n, 5)
    colors = greedy_coloring(n, prob.edges)
    m_ex, _, _ = exact_stats(prob, beta=2.0)
    res = run_ising(prob, colors, (50, 300, 2), 64, jax.random.key(1), beta=2.0)
    m = np.where(res["samples"], 1.0, -1.0).mean(1)
    se = m.std(0, ddof=1) / np.sqrt(64)
    assert np.all(np.abs(m.mean(0) - m_ex) < 4.5 * se + 1e-3)


def test_backends_agree_in_distribution():
    n = 12
    prob = random_ising(n, 6, density=0.4)
    colors = greedy_coloring(n, prob.edges)
    sched = (100, 300, 2)
    r1 = run_ising(prob, colors, sched, 64, jax.random.key(2), backend="thrml")
    r2 = run_ising(prob, colors, sched, 64, jax.random.key(3), backend="jax")
    m1 = np.where(r1["samples"], 1.0, -1.0).mean(1)
    m2 = np.where(r2["samples"], 1.0, -1.0).mean(1)
    se = np.sqrt(m1.var(0, ddof=1) / 64 + m2.var(0, ddof=1) / 64)
    assert np.all(np.abs(m1.mean(0) - m2.mean(0)) < 4.5 * se + 1e-3)
    assert r1["n_blocks"] == r2["n_blocks"]


def test_jax_backend_ell_path_matches_dense():
    n = 10
    prob = random_ising(n, 7)
    colors = greedy_coloring(n, prob.edges)
    m_ex, _, _ = exact_stats(prob)
    for dense in (True, False):
        sm = IsingSampler(prob, colors, backend="jax", dense=dense)
        res = run_ising(prob, colors, (50, 300, 2), 64, jax.random.key(4), sampler=sm)
        m = np.where(res["samples"], 1.0, -1.0).mean(1)
        se = m.std(0, ddof=1) / np.sqrt(64)
        assert np.all(np.abs(m.mean(0) - m_ex) < 4.5 * se + 1e-3)


def test_init_array_and_modes():
    n = 6
    prob = random_ising(n, 8)
    colors = greedy_coloring(n, prob.edges)
    init = np.zeros((3, n), bool)
    for mode in (init, "random", "hinton", None):
        r = run_ising(prob, colors, (1, 2, 1), 3, jax.random.key(0), init=mode, backend="jax")
        assert r["samples"].shape == (3, 2, n)
    with pytest.raises(ValueError):
        run_ising(prob, np.zeros(n, int), (1, 2, 1), 3, jax.random.key(0), backend="jax")


# ------------------------------------------------------------------ annealing / tempering
@pytest.mark.parametrize("backend", ["thrml", "jax"])
def test_anneal_finds_ground_state(backend):
    n = 12
    prob = random_ising(n, 9, density=0.8, hs=0.3, js=1.0)
    colors = greedy_coloring(n, prob.edges)
    E_min = prob.energy(all_states(n)).min()
    betas = geometric_betas(0.1, 20.0, 25)
    res = anneal_ising(prob, colors, betas, 10, 16, jax.random.key(5), backend=backend)
    assert res["final"].shape == (16, n) and res["energy_trace"].shape == (16, 25)
    assert np.isclose(res["best_energy"].min(), E_min, atol=1e-3)
    assert np.allclose(res["best_energy"], prob.energy(res["best"]), atol=1e-3)
    assert res["energy_trace"][:, -1].mean() < res["energy_trace"][:, 0].mean()


def test_anneal_ferromagnet():
    n = 12
    r = np.arange(n - 1)
    prob = IsingProblem(np.zeros(n), r, r + 1, np.ones(n - 1), 0.0)  # chain
    colors = greedy_coloring(n, prob.edges)
    res = anneal_ising(prob, colors, geometric_betas(0.2, 10, 20), 10, 8, jax.random.key(0))
    assert np.isclose(res["best_energy"].min(), -(n - 1), atol=1e-4)


@pytest.mark.parametrize("backend", ["thrml", "jax"])
def test_parallel_tempering_visits_both_modes(backend):
    # strong ferromagnet, tiny field: two modes (all up / all down) separated by a barrier
    n = 8
    Jd = np.ones((n, n)) - np.eye(n)
    prob = IsingProblem.from_dense(np.full(n, 0.01), Jd * 0.6)
    colors = greedy_coloring(n, prob.edges)
    betas = np.array([0.05, 0.1, 0.2, 0.4, 0.8, 1.0])
    res = parallel_tempering(prob, colors, betas, 600, 2, jax.random.key(0), n_chains=4, backend=backend)
    assert res["samples"].shape == (4, 300, n)
    mag = np.where(res["samples"], 1.0, -1.0).mean(-1)
    assert np.all(res["swap_rate"] > 0.05)
    # every chain visits both magnetisation signs
    assert np.all((mag > 0.5).any(1)) and np.all((mag < -0.5).any(1))
    assert res["betas"][res["target"]] == 1.0
    # cold replica is not stuck: crossing count large
    assert np.mean((np.diff(np.sign(mag), axis=1) != 0).sum(1)) >= 2
