"""Tests for tomo.ebm_binary."""
import itertools

import jax
import numpy as np
import pytest

from tomo import ebm_common as ec
from tomo.config import load_config
from tomo.ebm_binary import binary_decode, binary_encode, build_ising_binary, sample_binary
from tomo.sampling import check_coloring


def test_encode_roundtrip():
    x = np.arange(8)
    assert np.array_equal(binary_decode(binary_encode(x, 8)), x)
    with pytest.raises(ValueError):
        binary_encode(x, 6)


def test_energy_matches_pixel_energy(tiny_problem):
    prob, meta = build_ising_binary(tiny_problem, 8)
    s = meta["setup"]
    assert prob.n == 3 * s.N
    rng = np.random.default_rng(0)
    u = rng.random((30, prob.n)) < 0.5
    x = binary_decode(u.reshape(30, s.N, 3))
    np.testing.assert_allclose(prob.energy(u), ec.pixel_energy(x, s.Q, s.c, s.const), rtol=1e-8, atol=1e-6)
    x1 = np.vstack([s.x0, np.zeros(s.N, int), np.full(s.N, 7)])
    u1 = binary_encode(x1, 8).reshape(3, -1).astype(bool)
    np.testing.assert_allclose(prob.energy(u1), ec.pixel_energy(x1, s.Q, s.c, s.const), rtol=1e-8, atol=1e-6)


def test_colouring(tiny_problem):
    prob, meta = build_ising_binary(tiny_problem, 8)
    assert check_coloring(prob.n, prob.edges, meta["colors"])
    assert meta["n_blocks"] == len(np.unique(meta["colors"])) >= 3


def test_marginals_match_enumeration(tiny_problem, tiny_cfg):
    """4 pixels, K=4: exact enumeration of 4^4 states vs sampled marginals."""
    rng = np.random.default_rng(1)
    A = rng.normal(size=(4, 4)) * 0.3
    Q = A @ A.T + np.diag(rng.uniform(1.0, 2.0, 4))
    c = rng.normal(size=4)
    s = ec.EBMSetup(problem=tiny_problem, K=4, eps_max=1.0, Delta=1.0 / 3, lam=0.0, w=None,
                    x0=np.array([1, 2, 0, 3]), eps_tik=None, Q=Q, c=c, const=0.0)
    prob, meta = build_ising_binary(tiny_problem, 4, setup=s)
    states = np.array(list(itertools.product(range(4), repeat=4)))
    E = ec.pixel_energy(states, Q, c, 0.0)
    p = np.exp(-(E - E.min())); p /= p.sum()
    mean_ex = (p[:, None] * states).sum(0)
    var_ex = (p[:, None] * states ** 2).sum(0) - mean_ex ** 2
    cfg = load_config(tiny_cfg)
    r = sample_binary(tiny_problem, cfg, jax.random.PRNGKey(3), setup=s, schedule=dict(n_warmup=200, n_samples=1500, steps_per_sample=2),
                      n_chains=8, K=4)
    lev = r["levels"].reshape(-1, 4)
    np.testing.assert_allclose(lev.mean(0), mean_ex, atol=0.06)
    np.testing.assert_allclose(lev.std(0), np.sqrt(var_ex), atol=0.06)
    assert r["invalid_frac"] == 0 and r["n_spins"] == 8


def test_sample_result_dict(tiny_problem, tiny_cfg):
    r = sample_binary(tiny_problem, tiny_cfg, jax.random.PRNGKey(0), schedule=dict(n_warmup=50, n_samples=20, steps_per_sample=2), n_chains=4)
    N = tiny_problem.T.shape[1]
    assert r["mean"].shape == (N,) and r["samples"].shape == (4, 20, N)
    for k in ("std", "rhat", "n_spins", "n_blocks", "max_degree", "time", "invalid_frac"):
        assert k in r
