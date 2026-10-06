import numpy as np
import pytest
from scipy import integrate, stats

from tomo.baselines import tuned_tikhonov
from tomo.positive import (gaussian_posterior_eps, log_laplace, truncated_gaussian_posterior,
                           truncated_gaussian_sample)


@pytest.mark.parametrize("method", ["gibbs", "ess", "mixed"])
def test_1d_truncnorm(method):
    # N(mean 0.3, sd 0.5) truncated to [0, 0.8]
    mu, sd, lo, hi = 0.3, 0.5, 0.0, 0.8
    P, h = np.array([[1 / sd ** 2]]), np.array([mu / sd ** 2])
    r = truncated_gaussian_sample(P, h, lo, hi, n_samples=4000, n_warmup=50, n_chains=4, key=1, method=method)
    ref = stats.truncnorm((lo - mu) / sd, (hi - mu) / sd, loc=mu, scale=sd)
    assert abs(r["mean"][0] - ref.mean()) < 0.01
    assert abs(r["std"][0] - ref.std()) < 0.01
    assert r["samples"].min() >= lo and r["samples"].max() <= hi
    assert r["rhat"][0] < 1.05


@pytest.mark.parametrize("method", ["gibbs", "ess", "mixed"])
def test_2d_vs_quadrature(method):
    Sig = np.array([[1.0, 0.8], [0.8, 1.0]])
    mu = np.array([0.2, -0.1])
    P = np.linalg.inv(Sig)
    h = P @ mu
    lo, hi = 0.0, 1.5
    r = truncated_gaussian_sample(P, h, lo, hi, n_samples=6000, n_warmup=100, n_chains=4, key=2, method=method)
    pdf = lambda y, x: np.exp(-0.5 * (np.array([x, y]) - mu) @ P @ (np.array([x, y]) - mu))  # noqa: E731
    I = lambda f: integrate.dblquad(lambda y, x: f(x, y) * pdf(y, x), lo, hi, lo, hi)[0]  # noqa: E731
    Z = I(lambda x, y: 1.0)
    m = np.array([I(lambda x, y: x), I(lambda x, y: y)]) / Z
    v = np.array([I(lambda x, y: x * x), I(lambda x, y: y * y)]) / Z - m ** 2
    c = I(lambda x, y: x * y) / Z - m[0] * m[1]
    S = r["samples"].reshape(-1, 2)
    assert np.allclose(S.mean(0), m, atol=0.02)
    assert np.allclose(S.var(0), v, atol=0.02)
    assert abs(np.cov(S.T)[0, 1] - c) < 0.02


def test_untruncated_limit_equals_gaussian():
    rng = np.random.default_rng(0)
    B = rng.standard_normal((6, 6))
    P = B @ B.T + 2 * np.eye(6)
    h = rng.standard_normal(6)
    Sig = np.linalg.inv(P)
    r = truncated_gaussian_sample(P, h, -50.0, 50.0, n_samples=6000, n_warmup=100, n_chains=4, key=3, method="mixed")
    assert np.allclose(r["mean"], Sig @ h, atol=0.06)
    assert np.allclose(r["std"], np.sqrt(np.diag(Sig)), atol=0.05)


def test_problem_wrapper(tiny_problem):
    p = tiny_problem
    lam = tuned_tikhonov(p.T, p.b, p.sigma, p.L)["lam"]
    g = gaussian_posterior_eps(p, lam)
    # huge box -> Gaussian posterior moments
    r = truncated_gaussian_posterior(p, lam, lower=-1e6, upper=None, n_samples=1500, n_warmup=200, n_chains=4, seed=0)
    assert np.all(np.abs(r["mean"] - g["mean"]) < 0.5 * g["std"])
    pos = truncated_gaussian_posterior(p, lam, 0.0, None, n_samples=300, n_warmup=100, n_chains=4, seed=0)
    assert pos["samples"].min() >= 0 and pos["samples"].shape == (4, 300, g["mean"].size)
    box = truncated_gaussian_posterior(p, lam, 0.0, 0.5 * g["mean"].max(), n_samples=300, n_warmup=100, n_chains=4, seed=0)
    assert box["samples"].max() <= 0.5 * g["mean"].max() * (1 + 1e-5)


def test_log_laplace(tiny_problem):
    p = tiny_problem
    r = log_laplace(p, n_samples=100)
    assert r["mean"].min() > 0 and np.all(np.isfinite(r["std"]))
    rel = np.linalg.norm(r["mean"] - p.eps_true) / np.linalg.norm(p.eps_true)
    assert rel < 0.6


def test_tail_sampler():
    import jax
    import jax.numpy as jnp
    from tomo.positive import _trunc_std_normal
    n = 20000
    for a, b in [(6.0, np.inf), (-np.inf, -3.0), (12.0, 13.0), (-0.5, 2.0)]:
        z = np.asarray(_trunc_std_normal(jax.random.PRNGKey(0), jnp.full(n, a, jnp.float32), jnp.full(n, b, jnp.float32)))
        ref = stats.truncnorm(a, b)
        assert np.all(z >= a) and np.all(z <= b)
        assert abs(z.mean() - ref.mean()) < 0.03 * max(ref.std(), 0.05) * 10 and abs(z.std() - ref.std()) < 0.1 * max(ref.std(), 0.05) + 0.01
