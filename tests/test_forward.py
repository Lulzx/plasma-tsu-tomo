import numpy as np
import pytest

from tomo.forward import eps_max_from_estimate, make_problem, pixel_area
from tomo.geometry import d_shape_mask, make_grid
from tomo.phantoms import PHANTOMS, make_phantom


@pytest.mark.parametrize("name", PHANTOMS + ["random"])
@pytest.mark.parametrize("n", [32, 128])
def test_phantoms(cfg, name, n):
    g = make_grid(n)
    m = d_shape_mask(g, **cfg["plasma"])
    img = make_phantom(name, g, m, rng=np.random.default_rng(0))
    assert img.shape == (n, n) and (img >= 0).all()
    assert (img[~m] == 0).all() and img[m].max() > 0.5
    assert np.isclose(img.max(), 1.0, atol=0.05) or name == "edge"


def test_random_phantom_rng():
    g = make_grid(32)
    m = d_shape_mask(g, 0.5, 0.5, 0.495, 1.03, 0.35)
    a = make_phantom("random", g, m, rng=np.random.default_rng(1))
    b = make_phantom("random", g, m, rng=np.random.default_rng(1))
    c = make_phantom("random", g, m, rng=np.random.default_rng(2))
    assert np.array_equal(a, b) and not np.allclose(a, c)


def test_problem_shapes_and_truth(problem):
    p = problem
    N = p.mask.sum()
    assert p.T.shape == (72, N) and p.T.dtype == np.float32
    assert p.eps_true.shape == (N,) and p.L.shape == (N, N) and p.eps_max is None
    assert p.fine_img.shape == (128, 128) and p.eps_true_img.shape == (32, 32)
    # block averaging conserves the integral except pixels the coarse mask drops
    tot_f = p.fine_img.sum() * p.fine_grid.dx**2
    tot_c = p.eps_true.sum() * pixel_area(p)
    assert abs(tot_c / tot_f - 1) < 0.02
    assert (p.sigma > 0).all()
    # no inverse crime, but coarse model must still roughly predict the data
    rel = np.linalg.norm(p.T @ p.eps_true - p.b_clean) / np.linalg.norm(p.b_clean)
    assert 0 < rel < 0.1


def test_noise_statistics(cfg):
    p = make_problem(cfg, "hollow", 5)
    assert np.allclose(p.sigma, 0.03 * np.abs(p.b_clean) + 0.01 * p.b_clean.max())
    z = (p.b - p.b_clean) / p.sigma
    assert abs(z.mean()) < 0.5 and 0.7 < z.std() < 1.3


def test_determinism(cfg):
    a, b = make_problem(cfg, "random", 7), make_problem(cfg, "random", 7)
    c = make_problem(cfg, "random", 8)
    assert np.array_equal(a.b, b.b) and np.array_equal(a.eps_true, b.eps_true)
    assert not np.allclose(a.b, c.b)


def test_calibration_offset(cfg):
    from tomo.config import load_config
    c = load_config({"noise": {"calib_offset": 0.05}})
    p = make_problem(c, "peaked", 0)
    assert len(np.unique(p.calib_gain)) == 3 and not np.allclose(p.calib_gain, 1)


def test_tiny_problem(tiny_problem):
    p = tiny_problem
    assert p.grid.n == 12 and p.T.shape[0] == 24 and 80 < p.mask.sum() < 130
    assert p.fine_grid.n == 48


def test_eps_max_helper():
    assert eps_max_from_estimate(np.array([-5.0, 1.0, 2.0]), 1.2) == pytest.approx(2.4)
