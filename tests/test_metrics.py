import types
import numpy as np
from tomo import metrics as M


def _grid(n):
    xc = (np.arange(n) + 0.5) / n
    return types.SimpleNamespace(n=n, xc=xc, yc=xc, dx=1 / n, dy=1 / n)


def _setup(n=16):
    g = _grid(n)
    X, Y = np.meshgrid(g.xc, g.xc)
    mask = (X - .5) ** 2 + (Y - .5) ** 2 < .2 ** 2
    img = np.exp(-((X - .6) ** 2 + (Y - .4) ** 2) / .02) * mask
    return g, mask, img


def test_basic():
    g, mask, img = _setup()
    t = img[mask]
    assert M.rel_l2(t, t) == 0
    assert abs(M.rel_l2(2 * t, t) - 1) < 1e-12
    assert M.ssim_img(M.to_img(t, mask), img, mask) > 0.999
    assert M.ssim_img(M.to_img(t[::-1], mask), img, mask) < 0.9
    assert M.peak_error_cm(t, t, g, mask) == 0
    e = np.roll(img, 1, axis=1)[mask]
    assert abs(M.peak_error_cm(e, t, g, mask) - 100 * 1 / 16) < 1e-9
    assert M.power_error(1.1 * t, t, 0.01) < 0.1 + 1e-9
    T = np.eye(3); b = np.array([1., 2, 3])
    assert abs(M.reduced_chi2(T, b, np.ones(3), np.zeros(3)) - 14 / 3) < 1e-12


def test_coverage():
    rng = np.random.default_rng(0)
    true = rng.normal(size=4000)
    s = rng.normal(size=(500, 4000))
    assert abs(M.coverage(s, true, 0.68) - 0.68) < 0.03
    assert abs(M.coverage((np.zeros(4000), np.ones(4000)), rng.normal(size=4000), 0.95) - 0.95) < 0.02


def test_rhat_and_convergence():
    rng = np.random.default_rng(1)
    r = M.split_rhat(rng.normal(size=(4, 400, 20)))
    assert np.all(np.abs(r - 1) < 0.05)
    x = rng.normal(size=(4, 400, 5)) + np.arange(4)[:, None, None] * 3
    assert np.all(M.split_rhat(x) > 1.05)
    assert M.sweeps_to_converge(rng.normal(size=(4, 400, 5)), 10) is not None
    assert M.sweeps_to_converge(x, 10) is None


def test_summarize():
    g, mask, img = _setup()
    n = mask.sum()
    T = np.random.default_rng(0).random((5, n))
    t = img[mask]
    p = types.SimpleNamespace(grid=g, mask=mask, T=T, b=T @ t, sigma=np.ones(5), eps_true=t)
    s = M.summarize(p, {"mean": t, "std": np.full(n, .1)})
    assert s["rel_l2"] == 0 and "coverage68" in s and "coverage95" in s
    smp = t + 0.1 * np.random.default_rng(2).normal(size=(4, 50, n))
    s = M.summarize(p, {"mean": t, "samples": smp})
    assert "coverage95" in s
