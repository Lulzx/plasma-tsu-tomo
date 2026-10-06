"""TCV real-geometry tests (vendored data in tomo/data/tcv; optional 1000-phantom set skipped if absent)."""
import numpy as np
import pytest

from tomo import baselines, metrics
from tomo.config import load_config
from tomo.geometry import _points_in_polygon, geometry_matrix
from tomo.tcv import (fetched_path, make_tcv_grid, make_tcv_phantom, make_tcv_problem, tcv_chords,
                      tcv_mask, tcv_vessel_polygon, coverage_stats, TCV_PHANTOMS)

SMALL = {"grid": {"n": 10, "ny": 30}, "data_grid": {"n": 20, "ny": 60}}


def _cfg(small=True):
    cfg = load_config("configs/tcv.yaml")
    if small:
        cfg["grid"].update(SMALL["grid"]); cfg["data_grid"].update(SMALL["data_grid"])
    return cfg


def test_chords_cross_vessel():
    ch = tcv_chords()
    assert len(ch.p0) == 120 and set(ch.camera) == {0, 1, 2, 3}
    g = make_tcv_grid(41, 120)
    T = geometry_matrix(ch, g, tcv_mask(g))
    assert (T.sum(1) > 0.25).all()         # every LoS has > 0.25 m inside the vessel
    # every pinhole lies outside the vessel polygon or on its boundary side (outside the pixel mask)
    assert not _points_in_polygon(ch.p0[:, 0], ch.p0[:, 1], tcv_vessel_polygon()).any()


def test_mask_and_shapes():
    cfg = _cfg(small=False)
    p = make_tcv_problem(cfg, "peaked", 0)
    assert p.grid.shape == (60, 20) and p.mask.shape == (60, 20)
    N = int(p.mask.sum())
    assert 800 <= N <= 1500
    assert p.T.shape == (120, N) and p.L.shape == (N, N) and p.eps_true.shape == (N,)
    assert p.fine_img.shape == (240, 80) and p.T_fine.shape[0] == 120
    assert abs(p.T.sum() - p.T_fine.sum()) / p.T.sum() < 0.02
    cov = coverage_stats(p.T)
    assert cov["frac_crossed"] > 0.95 and cov["median_chords_per_pixel"] >= 2


@pytest.mark.parametrize("name", TCV_PHANTOMS + ["random"])
def test_phantoms(name):
    g = make_tcv_grid(20, 60); m = tcv_mask(g)
    img = make_tcv_phantom(name, g, m, rng=1)
    assert img.shape == g.shape and img.min() >= 0 and abs(img.max() - 1) < 1e-9 and (img[~m] == 0).all()


def test_problem_deterministic_and_baselines():
    cfg = _cfg()
    p1, p2 = make_tcv_problem(cfg, "random", 3), make_tcv_problem(cfg, "random", 3)
    assert np.array_equal(p1.b, p2.b)
    tk = baselines.tikhonov(p1.T, p1.b, p1.sigma, p1.L, method="evidence")
    assert metrics.rel_l2(tk["mean"], p1.eps_true) < 0.6


def test_ebm_prepare():
    from tomo.ebm_common import prepare
    p = make_tcv_problem(_cfg(), "peaked", 0)
    s = prepare(p, 4)
    assert p.eps_max is not None or s is not None


@pytest.mark.skipif(fetched_path("phantoms.npy") is None, reason="run `python -m tomo.tcv --fetch`")
def test_hamm_phantom():
    g = make_tcv_grid(20, 60); m = tcv_mask(g)
    img = make_tcv_phantom("hamm:0", g, m)
    assert img.max() > 0 and (img[~m] == 0).all()
