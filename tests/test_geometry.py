import numpy as np

from tomo.config import load_config, save_config
from tomo.geometry import (checkerboard, d_shape_mask, geometry_matrix, laplacian, make_fans,
                           make_grid, neighbor_edges, siddon, Chords)


def test_siddon_sum_equals_clipped_length():
    g = make_grid(17)
    rng = np.random.default_rng(1)
    for _ in range(200):
        p0, p1 = rng.uniform(-0.5, 1.5, (2, 2))
        idx, ln = siddon(p0, p1, g)
        # reference: slab clip
        d = p1 - p0
        lo, hi = 0.0, 1.0
        ok = True
        for k in range(2):
            if d[k] == 0:
                ok &= 0 <= p0[k] <= 1
            else:
                a, b = sorted(((0 - p0[k]) / d[k], (1 - p0[k]) / d[k]))
                lo, hi = max(lo, a), min(hi, b)
        ref = max(hi - lo, 0.0) * np.hypot(*d) if ok else 0.0
        assert abs(ln.sum() - ref) < 1e-12
        assert len(np.unique(idx)) == len(idx)


def test_axis_aligned_and_outside():
    g = make_grid(10)
    idx, ln = siddon([-1.0, 0.25], [2.0, 0.25], g)  # horizontal, starts outside
    assert len(idx) == 10 and np.allclose(ln, 0.1)
    assert np.all(idx // 10 == 2)
    idx, ln = siddon([0.55, 2.0], [0.55, -1.0], g)  # vertical, reversed
    assert len(idx) == 10 and np.allclose(ln, 0.1) and np.all(idx % 10 == 5)
    idx, ln = siddon([0.0, 0.0], [1.0, 1.0], g)  # exact diagonal through corners
    assert np.allclose(ln.sum(), np.sqrt(2)) and len(idx) == 10
    assert np.allclose(ln, np.sqrt(2) / 10)
    idx, ln = siddon([2, 2], [3, 3], g)  # misses
    assert len(idx) == 0
    idx, ln = siddon([0.3, 0.3], [0.3, 0.3], g)  # degenerate
    assert len(idx) == 0
    idx, ln = siddon([0.5, 0.5], [0.5, 0.5], g)
    assert len(idx) == 0


def test_uniform_disk_chord_length():
    g = make_grid(256)
    cx, cy, r = 0.5, 0.5, 0.4
    img = (((g.X - cx) ** 2 + (g.Y - cy) ** 2) <= r**2).astype(float).ravel()
    rng = np.random.default_rng(3)
    p0 = rng.uniform(0, 1, (400, 2))
    p1 = rng.uniform(0, 1, (400, 2))
    T = geometry_matrix(Chords(p0, p1, np.zeros(400, int)), g, dtype=np.float64)
    num, ana = T @ img, np.zeros(400)
    for i in range(400):
        d = p1[i] - p0[i]
        L = np.hypot(*d)
        u = d / L
        w = p0[i] - [cx, cy]
        bq, cq = w @ u, w @ w - r * r
        disc = bq * bq - cq
        if disc > 0:
            s0, s1 = max(-bq - np.sqrt(disc), 0), min(-bq + np.sqrt(disc), L)
            ana[i] = max(s1 - s0, 0)
    hit = ana > 0.05
    assert hit.sum() > 100
    assert abs(num[hit].sum() / ana[hit].sum() - 1) < 0.005
    assert np.median(np.abs(num[hit] / ana[hit] - 1)) < 0.005


def test_default_mask_and_cameras(cfg):
    g = make_grid(cfg["grid"]["n"])
    m = d_shape_mask(g, **cfg["plasma"])
    assert 750 <= m.sum() <= 850
    # D shape: top/bottom extreme points shifted to the inboard (small R) side
    ys, xs = np.nonzero(m)
    assert xs[ys == ys.min()].mean() < xs[ys == (ys.min() + ys.max()) // 2].mean()
    ch = make_fans(cfg["cameras"])
    assert len(ch.p0) == 72 and set(ch.camera) == {0, 1, 2}
    T = geometry_matrix(ch, g, m)
    assert T.dtype == np.float32 and T.shape == (72, m.sum())
    assert (T.sum(1) > 0.2).all()  # every chord intersects the plasma
    uncovered = ((T > 0).sum(0) == 0).mean()
    print("active px", m.sum(), "uncovered fraction", uncovered)
    assert uncovered < 0.02


def test_graph_helpers():
    m = np.zeros((5, 5), bool)
    m[1:4, 1:4] = True
    m[2, 2] = False
    e = neighbor_edges(m)
    assert (e[:, 0] < e[:, 1]).all() and len(e) == len({tuple(x) for x in e}) == 8
    L = laplacian(m)
    x = np.random.default_rng(0).normal(size=m.sum())
    assert np.isclose(x @ L @ x, ((x[e[:, 0]] - x[e[:, 1]]) ** 2).sum())
    c = checkerboard(m)
    assert (c[e[:, 0]] != c[e[:, 1]]).all()


def test_config_roundtrip(tmp_path):
    c = load_config({"schedule": {"n_chains": 4}})
    assert c["schedule"]["n_chains"] == 4 and c["schedule"]["init"] == "tikhonov"
    p = save_config(c, tmp_path / "out")
    assert load_config(p) == c
