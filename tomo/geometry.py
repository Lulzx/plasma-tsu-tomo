"""Grid, D-shaped plasma mask, bolometer chord fans and Siddon ray tracing."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Grid:
    """Square pixel grid over [xmin,xmax] x [ymin,ymax]; flat index = iy*n + ix."""
    n: int
    xmin: float = 0.0
    xmax: float = 1.0
    ymin: float = 0.0
    ymax: float = 1.0

    @property
    def dx(self) -> float:
        return (self.xmax - self.xmin) / self.n

    @property
    def dy(self) -> float:
        return (self.ymax - self.ymin) / self.n

    @property
    def xc(self) -> np.ndarray:
        return self.xmin + (np.arange(self.n) + 0.5) * self.dx

    @property
    def yc(self) -> np.ndarray:
        return self.ymin + (np.arange(self.n) + 0.5) * self.dy

    @property
    def X(self) -> np.ndarray:
        return np.meshgrid(self.xc, self.yc, indexing="xy")[0]

    @property
    def Y(self) -> np.ndarray:
        return np.meshgrid(self.xc, self.yc, indexing="xy")[1]


def make_grid(n: int, size: float = 1.0) -> Grid:
    """Grid of n x n pixels over the box [0,size]^2."""
    return Grid(int(n), 0.0, float(size), 0.0, float(size))


def d_boundary(R0, Z0, a, kappa, delta, npts: int = 4096) -> np.ndarray:
    """Closed boundary polygon (npts,2): R=R0+a cos(t+delta sin t), Z=Z0+kappa a sin t."""
    t = np.linspace(0.0, 2 * np.pi, npts, endpoint=False)
    return np.stack([R0 + a * np.cos(t + delta * np.sin(t)), Z0 + kappa * a * np.sin(t)], axis=1)


def _points_in_polygon(px, py, poly) -> np.ndarray:
    """Even-odd ray casting; px, py arrays of equal shape."""
    inside = np.zeros(px.shape, dtype=bool)
    x1, y1 = poly[:, 0], poly[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    for xa, ya, xb, yb in zip(x1, y1, x2, y2):
        if ya == yb:
            continue
        cond = (ya > py) != (yb > py)
        xint = xa + (py - ya) * (xb - xa) / (yb - ya)
        inside ^= cond & (px < xint)
    return inside


def d_shape_mask(grid: Grid, R0, Z0, a, kappa, delta) -> np.ndarray:
    """Boolean (n,n) mask; pixel active if its centre lies inside the D-shaped boundary."""
    poly = d_boundary(R0, Z0, a, kappa, delta)
    return _points_in_polygon(grid.X, grid.Y, poly)


@dataclass
class Chords:
    """Line-of-sight segments: start p0 (M,2), end p1 (M,2), camera index (M,)."""
    p0: np.ndarray
    p1: np.ndarray
    camera: np.ndarray


def _clip_param(p0, p1, box):
    """Slab clipping of p0 + t (p1-p0), t in [0,1], to box=(xmin,xmax,ymin,ymax).

    Returns (tmin, tmax) or None if the segment misses the box.
    """
    d = p1 - p0
    tmin, tmax = 0.0, 1.0
    for k in range(2):
        lo, hi = box[2 * k], box[2 * k + 1]
        if d[k] == 0.0:
            if p0[k] < lo or p0[k] > hi:
                return None
        else:
            ta, tb = (lo - p0[k]) / d[k], (hi - p0[k]) / d[k]
            if ta > tb:
                ta, tb = tb, ta
            tmin, tmax = max(tmin, ta), min(tmax, tb)
            if tmin >= tmax:
                return None
    return tmin, tmax


def make_fans(cfg_cameras, box=(0.0, 1.0, 0.0, 1.0)) -> Chords:
    """Build chord fans. Each camera dict: name, origin, n_chords, fan_center_deg, fan_width_deg.

    Chord angles are evenly spaced (pixel-centred, i.e. at bin centres of the fan) over
    the fan width; directions use the maths convention (0 deg = +x, 90 deg = +y). Chords
    run from the camera origin until they leave ``box`` and are clipped to it.
    """
    p0s, p1s, cams = [], [], []
    diag = 2.0 * np.hypot(box[1] - box[0], box[3] - box[2])
    for ci, cam in enumerate(cfg_cameras):
        n = int(cam["n_chords"])
        w = float(cam["fan_width_deg"])
        ang = np.deg2rad(cam["fan_center_deg"] + (np.arange(n) + 0.5) / n * w - w / 2)
        o = np.asarray(cam["origin"], float)
        for th in ang:
            far = o + diag * np.array([np.cos(th), np.sin(th)])
            tt = _clip_param(o, far, box)
            if tt is None:
                continue
            p0s.append(o + tt[0] * (far - o))
            p1s.append(o + tt[1] * (far - o))
            cams.append(ci)
    return Chords(np.array(p0s), np.array(p1s), np.array(cams, dtype=int))


def siddon(p0, p1, grid: Grid):
    """Exact chord lengths of segment p0->p1 through each pixel (Siddon).

    All grid-line crossings (x and y planes) inside the clipped segment are sorted;
    each sub-segment lies in exactly one pixel, identified from its midpoint, which
    makes vertical/horizontal rays, corner crossings and rays starting outside the
    box all work. Returns (idx, length): sorted unique flat pixel indices
    (iy*n+ix) and lengths in grid units (metres). Empty arrays if the box is missed.
    """
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    d = p1 - p0
    seg_len = float(np.hypot(*d))
    empty = (np.zeros(0, dtype=np.int64), np.zeros(0))
    if seg_len == 0.0:
        return empty
    tt = _clip_param(p0, p1, (grid.xmin, grid.xmax, grid.ymin, grid.ymax))
    if tt is None:
        return empty
    tmin, tmax = tt
    parts = [np.array([tmin, tmax])]
    if d[0] != 0.0:
        t = (grid.xmin + np.arange(grid.n + 1) * grid.dx - p0[0]) / d[0]
        parts.append(t[(t > tmin) & (t < tmax)])
    if d[1] != 0.0:
        t = (grid.ymin + np.arange(grid.n + 1) * grid.dy - p0[1]) / d[1]
        parts.append(t[(t > tmin) & (t < tmax)])
    ts = np.sort(np.concatenate(parts))
    dt = np.diff(ts)
    mid = p0[None, :] + (0.5 * (ts[:-1] + ts[1:]))[:, None] * d[None, :]
    ix = np.clip(np.floor((mid[:, 0] - grid.xmin) / grid.dx).astype(np.int64), 0, grid.n - 1)
    iy = np.clip(np.floor((mid[:, 1] - grid.ymin) / grid.dy).astype(np.int64), 0, grid.n - 1)
    flat = iy * grid.n + ix
    keep = dt > 0
    uniq, inv = np.unique(flat[keep], return_inverse=True)
    return uniq, np.bincount(inv, weights=dt[keep] * seg_len, minlength=len(uniq))


def geometry_matrix(chords: Chords, grid: Grid, mask=None, dtype=np.float32) -> np.ndarray:
    """T[i,j] = length of chord i in pixel j. Shape (M, N_active) if mask else (M, n*n)."""
    M = len(chords.p0)
    full = np.zeros((M, grid.n * grid.n), dtype=np.float64)
    for i in range(M):
        idx, ln = siddon(chords.p0[i], chords.p1[i], grid)
        full[i, idx] = ln
    if mask is not None:
        full = full[:, np.flatnonzero(np.asarray(mask).ravel())]
    return full.astype(dtype)


def neighbor_edges(mask) -> np.ndarray:
    """4-neighbour pairs (j<k) among active pixels, in active-pixel indexing, shape (E,2)."""
    mask = np.asarray(mask, bool)
    n_r, n_c = mask.shape
    amap = -np.ones(mask.shape, dtype=np.int64)
    amap[mask] = np.arange(mask.sum())
    e = []
    for a, b in ((amap[:, :-1], amap[:, 1:]), (amap[:-1, :], amap[1:, :])):
        ok = (a >= 0) & (b >= 0)
        e.append(np.stack([a[ok], b[ok]], axis=1))
    e = np.concatenate(e, axis=0)
    e = np.sort(e, axis=1)
    return e[np.lexsort((e[:, 1], e[:, 0]))]


def laplacian(mask) -> np.ndarray:
    """Dense float64 graph Laplacian with x^T L x = sum_edges (x_j - x_k)^2."""
    e = neighbor_edges(mask)
    N = int(np.asarray(mask).sum())
    L = np.zeros((N, N))
    np.add.at(L, (e[:, 0], e[:, 0]), 1.0)
    np.add.at(L, (e[:, 1], e[:, 1]), 1.0)
    np.add.at(L, (e[:, 0], e[:, 1]), -1.0)
    np.add.at(L, (e[:, 1], e[:, 0]), -1.0)
    return L


def checkerboard(mask) -> np.ndarray:
    """(N_active,) int in {0,1}: (ix+iy) % 2 of each active pixel (row-major order)."""
    iy, ix = np.nonzero(np.asarray(mask, bool))
    return ((ix + iy) % 2).astype(int)
