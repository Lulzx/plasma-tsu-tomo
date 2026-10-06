"""Synthetic emissivity phantoms, analytic in (x, y) so they evaluate on any grid.

All phantoms are evaluated at pixel centres of the given grid, have peak value ~1
(scaled by ``amplitude``), are >= 0 and are exactly zero outside ``mask``. The
normalised radius ``rho`` (0 at the magnetic axis, 1 on the boundary) is computed
from the analytic D-shaped boundary, so fine (128) and coarse (32) grids see the
same underlying field.
"""
from __future__ import annotations

import numpy as np

from .geometry import d_boundary

PHANTOMS = ["peaked", "hollow", "blob", "edge"]

#: Default plasma shape (matches configs/default.yaml); override with ``plasma=`` kwarg.
DEFAULT_PLASMA = dict(R0=0.5, Z0=0.5, a=0.495, kappa=1.03, delta=0.35)


def normalized_radius(grid, plasma=None, axis_shift=0.0):
    """Return (rho, theta) on ``grid``: rho = r / r_boundary(theta) about the axis.

    The axis is (R0 + axis_shift, Z0). The boundary is star-shaped about it for the
    default shape; r_boundary(theta) is interpolated from the dense boundary polygon.
    """
    p = dict(DEFAULT_PLASMA if plasma is None else plasma)
    poly = d_boundary(p["R0"], p["Z0"], p["a"], p["kappa"], p["delta"])
    cx, cy = p["R0"] + axis_shift, p["Z0"]
    bth = np.arctan2(poly[:, 1] - cy, poly[:, 0] - cx)
    br = np.hypot(poly[:, 0] - cx, poly[:, 1] - cy)
    o = np.argsort(bth)
    bth, br = bth[o], br[o]
    th = np.arctan2(grid.Y - cy, grid.X - cx)
    r = np.hypot(grid.X - cx, grid.Y - cy)
    rb = np.interp(th, bth, br, period=2 * np.pi)
    return r / rb, th


def _smooth_random(n, corr_len_px, rng):
    """Unit-variance Gaussian random field on an n x n grid (Gaussian spectral filter)."""
    w = rng.standard_normal((n, n))
    k = np.fft.fftfreq(n)
    k2 = k[:, None] ** 2 + k[None, :] ** 2
    f = np.fft.ifft2(np.fft.fft2(w) * np.exp(-0.5 * k2 * (2 * np.pi * corr_len_px) ** 2)).real
    return (f - f.mean()) / f.std()


def make_phantom(name, grid, mask, rng=None, plasma=None, amplitude=1.0, **kw):
    """Emissivity image (n,n), >= 0, zero outside ``mask``.

    name: 'peaked' (Gaussian core, kw: width=0.35 in rho), 'hollow' (annulus, kw:
    r0=0.55, width=0.15), 'blob' (off-axis bright spot, kw: rho0=0.5, angle_deg=35,
    sigma=0.06 m, background=0.05), 'edge' (band near boundary, kw: r0=0.85,
    width=0.07, background=0.02), 'random' (positive smooth Gaussian random field,
    exp of a GRF with correlation length ``corr_len`` metres (0.12) and log-std
    ``log_std`` (0.7), peaked toward the core by an envelope, normalised to max 1;
    depends on ``rng``, a numpy Generator or seed).
    """
    mask = np.asarray(mask, bool)
    rho, th = normalized_radius(grid, plasma, kw.get("axis_shift", 0.03 if name == "peaked" else 0.0))
    if name == "peaked":
        img = np.exp(-0.5 * (rho / kw.get("width", 0.35)) ** 2)
    elif name == "hollow":
        img = np.exp(-0.5 * ((rho - kw.get("r0", 0.55)) / kw.get("width", 0.15)) ** 2)
    elif name == "blob":
        p = DEFAULT_PLASMA if plasma is None else plasma
        # Blob centre in physical coordinates, placed at normalised radius rho0.
        rho_b, _ = normalized_radius(grid, plasma)
        ang = np.deg2rad(kw.get("angle_deg", 35.0))
        poly = d_boundary(p["R0"], p["Z0"], p["a"], p["kappa"], p["delta"])
        bth = np.arctan2(poly[:, 1] - p["Z0"], poly[:, 0] - p["R0"])
        o = np.argsort(bth)
        rb = np.interp(ang, bth[o], np.hypot(poly[:, 0] - p["R0"], poly[:, 1] - p["Z0"])[o], period=2 * np.pi)
        cx = p["R0"] + kw.get("rho0", 0.5) * rb * np.cos(ang)
        cy = p["Z0"] + kw.get("rho0", 0.5) * rb * np.sin(ang)
        s = kw.get("sigma", 0.06)
        img = np.exp(-0.5 * ((grid.X - cx) ** 2 + (grid.Y - cy) ** 2) / s**2)
        img = img + kw.get("background", 0.05) * np.exp(-0.5 * (rho / 0.6) ** 2)
    elif name == "edge":
        img = np.exp(-0.5 * ((rho - kw.get("r0", 0.85)) / kw.get("width", 0.07)) ** 2)
        img = img + kw.get("background", 0.02) * (rho < 1)
    elif name == "random":
        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)
        ell = kw.get("corr_len", 0.12) / (grid.xmax - grid.xmin) * grid.n  # in pixels
        g = _smooth_random(grid.n, ell, rng)
        env = np.exp(-0.5 * (np.clip(rho, 0, 1.5) / 0.8) ** 2)
        img = env * np.exp(kw.get("log_std", 0.7) * g)
        m = img[mask].max() if mask.any() else 1.0
        img = img / m
    else:
        raise ValueError(f"unknown phantom {name!r}")
    img = np.where(mask, img, 0.0)
    return np.maximum(amplitude * img, 0.0)
