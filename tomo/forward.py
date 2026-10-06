"""Forward model: synthetic data b = T eps + noise, generated on a fine grid.

Data come from a 128x128 fine grid with its own geometry matrix and fine mask (same
analytic D-shape); the reconstruction truth is the area average of the fine image
over 4x4 blocks restricted to the coarse mask. This avoids the inverse crime.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import load_config
from .geometry import (Chords, Grid, checkerboard, d_shape_mask, geometry_matrix, laplacian,
                       make_fans, make_grid, neighbor_edges)
from .phantoms import make_phantom


@dataclass
class Problem:
    """A tomography problem instance. Pixel vectors are over active pixels (row-major).

    eps_max is deliberately ``None`` here: setting it from the ground truth would be
    cheating. The EBM must set it from data alone, e.g. with
    ``eps_max_from_estimate(tikhonov_mean)`` (factor * max of the clipped Tikhonov
    solution, factor = cfg['model']['eps_max_factor']) and store it on the problem.
    """
    grid: Grid
    mask: np.ndarray
    T: np.ndarray                  # (M, N) float32 coarse geometry matrix
    chords: Chords
    eps_true: np.ndarray           # (N,) area-averaged truth
    eps_true_img: np.ndarray       # (n, n), zero outside mask
    b_clean: np.ndarray            # (M,) noiseless data from the fine grid
    b: np.ndarray                  # (M,) noisy data
    sigma: np.ndarray              # (M,) per-chord noise std
    L: np.ndarray                  # (N, N) graph Laplacian
    edges: np.ndarray              # (E, 2) neighbour pairs
    eps_max: Optional[float]       # None until set from data (see class doc)
    fine_grid: Grid
    fine_img: np.ndarray           # (nf, nf)
    fine_mask: np.ndarray = None
    T_fine: np.ndarray = None      # (M, Nf) float64 fine geometry matrix
    calib_gain: np.ndarray = None  # (M,) multiplicative per-camera gain error (ones if none)
    cfg: dict = field(default_factory=dict)
    phantom: str = ""
    seed: int = 0
    colors: np.ndarray = None      # checkerboard colouring of active pixels


def pixel_area(problem) -> float:
    """Area of one coarse pixel (m^2)."""
    return problem.grid.dx * problem.grid.dy


def eps_max_from_estimate(est, factor: float = 1.2) -> float:
    """eps_max from data only: ``factor`` * max of an estimate (e.g. Tikhonov) clipped >= 0."""
    return float(factor * np.max(np.clip(np.asarray(est, float), 0.0, None)))


def block_average(img_fine: np.ndarray, n: int, ny: int = None) -> np.ndarray:
    """Area-average a fine image into (ny,n) blocks (default ny=n; sizes must divide)."""
    ny = n if ny is None else ny
    nfy, nf = img_fine.shape
    if nf % n or nfy % ny:
        raise ValueError(f"fine shape {img_fine.shape} is not a multiple of coarse shape {(ny, n)}")
    return img_fine.reshape(ny, nfy // ny, n, nf // n).mean(axis=(1, 3))


def make_problem(cfg, phantom, seed) -> Problem:
    """Build a Problem from config, phantom name (or an (nf,nf) fine image) and seed.

    Deterministic given (cfg, phantom, seed): phantom randomness uses
    default_rng([seed, 0]) and noise/calibration use default_rng([seed, 1]).
    Noise: sigma_i = rel*|b_clean_i| + floor*max(b_clean); b = gain_c * b_clean + N(0, sigma).
    ``noise.calib_offset`` (default 0) is the std of a per-camera multiplicative gain error.
    """
    cfg = load_config(cfg)
    grid = make_grid(cfg["grid"]["n"], cfg["grid"].get("size", 1.0))
    fine = make_grid(cfg["data_grid"]["n"], cfg["grid"].get("size", 1.0))
    pl = cfg["plasma"]
    mask = d_shape_mask(grid, **pl)
    fmask = d_shape_mask(fine, **pl)
    chords = make_fans(cfg["cameras"], box=(grid.xmin, grid.xmax, grid.ymin, grid.ymax))
    T = geometry_matrix(chords, grid, mask)
    T_fine = geometry_matrix(chords, fine, fmask, dtype=np.float64)

    if isinstance(phantom, str):
        rng_p = np.random.default_rng([seed, 0])
        fine_img = make_phantom(phantom, fine, fmask, rng=rng_p, plasma=pl)
        name = phantom
    else:
        fine_img, name = np.where(fmask, np.asarray(phantom, float), 0.0), "custom"

    eps_img = np.where(mask, block_average(fine_img, grid.n), 0.0)
    eps_true = eps_img[mask]

    b_clean = T_fine @ fine_img[fmask]
    nz = cfg["noise"]
    rng_n = np.random.default_rng([seed, 1])
    sigma = nz["rel"] * np.abs(b_clean) + nz["floor"] * b_clean.max()
    gain = np.ones(len(b_clean))
    if nz.get("calib_offset", 0.0):
        g_cam = 1.0 + nz["calib_offset"] * rng_n.standard_normal(len(cfg["cameras"]))
        gain = g_cam[chords.camera]
    b = gain * b_clean + sigma * rng_n.standard_normal(len(b_clean))

    return Problem(grid=grid, mask=mask, T=T, chords=chords, eps_true=eps_true,
                   eps_true_img=eps_img, b_clean=b_clean, b=b, sigma=sigma,
                   L=laplacian(mask), edges=neighbor_edges(mask), eps_max=None,
                   fine_grid=fine, fine_img=fine_img, fine_mask=fmask, T_fine=T_fine,
                   calib_gain=gain, cfg=cfg, phantom=name, seed=int(seed),
                   colors=checkerboard(mask))
