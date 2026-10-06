"""Real-geometry test case: TCV (EPFL tokamak) bolometer lines of sight and vessel.

Geometry (120 bolometer chords, vessel outline) and phantom ingredients come from the open
repository https://github.com/dhamm97/real-time-tomo-prad (Hamm et al., arXiv:2603.11856,
MIT licence), vendored in ``tomo/data/tcv`` (small files, see NOTICE.md there). The large
files (1000 anisotropic-diffusion-style SOLPS-inspired phantoms, the authors' etendue
geometry matrix) are downloaded on demand to ``data/tcv/`` (gitignored):
``python -m tomo.tcv --fetch``.

Coordinates are metres in the poloidal (R, Z) plane: x = R, y = Z, so every method in
``tomo/`` runs unchanged. The grid is rectangular (``n`` columns in R, ``ny`` rows in Z,
row 0 at the bottom). Chords are idealised lines of sight (pinhole to far wall); the
volume-of-sight/etendue weighting is ignored (Hamm et al., arXiv:2608.03835, find LoS
adequate for radiated power).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import urllib.request
from dataclasses import replace
from typing import Optional

import numpy as np
from scipy import ndimage

from .config import load_config
from .forward import Problem, block_average
from .geometry import (Chords, Grid, _points_in_polygon, checkerboard, geometry_matrix, laplacian,
                       neighbor_edges)

R_MIN, R_MAX = 0.624, 1.1376     # radial extent of the authors' pixel box (m)
Z_MIN, Z_MAX = -0.75, 0.75
_VESSEL_LR, _VESSEL_LZ = 0.511, 1.5   # normalisation of tcv_shape_coords (as in the source repo)
SOURCE_COMMIT = "7f6cee2c244ac0c30b9267e90052ce8a9060f627"
_RAW = f"https://raw.githubusercontent.com/dhamm97/real-time-tomo-prad/{SOURCE_COMMIT}/src/"
FETCH_FILES = {   # local name -> path in the source repo
    "phantoms.npy": "results/hyperparameter_study_results/phantoms/phantoms.npy",
    "geometry_matrix_NINO.npy": "tcv_geometry/geometry_matrix_NINO.npy",
}
#: Chord index ranges of the four cameras (from the authors' plotting notebook).
CAMERA_RANGES = {"top": (0, 20), "main_lfs": (20, 60), "divertor": (60, 100), "bottom": (100, 120)}

TCV_PHANTOMS = ["peaked", "hollow", "blob", "edge", "divertor"]

_VENDOR = os.path.join(os.path.dirname(__file__), "data", "tcv")
_FETCHED = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "tcv")


class TCVDataMissing(RuntimeError):
    pass


def _ld(name):
    return np.load(os.path.join(_VENDOR, name), allow_pickle=False)


def fetch_tcv_data(dest: str = _FETCHED, files=None, force: bool = False) -> dict:
    """Download the large optional files into ``dest``; returns {name: sha256}."""
    os.makedirs(dest, exist_ok=True)
    out = {}
    for name in files or FETCH_FILES:
        path = os.path.join(dest, name)
        if force or not os.path.exists(path):
            urllib.request.urlretrieve(_RAW + FETCH_FILES[name], path + ".part")
            os.replace(path + ".part", path)
        with open(path, "rb") as f:
            out[name] = hashlib.sha256(f.read()).hexdigest()
    return out


def fetched_path(name: str) -> Optional[str]:
    p = os.path.join(_FETCHED, name)
    return p if os.path.exists(p) else None


# ----------------------------------------------------------------------------- geometry
def tcv_vessel_polygon() -> np.ndarray:
    """Closed vessel outline (8 vertices, chamfered rectangle) in (R, Z) metres."""
    s = _ld("tcv_shape_coords.npy")[:-1]
    return np.stack([R_MIN + _VESSEL_LR * s[:, 0], Z_MIN + _VESSEL_LZ * s[:, 1]], axis=1)


def tcv_chords() -> Chords:
    """120 bolometer lines of sight: p0 = pinhole, p1 = far end at the wall/box, (R, Z) metres."""
    r, z = _ld("bolo_coords_r.npy"), _ld("bolo_coords_z.npy")
    p0 = np.stack([r[:, 0], z[:, 0]], axis=1)
    p1 = np.stack([r[:, 1], z[:, 1]], axis=1)
    cam = np.zeros(len(p0), dtype=int)
    for ci, (a, b) in enumerate(CAMERA_RANGES.values()):
        cam[a:b] = ci
    return Chords(p0, p1, cam)


def tcv_etendues() -> np.ndarray:
    return _ld("etendues.npy").ravel()


def make_tcv_grid(nr: int = 20, nz: int = 60) -> Grid:
    return Grid(int(nr), R_MIN, R_MAX, Z_MIN, Z_MAX, ny=int(nz))


def tcv_mask(grid: Grid) -> np.ndarray:
    """(ny, n) bool: pixel centre inside the vessel outline."""
    return _points_in_polygon(grid.X, grid.Y, tcv_vessel_polygon())


def coverage_stats(T: np.ndarray) -> dict:
    """Chord coverage of the active pixels of a geometry matrix T (M, N)."""
    hits = (T > 0).sum(0)
    rows = T.sum(0)
    return dict(n_chords=T.shape[0], n_active=T.shape[1], frac_crossed=float((hits > 0).mean()),
                frac_ge3=float((hits >= 3).mean()), median_chords_per_pixel=float(np.median(hits)),
                min_chords=int(hits.min()), max_chords=int(hits.max()), mean_length_m=float(rows.mean()))


# ----------------------------------------------------------------------------- phantoms
def _sample_native(arr, R, Z, order=1):
    """Sample an array stored on the authors' native grid (row 0 = top, spanning the full box)."""
    ny0, nx0 = arr.shape
    rows = (Z_MAX - Z) / (Z_MAX - Z_MIN) * ny0 - 0.5
    cols = (R - R_MIN) / (R_MAX - R_MIN) * nx0 - 0.5
    return ndimage.map_coordinates(np.asarray(arr, float), [rows, cols], order=order, mode="nearest")


_EQ = {}


def _equilibrium():
    """Normalised-flux ingredients of the authors' base equilibrium (diverted, lower X-point)."""
    if not _EQ:
        psi = _ld("magnetic_equilibrium.npy")
        ny0, nx0 = psi.shape
        # core = closed surfaces above the X-point row (index 90 of 120 from the top)
        z_x = Z_MAX - 90.0 / 120.0 * (Z_MAX - Z_MIN)
        k = np.unravel_index(np.argmin(psi), psi.shape)
        _EQ.update(psi=psi, psi_min=float(psi.min()), z_x=z_x,
                   axis=(R_MIN + (k[1] + 0.5) / nx0 * (R_MAX - R_MIN),
                         Z_MAX - (k[0] + 0.5) / ny0 * (Z_MAX - Z_MIN)))
    return _EQ


def _core_coords(grid: Grid):
    """(rho, core) on grid: rho = sqrt(1 - psi/psi_min) (0 axis, 1 LCFS), core = inside LCFS above X-point."""
    eq = _equilibrium()
    psi = _sample_native(eq["psi"], grid.X, grid.Y)
    rho = np.sqrt(np.clip(1.0 - psi / eq["psi_min"], 0.0, None))
    core = (psi < 0) & (grid.Y > eq["z_x"])
    return rho, core


def _smooth_random(shape, corr_px, rng):
    w = rng.standard_normal(shape)
    ky, kx = np.fft.fftfreq(shape[0]), np.fft.fftfreq(shape[1])
    k2 = ky[:, None] ** 2 + kx[None, :] ** 2
    f = np.fft.ifft2(np.fft.fft2(w) * np.exp(-0.5 * k2 * (2 * np.pi * corr_px) ** 2)).real
    return (f - f.mean()) / f.std()


def make_tcv_phantom(name, grid: Grid, mask, rng=None, amplitude: float = 1.0, **kw) -> np.ndarray:
    """TCV-shaped emissivity (ny, n) >= 0, zero outside ``mask``, max ~ amplitude.

    peaked   : Gaussian core in sqrt-flux rho (kw width=0.45), smoothly cut at the LCFS/X-point height.
    hollow   : core annulus at rho0=0.6 (kw r0, width=0.15).
    blob     : off-axis Gaussian (kw dR=0.07, dZ=0.2 from the axis, sigma=0.05 m) on a faint core.
    edge     : ring just inside the LCFS (kw r0=0.9, width=0.07) plus faint core.
    divertor : SOLPS-inspired inner/outer legs, ring and X-point radiation (authors' components,
               kw coeffs=(1.0, 0.6, 0.4, 0.3), core coefficient kw core=0.3).
    random   : positive log-GRF (kw corr_len=0.12 m, log_std=0.7) times a vessel envelope; ``rng`` dependent.
    hamm:<k> : k-th of the 1000 downloaded phantoms (see fetch_tcv_data).
    """
    mask = np.asarray(mask, bool)
    eq = _equilibrium()
    rho, core = _core_coords(grid)
    sig_px = 0.012 / grid.dx  # ~12 mm edge softening
    if name == "peaked":
        img = np.exp(-0.5 * (rho / kw.get("width", 0.45)) ** 2) * core
    elif name == "hollow":
        img = np.exp(-0.5 * ((rho - kw.get("r0", 0.6)) / kw.get("width", 0.15)) ** 2) * core
    elif name == "blob":
        cR, cZ = eq["axis"][0] + kw.get("dR", 0.07), eq["axis"][1] + kw.get("dZ", 0.2)
        s = kw.get("sigma", 0.05)
        img = np.exp(-0.5 * ((grid.X - cR) ** 2 + (grid.Y - cZ) ** 2) / s**2)
        img = (img + kw.get("background", 0.05) * np.exp(-0.5 * (rho / 0.6) ** 2)) * core
    elif name == "edge":
        img = (np.exp(-0.5 * ((rho - kw.get("r0", 0.9)) / kw.get("width", 0.07)) ** 2)
               + kw.get("background", 0.05) * np.exp(-0.5 * (rho / 0.6) ** 2)) * core
    elif name == "divertor":
        comps = [_sample_native(_ld(f), grid.X, grid.Y) for f in
                 ("solps_phantom_inner_leg.npy", "solps_phantom_outer_leg.npy", "solps_phantom_ring_and_core.npy")]
        comps = [c / max(c.max(), 1e-30) for c in comps]
        xpt = _sample_native(_ld("xpt_rad.npy"), grid.X, grid.Y)
        a = kw.get("coeffs", (1.0, 0.6, 0.4, 0.3))
        img = a[0] * comps[0] + a[1] * comps[1] + a[2] * comps[2] + a[3] * xpt
        img = img + kw.get("core", 0.3) * np.exp(-0.5 * (rho / 0.45) ** 2) * core
    elif name == "random":
        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)
        g = _smooth_random(grid.shape, kw.get("corr_len", 0.12) / grid.dx, rng)
        env = np.exp(-0.5 * (np.hypot((grid.X - eq["axis"][0]) / 0.25, (grid.Y - eq["axis"][1]) / 0.55)) ** 2)
        img = (0.1 + env) * np.exp(kw.get("log_std", 0.7) * g)
    elif name.startswith("hamm:"):
        p = fetched_path("phantoms.npy")
        if p is None:
            raise TCVDataMissing("run `python -m tomo.tcv --fetch` to download the 1000 phantoms")
        ph = np.load(p, mmap_mode="r", allow_pickle=False)
        img = _sample_native(ph[int(name[5:])], grid.X, grid.Y)
    else:
        raise ValueError(f"unknown TCV phantom {name!r}")
    if name != "random" and not name.startswith("hamm:") and sig_px > 0.3:
        img = ndimage.gaussian_filter(np.where(mask, img, 0.0), sig_px)
    img = np.where(mask, img, 0.0)
    m = img.max()
    return np.maximum(amplitude * img / m, 0.0) if m > 0 else img


# ----------------------------------------------------------------------------- problem
def make_tcv_problem(cfg, phantom, seed) -> Problem:
    """Same contract as ``tomo.forward.make_problem`` on the TCV geometry.

    Config: ``grid.n`` (R pixels), ``grid.ny`` (Z pixels), ``data_grid.n``/``data_grid.ny`` (fine
    grid, integer multiples), ``noise``. Defaults (see configs/tcv.yaml): 20x60 coarse, 80x240 fine.
    Data come from the fine grid with its own geometry matrix (no inverse crime); noise model
    and RNG streams are identical to make_problem (phantom: [seed,0], noise: [seed,1]).
    """
    cfg = load_config(cfg)
    g, d = cfg["grid"], cfg["data_grid"]
    grid = make_tcv_grid(g["n"], g.get("ny", 3 * g["n"]))
    fine = make_tcv_grid(d["n"], d.get("ny", 3 * d["n"]))
    mask, fmask = tcv_mask(grid), tcv_mask(fine)
    chords = tcv_chords()
    T = geometry_matrix(chords, grid, mask)
    T_fine = geometry_matrix(chords, fine, fmask, dtype=np.float64)

    if isinstance(phantom, str):
        fine_img = make_tcv_phantom(phantom, fine, fmask, rng=np.random.default_rng([seed, 0]))
        name = phantom
    else:
        fine_img, name = np.where(fmask, np.asarray(phantom, float), 0.0), "custom"

    eps_img = np.where(mask, block_average(fine_img, grid.n, grid.ny), 0.0)
    b_clean = T_fine @ fine_img[fmask]
    nz = cfg["noise"]
    rng_n = np.random.default_rng([seed, 1])
    sigma = nz["rel"] * np.abs(b_clean) + nz["floor"] * b_clean.max()
    gain = np.ones(len(b_clean))
    if nz.get("calib_offset", 0.0):
        n_cam = int(chords.camera.max()) + 1
        gain = (1.0 + nz["calib_offset"] * rng_n.standard_normal(n_cam))[chords.camera]
    b = gain * b_clean + sigma * rng_n.standard_normal(len(b_clean))
    return Problem(grid=grid, mask=mask, T=T, chords=chords, eps_true=eps_img[mask],
                   eps_true_img=eps_img, b_clean=b_clean, b=b, sigma=sigma, L=laplacian(mask),
                   edges=neighbor_edges(mask), eps_max=None, fine_grid=fine, fine_img=fine_img,
                   fine_mask=fmask, T_fine=T_fine, calib_gain=gain, cfg=cfg, phantom=name,
                   seed=int(seed), colors=checkerboard(mask))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fetch", action="store_true", help="download large optional files to data/tcv/")
    a = ap.parse_args()
    if a.fetch:
        for k, v in fetch_tcv_data().items():
            print(k, v)


if __name__ == "__main__":
    main()
