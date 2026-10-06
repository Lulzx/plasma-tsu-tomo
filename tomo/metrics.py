"""Reconstruction metrics: accuracy, calibration and convergence diagnostics.

All vectors are over *active* (masked) pixels in row-major order of the full
grid.  Nothing here imports tomo.geometry/forward; ``problem`` is duck-typed
(attributes ``grid, mask, T, b, sigma, eps_true``).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm
from skimage.metrics import structural_similarity


def to_img(vec, mask, fill=0.0):
    """Scatter an active-pixel vector (N,) into an (n,n) image; ``fill`` outside mask."""
    mask = np.asarray(mask, bool)
    img = np.full(mask.shape, fill, dtype=float)
    img[mask] = np.asarray(vec, float)
    return img


def rel_l2(est, true):
    """Relative L2 error ||est - true|| / ||true||."""
    est, true = np.asarray(est, float), np.asarray(true, float)
    return float(np.linalg.norm(est - true) / max(np.linalg.norm(true), 1e-300))


def ssim_img(est_img, true_img, mask):
    """SSIM on the full image, with zeros outside ``mask``.

    data_range is the dynamic range of the (masked) truth, so the score does not
    depend on the estimate's scale.  Window 7 (skimage default), clipped to the
    image size for tiny images.
    """
    mask = np.asarray(mask, bool)
    e = np.where(mask, np.asarray(est_img, float), 0.0)
    t = np.where(mask, np.asarray(true_img, float), 0.0)
    dr = float(t.max() - t.min())
    if dr <= 0:
        dr = 1.0
    win = min(7, min(t.shape) // 2 * 2 - 1)
    return float(structural_similarity(t, e, data_range=dr, win_size=max(win, 3)))


def _peak_xy(vec, grid, mask):
    img = to_img(vec, mask, fill=-np.inf)
    iy, ix = np.unravel_index(np.argmax(img), img.shape)
    return np.array([np.asarray(grid.xc)[ix], np.asarray(grid.yc)[iy]])


def peak_error_cm(est, true, grid, mask):
    """Distance in cm between pixel-centre coordinates of the argmax of est and true.

    Grid coordinates are in metres, so the result is multiplied by 100.
    """
    return float(100.0 * np.linalg.norm(_peak_xy(est, grid, mask) - _peak_xy(true, grid, mask)))


def power_error(est, true, pixel_area=1.0):
    """Relative error of the area-weighted total (sum * pixel_area); area cancels."""
    pe, pt = float(np.sum(est)) * pixel_area, float(np.sum(true)) * pixel_area
    return float(abs(pe - pt) / max(abs(pt), 1e-300))


def reduced_chi2(T, b, sigma, est):
    """sum(((b - T est)/sigma)^2) / M."""
    r = (np.asarray(b, float) - np.asarray(T, float) @ np.asarray(est, float)) / np.asarray(sigma, float)
    return float(np.sum(r**2) / r.size)


def coverage(samples_or_mean_std, true, level):
    """Fraction of pixels whose truth lies in the central credible interval.

    Accepts samples (S,N) (empirical quantile interval) or a (mean, std) tuple
    (Gaussian interval, z = Phi^-1((1+level)/2)).  Pixels with std 0 count only if exact.
    """
    true = np.asarray(true, float)
    if isinstance(samples_or_mean_std, (tuple, list)):
        mean, std = (np.asarray(a, float) for a in samples_or_mean_std)
        z = norm.ppf(0.5 + level / 2)
        lo, hi = mean - z * std, mean + z * std
    else:
        s = np.asarray(samples_or_mean_std, float)
        if s.ndim == 3:  # (C,S,N) -> pool chains
            s = s.reshape(-1, s.shape[-1])
        lo = np.quantile(s, 0.5 - level / 2, axis=0)
        hi = np.quantile(s, 0.5 + level / 2, axis=0)
    return float(np.mean((true >= lo) & (true <= hi)))


def split_rhat(chains):
    """Split-chain Gelman-Rubin R-hat per pixel for chains (C, S, N).

    Each chain is cut in half (odd sample dropped) giving 2C chains of length S//2.
    Constant pixels (W = B = 0) return 1.0.
    """
    x = np.asarray(chains, float)
    if x.ndim == 2:
        x = x[..., None]
    C, S, N = x.shape
    h = S // 2
    if h < 2:
        return np.full(N, np.nan)
    sp = np.concatenate([x[:, :h], x[:, h:2 * h]], axis=0)  # (2C, h, N)
    m = sp.mean(1)
    W = sp.var(1, ddof=1).mean(0)
    B = h * m.var(0, ddof=1)
    var_plus = (h - 1) / h * W + B / h
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.sqrt(var_plus / W)
    r = np.where((W == 0) & (B == 0), 1.0, r)
    return np.where((W == 0) & (B > 0), np.inf, r)


def sweeps_to_converge(trace, sweeps_per_sample, thresh=1.05):
    """Smallest prefix with max-over-pixels split R-hat < thresh.

    ``trace`` is (C,S,N).  Prefix lengths are tried on a geometric-ish grid
    (min 4 samples) and the first passing one is returned as
    n_samples_prefix * sweeps_per_sample, or None if never converged.
    """
    trace = np.asarray(trace)
    S = trace.shape[1]
    cands = sorted({min(S, int(round(v))) for v in np.unique(np.r_[np.arange(4, 17), np.geomspace(16, S, 40)]) if v <= S})
    for n in cands:
        if n < 4:
            continue
        r = split_rhat(trace[:, :n])
        if np.all(np.isfinite(r)) and np.max(r) < thresh:
            return int(n * sweeps_per_sample)
    return None


def summarize(problem, est_dict):
    """All scalar metrics for one estimate dict ('mean' required; optional 'samples', 'std', 'time').

    Coverage at 68/95% is added when samples (C,S,N) or std is present.
    """
    mean = np.asarray(est_dict["mean"], float)
    true = np.asarray(problem.eps_true, float)
    mask, grid = problem.mask, problem.grid
    area = float(grid.dx * grid.dy) if hasattr(grid, "dx") else 1.0
    out = {
        "rel_l2": rel_l2(mean, true),
        "ssim": ssim_img(to_img(mean, mask), to_img(true, mask), mask),
        "peak_err_cm": peak_error_cm(mean, true, grid, mask),
        "power_err": power_error(mean, true, area),
        "chi2_red": reduced_chi2(problem.T, problem.b, problem.sigma, mean),
    }
    samples = est_dict.get("samples")
    for lev in (0.68, 0.95):
        key = f"coverage{int(round(lev * 100))}"
        if samples is not None:
            s = np.asarray(samples)
            out[key] = coverage(s.reshape(-1, s.shape[-1]), true, lev)
        elif est_dict.get("std") is not None:
            out[key] = coverage((mean, est_dict["std"]), true, lev)
    if samples is not None and np.asarray(samples).ndim == 3:
        r = split_rhat(samples)
        out["rhat_max"] = float(np.nanmax(r)) if np.any(np.isfinite(r)) else float("nan")
    for k in ("time", "lam"):
        if est_dict.get(k) is not None:
            out[k] = float(est_dict[k])
    return out
