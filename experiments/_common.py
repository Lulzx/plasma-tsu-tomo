"""Shared helpers for the experiment scripts (kept out of tomo/ on purpose).

Every script: ``--config`` (default configs/default.yaml), ``--out`` (default results/<script>/),
``--quick`` (tiny settings for smoke tests).  The merged config is written next to the outputs.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tomo.config import DEFAULT_PATH, load_config, save_config  # noqa: E402
from tomo.metrics import to_img  # noqa: E402

# Okabe-Ito colour-blind-safe palette
OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "verm": "#D55E00", "purple": "#CC79A7"}
METHOD_COLORS = {"tikhonov_gcv": OI["sky"], "tikhonov_tuned": OI["blue"], "mfi": OI["green"],
                 "gp": OI["orange"], "potts": OI["verm"], "dense": OI["purple"],
                 "sparse": OI["sky"], "chain": OI["black"]}
CMAP_FIELD, CMAP_STD, CMAP_ERR = "viridis", "magma", "cividis"
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "font.size": 9, "axes.grid": False})

EBM_VARIANTS = ["potts", "dense", "sparse", "chain"]

# tiny settings for --quick smoke tests (deep-merged over the config)
QUICK = {
    "model": {"K": 4, "Kz": 4},
    "schedule": {"posterior": {"n_warmup": 10, "n_samples": 16, "steps_per_sample": 2},
                 "anneal": {"beta_min": 0.5, "beta_max": 10.0, "n_betas": 3, "sweeps_per_beta": 2},
                 "n_chains": 4, "tempering": {"n_replicas": 3, "beta_min": 0.5}},
    "experiments": {"n_random": 3, "n_random_ebm": 1, "phantoms": ["peaked", "hollow"],
                    "K_values": [4, 6], "tau_values": [0.5, 1.0], "sparse_thresholds": [0.05, 0.2],
                    "noise_levels": [0.03, 0.10]},
}


def make_parser(script, desc=""):
    ap = argparse.ArgumentParser(description=desc or script)
    ap.add_argument("--config", default=DEFAULT_PATH, help="YAML merged over configs/default.yaml")
    ap.add_argument("--out", default=None, help="output dir (default results/<script>/)")
    ap.add_argument("--quick", action="store_true", help="tiny fast settings for smoke testing")
    return ap


def setup(args, script, extra_overrides=None):
    """Load config (+quick overrides), create out dir, save merged config. Returns (cfg, out)."""
    from tomo.config import deep_merge
    cfg = load_config(args.config)
    if args.quick:
        cfg = deep_merge(cfg, QUICK)
    if extra_overrides:
        cfg = deep_merge(cfg, extra_overrides)
    out = args.out or os.path.join(ROOT, "results", script + ("_quick" if args.quick else ""))
    os.makedirs(out, exist_ok=True)
    save_config(cfg, out)
    log(f"[{script}] out={out} quick={args.quick} load={os.getloadavg()[0]:.1f}")
    return cfg, out


def log(*a):
    print(*a, flush=True)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist() if o.size <= 4096 else {"__array__": list(o.shape)}
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if np.isfinite(f) else (None if np.isnan(f) else ("inf" if f > 0 else "-inf"))
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if o is None or isinstance(o, (str, int, bool)):
        return o
    return str(o)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(jsonable(obj), f, indent=1)
    return path


def save_csv(rows, path, fields=None):
    rows = [jsonable(r) for r in rows]
    if fields is None:
        fields = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


def fmt(v, nd=3):
    if v is None:
        return "-"
    if isinstance(v, str):
        return v
    if isinstance(v, (int, np.integer)):
        return str(v)
    v = float(v)
    if not np.isfinite(v):
        return str(v)
    return f"{v:.{nd}g}"


def md_table(rows, cols, headers=None):
    headers = headers or cols
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(c)) if not isinstance(r.get(c), str) else r[c] for c in cols) + " |")
    return "\n".join(lines) + "\n"


def save_fig(fig, out, name):
    path = os.path.join(out, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------------------
# problems and baselines
# ----------------------------------------------------------------------------------------
def problem_list(cfg, n_random=0, phantoms=None):
    """[(name, phantom, seed)]: named phantoms use seed=cfg seed, random field k uses seed0+k."""
    ex = cfg["experiments"]
    ph = ex["phantoms"] if phantoms is None else phantoms
    lst = [(p, p, int(cfg["seed"])) for p in ph]
    lst += [(f"random{k:03d}", "random", int(ex["seed0"]) + k) for k in range(int(n_random))]
    return lst


def build_problem(cfg, phantom, seed, noise_rel=None):
    from tomo.forward import make_problem
    if noise_rel is not None:
        from tomo.config import deep_merge
        cfg = deep_merge(cfg, {"noise": {"rel": float(noise_rel)}})
    return make_problem(cfg, phantom, seed)


BASELINES = ["tikhonov_gcv", "tikhonov_tuned", "mfi", "gp"]


def run_baselines(p, which=BASELINES):
    """Run baselines on problem p. Returns {name: est_dict (with 'mean', maybe 'std')}."""
    from tomo import baselines as B
    out = {}
    for w in which:
        try:
            if w == "tikhonov_gcv":
                out[w] = B.tikhonov(p.T, p.b, p.sigma, p.L)
            elif w == "tikhonov_tuned":
                out[w] = B.tuned_tikhonov(p.T, p.b, p.sigma, p.L)
            elif w == "mfi":
                out[w] = B.mfi(p.T, p.b, p.sigma, p.mask, p.grid)
            elif w == "gp":
                r = B.gp_tomography(p.T, p.b, p.sigma, p.grid, p.mask)
                r.pop("cov", None)
                out[w] = r
        except Exception as e:  # noqa: BLE001
            out[w] = {"error": f"{type(e).__name__}: {e}"}
            log(f"  baseline {w} FAILED: {e}")
    return out


def metric_row(p, est, **extra):
    from tomo.metrics import summarize
    row = dict(extra)
    if "error" in est:
        row["error"] = est["error"]
        return row
    d = {k: est[k] for k in ("mean", "std", "samples", "time", "lam") if est.get(k) is not None}
    row.update(summarize(p, d))
    return row


# ----------------------------------------------------------------------------------------
# EBM runner
# ----------------------------------------------------------------------------------------
def run_ebm(p, variant, cfg, key_seed=0, **kw):
    """Run one EBM variant; never raises. Returns (result_or_None, info dict with diagnostics/failure)."""
    import jax
    from tomo.metrics import sweeps_to_converge
    key = jax.random.PRNGKey(int(key_seed))
    sc = cfg["schedule"]
    t0 = time.perf_counter()
    try:
        if variant == "potts":
            from tomo.ebm_potts import sample_potts
            res = sample_potts(p, int(cfg["model"]["K"]), cfg["model"].get("lam"), cfg, key, **kw)
        elif variant in ("dense", "sparse"):
            from tomo.ebm_ising import sample_ising_variant
            res = sample_ising_variant(p, variant, cfg, key, **kw)
        elif variant == "chain":
            from tomo.ebm_chain import sample_chain
            res = sample_chain(p, cfg, key, **kw)
        else:
            raise ValueError(variant)
    except Exception as e:  # noqa: BLE001  (spec: report failures explicitly)
        log(f"  EBM {variant} FAILED: {type(e).__name__}: {e}")
        return None, {"variant": variant, "failed": True, "error": f"{type(e).__name__}: {e}",
                      "traceback": traceback.format_exc()[-1500:], "wall_s": time.perf_counter() - t0}
    wall = time.perf_counter() - t0
    post, ann = sc["posterior"], sc["anneal"]
    rh = np.asarray(res["rhat"], float)
    finite = rh[np.isfinite(rh)]
    sps = int(post["steps_per_sample"])
    s2c = res.get("sweeps_to_converge")
    if s2c is None and "sweeps_to_converge" not in res:
        try:
            s2c = sweeps_to_converge(res["samples"], sps)
        except Exception:  # noqa: BLE001
            s2c = None
    info = {"variant": variant, "failed": False,
            "rhat_max": float(np.max(rh)) if len(rh) and np.all(np.isfinite(rh)) else (float("inf") if len(rh) else float("nan")),
            "rhat_median": float(np.median(finite)) if len(finite) else float("nan"),
            "rhat_n_nonfinite": int((~np.isfinite(rh)).sum()),
            "invalid_frac": float(res.get("invalid_frac") or 0.0),
            "n_spins": res.get("n_spins"), "n_blocks": res.get("n_blocks"), "max_degree": res.get("max_degree"),
            "sweeps_posterior": int(post["n_warmup"]) + int(post["n_samples"]) * sps,
            "n_warmup": int(post["n_warmup"]), "steps_per_sample": sps,
            "sweeps_map": int(ann["n_betas"]) * int(ann["sweeps_per_beta"]),
            "sweeps_to_converge": s2c,
            "n_chains": int(kw.get("n_chains") or sc["n_chains"]),
            "time_s": res.get("time"), "wall_incl_compile_s": wall,
            "compile_time_s": res.get("compile_time"), "build_time_s": res.get("build_time")}
    for k in ("lam", "tau", "tau_over_dz", "eff_noise_ratio", "frac_top", "chain_invalid_frac", "converged", "map_energy"):
        if res.get(k) is not None:
            info[k] = res[k]
    return res, info


def ebm_estimates(res):
    """Estimate dicts for mean (with samples -> coverage) and MAP."""
    d = {"mean": res["mean"], "std": res["std"], "samples": res["samples"], "time": res.get("time")}
    m = None if res.get("map") is None else {"mean": res["map"], "time": res.get("time")}
    return d, m


def energy_trace_1d(tr):
    """Reduce an energy trace of any shape (C,S[,R]) to a (S,) chain-mean curve."""
    if tr is None:
        return None
    a = np.asarray(tr, float)
    while a.ndim > 2:
        a = a[..., -1]
    return a.mean(0) if a.ndim == 2 else a


# ----------------------------------------------------------------------------------------
# plotting
# ----------------------------------------------------------------------------------------
def show_field(ax, vec, mask, vmin=None, vmax=None, cmap=CMAP_FIELD, title=None, grid=None):
    img = to_img(vec, mask, fill=np.nan)
    ext = None if grid is None else (grid.xmin, grid.xmax, grid.ymin, grid.ymax)
    im = ax.imshow(img, origin="lower", extent=ext, vmin=vmin, vmax=vmax, cmap=cmap, interpolation="nearest")
    if title:
        ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def field_grid(path_out, name, rows, col_labels, mask, grid, row_labels, cbar_label="emissivity (a.u.)"):
    """rows: list (per row) of list of (vec, kind) with kind in {'field','std'}; None entries are skipped.

    Fields in a row share the colour scale (0..max over the row); std panels have their own scale. Each panel has a colourbar (units: cbar_label).
    """
    nr, nc = len(rows), len(col_labels)
    fig, axs = plt.subplots(nr, nc, figsize=(1.9 * nc + 0.4, 2.0 * nr + 0.3), squeeze=False)
    for i, row in enumerate(rows):
        vmax = max(float(np.nanmax(v)) for v, k in row if v is not None and k == "field")
        for j, (v, k) in enumerate(row):
            ax = axs[i, j]
            if v is None:
                ax.axis("off")
                continue
            im = show_field(ax, v, mask, 0.0, vmax if k == "field" else None,
                            CMAP_FIELD if k == "field" else CMAP_STD, grid=grid)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.ax.tick_params(labelsize=6)
            if i == 0:
                ax.set_title(col_labels[j], fontsize=8)
            if j == 0:
                ax.set_ylabel(row_labels[i], fontsize=8)
    fig.tight_layout()
    return save_fig(fig, path_out, name)
