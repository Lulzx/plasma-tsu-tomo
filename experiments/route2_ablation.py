"""Route-2 ablation: where does the discrete Potts posterior's gain over the Gaussian posterior come from?

All posterior comparators share the SAME lambda (evidence-tuned Tikhonov lambda), the same data/noise
and the same smoothness prior P = A^T A + lam L:

  gaussian       exact unconstrained Gaussian posterior                       (no positivity, no bound, continuous)
  trunc_pos      same density truncated to eps >= 0                           (+ positivity)
  trunc_box      truncated to [0, eps_max], eps_max = Potts bound              (+ positivity + upper bound; K -> inf limit of Potts)
  potts_K{4,8,16,32}  K-level Potts posterior on {0, Delta, .., eps_max}      (+ discreteness)
  log_laplace    exp(f) positivity baseline (our re-implementation of the idea of arXiv:2410.11454; own lambda)
  gp, mfi, tikhonov_tuned   classical baselines.

Reading the table: gaussian -> trunc_pos isolates positivity, trunc_pos -> trunc_box the upper bound,
trunc_box -> potts_K the discretisation (K small = coarse; K large should approach trunc_box).
"""
from __future__ import annotations

import os
import time

from _common import (OI, build_problem, field_grid, log, make_parser, md_table, metric_row, np, plt,
                     problem_list, run_baselines, save_csv, save_fig, save_json, setup)

SCRIPT = "route2_ablation"
METRICS = ["rel_l2", "ssim", "peak_err_cm", "coverage68", "coverage95", "rhat_max", "rhat_med", "time"]
FULL = {"trunc": {"n_warmup": 300, "n_samples": 200, "thin": 5, "n_chains": 8},
        "potts": {"n_warmup": 500, "n_samples": 200, "steps_per_sample": 5, "n_chains": 8}}
QUICKS = {"trunc": {"n_warmup": 20, "n_samples": 20, "thin": 2, "n_chains": 4},
          "potts": {"n_warmup": 20, "n_samples": 20, "steps_per_sample": 2, "n_chains": 4}}


def _row(p, est, problem, kind, seed, method, **extra):
    r = metric_row(p, est, problem=problem, kind=kind, seed=seed, method=method)
    if est.get("rhat") is not None and "samples" in est:
        rh = np.asarray(est["rhat"], float)
        r["rhat_med"] = float(np.nanmedian(rh[np.isfinite(rh)])) if np.any(np.isfinite(rh)) else float("nan")
        r["rhat_max"] = float(np.max(rh)) if np.all(np.isfinite(rh)) else float("inf")
    else:
        r.pop("rhat_max", None)  # iid / closed-form samples: no R-hat
    r.update(extra)
    return r


def main():
    ap = make_parser(SCRIPT, __doc__)
    ap.add_argument("--phantoms", default=None, help="comma list (default config experiments.phantoms)")
    ap.add_argument("--n-random", type=int, default=3, help="random fields (seeds experiments.seed0 + k)")
    ap.add_argument("--Ks", default="4,8,16,32")
    args = ap.parse_args()
    cfg, out = setup(args, SCRIPT)
    import jax
    from tomo.ebm_common import prepare
    from tomo.ebm_potts import sample_potts
    from tomo.positive import gaussian_posterior_eps, log_laplace, truncated_gaussian_posterior

    phantoms = args.phantoms.split(",") if args.phantoms else None
    Ks = [int(k) for k in args.Ks.split(",")]
    n_rand = args.n_random
    if args.quick:
        phantoms, n_rand, Ks = ["peaked"], 0, [4, 8]
    sched = QUICKS if args.quick else FULL
    probs = problem_list(cfg, n_rand, phantoms)
    log(f"problems {[x[0] for x in probs]}, Ks={Ks}")
    rows, keep = [], {}
    t_start = time.time()
    for pi, (name, ph, seed) in enumerate(probs):
        p = build_problem(cfg, ph, seed)
        kind = "phantom" if ph != "random" else "random"
        st = prepare(p, max(Ks), None, cfg["model"]["eps_max_factor"])
        lam, em = st.lam, st.eps_max
        log(f"[{pi + 1}/{len(probs)}] {name} seed={seed} lam={lam:.4g} eps_max={em:.4g} load={os.getloadavg()[0]:.1f}")
        est = {}
        g = gaussian_posterior_eps(p, lam)
        est["gaussian"] = g
        t = sched["trunc"]
        est["trunc_pos"] = truncated_gaussian_posterior(p, lam, 0.0, None, seed=seed + 1, **t)
        est["trunc_box"] = truncated_gaussian_posterior(p, lam, 0.0, em, seed=seed + 2, **t)
        for K in Ks:
            r = sample_potts(p, K, lam, {"schedule": {"posterior": {}, "anneal": {}, "n_chains": t["n_chains"]}},
                             jax.random.PRNGKey(seed * 7 + K), setup=None, n_chains=sched["potts"]["n_chains"],
                             posterior={k: sched["potts"][k] for k in ("n_warmup", "n_samples", "steps_per_sample")},
                             do_map=False, frac_random=0.5, eps_max_factor=cfg["model"]["eps_max_factor"])
            assert abs(r["lam"] - lam) < 1e-9 * max(lam, 1) and abs(r["eps_max"] - em) < 1e-9 * em
            est[f"potts_K{K}"] = r
        try:
            est["log_laplace"] = log_laplace(p, lam_ref=lam)
        except Exception as e:  # noqa: BLE001
            log(f"  log_laplace FAILED {e}")
        base = run_baselines(p, ["tikhonov_tuned", "mfi", "gp"])
        est.update({k: v for k, v in base.items() if "error" not in v})
        for m, e in est.items():
            extra = {"frac_top": e.get("frac_top")} if m.startswith("potts") else {}
            if m.startswith("trunc"):
                extra = {"frac_at_upper": float(np.mean(e["samples"] >= em * (1 - 1e-4))) if m == "trunc_box" else 0.0,
                         "frac_at_zero": float(np.mean(e["samples"] <= 1e-6 * em))}
            rows.append(_row(p, e, name, kind, seed, m, lam=lam, eps_max=em, **extra))
            log(f"   {m:15s} rel_l2={rows[-1].get('rel_l2', float('nan')):.3f} cov95={rows[-1].get('coverage95', float('nan')):.2f} "
                f"rhat_max={rows[-1].get('rhat_max', float('nan')):.2f} t={rows[-1].get('time', float('nan')):.1f}")
        if kind == "phantom":
            keep[name] = (p, {m: est[m]["mean"] for m in est})
        save_csv(rows, f"{out}/ablation_all.csv")
    log(f"total {time.time() - t_start:.0f}s")

    order = ["tikhonov_tuned", "gaussian", "trunc_pos", "trunc_box"] + [f"potts_K{k}" for k in Ks] + ["log_laplace", "gp", "mfi"]
    names = [x[0] for x in probs]
    agg = []
    for m in order:
        rr = [r for r in rows if r["method"] == m and "error" not in r]
        if not rr:
            continue
        a = {"method": m, "n": len(rr)}
        for k in METRICS:
            v = [r[k] for r in rr if r.get(k) is not None and np.isfinite(r[k])]
            if v:
                a[k] = float(np.mean(v)) if k != "rhat_max" else float(np.max(v))
        agg.append(a)
    save_csv(agg, f"{out}/ablation_mean.csv")
    md = ["# Route-2 ablation: positivity vs upper bound vs discreteness\n",
          f"Same lambda (evidence) and same smoothness prior for gaussian / trunc_* / potts_K*; eps_max is the Potts bound "
          f"(1.2 x max clipped Tikhonov, data only). Problems: {names}. Schedules: {sched}. "
          "`rhat_max` = worst over pixels (max over problems in the mean table), `rhat_med` = median over pixels; "
          "log_laplace samples are iid Laplace draws (no R-hat); log_laplace is our re-implementation of the idea of arXiv:2410.11454 "
          "with its own lambda (prior acts on log-emissivity), gp/mfi/tikhonov_tuned use their own hyperparameters.\n"]
    for name in names:
        rr = [r for r in rows if r["problem"] == name]
        md.append(f"## {name}\n")
        md.append(md_table(sorted(rr, key=lambda r: order.index(r["method"]) if r["method"] in order else 99),
                           ["method"] + METRICS + ["frac_top", "frac_at_upper", "frac_at_zero"]))
    md.append(f"## Mean over {len(names)} problems\n")
    md.append(md_table(agg, ["method", "n"] + METRICS))
    with open(f"{out}/ablation_table.md", "w") as f:
        f.write("\n".join(md))

    # bar figure: rel-L2 per problem and mean
    show = [m for m in order if any(r["method"] == m for r in rows)]
    cols = [OI[c] for c in ("sky", "blue", "green", "orange", "verm", "purple", "black", "yellow")]
    fig, axs = plt.subplots(1, 2, figsize=(max(7, 1.0 * len(names) + 6), 3.4), gridspec_kw={"width_ratios": [max(2, len(names)), 1.4]})
    w = 0.8 / len(show)
    for i, m in enumerate(show):
        v = [next((r.get("rel_l2", np.nan) for r in rows if r["problem"] == n and r["method"] == m), np.nan) for n in names]
        axs[0].bar(np.arange(len(names)) + i * w, v, w, label=m, color=cols[i % len(cols)], hatch="//" if m.startswith("potts") else None)
    axs[0].set_xticks(np.arange(len(names)) + 0.4 - w / 2)
    axs[0].set_xticklabels(names, rotation=20, fontsize=7)
    axs[0].set_ylabel("relative L2 error")
    axs[0].legend(fontsize=5.5, ncol=2)
    mean_l2 = [next(a.get("rel_l2", np.nan) for a in agg if a["method"] == m) for m in show]
    axs[1].barh(range(len(show)), mean_l2, color=[cols[i % len(cols)] for i in range(len(show))])
    axs[1].set_yticks(range(len(show)))
    axs[1].set_yticklabels(show, fontsize=6)
    axs[1].invert_yaxis()
    axs[1].set_xlabel(f"mean rel-L2 ({len(names)} problems)")
    fig.tight_layout()
    save_fig(fig, out, "ablation_bars.png")
    save_json({"rows": rows, "mean": agg, "sched": sched}, f"{out}/results.json")
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
