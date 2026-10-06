"""Multi-seed statistics for the converging EBMs against the baselines.

Runs the 4 named phantoms with several noise seeds each, plus N random fields, one noise draw each. Every method sees
the same problems. The EBMs are Potts and I-sparse (threshold from --sparse-threshold, default 0.2); both converge.
MAP annealing is skipped because only posterior quantities are reported. The summary gives:
- per phantom, mean ± std over seeds;
- over random fields, mean and median;
- paired win rates of each EBM against each baseline, with a two-sided sign-test p-value.
"""
import os
import sys
import time
from math import comb

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (build_problem, ebm_estimates, log, make_parser, md_table, metric_row, run_baselines,  # noqa: E402
                     run_ebm, save_csv, save_json, setup)

SCRIPT = "multiseed"
METRICS = ["rel_l2", "ssim", "coverage68", "coverage95"]


def sign_test(wins, losses):
    """Two-sided exact sign test (ties dropped)."""
    n = wins + losses
    if n == 0:
        return float("nan")
    k = min(wins, losses)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    ap = make_parser(SCRIPT, __doc__)
    ap.add_argument("--seeds", type=int, default=5, help="noise seeds per named phantom")
    ap.add_argument("--n-random", type=int, default=30, help="random fields")
    ap.add_argument("--variants", default="potts,sparse")
    ap.add_argument("--sparse-threshold", type=float, default=0.2)
    ap.add_argument("--geometry", choices=["default", "tcv"], default="default",
                    help="'tcv' builds problems with tomo.tcv.make_tcv_problem (use with --config configs/tcv.yaml)")
    args = ap.parse_args()
    over = {"model": {"sparse_threshold": args.sparse_threshold}}
    if args.geometry == "tcv" and args.out is None:
        args.out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results",
                                "multiseed_tcv" + ("_quick" if args.quick else ""))
    cfg, out = setup(args, SCRIPT, extra_overrides=over)
    if args.geometry == "tcv":
        from tomo.tcv import make_tcv_problem
        builder = make_tcv_problem
    else:
        builder = build_problem
    variants = args.variants.split(",")
    ex = cfg["experiments"]
    seeds = 1 if args.quick else args.seeds
    nr = 1 if args.quick else args.n_random
    probs = [(f"{ph}_s{s}", ph, int(cfg["seed"]) + s, ph) for ph in ex["phantoms"] for s in range(seeds)]
    probs += [(f"random{k:03d}", "random", int(ex["seed0"]) + k, "random") for k in range(nr)]
    log(f"{len(probs)} problems: {len(ex['phantoms'])} phantoms x {seeds} seeds + {nr} random fields; variants={variants}; "
        f"sparse_threshold={args.sparse_threshold}; MAP anneal skipped.")

    rows, runs = [], []
    t0 = time.time()
    for i, (name, ph, seed, group) in enumerate(probs):
        p = builder(cfg, ph, seed)
        log(f"[{i + 1}/{len(probs)}] {name} seed={seed} load={os.getloadavg()[0]:.1f} elapsed={time.time() - t0:.0f}s")
        for m, e in run_baselines(p).items():
            rows.append(metric_row(p, e, problem=name, group=group, seed=seed, method=m))
        for vi, v in enumerate(variants):
            kw = {"do_map": False}
            res, info = run_ebm(p, v, cfg, key_seed=seed * 10 + vi, **kw)
            info.update(problem=name, group=group, seed=seed)
            runs.append(info)
            if res is None:
                continue
            d, _ = ebm_estimates(res)
            r = metric_row(p, d, problem=name, group=group, seed=seed, method=v)
            r.update(rhat_max=info.get("rhat_max"), time_s=info.get("time_s"))
            rows.append(r)
            log(f"    {v}: rel_l2={r['rel_l2']:.3f} cov95={r.get('coverage95', float('nan')):.2f} "
                f"rhat_max={info['rhat_max']:.3g} t={info['time_s']:.1f}s")
            del res
        save_csv(rows, f"{out}/metrics_all.csv")
        save_json({"rows": rows, "runs": runs}, f"{out}/results.json")

    methods = list(dict.fromkeys(r["method"] for r in rows))
    groups = list(dict.fromkeys(r["group"] for r in rows))
    summ = []
    for g in groups:
        for m in methods:
            sel = [r for r in rows if r["group"] == g and r["method"] == m]
            row = {"group": g, "method": m, "n": len(sel)}
            for k in METRICS + ["rhat_max", "time_s"]:
                v = np.array([r[k] for r in sel if r.get(k) is not None and np.isfinite(r[k])], float)
                row[k] = f"{v.mean():.3f} ± {v.std(ddof=1):.3f}" if len(v) > 1 else (f"{v[0]:.3f}" if len(v) else "-")
                if k == "rel_l2" and len(v):
                    row["rel_l2_median"] = f"{np.median(v):.3f}"
            summ.append(row)

    by = {}
    for r in rows:
        by.setdefault((r["problem"], r["method"]), r["rel_l2"])
    baselines = [m for m in methods if m not in variants]
    wins = []
    for v in variants:
        for b in baselines:
            for g in groups + ["all"]:
                probs_g = [n for (n, _, _, gg) in probs if g == "all" or gg == g]
                w = sum(1 for n in probs_g if (n, v) in by and (n, b) in by and by[(n, v)] < by[(n, b)])
                l_ = sum(1 for n in probs_g if (n, v) in by and (n, b) in by and by[(n, v)] > by[(n, b)])
                wins.append({"ebm": v, "baseline": b, "group": g, "wins": w, "losses": l_,
                             "win_rate": f"{w / max(w + l_, 1):.2f}", "sign_test_p": f"{sign_test(w, l_):.3g}"})

    save_json({"summary": summ, "wins": wins}, f"{out}/summary.json")
    with open(f"{out}/summary.md", "w") as f:
        f.write(f"# Multi-seed statistics ({args.geometry} geometry)\n\n{len(probs)} problems; phantoms x {seeds} noise seeds; {nr} random fields. "
                f"EBMs: {variants} (sparse threshold {args.sparse_threshold}). Mean ± std over problems in each group.\n\n")
        f.write(md_table(summ, ["group", "method", "n", "rel_l2", "rel_l2_median", "ssim", "coverage68", "coverage95",
                                "rhat_max", "time_s"]))
        f.write("\n\n## Paired rel-L2 win rates (EBM better than baseline), two-sided sign test\n\n")
        f.write(md_table(wins, ["ebm", "baseline", "group", "wins", "losses", "win_rate", "sign_test_p"]))
        f.write("\n")
    log(f"done in {time.time() - t0:.0f}s -> {out}/summary.md")


if __name__ == "__main__":
    main()
