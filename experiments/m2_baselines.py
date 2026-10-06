"""M2: classical baselines (Tikhonov GCV / tuned (evidence), MFI, GP) on the 4 phantoms + N random fields."""
from __future__ import annotations

import time

from _common import (BASELINES, METHOD_COLORS, OI, build_problem, field_grid, log, make_parser, md_table, metric_row,
                     np, plt, problem_list, run_baselines, save_csv, save_fig, save_json, setup)

SCRIPT = "m2_baselines"
METRICS = ["rel_l2", "ssim", "peak_err_cm", "power_err", "chi2_red", "coverage68", "coverage95"]


def main():
    ap = make_parser(SCRIPT, __doc__)
    ap.add_argument("--n-random", type=int, default=None, help="random fields (default experiments.n_random=200)")
    args = ap.parse_args()
    cfg, out = setup(args, SCRIPT)
    nr = cfg["experiments"]["n_random"] if args.n_random is None else args.n_random
    probs = problem_list(cfg, nr)
    log(f"{len(probs)} problems ({len(cfg['experiments']['phantoms'])} phantoms + {nr} random fields), baselines {BASELINES}")
    rows, keep = [], {}
    t0 = time.time()
    for i, (name, ph, seed) in enumerate(probs):
        p = build_problem(cfg, ph, seed)
        est = run_baselines(p)
        for m, e in est.items():
            r = metric_row(p, e, problem=name, kind="phantom" if ph != "random" else "random", seed=seed, method=m)
            if "hyper" in e:
                r.update({f"gp_{k}": v for k, v in e["hyper"].items()})
            rows.append(r)
        if ph != "random" or name == "random000":
            keep[name] = (p, est)
        if (i + 1) % 20 == 0 or i == len(probs) - 1:
            log(f"  {i + 1}/{len(probs)}  {time.time() - t0:.0f}s")
    save_csv(rows, f"{out}/metrics_all.csv")
    save_json(rows, f"{out}/metrics_all.json")

    # --- summary tables
    def table(sel_kind):
        res = []
        for m in BASELINES:
            rr = [r for r in rows if r["method"] == m and r["kind"] == sel_kind and "error" not in r]
            if not rr:
                continue
            row = {"method": m, "n": len(rr)}
            for k in METRICS:
                v = np.array([r[k] for r in rr if r.get(k) is not None], float)
                if len(v):
                    row[k] = float(np.mean(v))
                    if sel_kind == "random":
                        row[k + "_med"] = float(np.median(v))
                        row[k + "_std"] = float(np.std(v))
            res.append(row)
        return res

    md = ["# M2 baselines\n", "## Named phantoms (per phantom)\n"]
    for name in cfg["experiments"]["phantoms"]:
        rr = [r for r in rows if r["problem"] == name]
        md.append(f"### {name}\n")
        md.append(md_table(rr, ["method"] + METRICS + ["lam", "time"]))
    summ = {}
    rand = table("random")
    if rand:
        md.append(f"## Random fields (n={rand[0]['n']}): mean over fields\n")
        md.append(md_table(rand, ["method"] + METRICS))
        md.append("Median / std of rel-L2 over random fields:\n")
        md.append(md_table(rand, ["method", "rel_l2_med", "rel_l2_std", "ssim_med", "peak_err_cm_med"]))
        summ["random"] = rand
        wins = {}
        for m in BASELINES:
            wins[m] = 0
        by_seed = {}
        for r in rows:
            if r["kind"] == "random" and "error" not in r:
                by_seed.setdefault(r["problem"], {})[r["method"]] = r["rel_l2"]
        for d in by_seed.values():
            wins[min(d, key=d.get)] += 1
        md.append("Best rel-L2 count over random fields: " + ", ".join(f"{k}={v}" for k, v in wins.items()) + "\n")
        summ["best_count"] = wins
    summ["phantoms"] = {n: [r for r in rows if r["problem"] == n] for n in cfg["experiments"]["phantoms"]}
    with open(f"{out}/metrics_table.md", "w") as f:
        f.write("\n".join(md))
    save_json(summ, f"{out}/summary.json")

    # --- figure: reconstruction grid (phantoms + one random field)
    names = [n for n in keep]
    cols = ["truth", "Tikhonov (GCV)", "Tikhonov (tuned)", "MFI", "GP mean", "GP std"]
    grid_rows = []
    for n in names:
        p, est = keep[n]
        g = lambda k, f="mean": est[k].get(f) if "error" not in est[k] else None  # noqa: E731
        grid_rows.append([(p.eps_true, "field"), (g("tikhonov_gcv"), "field"), (g("tikhonov_tuned"), "field"),
                          (g("mfi"), "field"), (g("gp"), "field"), (g("gp", "std"), "std")])
    p0 = keep[names[0]][0]
    field_grid(out, "reconstructions.png", grid_rows, cols, p0.mask, p0.grid, names)

    # --- figure: error distributions over random fields
    if rand:
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
        for ax, key, lab in zip(axs, ["rel_l2", "ssim"], ["relative L2 error", "SSIM"]):
            data = [[r[key] for r in rows if r["method"] == m and r["kind"] == "random" and "error" not in r] for m in BASELINES]
            bp = ax.boxplot(data, patch_artist=True, showfliers=False)
            for patch, m in zip(bp["boxes"], BASELINES):
                patch.set_facecolor(METHOD_COLORS[m])
                patch.set_alpha(0.8)
            ax.set_xticks(range(1, len(BASELINES) + 1))
            ax.set_xticklabels(BASELINES, rotation=20, fontsize=8)
            ax.set_ylabel(lab)
            ax.set_title(f"{lab}, {len(data[0])} random fields")
        save_fig(fig, out, "random_fields_boxplot.png")
    # GP coverage
    cov = [r for r in rows if r["method"] == "gp" and "coverage95" in r]
    if cov:
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        ax.hist([r["coverage95"] for r in cov], bins=20, color=OI["orange"])
        ax.axvspan(0.90, 0.98, color="0.85", zorder=0, label="target 90-98%")
        ax.set_xlabel("fraction of pixels inside GP 95% interval")
        ax.set_ylabel("count of problems")
        ax.legend(fontsize=7)
        save_fig(fig, out, "gp_coverage95.png")
    log("done")


if __name__ == "__main__":
    main()
