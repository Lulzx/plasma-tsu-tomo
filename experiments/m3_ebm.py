"""M3: EBM variants (Potts, dense/sparse Ising, I-chain, I-tree) vs baselines on the SAME problems and noise draws."""
from __future__ import annotations

import os
import time

from _common import (EBM_VARIANTS, METHOD_COLORS, OI, build_problem, ebm_estimates, field_grid, log, make_parser,
                     md_table, metric_row, np, plt, problem_list, run_baselines, run_ebm, save_csv, save_fig,
                     save_json, setup)

SCRIPT = "m3_ebm"
METRICS = ["rel_l2", "ssim", "peak_err_cm", "power_err", "chi2_red", "coverage68", "coverage95"]
DIAG = ["rhat_max", "rhat_median", "invalid_frac", "n_spins", "n_blocks", "max_degree", "sweeps_posterior",
        "sweeps_to_converge", "time_s", "wall_incl_compile_s"]


def main():
    ap = make_parser(SCRIPT, __doc__)
    ap.add_argument("--variants", default=None, help="comma list from potts,dense,sparse,chain,tree (default: config experiments.variants)")
    ap.add_argument("--n-random", type=int, default=None,
                    help="random fields in addition to the 4 phantoms (default experiments.n_random_ebm=20; EBMs cost ~1 min each)")
    args = ap.parse_args()
    cfg, out = setup(args, SCRIPT)
    variants = args.variants.split(",") if args.variants else list(cfg["experiments"]["variants"])
    bad = [v for v in variants if v not in EBM_VARIANTS]
    if bad:
        raise SystemExit(f"unknown variants {bad}")
    nr = cfg["experiments"]["n_random_ebm"] if args.n_random is None else args.n_random
    probs = problem_list(cfg, nr)
    os.makedirs(f"{out}/runs", exist_ok=True)
    log(f"variants={variants}; {len(probs)} problems = {len(cfg['experiments']['phantoms'])} phantoms + {nr} random fields.")
    log(f"SUBSAMPLING NOTE: EBM runs use only {nr} random fields (baselines-only study m2 uses n_random={cfg['experiments']['n_random']}); "
        f"baselines here are run on exactly the same problem/noise draws.")

    rows, runs, failures = [], [], []
    keep = {}
    sc = cfg["schedule"]
    meta = {"variants": variants, "n_random_ebm": nr, "n_problems": len(probs), "quick": args.quick,
            "subsampling": f"{nr} random fields for EBMs (baseline study m2 uses {cfg['experiments']['n_random']})",
            "schedule": sc, "model": cfg["model"]}

    def flush():
        save_json({"meta": meta, "rows": rows, "runs": runs, "failures": failures}, f"{out}/results.json")
        save_csv(rows, f"{out}/metrics_all.csv")

    t_start = time.time()
    for pi, (name, ph, seed) in enumerate(probs):
        p = build_problem(cfg, ph, seed)
        kind = "phantom" if ph != "random" else "random"
        log(f"[{pi + 1}/{len(probs)}] {name} (seed {seed}) load={os.getloadavg()[0]:.1f}")
        est = run_baselines(p)
        for m, e in est.items():
            rows.append(metric_row(p, e, problem=name, kind=kind, seed=seed, method=m))
        item = {"p": p, "base": est, "ebm": {}} if kind == "phantom" else None
        for vi, v in enumerate(variants):
            log(f"  running {v} ...")
            res, info = run_ebm(p, v, cfg, key_seed=seed * 10 + vi)
            info.update(problem=name, kind=kind, seed=seed)
            runs.append(info)
            if res is None:
                failures.append(info)
                continue
            d, mp = ebm_estimates(res)
            r = metric_row(p, d, problem=name, kind=kind, seed=seed, method=v)
            r.update({k: info.get(k) for k in DIAG})
            rows.append(r)
            if mp is not None:
                rows.append(metric_row(p, mp, problem=name, kind=kind, seed=seed, method=f"{v}_map"))
            log(f"    {v}: rel_l2={r.get('rel_l2', float('nan')):.3f} rhat_max={info['rhat_max']:.3g} "
                f"time={info['time_s']:.1f}s spins={info['n_spins']} blocks={info['n_blocks']}")
            np.savez_compressed(f"{out}/runs/{name}_{v}.npz", mean=res["mean"], std=res["std"],
                                map=res["map"] if res.get("map") is not None else np.zeros(0),
                                energy_trace=np.asarray(res["energy_trace"], np.float32) if res.get("energy_trace") is not None else np.zeros(0),
                                rhat=np.asarray(res["rhat"], np.float32))
            if item is not None:
                item["ebm"][v] = {"mean": res["mean"], "std": res["std"], "map": res.get("map"),
                                  "trace": np.asarray(res["energy_trace"]) if res.get("energy_trace") is not None else None}
            del res
        if item is not None:
            keep[name] = item
        flush()
    log(f"total {time.time() - t_start:.0f}s, load={os.getloadavg()[0]:.1f}; failures: {len(failures)}")
    for f_ in failures:
        log(f"  FAILED {f_['problem']}/{f_['variant']}: {f_['error']}")

    # ---- tables
    methods = ["tikhonov_gcv", "tikhonov_tuned", "mfi", "gp"] + [x for v in variants for x in (v, f"{v}_map")]
    md = ["# M3 EBM comparison\n",
          f"Variants: {variants}. Schedule: posterior {sc['posterior']}, {sc['n_chains']} chains, anneal {sc['anneal']}, K={cfg['model']['K']}.\n",
          f"**Subsampling:** EBM runs on the 4 phantoms + {nr} random fields (m2 baselines use {cfg['experiments']['n_random']}). "
          "Baselines in this table use the same problems and noise draws. `*_map` rows are the annealed MAP estimate. "
          "time_s = sampling+anneal time reported by the sampler; for potts/dense/sparse this excludes JIT compile, but for **chain** it includes model build, "
          "Tikhonov warm start and JIT compile (tomo/ebm_chain.py measures from function entry), so chain time_s is NOT comparable and overstates sampling cost.\n"]
    if failures:
        md.append("## FAILURES\n")
        md.append(md_table(failures, ["problem", "variant", "error"]))
    for name in [p_[0] for p_ in probs if p_[1] != "random"]:
        rr = [r for r in rows if r["problem"] == name]
        md.append(f"## {name}\n")
        md.append(md_table(sorted(rr, key=lambda r: methods.index(r["method"]) if r["method"] in methods else 99),
                           ["method"] + METRICS + ["rhat_max", "invalid_frac", "n_spins", "n_blocks", "max_degree", "time_s"]))
    if nr:
        md.append(f"## Random fields (n={nr}): mean over fields\n")
        agg = []
        for m in methods:
            rr = [r for r in rows if r["method"] == m and r["kind"] == "random" and "error" not in r]
            if rr:
                a = {"method": m, "n": len(rr)}
                for k in METRICS + ["rhat_max", "time_s"]:
                    v = [r[k] for r in rr if r.get(k) is not None and np.isfinite(r[k])]
                    if v:
                        a[k] = float(np.mean(v))
                agg.append(a)
        md.append(md_table(agg, ["method", "n"] + METRICS + ["rhat_max", "time_s"]))
        save_json(agg, f"{out}/random_summary.json")
    with open(f"{out}/results_table.md", "w") as f:
        f.write("\n".join(md))

    # ---- figures
    if keep:
        names = list(keep)
        p0 = keep[names[0]]["p"]
        cols = ["truth", "Tikhonov (tuned)", "GP mean", "GP std"]
        for v in variants:
            cols += [f"{v} mean", f"{v} std", f"{v} MAP"]
        grid_rows = []
        for n in names:
            it = keep[n]
            gp = it["base"].get("gp", {})
            row = [(it["p"].eps_true, "field"), (it["base"]["tikhonov_tuned"].get("mean"), "field"),
                   (gp.get("mean"), "field"), (gp.get("std"), "std")]
            for v in variants:
                e = it["ebm"].get(v)
                row += [(e["mean"], "field"), (e["std"], "std"), (e["map"], "field")] if e else [(None, "field")] * 3
            grid_rows.append(row)
        field_grid(out, "ebm_reconstructions.png", [[(v, k) if v is not None else (None, k) for v, k in r] for r in grid_rows],
                   cols, p0.mask, p0.grid, names)
        # energy traces
        have = [v for v in variants if any(v in keep[n]["ebm"] for n in names)]
        if have:
            fig, axs = plt.subplots(len(names), len(have), figsize=(3.2 * len(have), 2.3 * len(names)), squeeze=False)
            sps = int(sc["posterior"]["steps_per_sample"])
            for i, n in enumerate(names):
                for j, v in enumerate(have):
                    ax = axs[i, j]
                    e = keep[n]["ebm"].get(v)
                    if e is not None and e["trace"] is not None and e["trace"].ndim >= 2:
                        tr = np.asarray(e["trace"], float)
                        while tr.ndim > 2:
                            tr = tr[..., -1]
                        x = (np.arange(tr.shape[1]) + 1) * sps
                        for c in range(tr.shape[0]):
                            ax.plot(x, tr[c], lw=0.6, alpha=0.7, color=METHOD_COLORS[v])
                    if i == 0:
                        ax.set_title(v, fontsize=9)
                    if i == len(names) - 1:
                        ax.set_xlabel("posterior sweeps after warm-up")
                    if j == 0:
                        ax.set_ylabel(f"{n}\nenergy E (units of kT)")
            fig.tight_layout()
            save_fig(fig, out, "energy_traces.png")
        # R-hat summary bar
        rr = [r for r in runs if not r.get("failed") and r["kind"] == "phantom"]
        if rr:
            fig, ax = plt.subplots(figsize=(5, 3.2))
            for k, v in enumerate(variants):
                vals = [min(r["rhat_max"], 10) for r in rr if r["variant"] == v]
                ax.scatter([k] * len(vals), vals, color=METHOD_COLORS[v], s=25)
            ax.axhline(1.05, color="0.4", ls="--", lw=0.8, label="1.05")
            ax.set_xticks(range(len(variants)))
            ax.set_xticklabels(variants)
            ax.set_ylabel("max split R-hat over pixels (clipped at 10)")
            ax.legend(fontsize=7)
            save_fig(fig, out, "rhat.png")
    flush()
    log("done")


if __name__ == "__main__":
    main()
