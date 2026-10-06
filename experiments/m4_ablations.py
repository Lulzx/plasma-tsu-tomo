"""M4: ablations. --which K,tau,sparse,noise (default all). Each runs EBMs on experiments.ablation_phantoms."""
from __future__ import annotations

import time

from _common import (METHOD_COLORS, OI, build_problem, ebm_estimates, log, make_parser, md_table, metric_row, np, plt,
                     run_baselines, run_ebm, save_csv, save_fig, save_json, setup)
from tomo.config import deep_merge

SCRIPT = "m4_ablations"
ALL = ["K", "tau", "sparse", "noise"]


def one(cfg, p, variant, label, rows, seed, **kw):
    log(f"  {dict(label)}: {variant}")
    res, info = run_ebm(p, variant, cfg, key_seed=seed, **kw)
    base = dict(problem=p.phantom, seed=p.seed, variant=variant, **label_dict(label))
    if res is None:
        rows.append({**base, "error": info["error"]})
        return
    d, mp = ebm_estimates(res)
    r = metric_row(p, d, **base)
    r.update({k: info.get(k) for k in ("rhat_max", "invalid_frac", "n_spins", "n_blocks", "max_degree", "time_s",
                                       "tau_over_dz", "eff_noise_ratio", "sweeps_to_converge")})
    if mp is not None:
        r["map_rel_l2"] = metric_row(p, mp)["rel_l2"]
    rows.append(r)


def label_dict(label):
    return dict(label)


def main():
    ap = make_parser(SCRIPT, __doc__)
    ap.add_argument("--which", default=",".join(ALL))
    ap.add_argument("--variants", default="potts,chain", help="variants for the K and noise ablations")
    args = ap.parse_args()
    cfg, out = setup(args, SCRIPT)
    which = args.which.split(",")
    if [w for w in which if w not in ALL]:
        raise SystemExit(f"unknown --which {[w for w in which if w not in ALL]}; choose from {ALL}")
    kv = args.variants.split(",")
    ex = cfg["experiments"]
    phs = ex["ablation_phantoms"] if not args.quick else ex["ablation_phantoms"][:1]
    t0 = time.time()
    tables = {}

    def plot(name, rows, xkey, xlabel, groupkey="variant", logx=False, ref=None):
        ok = [r for r in rows if "error" not in r]
        if not ok:
            return
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
        for g in sorted({r[groupkey] for r in ok}):
            rr = sorted([r for r in ok if r[groupkey] == g and r.get(xkey) is not None], key=lambda r: r[xkey])
            col = METHOD_COLORS.get(g, OI["black"])
            axs[0].plot([r[xkey] for r in rr], [r["rel_l2"] for r in rr], "o-", color=col, label=g)
            if any(r.get("coverage95") is not None for r in rr):
                axs[1].plot([r[xkey] for r in rr], [r.get("coverage95", np.nan) for r in rr], "o-", color=col, label=g)
        if ref:
            axs[0].plot(*ref, "k--", lw=1, label="tuned Tikhonov")
        axs[1].axhspan(0.90, 0.98, color="0.85", zorder=0, label="target")
        for ax, yl in zip(axs, ["relative L2 error", "95% interval coverage"]):
            ax.set_xlabel(xlabel)
            ax.set_ylabel(yl)
            if logx:
                ax.set_xscale("log")
            ax.legend(fontsize=7)
        fig.suptitle(f"{name} ablation ({', '.join(phs)})", fontsize=9)
        save_fig(fig, out, f"ablation_{name}.png")

    for ph in phs:
        p = build_problem(cfg, ph, cfg["seed"])
        tik = metric_row(p, run_baselines(p, ["tikhonov_tuned"])["tikhonov_tuned"])["rel_l2"]
        if "K" in which:
            rows = tables.setdefault("K", [])
            for K in ex["K_values"]:
                c = deep_merge(cfg, {"model": {"K": int(K)}})
                for v in kv:
                    one(c, p, v, [("K", int(K))], rows, 1)
        if "tau" in which:
            rows = tables.setdefault("tau", [])
            for tau in ex["tau_values"]:
                one(cfg, p, "chain", [("tau", float(tau))], rows, 2, tau=float(tau))
        if "sparse" in which:
            rows = tables.setdefault("sparse", [])
            for th in ex["sparse_thresholds"]:
                one(cfg, p, "sparse", [("threshold", float(th))], rows, 3, threshold=float(th))
        if "noise" in which:
            rows = tables.setdefault("noise", [])
            for nz in ex["noise_levels"]:
                pn = build_problem(cfg, ph, cfg["seed"], noise_rel=nz)
                tn = metric_row(pn, run_baselines(pn, ["tikhonov_tuned"])["tikhonov_tuned"])
                rows.append({"problem": ph, "variant": "tikhonov_tuned", "noise": float(nz), "rel_l2": tn["rel_l2"]})
                c = deep_merge(cfg, {"noise": {"rel": float(nz)}})
                for v in kv:
                    one(c, pn, v, [("noise", float(nz))], rows, 4)
        tik_ref = tik
    # tables + plots
    md = ["# M4 ablations\n", f"phantoms: {phs}; schedule: {cfg['schedule']['posterior']}, n_chains={cfg['schedule']['n_chains']}\n"]
    cols = ["problem", "variant", "rel_l2", "map_rel_l2", "coverage95", "rhat_max", "invalid_frac", "n_spins", "n_blocks", "time_s", "error"]
    for k, rows in tables.items():
        save_csv(rows, f"{out}/ablation_{k}.csv")
        xkey = {"K": "K", "tau": "tau", "sparse": "threshold", "noise": "noise"}[k]
        md.append(f"## {k}\n")
        md.append(md_table(rows, [xkey] + [c for c in cols if c != "error"] + ["error"]) if rows else "")
        tix = [r for r in rows if r["variant"] == "tikhonov_tuned"]
        plot(k, [r for r in rows if r["variant"] != "tikhonov_tuned"], xkey, {"K": "levels K", "tau": "tau (units of dz)",
             "sparse": "sparsification threshold (fraction of max |Q_jk|)", "noise": "relative noise level"}[k],
             logx=(k in ("tau", "sparse")),
             ref=([r[xkey] for r in tix], [r["rel_l2"] for r in tix]) if tix else
             ([min(r[xkey] for r in rows), max(r[xkey] for r in rows)], [tik_ref] * 2))
    save_json(tables, f"{out}/ablations.json")
    with open(f"{out}/ablations_table.md", "w") as f:
        f.write("\n".join(md))
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
