"""M5: TSU energy/latency estimate from measured n_spins / n_blocks / sweeps (m3 results), sensitivity, laptop and GPU comparison."""
from __future__ import annotations

import json
import os

from _common import (OI, ROOT, fmt, log, make_parser, md_table, np, plt, save_csv, save_fig, save_json, setup)
from tomo import energy as En

SCRIPT = "m5_energy"
POWERMETRICS_NOTE = ("Laptop energy = measured wall time x CPU package power. Default power is an ASSUMED constant (--cpu-power-w, 20 W). "
                     "For a measured value run, in a separate terminal during an m3 run: "
                     "`sudo powermetrics --samplers cpu_power -i 200 > pm.log` and pass `--powermetrics-log pm.log` "
                     "(parsed for 'CPU Power: N mW' lines; the mean is used). This script never runs sudo itself.")


def load_measured(path, cfg):
    """Per-variant measured numbers from m3 results.json, or None (placeholder fallback)."""
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    runs = [r for r in d["runs"] if not r.get("failed")]
    out = {}
    for v in sorted({r["variant"] for r in runs}):
        rr = [r for r in runs if r["variant"] == v]
        med = lambda k: float(np.median([r[k] for r in rr if r.get(k) is not None])) if any(r.get(k) is not None for r in rr) else None  # noqa: E731
        conv = [r["sweeps_to_converge"] for r in rr if r.get("sweeps_to_converge") is not None]
        out[v] = {"n_runs": len(rr), "n_spins": med("n_spins"), "n_blocks": med("n_blocks"), "max_degree": med("max_degree"),
                  "time_s": med("time_s"), "n_warmup": med("n_warmup"), "sweeps_posterior_full": med("sweeps_posterior"),
                  "sweeps_map": med("sweeps_map"), "frac_converged": len(conv) / len(rr),
                  "sweeps_to_converge_median": float(np.median(conv)) if conv else None,
                  "n_chains": rr[0].get("n_chains", cfg["schedule"]["n_chains"])}
    return out or None


def main():
    ap = make_parser(SCRIPT, __doc__)
    ap.add_argument("--m3-results", default=None, help="m3 results.json (default results/m3_ebm[_quick]/results.json)")
    ap.add_argument("--cpu-power-w", type=float, default=None, help="assumed CPU package power (default experiments.cpu_power_W=20)")
    ap.add_argument("--powermetrics-log", default=None, help="powermetrics log to parse for CPU power")
    ap.add_argument("--sens-variant", default="chain")
    args = ap.parse_args()
    cfg, out = setup(args, SCRIPT)
    m3 = args.m3_results or os.path.join(ROOT, "results", "m3_ebm" + ("_quick" if args.quick else ""), "results.json")
    meas = load_measured(m3, cfg)
    watts = args.cpu_power_w or cfg["experiments"]["cpu_power_W"]
    C = int(cfg["schedule"]["n_chains"])
    rows, warn = [], []
    if meas is None:
        msg = f"WARNING: {m3} not found; using spec placeholders (I-chain 40,600 spins, 44 blocks, 16 chains, 2500/600 sweeps)."
        log(msg)
        warn.append(msg)
        meas = {"chain (spec placeholder)": {"n_spins": 40600, "n_blocks": 44, "max_degree": 364, "time_s": None,
                                             "sweeps_posterior_full": 2500, "n_warmup": 0, "sweeps_map": 600,
                                             "frac_converged": None, "sweeps_to_converge_median": None, "n_chains": 16}}
    for v, m in meas.items():
        conv = m["sweeps_to_converge_median"]
        if conv is not None and m["frac_converged"] == 1.0:
            sw, basis = int((m["n_warmup"] or 0) + conv), "measured sweeps-to-converge (R-hat<1.05, median over runs) + warm-up"
        else:
            sw = int(m["sweeps_posterior_full"])
            basis = ("full configured schedule (NOT converged in %.0f%% of runs; value is a lower bound on cost to converge)" % (100 * (1 - m["frac_converged"]))
                     if m["frac_converged"] is not None else "spec placeholder")
            if m["frac_converged"] is not None:
                warn.append(f"{v}: not converged in all runs; using full schedule ({sw} sweeps) -- not a converged-cost estimate.")
        for kind, nsw in (("posterior", sw), ("MAP anneal", int(m["sweeps_map"] or 0))):
            if nsw <= 0:
                continue
            t = En.tsu_estimate(m["n_spins"], nsw, m["n_chains"], m["n_blocks"])
            gpu = En.gpu_mcmc_estimate(m["n_spins"], nsw, m["n_chains"], m["max_degree"] or 100)
            row = {"variant": v, "task": kind, "n_spins": m["n_spins"], "n_blocks": m["n_blocks"], "max_degree": m["max_degree"],
                   "n_chains": m["n_chains"], "sweeps": nsw, "sweeps_basis": basis if kind == "posterior" else "configured anneal schedule",
                   "tsu_energy_J": t["energy_J"], "tsu_latency_s": t["latency_s"],
                   "gpu_energy_J": gpu["energy_J"], "gpu_latency_s": gpu["latency_s"], "gpu_memory_bound": gpu["memory_bound"]}
            # laptop: measured wall time (posterior sampling+anneal total is m['time_s']); apportion by sweeps
            if m["time_s"] is not None:
                tot_sw = (m["sweeps_posterior_full"] or 0) + (m["sweeps_map"] or 0)
                t_task = m["time_s"] * (nsw / tot_sw) if tot_sw else None
                if t_task is not None:
                    src = args.powermetrics_log if args.powermetrics_log else watts
                    le = En.cpu_energy_from_powermetrics(src, t_task)
                    row.update(laptop_time_s=t_task, laptop_power_W=le["avg_power_W"], laptop_energy_J=le["energy_J"],
                               laptop_basis="powermetrics" if args.powermetrics_log else f"assumed {watts} W")
            rows.append(row)
    save_csv(rows, f"{out}/energy_table.csv")
    save_json({"rows": rows, "warnings": warn, "m3_results": m3, "cpu_power_W": watts}, f"{out}/energy.json")

    cols = ["variant", "task", "n_spins", "n_blocks", "n_chains", "sweeps", "tsu_energy_J", "tsu_latency_s",
            "laptop_time_s", "laptop_energy_J", "gpu_energy_J", "gpu_latency_s"]
    md = ["# TSU energy / latency estimates\n",
          "Model (tomo/energy.py): E = N_spins x N_sweeps x N_chains x E_cell (E_cell = 1.3 fJ), t = N_sweeps x N_blocks x t_update (t_update = 100 ns; chains in parallel on separate chip area). "
          "These are assumptions, not measurements of TSU hardware.\n"]
    for w in warn:
        md.append(f"> {w}\n")
    md.append(md_table(rows, cols))
    md.append("\nSweeps basis per variant:\n")
    for r in rows:
        if r["task"] == "posterior":
            md.append(f"- {r['variant']}: {r['sweeps']} sweeps -- {r['sweeps_basis']}")
    md.append("\n## Laptop and GPU comparison\n")
    md.append(POWERMETRICS_NOTE + "\n")
    md.append("GPU: tomo.energy.gpu_mcmc_estimate (op-count model: 2*degree+10 flops/update, 1e-11 J/flop, 10 TFLOP/s peak at 10% efficiency, 400 GB/s memory); "
              "energy is flop-based (lower bound when memory bound). Laptop time apportioned between posterior and MAP by sweeps; spec spins/blocks for Potts count 'categorical' sites with K levels, "
              "so TSU numbers for Potts assume a (hypothetical) categorical sampler cell and are not hardware-faithful.\n")
    # sensitivity
    sv = args.sens_variant if args.sens_variant in meas else next(iter(meas))
    m = meas[sv]
    base = [r for r in rows if r["variant"] == sv and r["task"] == "posterior"][0]
    sens = En.sensitivity(m["n_spins"], base["sweeps"], base["n_chains"], m["n_blocks"])
    save_json(sens, f"{out}/sensitivity.json")
    save_csv(sens["rows"], f"{out}/sensitivity.csv")
    md.append(f"## Sensitivity ({sv}, posterior, {base['sweeps']} sweeps, {base['n_chains']} chains)\n")
    md.append(md_table(sens["rows"], ["E_cell_factor", "E_cell_J", "t_update_s", "energy_J", "latency_s"]))
    with open(f"{out}/energy_table.md", "w") as f:
        f.write("\n".join(md))

    efs, tus = (0.5, 1, 3), (50e-9, 100e-9, 500e-9)
    mult = (0.25, 0.5, 1, 2, 4)
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    E1 = np.array([[En.tsu_estimate(m["n_spins"], base["sweeps"] * s, base["n_chains"], m["n_blocks"], E_cell=En.E_CELL_DEFAULT * f)["energy_J"] for s in mult] for f in efs])
    L1 = np.array([[En.tsu_estimate(m["n_spins"], base["sweeps"] * s, base["n_chains"], m["n_blocks"], t_update=tu)["latency_s"] for s in mult] for tu in tus])
    for ax, Z, ylab, ylabels, ttl, cb in [
            (axs[0], E1, "E_cell factor (x 1.3 fJ)", efs, "energy per frame", "energy (J)"),
            (axs[1], L1, "t_update (ns)", [int(t * 1e9) for t in tus], "latency per frame", "latency (s)")]:
        im = ax.imshow(np.log10(Z), cmap="viridis", aspect="auto", origin="lower")
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                ax.text(j, i, f"{Z[i, j]:.2g}", ha="center", va="center", fontsize=7, color="w")
        ax.set_xticks(range(len(mult)))
        ax.set_xticklabels([f"{s}x" for s in mult])
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels)
        ax.set_xlabel("sweeps (multiple of baseline)")
        ax.set_ylabel(ylab)
        ax.set_title(f"{sv}: {ttl}")
        fig.colorbar(im, ax=ax, label=f"log10 {cb}")
    ax = axs[2]
    pr = [r for r in rows if r["task"] == "posterior"]
    xs = np.arange(len(pr))
    for k, (key, lab, col) in enumerate([("tsu_energy_J", "TSU (est.)", OI["blue"]), ("gpu_energy_J", "GPU (est.)", OI["orange"]),
                                         ("laptop_energy_J", "laptop CPU", OI["verm"])]):
        vals = [r.get(key) or np.nan for r in pr]
        ax.bar(xs + (k - 1) * 0.27, vals, 0.27, label=lab, color=col)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([r["variant"] for r in pr], rotation=20, fontsize=7)
    ax.set_ylabel("energy per frame, posterior (J)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    save_fig(fig, out, "energy_sensitivity.png")
    log(open(f"{out}/energy_table.md").read()[:1500])


if __name__ == "__main__":
    main()
