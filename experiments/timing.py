"""Clean wall-clock timings of the converging EBMs, with the machine load and (optionally) measured CPU power.

Runs Potts (balanced and greedy colour blocks) and I-sparse on the synthetic and TCV geometries with the full
posterior schedule (no MAP anneal), --repeats times each, recording the 1-minute load average before and after,
epoch start/end times, wall time, compile time, R-hat and rel-L2.

CPU power needs root, so this script never measures it. In another terminal run
    sudo powermetrics --samplers cpu_power -i 1000 > pm_run.log
for the whole run, and (once, with nothing running) a short idle log
    sudo powermetrics --samplers cpu_power -i 1000 -n 30 > pm_idle.log
then attach the energy per run with
    python experiments/timing.py --attach-power pm_run.log --idle-log pm_idle.log
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import build_problem, ebm_estimates, log, make_parser, md_table, metric_row, run_ebm, save_json, setup  # noqa: E402

SCRIPT = "timing"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLS = ["geometry", "variant", "repeat", "n_pixels", "n_blocks", "load_before", "load_after", "wall_s", "sample_s",
        "compile_s", "rhat_max", "rel_l2", "coverage95"]


def attach_power(out, run_log, idle_log):
    from tomo.energy import cpu_energy_in_window, powermetrics_samples
    rows = json.load(open(f"{out}/timing.json"))["rows"]
    idle = None
    if idle_log:
        smp = powermetrics_samples(idle_log)
        idle = float(np.mean([w for _, _, w in smp]))
    for r in rows:
        try:
            e = cpu_energy_in_window(run_log, r["t_start"], r["t_end"], idle_W=idle)
        except ValueError:
            continue
        r.update(cpu_W=round(e["avg_power_W"], 2), cpu_J=round(e["energy_J"], 1),
                 pm_covered=round(e["covered_s"] / e["seconds"], 2))
        if idle is not None:
            r.update(idle_W=round(idle, 2), net_J=round(e["net_energy_J"], 1))
    write(out, rows, idle)


def write(out, rows, idle=None):
    save_json({"rows": rows}, f"{out}/timing.json")
    cols = COLS + [c for c in ("pm_covered", "cpu_W", "cpu_J", "idle_W", "net_J") if any(c in r for r in rows)]
    with open(f"{out}/timing.md", "w") as f:
        f.write("# Wall-clock timings (posterior only, no MAP anneal)\n\n"
                "load = 1-minute load average; sample_s = sampling only (excl. compile/build); wall_s = whole call. "
                "potts = balanced colour blocks, potts_greedy = unbalanced greedy blocks (previous default).\n\n")
        if idle is not None:
            f.write(f"CPU power from powermetrics; idle baseline {idle:.2f} W subtracted for net_J.\n\n")
        f.write(md_table(rows, cols))
        f.write("\n")


def main():
    ap = make_parser(SCRIPT, __doc__)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--variants", default="potts,potts_greedy,sparse")
    ap.add_argument("--geometries", default="default,tcv")
    ap.add_argument("--phantom", default="peaked")
    ap.add_argument("--attach-power", default=None, help="powermetrics log covering the runs")
    ap.add_argument("--idle-log", default=None, help="powermetrics log taken with the machine idle")
    ap.add_argument("--wait-powermetrics", action="store_true", help="start only once a powermetrics process runs")
    args = ap.parse_args()
    if args.wait_powermetrics:
        import subprocess
        log("waiting for a powermetrics process ...")
        while subprocess.run(["pgrep", "-x", "powermetrics"], capture_output=True).returncode != 0:
            time.sleep(5)
        log("powermetrics detected; settling 10 s")
        time.sleep(10)
    if args.attach_power:
        out = args.out or os.path.join(ROOT, "results", SCRIPT)
        return attach_power(out, args.attach_power, args.idle_log)
    from tomo.config import deep_merge, load_config
    from tomo.tcv import make_tcv_problem
    cfg0, out = setup(args, SCRIPT, extra_overrides={"model": {"sparse_threshold": 0.2}})
    rows = []
    for geom in args.geometries.split(","):
        if geom == "tcv":
            cfg = deep_merge(load_config(os.path.join(ROOT, "configs", "tcv.yaml")), {"model": {"sparse_threshold": 0.2}})
            if args.quick:
                cfg = deep_merge(cfg, {"schedule": cfg0["schedule"]})
            p = make_tcv_problem(cfg, args.phantom, int(cfg["seed"]))
        else:
            cfg = cfg0
            p = build_problem(cfg, args.phantom, int(cfg["seed"]))
        for v in args.variants.split(","):
            for rep in range(args.repeats):
                kw = {"do_map": False}
                if v == "potts_greedy":
                    kw["balance"] = False
                l0, t0 = os.getloadavg()[0], time.time()
                res, info = run_ebm(p, "potts" if v.startswith("potts") else v, cfg, key_seed=rep, **kw)
                t1, l1 = time.time(), os.getloadavg()[0]
                r = {"geometry": geom, "variant": v, "repeat": rep, "n_pixels": int(p.mask.sum()),
                     "load_before": round(l0, 2), "load_after": round(l1, 2), "t_start": t0, "t_end": t1,
                     "wall_s": round(t1 - t0, 1)}
                if res is not None:
                    d, _ = ebm_estimates(res)
                    m = metric_row(p, d)
                    r.update(n_blocks=res.get("n_blocks"), sample_s=round(float(res["time"]), 1),
                             compile_s=round(float(res.get("compile_time") or 0), 1),
                             rhat_max=round(float(info["rhat_max"]), 4), rel_l2=round(m["rel_l2"], 4),
                             coverage95=round(m.get("coverage95", float("nan")), 3))
                    del res
                rows.append(r)
                log(r)
                write(out, rows)
    log(f"-> {out}/timing.md")


if __name__ == "__main__":
    main()
