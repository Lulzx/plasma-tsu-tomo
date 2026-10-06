"""Quick TCV sanity check: Tikhonov (evidence) and GP on a peaked and a random phantom; saves a figure."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tomo import baselines, metrics
from tomo.tcv import coverage_stats, make_tcv_problem, tcv_vessel_polygon

out = Path(__file__).resolve().parent.parent / "results" / "tcv_check"
out.mkdir(parents=True, exist_ok=True)
cfg = str(Path(__file__).resolve().parent.parent / "configs" / "tcv.yaml")
res, rows = {}, []
for ph in ["peaked", "random", "divertor"]:
    p = make_tcv_problem(cfg, ph, 1000)
    if ph == "peaked":
        res["coverage"] = coverage_stats(p.T)
    tk = baselines.tikhonov(p.T, p.b, p.sigma, p.L, method="evidence")
    gp = baselines.gp_tomography(p.T, p.b, p.sigma, p.grid, p.mask)
    res[ph] = {"tikhonov_evidence": float(metrics.rel_l2(tk["mean"], p.eps_true)),
               "gp": float(metrics.rel_l2(gp["mean"], p.eps_true))}
    rows.append((ph, p, tk, gp))
    print(ph, res[ph], flush=True)
json.dump(res, open(out / "sanity.json", "w"), indent=1)

fig, ax = plt.subplots(len(rows), 4, figsize=(11, 4.2 * len(rows)))
poly = np.vstack([tcv_vessel_polygon(), tcv_vessel_polygon()[:1]])
g = rows[0][1].grid
ext = [g.xmin, g.xmax, g.ymin, g.ymax]
for i, (ph, p, tk, gp) in enumerate(rows):
    a = ax[i, 0]
    for c in range(len(p.chords.p0)):
        a.plot(*np.stack([p.chords.p0[c], p.chords.p1[c]]).T, lw=0.3, color=f"C{p.chords.camera[c]}")
    a.plot(poly[:, 0], poly[:, 1], "k", lw=1)
    a.set_xlim(0.45, 1.35); a.set_ylim(-0.95, 0.95); a.set_aspect("equal"); a.set_title(f"{ph}: 120 LoS + vessel")
    vmax = p.eps_true.max()
    for a, v, t in ((ax[i, 1], p.eps_true, "truth"),
                    (ax[i, 2], tk["mean"], f"Tikhonov rel-L2={res[ph]['tikhonov_evidence']:.2f}"),
                    (ax[i, 3], gp["mean"], f"GP rel-L2={res[ph]['gp']:.2f}")):
        im = a.imshow(np.ma.masked_where(~p.mask, metrics.to_img(v, p.mask)), origin="lower", extent=ext,
                      vmin=0, vmax=vmax, cmap="inferno")
        a.set_title(t)
        a.plot(poly[:, 0], poly[:, 1], "k", lw=0.6)
fig.tight_layout()
fig.savefig(out / "tcv_check.png", dpi=110)
