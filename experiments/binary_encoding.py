"""Binary vs thermometer (I-dense) encoding: sampling quality, hardware cost, embedded check.

Run: .venv/bin/python experiments/binary_encoding.py [--quick]   -> results/binary_encoding/{results.json,results.md}
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax  # noqa: E402
import numpy as np  # noqa: E402

from _common import ROOT, jsonable, log, make_parser, save_json, setup  # noqa: E402
from tomo import ebm_common as ec  # noqa: E402
from tomo.config import load_config  # noqa: E402
from tomo.ebm_binary import binary_decode, binary_encode, build_ising_binary, sample_binary  # noqa: E402
from tomo.ebm_ising import build_ising_dense, sample_ising_variant  # noqa: E402
from tomo.embed import (coupling_dynamic_range, coupling_quantisation, decode, embed_bounded_degree,  # noqa: E402
                        logical_to_physical)
from tomo.forward import make_problem  # noqa: E402
from tomo.metrics import coverage, rel_l2, split_rhat  # noqa: E402
from tomo.sampling import run_ising  # noqa: E402


def iat(x):
    """Integrated autocorrelation time (in samples) of (C,S) series, Sokal window, chain-averaged ACF."""
    x = x - x.mean(1, keepdims=True)
    S = x.shape[1]
    f = np.fft.rfft(x, n=2 * S, axis=1)
    ac = np.fft.irfft(f * np.conj(f), axis=1)[:, :S].mean(0)
    if ac[0] <= 0:
        return float("nan")
    ac = ac / ac[0]
    tau = 1.0
    for t in range(1, S):
        tau += 2 * ac[t]
        if t > 5 * tau:
            break
    return float(tau)


def hw_stats(prob):
    r, bits, lo, hi = coupling_dynamic_range(prob)
    q = coupling_quantisation(prob)
    return dict(dyn_range_bits=bits, bits_max_over_p1=q["bits_max_over_p1"], bits_max_over_p50=q["bits_max_over_p50"],
                zero_frac_8bit=q["zero_frac"][8], zero_frac_6bit=q["zero_frac"][6], zero_frac_4bit=q["zero_frac"][4])


def quality(problem, res, stride):
    s = res["samples"]                                   # (C,S,N) emissivity
    T = np.asarray(problem.T)
    truth = problem.eps_true
    rh = np.asarray(res["rhat"])
    rh_f = rh[np.isfinite(rh)]
    pred = s @ T.T                                       # (C,S,M)
    iat_tot = iat(s.sum(2)) * stride
    iat_chord = np.array([iat(pred[:, :, m]) for m in range(pred.shape[2])]) * stride
    lev = res["levels"]
    iat_pix = np.array([iat(lev[:, :, k]) for k in range(0, lev.shape[2], max(1, lev.shape[2] // 60))]) * stride
    return dict(rel_l2=rel_l2(res["mean"], truth), coverage95=coverage(s, truth, 0.95),
                rhat_max=float(rh.max()) if not np.isnan(rh).all() else None, rhat_median=float(np.median(rh_f)),
                iat_total_power_sweeps=iat_tot, iat_chord_median_sweeps=float(np.nanmedian(iat_chord)),
                iat_chord_max_sweeps=float(np.nanmax(iat_chord)), iat_pixel_median_sweeps=float(np.nanmedian(iat_pix)),
                n_spins=res["n_spins"], n_blocks=res["n_blocks"], max_degree=res["max_degree"], wall_s=res["time"],
                invalid_frac=res["invalid_frac"])


def embedded_check(cfg, quick):
    """tiny problem (12x12, 8 chords/camera), K=8, degree-16 tree embedding of binary and thermometer models."""
    tcfg = load_config({**cfg, "grid": {"n": 12}, "data_grid": {"n": 48}})
    for cam in tcfg["cameras"]:
        cam["n_chords"] = 8
    problem = make_problem(tcfg, "peaked", 0)
    K, C = 8, 16
    nsw = 600 if quick else 6000
    stride = 5
    sched = (0, nsw // stride, stride)
    out = []
    rng = np.random.default_rng(0)
    models = {}
    pb, mb = build_ising_binary(problem, K)
    models["binary"] = (pb, mb["colors"], mb["setup"], lambda u: binary_decode(u.reshape(u.shape[:-1] + (-1, 3))))
    pt, mt = build_ising_dense(problem, K)
    models["thermometer"] = (pt, mt["colors"], mt["setup"], lambda u: ec.dw_decode(u.reshape(u.shape[:-1] + (-1, K - 1)).astype(np.int8), K)[0])
    key = jax.random.PRNGKey(1)
    for name, (prob, colors, s, dec) in models.items():
        init = rng.random((C, prob.n)) < 0.5
        r = run_ising(prob, colors, sched, C, key, init=init, backend="jax")
        ref_lv = dec(r["samples"]).astype(float)
        ref_mean, ref_sd = ref_lv[:, ref_lv.shape[1] // 2:].mean((0, 1)), ref_lv[:, ref_lv.shape[1] // 2:].std((0, 1))
        out.append(dict(model=name, embed="none", n_spins=prob.n, max_degree=prob.degrees.max(), dmean_over_sd_med=0.0,
                        broken_edge=0.0, rhat_med=float(np.median(split_rhat(ref_lv))), wall_s=r["time"]))
        for jf in (None, 1.0, 4.0):
            pp, em = embed_bounded_degree(prob, 16, J_F=jf, topology="tree")
            e = run_ising(pp, em["colors"], sched, C, key, init=logical_to_physical(init, em), backend="jax")
            logical, info = decode(e["samples"], em)
            lv = dec(logical).astype(float)
            m = lv[:, lv.shape[1] // 2:].mean((0, 1))
            d = np.abs(m - ref_mean) / np.maximum(ref_sd, 1e-3)
            # frozen = fraction of pixels whose level never changed in any chain over the run
            frozen = float(np.mean(lv.std(1).max(0) == 0))
            out.append(dict(model=name, embed=f"tree D=16 J_F={em['J_F']:.3g}" + (" (auto)" if jf is None else ""),
                            n_spins=pp.n, overhead=em["overhead"], max_degree=int(pp.degrees.max()),
                            dyn_range_bits=hw_stats(pp)["dyn_range_bits"], dmean_over_sd_med=float(np.median(d)),
                            dmean_over_sd_max=float(d.max()), broken_edge=float(np.mean(info["broken_edge_frac"])),
                            frozen_pixel_frac=frozen, rhat_med=float(np.median(split_rhat(lv))), wall_s=e["time"]))
            log("  embedded", out[-1])
    return out


def main():
    ap = make_parser("binary_encoding", "binary vs thermometer encoding")
    args = ap.parse_args()
    cfg, outdir = setup(args, "binary_encoding")
    outdir = args.out or os.path.join(ROOT, "results", "binary_encoding")
    os.makedirs(outdir, exist_ok=True)
    problem = make_problem(cfg, "peaked", 0)
    sc = cfg["schedule"]["posterior"]
    stride = int(sc["steps_per_sample"])
    K = int(cfg["model"]["K"])
    key = jax.random.PRNGKey(0)
    rows = {}
    res_t = sample_ising_variant(problem, "dense", cfg, key, do_map=False, overdispersed=True)
    pt, mt = build_ising_dense(problem, K, setup=None)
    rows["thermometer"] = {**quality(problem, res_t, stride), **hw_stats(pt)}
    log("thermometer", rows["thermometer"], f"load={os.getloadavg()[0]:.1f}")
    res_b = sample_binary(problem, cfg, key)
    rows["binary"] = {**quality(problem, res_b, stride), **hw_stats(build_ising_binary(problem, K, setup=res_b["setup"])[0])}
    log("binary", rows["binary"], f"load={os.getloadavg()[0]:.1f}")
    # posterior agreement
    agree = dict(mean_rel_l2_binary_vs_thermometer=rel_l2(res_b["mean"], res_t["mean"]),
                 std_ratio_median=float(np.median(res_b["std"] / np.maximum(res_t["std"], 1e-12))))
    emb = embedded_check(cfg, args.quick)
    save_json(dict(schedule=sc, K=K, main=rows, agreement=agree, embedded=emb), os.path.join(outdir, "results.json"))
    keys = [("rel_l2", "rel-L2", "{:.3f}"), ("coverage95", "coverage95", "{:.3f}"), ("rhat_max", "R-hat max", "{:.3f}"),
            ("rhat_median", "R-hat median", "{:.3f}"), ("iat_total_power_sweeps", "IAT total power [sweeps]", "{:.1f}"),
            ("iat_chord_median_sweeps", "IAT chord data, median [sweeps]", "{:.1f}"),
            ("iat_chord_max_sweeps", "IAT chord data, max [sweeps]", "{:.1f}"),
            ("iat_pixel_median_sweeps", "IAT pixel level, median [sweeps]", "{:.1f}"),
            ("n_spins", "n_spins", "{:d}"), ("n_blocks", "n_blocks", "{:d}"), ("max_degree", "max_degree", "{:d}"),
            ("wall_s", "wall time [s]", "{:.1f}"), ("dyn_range_bits", "coupling dyn. range [bits] (max/min)", "{:.1f}"),
            ("bits_max_over_p1", "bits max/p1", "{:.1f}"), ("zero_frac_8bit", "frac couplings -> 0 at 8 bit", "{:.3f}"),
            ("zero_frac_6bit", "frac -> 0 at 6 bit", "{:.3f}")]
    md = [f"# Binary vs thermometer encoding (default problem, peaked seed 0, K={K}, schedule {sc}, 16 chains, overdispersed)\n",
          "| metric | thermometer (I-dense) | binary |", "|---|---|---|"]
    for k, lab, f in keys:
        v = [rows[m][k] for m in ("thermometer", "binary")]
        md.append(f"| {lab} | " + " | ".join("n/a" if x is None else f.format(int(x) if f == "{:d}" else x) for x in v) + " |")
    md += ["", f"Posterior agreement: {agree}", "", "## Degree-16 embedded models (tiny problem 12x12, K=8, tree embedding)\n",
           "| model | embedding | n_spins | max_deg | dyn range [bits] | median |dmean|/sd | broken bonds | frozen pixels | R-hat med | wall [s] |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in emb:
        md.append(f"| {e['model']} | {e['embed']} | {e['n_spins']} | {e['max_degree']} | {e.get('dyn_range_bits', float('nan')):.1f} | "
                  f"{e['dmean_over_sd_med']:.2f} | {e['broken_edge']:.3f} | {e.get('frozen_pixel_frac', 0):.3f} | {e['rhat_med']:.3f} | {e['wall_s']:.1f} |")
    open(os.path.join(outdir, "results.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
