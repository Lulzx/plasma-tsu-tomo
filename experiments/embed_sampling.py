"""Sampling sanity check: embedded vs unembedded I-chain (K_z=16, tiny problem) and mixing slowdown.

Run: .venv/bin/python experiments/embed_sampling.py [--D 16] [--sweeps 6000]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jax  # noqa: E402
import numpy as np  # noqa: E402

from tomo.config import load_config  # noqa: E402
from tomo.ebm_chain import build_ising_chain, decode_chain  # noqa: E402
from tomo.embed import decode, embed_bounded_degree, logical_to_physical  # noqa: E402
from tomo.forward import make_problem  # noqa: E402
from tomo.metrics import split_rhat, sweeps_to_converge  # noqa: E402
from tomo.sampling import run_ising  # noqa: E402


def iat(x):
    """Integrated autocorrelation time of a (C,S) series (Sokal window), averaged over chains."""
    x = x - x.mean(1, keepdims=True)
    S = x.shape[1]
    f = np.fft.rfft(x, n=2 * S, axis=1)
    ac = np.fft.irfft(f * np.conj(f), axis=1)[:, :S].mean(0)
    ac = ac / ac[0]
    tau = 1.0
    for t in range(1, S):
        tau += 2 * ac[t]
        if t > 5 * tau:
            break
    return tau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, default=16)
    ap.add_argument("--sweeps", type=int, default=6000)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--chains", type=int, default=16)
    a = ap.parse_args()
    cfg = load_config({"grid": {"n": 12}, "data_grid": {"n": 48}})
    for cam in cfg["cameras"]:
        cam["n_chords"] = 8
    problem = make_problem(cfg, "peaked", 0)
    m = cfg["model"]
    prob, meta = build_ising_chain(problem, 8, None, None, m["tau"], 16, tau_mode=m["tau_mode"],
                                   compensate=m["chain_compensate"], comp_floor=m["chain_comp_floor"])
    print(f"logical: n={prob.n} maxdeg={prob.degrees.max()} blocks={meta['n_blocks']}")
    ns = a.sweeps // a.stride
    sched = (0, ns, a.stride)
    key = jax.random.PRNGKey(1)

    def logical_levels(bits):
        return decode_chain(bits, meta)["x"].astype(float)       # (C,S,N)

    def report(name, S, dec=None, info=None, nspin=None, t=None):
        lv = logical_levels(S if dec is None else dec)
        h = lv.shape[1] // 2
        rh = split_rhat(lv)
        tr = lv.mean(2)           # mean level over pixels (scalar summary)
        pix = lv[:, :, :: max(1, lv.shape[2] // 40)]
        out = dict(name=name, iat_pix=float(np.median([iat(pix[:, :, k]) for k in range(pix.shape[2])])) * a.stride, n=nspin, mean_lv=lv[:, h:].mean((0, 1)), rhat_med=float(np.median(rh)), rhat_max=float(np.max(rh)),
                   s2c=sweeps_to_converge(lv, a.stride), iat_sweeps=iat(tr) * a.stride, wall=t)
        if info is not None:
            out["broken_edge"] = float(np.mean(info["broken_edge_frac"]))
            out["broken_spin"] = float(np.mean(info["broken_spin_frac"]))
        return out

    res = []
    rng = np.random.default_rng(0)
    init_l = rng.random((a.chains, prob.n)) < 0.5          # overdispersed *logical* starts, shared by all runs
    r = run_ising(prob, meta["colors"], sched, a.chains, key, init=init_l, backend="jax")
    res.append(report("unembedded", r["samples"], nspin=prob.n, t=r["time"]))
    ref = res[0]["mean_lv"]
    cfgs = [("tree", jf) for jf in (1.0, 2.0, 4.0, 8.0, 16.0)] + [("chain", 4.0), ("tree", "copy"), ("tree", "cut")]
    for topo, jf in cfgs:
        if jf in ("copy", "cut"):
            pp, em = embed_bounded_degree(prob, a.D, topology=topo, jf_rule=jf)
        else:
            pp, em = embed_bounded_degree(prob, a.D, J_F=jf, topology=topo)
        e = run_ising(pp, em["colors"], sched, a.chains, key, init=logical_to_physical(init_l, em), backend="jax")
        dec, info = decode(e["samples"], em)
        res.append(report(f"{topo} J_F={em['J_F']:.1f}{'' if not isinstance(jf, str) else ' (' + jf + ' rule)'}", e["samples"], dec, info, pp.n, e["time"]))
    sd = np.sqrt(np.var(logical_levels(r["samples"])[:, r["samples"].shape[1] // 2:], axis=(0, 1)))
    print(f"{'config':58s} {'nspin':>7s} {'Rhat med/max':>14s} {'sw->R<1.05':>10s} {'IAT[sweeps]':>11s} {'IATpix':>7s} {'dmean/sd (med,max)':>20s} {'broken e/s':>12s} {'wall':>6s}")
    for x in res:
        d = np.abs(x["mean_lv"] - ref) / np.maximum(sd, 1e-3)
        print(f"{x['name']:58s} {x['n']:7d} {x['rhat_med']:7.3f}/{x['rhat_max']:6.3f} {str(x['s2c']):>10s} {x['iat_sweeps']:11.1f} {x['iat_pix']:7.1f} "
              f"{np.median(d):9.3f},{d.max():6.2f} {x.get('broken_edge', 0):7.4f}/{x.get('broken_spin', 0):.3f} {x['wall']:6.1f}")


if __name__ == "__main__":
    main()
