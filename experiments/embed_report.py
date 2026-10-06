"""Degree-16 embedding table for the real models (tomo.embed) + binary-encoding comparison.

Run: .venv/bin/python experiments/embed_report.py [--D 16]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from tomo import ebm_chain, ebm_ising  # noqa: E402
from tomo.config import load_config  # noqa: E402
from tomo.embed import (bipartite_relay_estimate, binary_bit_qubo, coupling_quantisation,  # noqa: E402
                        embed_bounded_degree)
from tomo.forward import make_problem  # noqa: E402
from tomo.sampling import bits_to_spins_qubo, greedy_coloring  # noqa: E402


def models(problem, cfg):
    K, m = cfg["model"]["K"], cfg["model"]
    out = {}
    for Kz in (16, 32):
        out[f"I-chain Kz={Kz}"] = ebm_chain.build_ising_chain(
            problem, K, None, None, m["tau"], Kz, tau_mode=m["tau_mode"],
            compensate=m["chain_compensate"], comp_floor=m["chain_comp_floor"])
    out["I-dense"] = ebm_ising.build_ising_dense(problem, K)
    out["I-sparse"] = ebm_ising.build_ising_sparse(problem, K)
    try:
        from tomo import ebm_tree
        out["I-tree"] = ebm_tree.build_ising_tree(problem, K)
    except Exception as e:  # noqa: BLE001
        print("ebm_tree unavailable:", repr(e)[:100])
    return out


def row(name, prob, D, topology):
    t0 = time.time()
    pp, m = embed_bounded_degree(prob, D, topology=topology)
    q, q0 = coupling_quantisation(pp), coupling_quantisation(prob)
    relay, _ = bipartite_relay_estimate(pp)
    return dict(model=name, topo=topology, n_log=prob.n, deg_log=int(prob.degrees.max()), n_phys=pp.n,
                overhead=round(pp.n / prob.n, 2), n_emb=m["n_embedded"], max_chain=m["max_chain_len"],
                max_path=m["max_path_len"], colours=m["n_colors"], J_F=m["J_F"],
                dr_log=round(q0['bits_max_over_p1'], 1), dr_phys=round(q['bits_max_over_p1'], 1),
                zero8=round(q['zero_frac'][8], 2), relays_2col=relay, sec=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, default=16)
    a = ap.parse_args()
    cfg = load_config()
    problem = make_problem(cfg, "peaked", 0)
    rows = []
    for name, (prob, meta) in models(problem, cfg).items():
        for topo in ("tree", "chain"):
            rows.append(row(name, prob, a.D, topo))
            print(rows[-1], flush=True)
    keys = list(rows[0])
    print("\n| " + " | ".join(keys) + " |\n|" + "---|" * len(keys))
    for r in rows:
        print("| " + " | ".join(f"{r[k]:.3g}" if isinstance(r[k], float) else str(r[k]) for k in keys) + " |")

    # ---- binary encoding alternative ----
    print("\nBinary (power-of-two) encoding of levels:")
    K = cfg["model"]["K"]
    setup = ebm_ising.ec.prepare(problem, K) if hasattr(ebm_ising, "ec") else None
    Qy, c, const = setup.Q, setup.c, setup.const          # pixel quadratic in level units: see ebm_common
    nb = int(np.log2(K))
    Qb, cb, cc, rowvar, off = binary_bit_qubo(Qy, c, const, nb)
    pb = bits_to_spins_qubo(Qb, cb, cc)
    dense_th, _ = models_dense(problem, K)
    for nm, p in (("I-dense thermometer", dense_th), (f"I-dense binary ({nb} bits)", pb)):
        q = coupling_quantisation(p)
        pp, m = embed_bounded_degree(p, a.D)
        print(f"{nm}: spins={p.n} edges={p.n_edges} maxdeg={p.degrees.max()} dynrange(max/p1)={q['bits_max_over_p1']:.1f} bits zero@8b={q['zero_frac'][8]:.2f}"
              f" | D={a.D}: phys={pp.n} overhead={pp.n/p.n:.1f} maxchain={m['max_chain_len']} colours={m['n_colors']}")
    # chain model in binary: reuse meta Qy/cy
    for Kz in (16, 32):
        prob, meta = ebm_chain.build_ising_chain(problem, K, None, None, cfg["model"]["tau"], Kz, tau_mode=cfg["model"]["tau_mode"],
                                                 compensate=cfg["model"]["chain_compensate"], comp_floor=cfg["model"]["chain_comp_floor"])
        N, L = meta["N"], meta["L"]
        nbits = np.concatenate([np.full(N, int(np.log2(K))), np.full(L, int(np.log2(Kz)))])
        Qb, cb, cc, rowvar, off = binary_bit_qubo(meta["Qy"], meta["cy"], meta["const_y"], nbits)
        pb = bits_to_spins_qubo(Qb, cb, cc)
        b0 = coupling_quantisation(prob)['bits_max_over_p1']
        b1 = coupling_quantisation(pb)['bits_max_over_p1']
        pp, m = embed_bounded_degree(pb, a.D)
        print(f"I-chain Kz={Kz} thermometer: spins={prob.n} maxdeg={prob.degrees.max()} dyn={b0:.1f} bits | binary: spins={pb.n} "
              f"edges={pb.n_edges} maxdeg={pb.degrees.max()} dyn={b1:.1f} bits | D={a.D}: phys={pp.n} overhead={pp.n/pb.n:.1f} "
              f"maxchain={m['max_chain_len']} colours={m['n_colors']}  (note: no DW penalty; stiff z-chain not freezing-tested)")


def models_dense(problem, K):
    return ebm_ising.build_ising_dense(problem, K)


if __name__ == "__main__":
    main()
