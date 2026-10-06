"""How fine must the z grid be for the compensated chain/tree to stay (nearly) exact under the evidence lambda?

Builds the models only, with no sampling. Reports, per K_z:
- chord-variance inflation after compensation (eff_noise_ratio);
- the fraction of chords where compensation clamps;
- spins, colours and maximum degree.
Defaults: compensation on, tau/dz = 0.75.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import build_problem, log, make_parser, md_table, save_csv, save_json, setup  # noqa: E402

SCRIPT = "kz_inflation"


def main():
    ap = make_parser(SCRIPT, __doc__)
    ap.add_argument("--Kzs", default="16,32,64,128")
    ap.add_argument("--phantoms", default="peaked,hollow")
    args = ap.parse_args()
    cfg, out = setup(args, SCRIPT)
    from tomo.ebm_chain import build_ising_chain
    from tomo.ebm_tree import build_ising_tree
    m = cfg["model"]
    kzs = [16] if args.quick else [int(k) for k in args.Kzs.split(",")]
    rows = []
    for ph in args.phantoms.split(","):
        p = build_problem(cfg, ph, int(cfg["seed"]))
        for kz in kzs:
            for name, f in [("chain", build_ising_chain), ("tree", build_ising_tree)]:
                prob, meta = f(p, int(m["K"]), None, None, float(m["tau"]), kz, tau_mode="dz", compensate=True)
                r = {"phantom": ph, "variant": name, "Kz": kz, "n_spins": prob.n,
                     "n_blocks": int(meta.get("n_blocks", len(set(meta["colors"])))),
                     "max_degree": int(prob.degrees.max()),
                     "inflation": round(float(meta["eff_noise_ratio"]), 3),
                     "clamped_frac": round(float(meta["comp_clamped_frac"]), 3)}
                rows.append(r)
                log(r)
    save_csv(rows, f"{out}/kz_inflation.csv")
    save_json(rows, f"{out}/kz_inflation.json")
    with open(f"{out}/kz_inflation.md", "w") as fh:
        fh.write("# Chord-variance inflation vs K_z (evidence lambda, compensation on, tau/dz=0.75)\n\n"
                 "inflation = mean (s^2 + sum tau^2)/sigma^2 - 1 over chords; clamped_frac = chords where "
                 "sum tau^2 >= sigma^2 so compensation cannot be exact.\n\n")
        fh.write(md_table(rows, ["phantom", "variant", "Kz", "n_spins", "n_blocks", "max_degree", "inflation",
                                 "clamped_frac"]))
        fh.write("\n")


if __name__ == "__main__":
    main()
