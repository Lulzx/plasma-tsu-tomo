"""Mixing theory experiments: linear-algebra predictions, validation against samplers, remedies, figures.

    python experiments/mixing_theory.py theory     # rho / DA rate sweeps (tau, f, lambda, chord length, 32x32 headline)
    python experiments/mixing_theory.py remedies   # SOR, grouping, admissibility limit
    python experiments/mixing_theory.py validate   # continuous Gibbs simulation + discrete Ising samplers (slow part)
    python experiments/mixing_theory.py figures    # docs/figures/mixing_*.png from results/mixing_theory/*.json

All sweeps use small problems (tiny 12x12, 16x16 with 8 chords per camera) so they are linear algebra only.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from _common import OI, ROOT, plt  # noqa: F401  (sets sys.path)

from tomo import mixing_theory as mt
from tomo.config import load_config
from tomo.forward import make_problem

OUT = os.path.join(ROOT, "results", "mixing_theory")
FIG = os.path.join(ROOT, "docs", "figures")


def small_problem(n, chords=8, phantom="peaked", seed=0):
    c = load_config({"grid": {"n": n}, "data_grid": {"n": 4 * n}})
    for cam in c["cameras"]:
        cam["n_chords"] = chords
    return c, make_problem(c, phantom, seed)


def save(name, obj):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name + ".json"), "w") as f:
        json.dump(obj, f, indent=1)


def load(name):
    with open(os.path.join(OUT, name + ".json")) as f:
        return json.load(f)


def metrics(pr, J, null=True):
    P = J["P"]
    rho = mt.gs_rho(P, J["cls"])
    out = dict(rho=rho, T=1 / (1 - rho), iat_slow=mt.iat_from_rho(rho), iat_sup=mt.iat_sup(P))
    if J["nz"]:
        da = mt.da_rate(P, J["nx"])
        out.update(da=da, T_da=1 / (1 - da), S_over_s2=float(np.mean(J["S"] / np.asarray(pr.sigma) ** 2)))
        if null and P.shape[0] <= 2500:
            out["null_bound"] = mt.null_space_bound(J, pr)
    return out


# ----------------------------------------------------------------------------------------
def stage_theory():
    res = {}
    c, pr = small_problem(16)
    lam_ev, lam_st = mt.build_joint(pr, "dense")["lam"], mt.stiff_lambda(pr)
    res["lam_evidence"], res["lam_stiff"] = lam_ev, lam_st
    res["n_chord_pixels"] = float((np.asarray(pr.T) > 0).sum(1).mean())
    res["dense"] = {name: metrics(pr, mt.build_joint(pr, "dense", lam=l)) for name, l in
                    [("evidence", lam_ev), ("stiff", lam_st)]}
    # A: vs tau/dz (sampler parametrisation), compensation on/off, evidence / stiff lambda
    A = []
    for lname, lam in [("evidence", lam_ev), ("stiff", lam_st)]:
        for kind in ("chain", "tree"):
            for comp in (True, False):
                for t in [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
                    J = mt.build_joint(pr, kind, tau=t, tau_mode="dz", Kz=32, lam=lam, compensate=comp)
                    A.append(dict(lam=lname, kind=kind, comp=comp, tau=t, **metrics(pr, J)))
    res["vs_tau_dz"] = A
    # B: vs admissibility fraction f = sum tau^2 / sigma^2 (exact embeddings need f < 1)
    B = []
    for lname, lam in [("evidence", lam_ev), ("stiff", lam_st)]:
        for kind in ("chain", "tree"):
            for f in [0.003, 0.01, 0.03, 0.1, 0.3, 0.6, 0.9, 0.99]:
                J = mt.build_joint(pr, kind, tau=f, tau_mode="frac", lam=lam)
                B.append(dict(lam=lname, kind=kind, f=f, **metrics(pr, J)))
    res["vs_f"] = B
    # C: vs lambda (multiples of the evidence lambda), f = 0.5
    C = [dict(kind="dense", lam_mult=m, **metrics(pr, mt.build_joint(pr, "dense", lam=m * lam_ev))) for m in
         [0.03, 0.1, 0.3, 1, 3, 10, 30, 100]]
    for kind in ("chain", "tree"):
        for m in [0.03, 0.1, 0.3, 1, 3, 10, 30, 100]:
            C.append(dict(kind=kind, lam_mult=m, **metrics(pr, mt.build_joint(pr, kind, tau=0.5, tau_mode="frac", lam=m * lam_ev))))
    res["vs_lambda"] = C
    # D: vs chord length (grid size at fixed chords per camera => n grows with the grid)
    D = []
    for n in [8, 10, 12, 16, 20, 24, 28]:
        c2, p2 = small_problem(n)
        npix = float((np.asarray(p2.T) > 0).sum(1).mean())
        for kind in ("dense", "chain", "tree"):
            J = mt.build_joint(p2, kind, tau=0.5, tau_mode="frac")
            D.append(dict(grid=n, n_pix=npix, N=int(J["nx"]), kind=kind, **metrics(p2, J, null=False)))
    res["vs_chord"] = D
    # E: headline at the repository default problem (32x32, 72 chords, evidence lambda, tau=0.75 dz, Kz=32, compensated)
    c3 = load_config()
    p3 = make_problem(c3, "peaked", 0)
    E = {}
    for kind in ("dense", "chain", "tree"):
        t0 = time.time()
        J = mt.build_joint(p3, kind)
        E[kind] = dict(n=int(J["P"].shape[0]), **metrics(p3, J, null=False), sec=time.time() - t0)
        print("default 32x32", kind, E[kind], flush=True)
    res["default32"] = E
    save("theory", res)


def stage_remedies():
    res = {}
    c, pr = small_problem(16)
    lam_ev = mt.build_joint(pr, "dense")["lam"]
    sor = []
    ws = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.7, 1.8, 1.9, 1.95, 1.98]
    for kind in ("dense", "chain", "tree"):
        J = mt.build_joint(pr, kind, tau=0.5, tau_mode="frac")
        for w in ws:
            sor.append(dict(kind=kind, omega=w, rho=mt.sor_rho(J["P"], J["cls"], w)))
    res["sor"] = sor
    G = []
    for g in [1, 2, 3, 4, 6, 8, 12, 16]:
        J = mt.build_joint(pr, "chain", tau=0.5, tau_mode="frac", group=g)
        P = J["P"]
        rho = mt.gs_rho(P, J["cls"])
        deg = np.diff(P.indptr) - 1
        G.append(dict(group=g, n_aux=int(J["nz"]), rho=rho, T=1 / (1 - rho), max_var_degree=int(deg.max()),
                      max_pixel_degree=int(deg[:J["nx"]].max()), n_classes=int(len(set(J["cls"]))),
                      da=mt.da_rate(P, J["nx"])))
    res["group"] = G
    # trace bound: sum_k d_k >= (sum_k t_k)^2 / sigma^2  (diag conditional precision a local embedding must add)
    J = mt.build_joint(pr, "chain", tau=0.99, tau_mode="frac")
    nx = J["nx"]
    T = np.asarray(pr.T, float)
    Lam = (J["P"][:nx, :nx].toarray() - J["lam"] * np.asarray(pr.L))
    dense_diag = np.diag(T.T * (1 / np.asarray(pr.sigma) ** 2) @ T)
    res["trace_bound"] = dict(chain_trace_Lambda=float(np.trace(Lam)), dense_trace=float(dense_diag.sum()),
                              bound=float(((T.sum(1)) ** 2 / np.asarray(pr.sigma) ** 2).sum()),
                              offdiag_max=float(np.abs(Lam - np.diag(np.diag(Lam))).max()))
    save("remedies", res)


def _mean_series(S, f, thin):
    """IAT in sweeps of f^T x for samples S (C, T, N) taken every ``thin`` sweeps (chains pooled)."""
    y = S @ f
    return thin * mt.iat_series(y.T)


def stage_validate(n_grid=12, sweeps=40000, thin=10, chains=4, warm=3000, cont_chains=128, cont_sweeps=40000, kinds=("dense", "chain", "tree"),
                   tag="tiny"):
    import jax
    from tomo.ebm_chain import sample_chain
    from tomo.ebm_ising import sample_ising_variant
    from tomo.ebm_tree import sample_tree
    c, pr = small_problem(n_grid)
    res = {}
    for kind in kinds:
        J = mt.build_joint(pr, kind)
        nx = J["nx"]
        F = mt.standard_functionals(pr, J)
        names = list(F)
        Fm = np.stack([F[k] for k in names], 1)
        th = mt.iat_functional(J["P"], Fm)
        met = metrics(pr, J)
        t0 = time.time()
        yc = mt.simulate_gs(J["P"], J["cls"], Fm, n_chains=cont_chains, n_sweeps=cont_sweeps, seed=1)
        sim = [mt.iat_series(yc[:, :, i], known_mean=0.0) for i in range(len(names))]
        t_sim = time.time() - t0
        print(kind, "continuous sim done", t_sim, flush=True)
        key = jax.random.PRNGKey(7)
        k = 1 if kind == "dense" else thin
        sched = dict(n_warmup=warm, n_samples=sweeps // k, steps_per_sample=k)
        t0 = time.time()
        if kind == "dense":
            r = sample_ising_variant(pr, "dense", c, key, schedule=sched, n_chains=chains * 2, do_map=False, anneal=False)
        elif kind == "chain":
            r = sample_chain(pr, c, key, tau=0.75, Kz=32, compensate=True, n_chains=chains, anneal=False, schedule=sched, init="half")
        else:
            r = sample_tree(pr, c, key, tau=0.75, Kz=32, compensate=True, n_chains=chains, anneal=False, schedule=sched, init="half")
        S = np.asarray(r["samples"])
        disc = [_mean_series(S, F[nm][:nx], k) for nm in names]
        # sd of each functional: continuous Gaussian vs discrete sampler
        Sig = np.linalg.inv(J["P"].toarray())[:nx, :nx]
        sd_th = [float(np.sqrt(F[nm][:nx] @ Sig @ F[nm][:nx])) for nm in names]
        sd_dis = [float((S @ F[nm][:nx]).std()) for nm in names]
        res[kind] = dict(names=names, theory_iat=[float(v) for v in th], sim_iat=sim, discrete_iat=disc, thin=k,
                         total_sweeps=sweeps, chains=int(S.shape[0]), sd_theory=sd_th, sd_discrete=sd_dis,
                         rhat_med=float(r.get("rhat_med", np.nan)) if kind != "dense" else None,
                         sample_time=float(time.time() - t0), **{kk: met[kk] for kk in ("rho", "T", "iat_sup")})
        print(kind, res[kind], flush=True)
        save("validate_" + tag, res)


def stage_rigidity(taus=(0.25, 0.5, 0.75, 1.5), sweeps=40000, thin=10):
    """Discrete chain sampler vs continuous prediction as tau/dz varies (tau < dz => rigid lattice)."""
    import jax
    from tomo.ebm_chain import sample_chain
    c, pr = small_problem(12)
    out = {}
    for t in taus:
        J = mt.build_joint(pr, "chain", tau=t, Kz=32)
        F = mt.standard_functionals(pr, J)
        names = ["total", "worst"]
        th = mt.iat_functional(J["P"], np.stack([F[k] for k in names], 1))
        r = sample_chain(pr, c, jax.random.PRNGKey(3), tau=t, Kz=32, compensate=True, n_chains=4, anneal=False,
                         schedule=dict(n_warmup=3000, n_samples=sweeps // thin, steps_per_sample=thin), init="half")
        S = np.asarray(r["samples"])
        out[str(t)] = dict(names=names, theory=[float(v) for v in th], discrete=[_mean_series(S, F[k][:J["nx"]], thin) for k in names],
                           frozen_frac=float(r["frozen_frac"]), rhat_med=float(r["rhat_med"]), tau_over_dz=float(r["tau_over_dz"]),
                           eff_noise_ratio=float(r["eff_noise_ratio"]), run_sweeps=sweeps)
        print(t, out[str(t)], flush=True)
        save("rigidity", out)


# ----------------------------------------------------------------------------------------
def stage_figures():
    os.makedirs(FIG, exist_ok=True)
    th, rem = load("theory"), load("remedies")
    col = {"chain": OI["blue"], "tree": OI["verm"], "dense": OI["black"]}
    dense_T = th["dense"]["evidence"]["T"]
    dense_Ts = th["dense"]["stiff"]["T"]

    # fig 1: relaxation time vs tau/dz and vs f
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for kind in ("chain", "tree"):
        for lname, ls in (("evidence", "-"), ("stiff", "--")):
            rows = [r for r in th["vs_tau_dz"] if r["kind"] == kind and r["lam"] == lname and r["comp"]]
            ax[0].plot([r["tau"] for r in rows], [r["T"] for r in rows], ls, color=col[kind], marker="o", ms=3,
                       label=f"{kind}, {lname} $\\lambda$")
    rows = [r for r in th["vs_tau_dz"] if r["kind"] == "chain" and r["lam"] == "evidence" and r["comp"]]
    for r in rows:
        if r["tau"] in (0.5, 1.0, 2.0):
            ax[0].annotate(f"S/$\\sigma^2$={r['S_over_s2']:.1f}", (r["tau"], r["T"]), fontsize=7, xytext=(3, 4), textcoords="offset points")
    ax[0].axhline(dense_T, color=col["dense"], lw=1, label="dense, evidence $\\lambda$")
    ax[0].axhline(dense_Ts, color=col["dense"], lw=1, ls="--", label="dense, stiff $\\lambda$")
    ax[0].set(xscale="log", yscale="log", xlabel=r"$\tau/\Delta z$ (sampler parametrisation, K$_z$=32)",
              ylabel=r"relaxation time $1/(1-\rho)$ [sweeps]", title="Gibbs relaxation time vs link width")
    ax[0].legend(fontsize=7)
    for kind in ("chain", "tree"):
        for lname, ls in (("evidence", "-"), ("stiff", "--")):
            rows = [r for r in th["vs_f"] if r["kind"] == kind and r["lam"] == lname]
            ax[1].plot([r["f"] for r in rows], [r["T"] for r in rows], ls, color=col[kind], marker="o", ms=3, label=f"{kind} GS, {lname}")
            ax[1].plot([r["f"] for r in rows], [r["T_da"] for r in rows], ls, color=col[kind], marker="s", ms=3, alpha=0.4,
                       label=f"{kind} two-block DA, {lname}" if lname == "evidence" else None)
    ax[1].axhline(dense_T, color=col["dense"], lw=1)
    ax[1].set(xscale="log", yscale="log", xlabel=r"admissibility fraction $f=\sum_k\tau_k^2/\sigma^2$ ($f<1$: exact)",
              ylabel=r"relaxation time [sweeps]", title="Exact embeddings: the f<1 wall")
    ax[1].axvline(1.0, color="grey", lw=0.8, ls=":")
    ax[1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "mixing_rho_vs_tau.png"))
    plt.close(fig)

    # fig 2: vs lambda
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for kind in ("dense", "chain", "tree"):
        rows = [r for r in th["vs_lambda"] if r["kind"] == kind]
        ax[0].plot([r["lam_mult"] for r in rows], [r["T"] for r in rows], color=col[kind], marker="o", ms=3, label=kind)
        if kind != "dense":
            ax[1].plot([r["lam_mult"] for r in rows], [r["T_da"] for r in rows], color=col[kind], marker="o", ms=3, label=f"{kind} DA")
            ax[1].plot([r["lam_mult"] for r in rows], [r["null_bound"] for r in rows], color=col[kind], ls=":", marker="x", ms=4,
                       label=f"{kind} null-space bound")
    for a in ax:
        a.axvline(th["lam_stiff"] / th["lam_evidence"], color="grey", ls=":", lw=1)
        a.set(xscale="log", yscale="log", xlabel=r"$\lambda/\lambda_\mathrm{evidence}$ (dotted: discrepancy $\lambda$)")
        a.legend(fontsize=7)
    ax[0].set(ylabel="relaxation time [sweeps]", title=r"Block-Gibbs vs prior strength ($f=0.5$)")
    ax[1].set(ylabel="two-block slowdown $1/(1-\\rho_{DA})$", title="Data-augmentation factor and its null-space bound")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "mixing_rho_vs_lambda.png"))
    plt.close(fig)

    # fig 3: vs chord length
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    fits = {}
    for kind in ("dense", "chain", "tree"):
        rows = [r for r in th["vs_chord"] if r["kind"] == kind]
        n = np.array([r["n_pix"] for r in rows])
        Tt = np.array([r["T"] for r in rows])
        ax[0].plot(n, Tt, color=col[kind], marker="o", ms=3, label=kind)
        fits[kind] = float(np.polyfit(np.log(n), np.log(Tt), 1)[0])
        if kind != "dense":
            ax[1].plot(n, Tt / np.array([r["T"] for r in th["vs_chord"] if r["kind"] == "dense"]), color=col[kind], marker="o", ms=3,
                       label=f"{kind} / dense")
    nn = np.array([r["n_pix"] for r in th["vs_chord"] if r["kind"] == "chain"])
    T0 = [r["T"] for r in th["vs_chord"] if r["kind"] == "chain"][0]
    ax[0].plot(nn, T0 * (nn / nn[0]) ** 2, color="grey", ls=":", label=r"$\propto n^2$")
    ax[0].set(xscale="log", yscale="log", xlabel="mean pixels per chord $n$", ylabel="relaxation time [sweeps]",
              title="Scaling with chord length ($f=0.5$)")
    ax[1].set(xscale="log", yscale="log", xlabel="mean pixels per chord $n$", ylabel="slowdown relative to dense Gibbs")
    ax[0].legend(fontsize=7)
    ax[1].legend(fontsize=7)
    th["fit_exponents"] = fits
    save("theory", th)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "mixing_rho_vs_chord_length.png"))
    plt.close(fig)

    # fig 4: predicted vs measured
    vp = [f for f in ("validate_tiny",) if os.path.exists(os.path.join(OUT, f + ".json"))]
    if vp:
        v = load(vp[0])
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
        mk = {"centre": "o", "total": "s", "pc1": "^", "worst": "D"}
        for kind, r in v.items():
            for i, nm in enumerate(r["names"]):
                ax[0].scatter(r["theory_iat"][i], r["sim_iat"][i], color=col[kind], marker=mk[nm], s=28,
                              label=f"{kind}" if i == 0 else None)
                ax[1].scatter(r["theory_iat"][i], r["discrete_iat"][i], color=col[kind], marker=mk[nm], s=28)
        lim = [2, 1e4]
        for a in ax:
            a.plot(lim, lim, color="grey", lw=0.8)
            a.set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="theory: continuous Gaussian IAT [sweeps]")
        ax[0].set(ylabel="simulated continuous Gibbs IAT", title="Linear theory vs exact simulation")
        ax[1].set(ylabel="measured discrete Ising sampler IAT", title="Continuous theory vs discrete Ising sampler")
        for nm, m in mk.items():
            ax[1].scatter([], [], color="k", marker=m, s=20, label=nm)
        ax[0].legend(fontsize=7)
        ax[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "mixing_predicted_vs_measured.png"))
        plt.close(fig)

    if os.path.exists(os.path.join(OUT, "rigidity.json")):
        rg = load("rigidity")
        taus = sorted(rg, key=float)
        fig, ax = plt.subplots(figsize=(5, 3.8))
        for i, nm in enumerate(["total", "worst"]):
            ax.plot([float(t) for t in taus], [rg[t]["theory"][i] for t in taus], "-o", ms=4, color=OI["blue"] if i == 0 else OI["sky"],
                    label=f"continuous theory ({nm})")
            ax.plot([float(t) for t in taus], [rg[t]["discrete"][i] for t in taus], "--s", ms=4, color=OI["verm"] if i == 0 else OI["orange"],
                    label=f"discrete I-chain ({nm})")
        ax.axhline(rg[taus[0]]["run_sweeps"], color="grey", ls=":", lw=1)
        ax.text(0.27, rg[taus[0]]["run_sweeps"] * 1.15, "run length (IAT beyond is a lower bound)", fontsize=7, color="grey")
        ax.set(xscale="log", yscale="log", xlabel=r"$\tau/\Delta z$", ylabel="IAT [sweeps]", title="Lattice rigidity: tiny problem, K$_z$=32")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "mixing_rigidity.png"))
        plt.close(fig)

    # fig 5: remedies
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for kind in ("dense", "chain", "tree"):
        rows = [r for r in rem["sor"] if r["kind"] == kind]
        ax[0].plot([r["omega"] for r in rows], [1 / (1 - r["rho"]) for r in rows], color=col[kind], marker="o", ms=3, label=kind)
    ax[0].set(yscale="log", xlabel=r"over-relaxation $\omega$", ylabel="relaxation time [sweeps]", title="Over-relaxed Gibbs (target-preserving)")
    ax[0].legend(fontsize=7)
    g = rem["group"]
    ax[1].plot([r["max_pixel_degree"] for r in g], [r["T"] for r in g], color=col["chain"], marker="o")
    for r in g:
        ax[1].annotate(f"g={r['group']}", (r["max_pixel_degree"], r["T"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax[1].axhline(dense_T, color="k", lw=1)
    ax[1].set(xscale="log", yscale="log", xlabel="max variable degree (pixel)", ylabel="relaxation time [sweeps]",
              title="Grouping g pixels per auxiliary")
    gg = np.array([r["group"] for r in g])
    ax[2].plot(gg, [r["da"] for r in g], color=col["chain"], marker="o")
    ax[2].set(xscale="log", xlabel="pixels per auxiliary g", ylabel=r"$\rho_{DA}$", title="Two-block rate")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "mixing_remedies.png"))
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["theory", "remedies", "validate", "validate16", "rigidity", "figures"])
    a = ap.parse_args()
    if a.stage == "theory":
        stage_theory()
    elif a.stage == "remedies":
        stage_remedies()
    elif a.stage == "validate":
        stage_validate()
    elif a.stage == "rigidity":
        stage_rigidity()
    elif a.stage == "validate16":
        stage_validate(n_grid=16, sweeps=30000, thin=10, chains=4, warm=3000, cont_chains=96, cont_sweeps=60000, kinds=("dense", "tree"), tag="g16")
    else:
        stage_figures()
