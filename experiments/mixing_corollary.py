"""Corollary 1 of the paper: the coordinate lower bound on the data-augmentation relaxation time,
T_DA >= max_j (lam L_jj + Pi_jj) / (lam L_jj + D_jj), against the exact value. -> results/mixing_theory/corollary.json"""
import json
import os
import sys
import time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from tomo import mixing_theory as mt
from tomo.config import load_config
from tomo.forward import make_problem

def small(n, chords=8):
    c = load_config({"grid": {"n": n}, "data_grid": {"n": 4 * n}})
    for cam in c["cameras"]: cam["n_chords"] = chords
    return make_problem(c, "peaked", 0)

def bound(pr, J, lam):
    nx = J["nx"]
    T = np.asarray(pr.T, float); sig = np.asarray(pr.sigma, float)
    L = np.asarray(pr.L, float)
    Pxx = J["P"][:nx, :nx].toarray()
    D = np.diag(T.T * (1/sig**2) @ T)
    Ldiag = np.diag(L)
    a = Pxx.diagonal(); b = lam*Ldiag + D
    Lam = a - lam*Ldiag
    r = a/b
    return dict(diag_bound=float(r.max()), argmax=int(r.argmax()), mediant=float((a.sum())/(b.sum())),
                trLam=float(Lam.sum()), trD=float(D.sum()), trLap_lam=float(lam*Ldiag.sum()),
                offdiag_Lam_max=float(np.abs(Pxx - lam*L - np.diag(Lam)).max()))
out = {}
t0=time.time()
pr = small(16)
for kind in ("dense","chain","tree"): pass
Jd = mt.build_joint(pr, "dense")
rho_d = mt.gs_rho(Jd["P"], Jd["cls"]); out["16x16_dense_T_GS"] = 1/(1-rho_d)
for kind, kw in [("chain", dict(tau=0.5, tau_mode="frac")), ("chain", dict(tau=0.99, tau_mode="frac")), ("tree", dict(tau=0.5, tau_mode="frac")), ("chain", dict())]:
    J = mt.build_joint(pr, kind, **kw)
    Pxx=J["P"][:J["nx"],:J["nx"]].toarray()
    d = bound(pr, J, J["lam"])
    d["T_DA_exact"] = 1/(1-mt.da_rate(J["P"], J["nx"]))
    out[f"16x16_{kind}_{kw}"] = d
    print(kind, kw, d, flush=True)
print(time.time()-t0)
c3 = load_config(); p3 = make_problem(c3, "peaked", 0)
th = json.load(open(os.path.join(ROOT, "results", "mixing_theory", "theory.json")))["default32"]
for kind in ("chain","tree"):
    J = mt.build_joint(p3, kind)
    d = bound(p3, J, J["lam"])
    d["T_DA_exact"] = 1/(1-th[kind]["da"])
    out[f"default32_{kind}"] = d; print(kind, d, flush=True)
print(time.time()-t0)
json.dump(out, open(os.path.join(ROOT, "results", "mixing_theory", "corollary.json"), "w"), indent=1)
