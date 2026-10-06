import itertools

import jax
import numpy as np
import pytest

from tomo.config import load_config
from tomo.ebm_chain import build_ising_chain
from tomo.ebm_common import dw_encode
from tomo.ebm_tree import (balanced_tree, build_ising_tree, decode_tree, sample_tree, tree_energy, tree_layout)
from tomo.forward import make_problem
from tomo.sampling import check_coloring

K, KZ = 4, 8


def _bits(x, w, K_, Kz_):
    return np.concatenate([dw_encode(x, K_).ravel(), dw_encode(w, Kz_).ravel()]).astype(bool)


@pytest.fixture(scope="module")
def built(tiny_problem):
    return build_ising_tree(tiny_problem, K, None, None, 0.5, KZ, tau_mode="dz")


@pytest.mark.parametrize("n", list(range(1, 46)))
def test_balanced_tree_shape(n):
    nodes = balanced_tree(n)
    assert len(nodes) == max(n - 1, 1)
    depth = max(nd["depth"] for nd in nodes)
    assert depth == (int(np.ceil(np.log2(n))) - 1 if n > 1 else 0)            # internal-node depth = ceil(log2 n) - 1
    leaves = sorted(c for nd in nodes for t, c in nd["kids"] if t == "p")
    assert leaves == list(range(n))                                            # every pixel exactly once, in order
    for q, nd in enumerate(nodes):
        assert len(nd["kids"]) == (2 if n > 1 else 1)
        for t, c in nd["kids"]:
            if t == "n":
                assert nodes[c]["parent"] == q and nodes[c]["depth"] == nd["depth"] + 1
        # contiguous leaf range
        assert nd["hi"] - nd["lo"] >= 1


def test_energy_matches_tree_formula(tiny_problem):
    """Bit-level Ising energy == tree formula evaluated node by node from T, b, sigma (independent of the builder's matrices)."""
    for kw in (dict(), dict(compensate=True), dict(window="chord"), dict(tau_mode="sigma", tau=0.3)):
        args = dict(tau=0.5, tau_mode="dz")
        args.update(kw)
        prob, meta = build_ising_tree(tiny_problem, K, None, None, args.pop("tau"), KZ, **args)
        T, b = np.asarray(tiny_problem.T, float), np.asarray(tiny_problem.b, float)
        tl, lay, D = meta["tl"], meta["lay"], meta["Delta"]
        rng = np.random.default_rng(0)
        for _ in range(3):
            x = rng.integers(0, K, meta["N"])
            w = rng.integers(0, KZ, meta["L"])
            z = meta["z0"] + meta["dz"] * w
            # formula from the explicit leaf ranges of balanced_tree, not from kids lists
            e = 0.0
            for i in range(T.shape[0]):
                vs = np.flatnonzero(tl["chord"] == i)
                if len(vs) == 0:
                    continue
                lo = lay["off"][i]
                pix = lay["link_pix"][lay["off"][i]:lay["off"][i + 1]]
                nodes = balanced_tree(len(pix))
                assert len(nodes) == len(vs)
                for q, nd in enumerate(nodes):
                    ssum = 0.0
                    for t, c in nd["kids"]:
                        ssum += z[vs[c]] if t == "n" else T[i, pix[c]] * D * x[pix[c]]
                    e += (z[vs[q]] - ssum) ** 2 / (2 * meta["tau_i"][vs[q]] ** 2)
                e += (b[i] - z[vs[0]]) ** 2 / (2 * meta["s2"][i])
            ed = tiny_problem.edges
            e += 0.5 * meta["lam"] * D ** 2 * np.sum((x[ed[:, 0]] - x[ed[:, 1]]) ** 2)
            assert prob.energy(_bits(x, w, K, KZ)) == pytest.approx(e, rel=1e-9)
            assert tree_energy(x, w, meta) == pytest.approx(e, rel=1e-9)
    # a non-thermometer pixel pattern costs +A per violation
    x = rng.integers(1, K, meta["N"])
    w = rng.integers(0, KZ, meta["L"])
    bits = _bits(x, w, K, KZ)
    bad = bits.copy()
    pix = bits[:meta["n_pix_bits"]].reshape(meta["N"], K - 1).copy()
    pix[0] = [0, 1, 0]
    bad[:meta["n_pix_bits"]] = pix.ravel()
    d = decode_tree(bad, meta)
    assert not d["valid_pix"][0] and d["valid_pix"].sum() == meta["N"] - 1
    assert prob.energy(bad) == pytest.approx(tree_energy(d["x"], w, meta) + meta["A"], rel=1e-9)


def test_aux_count_and_warm_start(built, tiny_problem):
    prob, meta = built
    T = np.asarray(tiny_problem.T)
    n_i = (T > 0).sum(1)
    assert meta["L"] == int(np.sum(np.maximum(n_i - 1, (n_i == 1))))           # n-1 per chord (1 for a single pixel)
    assert meta["n_spins"] == meta["N"] * (K - 1) + meta["L"] * (KZ - 1) == prob.n
    assert meta["depth_max"] <= int(np.ceil(np.log2(n_i.max())))
    d = decode_tree(meta["init"], meta)
    assert d["valid_pix"].all() and d["valid_z"].all() and (d["x"] == meta["setup"].x0).all()
    assert prob.energy(meta["init"]) == pytest.approx(tree_energy(d["x"], d["w"], meta), rel=1e-9)
    assert (d["w"] > 0).mean() > 0.3 and (d["w"] < KZ - 1).mean() > 0.5


def test_colouring_proper(built):
    prob, meta = built
    assert meta["colouring_used"] == "structured"
    assert check_coloring(prob.n, prob.edges, meta["colors"])
    assert len(np.unique(meta["colors"])) == meta["n_blocks"]
    # internal nodes use 3 classes, pixels a handful (greedy): blocks = (pixel classes) (K-1) + 3 (Kz-1)
    assert meta["n_blocks"] == meta["n_pix_colours"] * (K - 1) + 3 * (KZ - 1)
    assert meta["n_pix_colours"] <= 6


def test_no_dense_pixel_couplings(built):
    """Pixel variables couple only to Laplacian neighbours, sibling pixels of a tree leaf pair, and aux nodes (no T^T T)."""
    prob, meta = built
    N, rv, tl, lay = meta["N"], meta["rowvar"], meta["tl"], meta["lay"]
    e = prob.edges
    va, vb = rv[e[:, 0]], rv[e[:, 1]]
    pp = (va < N) & (vb < N) & (va != vb)
    allowed = {tuple(sorted(p)) for p in np.asarray(meta["setup"].problem.edges)}
    for v in range(tl["V"]):
        ps = [lay["link_pix"][p] for p in tl["kids_pix"][v]]
        if len(ps) == 2:
            allowed.add(tuple(sorted(ps)))
    got = {tuple(sorted((int(a), int(b)))) for a, b in zip(va[pp], vb[pp])}
    assert got <= allowed
    # aux-aux couplings only parent-child or siblings
    zz = (va >= N) & (vb >= N) & (va != vb)
    ok = set()
    for v in range(tl["V"]):
        kids = tl["kids_node"][v]
        ok |= {(min(v, c) + N, max(v, c) + N) for c in kids}
        if len(kids) == 2:
            ok.add((min(kids) + N, max(kids) + N))
    assert {tuple(sorted((int(a), int(b)))) for a, b in zip(va[zz], vb[zz])} <= ok


def test_degree_bounded_and_grid_independent():
    """Aux-bit degree is bounded by 4 variables (parent, 2 children, sibling..., pixels) and does not depend on the grid."""
    out = {}
    for n in (8, 12, 16):
        c = load_config({"grid": {"n": n}, "data_grid": {"n": 4 * n}})
        for cam in c["cameras"]:
            cam["n_chords"] = 8
        p = make_problem(c, "peaked", 0)
        prob, meta = build_ising_tree(p, K, None, None, 0.5, KZ, tau_mode="dz")
        deg, npb = meta["degrees"], meta["n_pix_bits"]
        # aux node: own bits (Kz-2) + parent (Kz-1) + 2 children (<= max(Kz,K)-1 each) + at most
        # (the sibling of v is also a neighbour: 1 more child-type variable) -> 4 neighbour variables
        assert deg[npb:].max() <= (KZ - 2) + 4 * (max(KZ, K) - 1)
        out[n] = deg[npb:].max()
        cnt = (np.asarray(p.T) > 0).sum(0)
        # pixel: own + 4 Laplacian + per chord: parent z + sibling (z or pixel)
        assert deg[:npb].max() <= (K - 2) + 4 * (K - 1) + cnt.max() * (KZ - 1 + max(KZ, K) - 1)
    assert out[8] == out[12] == out[16]


def _toy_energy_min(n, tau, s2, b, m):
    """min over continuous z of tree energy for one chord with leaf values m; returns minimum (Schur complement)."""
    nodes = balanced_tree(n)
    V = len(nodes)
    # residual r_v = z_v - sum_kids = A z - c
    A = np.zeros((V, V))
    c = np.zeros(V)
    for q, nd in enumerate(nodes):
        A[q, q] = 1.0
        for t, k in nd["kids"]:
            if t == "n":
                A[q, k] -= 1.0
            else:
                c[q] += m[k]
    root = np.zeros(V)
    root[0] = 1.0
    P = A.T @ A / tau ** 2 + np.outer(root, root) / s2
    h = A.T @ c / tau ** 2 + root * b / s2
    z = np.linalg.solve(P, h)
    r = A @ z - c
    return 0.5 * np.sum(r ** 2) / tau ** 2 + 0.5 * (b - z[0]) ** 2 / s2


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8, 13])
def test_effective_variance_continuous(n):
    """Continuous z: -log marginal(b | m) = (b - sum m)^2 / (2 (s^2 + sum_v tau_v^2)) + const; sum over n-1 internal nodes."""
    tau, s2, b = 0.07, 0.01, 0.9
    rng = np.random.default_rng(n)
    tot, e = [], []
    for _ in range(6):
        m = rng.uniform(0.0, 0.3, n)
        tot.append(m.sum())
        e.append(_toy_energy_min(n, tau, s2, b, m))
    tot, e = np.array(tot), np.array(e)
    n_int = max(n - 1, 1)
    veff = s2 + n_int * tau ** 2
    np.testing.assert_allclose(e, (b - tot) ** 2 / (2 * veff), rtol=1e-9, atol=1e-12)


def test_discrete_marginal_compensated():
    """Exact lattice sum over the discrete z of a 4-leaf tree: with s^2 = sigma^2 - sum tau^2 and tau/dz >= 0.75 the
    marginal is N(b; sum m, sigma^2) to ~1e-3 nats; uncompensated it is visibly inflated."""
    n, tau, sigma, b = 4, 0.03, 0.1, 0.9
    nodes = balanced_tree(n)
    V = len(nodes)                                           # 3

    def negmarg(m, dz, s2):
        grid = np.arange(0, int(1.6 / dz)) * dz
        zs = np.meshgrid(*([grid] * V), indexing="ij")
        E = np.zeros_like(zs[0])
        for q, nd in enumerate(nodes):
            ssum = 0.0
            for t, k in nd["kids"]:
                ssum = ssum + (zs[k] if t == "n" else m[k])
            E = E + (zs[q] - ssum) ** 2 / (2 * tau ** 2)
        E = E + (b - zs[0]) ** 2 / (2 * s2)
        return -np.log(np.sum(np.exp(-E)))

    rng = np.random.default_rng(0)
    ms = [rng.uniform(0.1, 0.25, n) for _ in range(6)]
    tot = np.array([m.sum() for m in ms])
    gauss = (b - tot) ** 2 / (2 * sigma ** 2)
    for ratio, tol in ((1.0, 2e-3), (0.75, 6e-3)):
        dz = tau / ratio
        s2c = sigma ** 2 - (n - 1) * tau ** 2
        d = np.array([negmarg(m, dz, s2c) for m in ms]) - gauss
        assert np.abs(d - d.mean()).max() < tol
        du = np.array([negmarg(m, dz, sigma ** 2) for m in ms]) - gauss
        assert np.abs(du - du.mean()).max() > 0.1 * (n - 1) * tau ** 2 / sigma ** 2 * 0.3


def test_compensation_build(tiny_problem):
    sig = np.asarray(tiny_problem.sigma)
    prob, meta = build_ising_tree(tiny_problem, K, None, None, 0.1, KZ, tau_mode="dz", compensate=True)
    ch, tau = meta["tl"]["chord"], meta["tau_i"]
    S = np.bincount(ch, weights=tau ** 2, minlength=len(sig))
    cw = meta["chords_with"]
    ok = S[cw] < 0.9 * sig[cw] ** 2
    assert ok.any()
    np.testing.assert_allclose((meta["s2"] + S)[cw][ok], sig[cw][ok] ** 2, rtol=1e-12)
    _, m2 = build_ising_tree(tiny_problem, K, None, None, 1e3, KZ, tau_mode="sigma", compensate=True)
    assert m2["comp_clamped_frac"] == 1.0 and m2["eff_noise_ratio"] > 1.0


def test_fewer_or_equal_aux_than_chain(tiny_problem):
    _, mt = build_ising_tree(tiny_problem, K, None, None, 0.5, KZ)
    _, mc = build_ising_chain(tiny_problem, K, None, None, 0.5, KZ)
    n_chords = int(np.sum(np.diff(mc["lay"]["off"]) > 0))
    assert mt["L"] == mc["L"] - n_chords + int(np.sum(np.diff(mc["lay"]["off"]) == 1))


def test_sample_tree_end_to_end(tiny_problem):
    cfg = load_config({"grid": {"n": 12}, "data_grid": {"n": 48}, "model": {"K": K, "Kz": KZ},
                       "schedule": {"n_chains": 4, "anneal": {"n_betas": 4, "sweeps_per_beta": 5}}})
    for cam in cfg["cameras"]:
        cam["n_chords"] = 8
    r = sample_tree(tiny_problem, cfg, jax.random.PRNGKey(0), backend="jax", tau=1.0, tau_mode="dz",
                    compensate=True, init="half", schedule=dict(n_warmup=40, n_samples=20, steps_per_sample=2))
    N = tiny_problem.eps_true.size
    for k in ("mean", "std", "map", "samples", "rhat", "energy_trace", "n_blocks", "n_spins", "max_degree", "time",
              "invalid_frac", "tau", "n_aux", "chain_invalid_frac", "frozen_frac", "rhat_med", "rhat_q95",
              "rhat_nonfinite_frac", "converged", "sweeps_to_converge", "eff_noise_ratio", "comp_clamped_frac",
              "invalid_ok"):
        assert k in r
    assert r["samples"].shape == (4, 20, N) and r["mean"].shape == (N,) and r["map"].shape == (N,)
    assert np.all(np.isfinite(r["mean"])) and r["mean"].min() >= 0
    assert r["invalid_frac"] < 0.01 and r["chain_invalid_frac"] < 0.05
    rel = np.linalg.norm(r["mean"] - tiny_problem.eps_true) / np.linalg.norm(tiny_problem.eps_true)
    assert rel < 1.0


def test_dispatcher_tree(tiny_problem):
    from tomo.ebm_ising import sample_ising_variant
    cfg = load_config({"grid": {"n": 12}, "data_grid": {"n": 48}, "model": {"K": K, "Kz": KZ},
                       "schedule": {"n_chains": 2, "anneal": {"n_betas": 2, "sweeps_per_beta": 2}}})
    for cam in cfg["cameras"]:
        cam["n_chords"] = 8
    r = sample_ising_variant(tiny_problem, "tree", cfg, jax.random.PRNGKey(0), backend="jax",
                             schedule=dict(n_warmup=5, n_samples=4, steps_per_sample=1))
    assert r["samples"].shape[:2] == (2, 4) and r["n_aux"] > 0
