import jax
import numpy as np
import pytest

from tomo.config import load_config
from tomo.ebm_chain import (build_ising_chain, chain_energy, chain_layout, decode_chain,
                            effective_noise_variance, sample_chain)
from tomo.ebm_common import dw_encode, prepare
from tomo.forward import make_problem
from tomo.sampling import check_coloring

K, KZ = 4, 8


@pytest.fixture(scope="module")
def built(tiny_problem):
    return build_ising_chain(tiny_problem, K, None, None, 0.5, KZ, tau_mode="dz")


def _bits(x, w, K_, Kz_):
    return np.concatenate([dw_encode(x, K_).ravel(), dw_encode(w, Kz_).ravel()]).astype(bool)


@pytest.mark.parametrize("kw", [dict(), dict(window="chord"), dict(group=2), dict(tau_mode="sigma", tau=0.3)])
def test_energy_matches_formula(tiny_problem, kw):
    args = dict(tau=0.5, tau_mode="dz")
    args.update(kw)
    prob, meta = build_ising_chain(tiny_problem, K, None, None, args.pop("tau"), KZ, **args)
    rng = np.random.default_rng(0)
    for _ in range(4):
        x = rng.integers(0, K, meta["N"])
        w = rng.integers(0, KZ, meta["L"])
        e_bits = prob.energy(_bits(x, w, K, KZ))
        assert e_bits == pytest.approx(chain_energy(x, w, meta), rel=1e-9)
    # invalid (non-thermometer) pixel pattern costs +A per violation on top of the level energy
    x = rng.integers(1, K, meta["N"])
    w = rng.integers(0, KZ, meta["L"])
    bits = _bits(x, w, K, KZ)
    base = prob.energy(bits)
    j = 0
    pix = bits[:meta["n_pix_bits"]].reshape(meta["N"], K - 1).copy()
    pix[j] = [0, 1, 0][:K - 1] if K - 1 == 3 else pix[j]
    bad = bits.copy()
    bad[:meta["n_pix_bits"]] = pix.ravel()
    d = decode_chain(bad, meta)
    assert not d["valid_pix"][j] and d["valid_pix"].sum() == meta["N"] - 1
    e_bad = prob.energy(bad)
    e_level = chain_energy(d["x"], w, meta)
    assert e_bad == pytest.approx(e_level + meta["A"], rel=1e-9)


def test_warm_start(built, tiny_problem):
    prob, meta = built
    d = decode_chain(meta["init"], meta)
    assert d["valid_pix"].all() and d["valid_z"].all()
    assert (d["x"] == meta["setup"].x0).all()
    assert prob.energy(meta["init"]) == pytest.approx(chain_energy(d["x"], d["w"], meta), rel=1e-9)
    # warm-start partial sums sit inside the z window (not clipped at an edge)
    assert (d["w"] > 0).mean() > 0.5 and (d["w"] < KZ - 1).mean() > 0.5


def test_counts_and_colouring(built):
    prob, meta = built
    assert meta["n_spins"] == meta["N"] * (K - 1) + meta["L"] * (KZ - 1) == prob.n
    assert meta["n_blocks"] == 2 * (K - 1) + 2 * (KZ - 1)
    assert check_coloring(prob.n, prob.edges, meta["colors"])
    assert len(np.unique(meta["colors"])) == meta["n_blocks"]
    # chord with a fixed warm start: aux per chord == number of crossed pixels
    lay = meta["lay"]
    assert lay["L"] == int(np.sum(np.asarray(meta["setup"].problem.T) > 0))


def test_no_dense_pixel_couplings(built):
    """Pixel bits couple only to their own bits, Laplacian neighbours and chain bits (no T^T T edges)."""
    prob, meta = built
    p = meta["setup"].problem
    rv, N = meta["rowvar"], meta["N"]
    e = prob.edges
    va, vb = rv[e[:, 0]], rv[e[:, 1]]
    pp = (va < N) & (vb < N) & (va != vb)
    pairs = {tuple(sorted(t)) for t in zip(va[pp], vb[pp])}
    lap = {tuple(sorted(t)) for t in p.edges.tolist()}
    assert pairs <= lap
    # chain-chain edges only between consecutive elements of one chord (or same element)
    cc = (va >= N) & (vb >= N) & (va != vb)
    d = np.abs(va[cc] - vb[cc])
    assert (d == 1).all()
    ch = meta["lay"]["chord"]
    assert (ch[va[cc] - N] == ch[vb[cc] - N]).all()


def test_chain_degree_grid_independent_and_pixel_degree_bounded(cfg):
    """Chain-bit degree is constant; pixel-bit degree is set by the chords through the pixel, not by the grid."""
    out = {}
    for n in (8, 12, 16):
        c = load_config({"grid": {"n": n}, "data_grid": {"n": 4 * n}})
        for cam in c["cameras"]:
            cam["n_chords"] = 6
        p = make_problem(c, "peaked", 0)
        prob, meta = build_ising_chain(p, K, None, None, 0.5, KZ, tau_mode="dz")
        deg, npb = meta["degrees"], meta["n_pix_bits"]
        cnt = (np.asarray(p.T) > 0).sum(0)                    # chords through each pixel
        nb = np.diff(meta["off"])
        # bound: own bits + 4 Laplacian neighbours + 2 aux per crossing chord
        pix_bound = (K - 2) + 4 * (K - 1) + 2 * cnt.max() * (KZ - 1)
        assert deg[:npb].max() <= pix_bound
        assert deg[npb:].max() <= (KZ - 2) + 2 * (KZ - 1) + 2 * (K - 1)
        out[n] = (deg[npb:].max(), deg[:npb].max(), cnt.max())
        assert nb.sum() == prob.n
    assert out[8][0] == out[12][0] == out[16][0]               # chain degree identical for every grid
    assert max(v[1] for v in out.values()) <= (K - 2) + 4 * (K - 1) + 2 * max(v[2] for v in out.values()) * (KZ - 1)


def test_effective_noise_toy_chord():
    """Marginalising z gives variance sigma^2 + n tau^2: continuous (exact) and discrete (fine grid) chain."""
    n, tau, sigma = 5, 0.07, 0.1
    # continuous: precision of z_1..z_n from link terms (z_0 = 0, shift m_k = T Delta x_k fixed) + data on z_n
    P = np.zeros((n, n))
    for k in range(n):
        P[k, k] += 1 / tau ** 2
        if k > 0:
            P[k - 1, k - 1] += 1 / tau ** 2
            P[k, k - 1] -= 1 / tau ** 2
            P[k - 1, k] -= 1 / tau ** 2
    cov = np.linalg.inv(P)          # prior-like chain (no data): z_n ~ N(sum m, n tau^2)
    assert cov[-1, -1] == pytest.approx(n * tau ** 2, rel=1e-9)
    assert effective_noise_variance(sigma, tau, n) == pytest.approx(sigma ** 2 + n * tau ** 2)
    # discrete: DP over fine z grid; -log Z(x) as a function of the increment total must equal
    # (b - sum m)^2 / (2 (sigma^2 + n tau^2)) + const
    dz = tau / 4
    grid = np.arange(0, 400) * dz
    rng = np.random.default_rng(0)
    b = 0.9

    def neg_log_marg(m):
        # z_0 = 0 -> alpha_1(z) = exp(-(z - m_1)^2 / 2tau^2) ; alpha_k(z) = sum_z' alpha_{k-1}(z') exp(-(z - z' - m_k)^2/2tau^2)
        a = np.exp(-(grid - m[0]) ** 2 / (2 * tau ** 2))
        for k in range(1, n):
            Kmat = np.exp(-(grid[:, None] - grid[None, :] - m[k]) ** 2 / (2 * tau ** 2))
            a = Kmat @ a
        z = np.sum(a * np.exp(-(b - grid) ** 2 / (2 * sigma ** 2)))
        return -np.log(z)

    totals, vals = [], []
    for _ in range(6):
        m = rng.uniform(0.1, 0.25, n)
        totals.append(m.sum())
        vals.append(neg_log_marg(m))
    totals, vals = np.array(totals), np.array(vals)
    gauss = (b - totals) ** 2 / (2 * (sigma ** 2 + n * tau ** 2))
    diff = (vals - gauss) - np.mean(vals - gauss)
    assert np.abs(diff).max() < 0.02
    naive = (b - totals) ** 2 / (2 * sigma ** 2)       # without inflation the shape is clearly different
    assert np.abs((vals - naive) - np.mean(vals - naive)).max() > 0.2


def test_sample_chain_end_to_end(tiny_problem, tiny_cfg):
    cfg = load_config({"grid": {"n": 12}, "data_grid": {"n": 48},
                       "model": {"K": K, "Kz": KZ},
                       "schedule": {"n_chains": 4, "anneal": {"n_betas": 4, "sweeps_per_beta": 5}}})
    for cam in cfg["cameras"]:
        cam["n_chords"] = 8
    r = sample_chain(tiny_problem, cfg, jax.random.PRNGKey(0), backend="jax", tau=1.0, tau_mode="dz",
                     schedule=dict(n_warmup=40, n_samples=20, steps_per_sample=2))
    N = tiny_problem.eps_true.size
    for k in ("mean", "std", "map", "samples", "rhat", "energy_trace", "n_blocks", "n_spins", "max_degree", "time",
              "invalid_frac", "tau", "n_aux", "chain_invalid_frac"):
        assert k in r
    assert r["samples"].shape == (4, 20, N) and r["mean"].shape == (N,) and r["map"].shape == (N,)
    assert np.all(np.isfinite(r["mean"])) and r["mean"].min() >= 0
    assert r["n_blocks"] == 2 * (K - 1) + 2 * (KZ - 1)
    assert r["invalid_frac"] < 0.01 and r["chain_invalid_frac"] < 0.05
    assert r["n_aux"] == int(np.sum(np.asarray(tiny_problem.T) > 0))
    # reconstruction is in the right ballpark of the truth (warm start is Tikhonov-like)
    rel = np.linalg.norm(r["mean"] - tiny_problem.eps_true) / np.linalg.norm(tiny_problem.eps_true)
    assert rel < 1.0


def test_energy_matches_spec_formula_independent(tiny_problem):
    """Brute-force spec formula from T, b, sigma (not via chain_energy / layout indexing), group=1, K=3."""
    Kc, Kzc = 3, 6
    prob, meta = build_ising_chain(tiny_problem, Kc, None, None, 0.7, Kzc, tau_mode="dz")
    T, b, sig = np.asarray(tiny_problem.T), np.asarray(tiny_problem.b), np.asarray(tiny_problem.sigma)
    lay, D = meta["lay"], meta["Delta"]
    rng = np.random.default_rng(3)
    x = rng.integers(0, Kc, meta["N"])
    w = rng.integers(0, Kzc, meta["L"])
    z = meta["z0"] + meta["dz"] * w
    e = 0.0
    for i in range(T.shape[0]):
        els = np.flatnonzero(lay["chord"] == i)
        if len(els) == 0:
            continue
        # chord pixels in chain order = link_pix of these elements
        prev = 0.0
        for a in els:
            j = lay["link_pix"][lay["link_of"] == a]
            assert len(j) == 1
            j = j[0]
            e += (z[a] - prev - T[i, j] * D * x[j]) ** 2 / (2 * meta["tau_i"][a] ** 2)
            prev = z[a]
        e += (b[i] - prev) ** 2 / (2 * sig[i] ** 2)
    ed = tiny_problem.edges
    e += 0.5 * meta["lam"] * D ** 2 * np.sum((x[ed[:, 0]] - x[ed[:, 1]]) ** 2)
    assert prob.energy(_bits(x, w, Kc, Kzc)) == pytest.approx(e, rel=1e-9)


def test_backends_agree_on_chain_model(tiny_problem):
    """jax and thrml backends sample the same chain model (posterior means agree within MC error)."""
    prob, meta = build_ising_chain(tiny_problem, 4, None, None, 1.0, 8, tau_mode="dz")
    from tomo.sampling import run_ising
    sched = dict(n_warmup=300, n_samples=60, steps_per_sample=2)
    init = np.tile(meta["init"], (6, 1))
    m, e = {}, {}
    for be in ("jax", "thrml"):
        r = run_ising(prob, meta["colors"], sched, 6, jax.random.PRNGKey(1), beta=1.0, init=init, backend=be)
        m[be] = decode_chain(r["samples"], meta)["x"].reshape(-1, meta["N"]).mean(0)
        e[be] = float(np.mean(r["energy_trace"]))
    assert np.sqrt(np.mean((m["jax"] - m["thrml"]) ** 2)) < 0.1
    assert abs(e["jax"] - e["thrml"]) < 0.05 * abs(e["jax"])


def test_diagnostics_and_init(tiny_problem):
    cfg = load_config({"grid": {"n": 12}, "data_grid": {"n": 48}, "model": {"K": K, "Kz": KZ},
                       "schedule": {"n_chains": 3, "anneal": {"n_betas": 3, "sweeps_per_beta": 3}}})
    for init in ("tikhonov", "random"):
        r = sample_chain(tiny_problem, cfg, jax.random.PRNGKey(2), tau=1.0, init=init, anneal=False,
                         schedule=dict(n_warmup=20, n_samples=10, steps_per_sample=2))
        for k in ("frozen_frac", "rhat_med", "rhat_q95", "converged", "invalid_ok"):
            assert k in r
        assert 0.0 <= r["frozen_frac"] <= 1.0


def _toy_marginal_dev(compensate, tau_over_dz=None, n=5, tau=0.03, sigma=0.1, b=0.9):
    """Max deviation (nats, up to a constant) of the exact-DP -log marginal from (b - sum m)^2/(2 sigma^2), and fitted variance ratio."""
    dz = tau / 4 if tau_over_dz is None else tau / tau_over_dz
    grid = np.arange(0, int(3.0 / dz)) * dz
    s2 = max(sigma ** 2 - n * tau ** 2, 0.1 * sigma ** 2) if compensate else sigma ** 2
    rng = np.random.default_rng(0)
    tot, vals = [], []
    for _ in range(8):
        m = rng.uniform(0.1, 0.25, n)
        a = np.exp(-(grid - m[0]) ** 2 / (2 * tau ** 2))
        for k in range(1, n):
            a = np.exp(-(grid[:, None] - grid[None, :] - m[k]) ** 2 / (2 * tau ** 2)) @ a
        vals.append(-np.log(np.sum(a * np.exp(-(b - grid) ** 2 / (2 * s2)))))
        tot.append(m.sum())
    tot, vals = np.array(tot), np.array(vals)
    d = vals - (b - tot) ** 2 / (2 * sigma ** 2)
    veff = 1 / (2 * np.polyfit((b - tot) ** 2, vals, 1)[0])
    return np.abs(d - d.mean()).max(), veff / sigma ** 2


def test_compensation_toy_chord():
    """s^2 = sigma^2 - n tau^2 makes the marginal exactly N(b; sum m, sigma^2); uncompensated is inflated by 1 + n tau^2/sigma^2."""
    n, tau, sigma = 5, 0.03, 0.1
    dev_u, v_u = _toy_marginal_dev(False)
    dev_c, v_c = _toy_marginal_dev(True)
    assert v_u == pytest.approx(1 + n * tau ** 2 / sigma ** 2, rel=1e-3)
    assert dev_u > 0.2 and dev_c < 1e-3 and dev_c < dev_u / 100
    assert v_c == pytest.approx(1.0, abs=1e-3)
    # discrete z: rounding error shrinks rapidly with tau/dz
    d = {r: _toy_marginal_dev(True, r) for r in (0.5, 0.75, 1.0)}
    assert d[1.0][0] < 1e-3 and d[0.75][0] < 5e-3 and d[0.5][0] < 0.1
    assert d[0.5][0] > d[0.75][0] > d[1.0][0]
    assert d[0.5][0] < _toy_marginal_dev(False, 0.5)[0] / 4


def test_compensation_build(tiny_problem):
    """Compensated model: s_i^2 + sum tau^2 == sigma^2 when admissible; clamp reported otherwise; energy consistent."""
    sig = np.asarray(tiny_problem.sigma)
    prob, meta = build_ising_chain(tiny_problem, K, None, None, 0.1, KZ, tau_mode="dz", compensate=True)
    ch, tau = meta["lay"]["chord"], meta["tau_i"]
    S = np.bincount(ch, weights=tau ** 2, minlength=len(sig))
    cw = meta["chords_with"]
    ok = S[cw] < 0.9 * sig[cw] ** 2
    assert ok.any()
    np.testing.assert_allclose((meta["s2"] + S)[cw][ok], sig[cw][ok] ** 2, rtol=1e-12)
    assert meta["eff_noise_ratio"] < meta["tau2_over_sigma2"] + 1e-12
    rng = np.random.default_rng(1)
    x, w = rng.integers(0, K, meta["N"]), rng.integers(0, KZ, meta["L"])
    assert prob.energy(_bits(x, w, K, KZ)) == pytest.approx(chain_energy(x, w, meta), rel=1e-9)
    # strongly clamped case: tau huge -> inflation reported
    _, m2 = build_ising_chain(tiny_problem, K, None, None, 1e3, KZ, tau_mode="sigma", compensate=True)
    assert m2["comp_clamped_frac"] == 1.0 and m2["eff_noise_ratio"] > 1.0
    _, m3 = build_ising_chain(tiny_problem, K, None, None, 0.1, KZ, tau_mode="dz", compensate=False)
    assert m3["eff_noise_ratio"] == pytest.approx(m3["tau2_over_sigma2"])
