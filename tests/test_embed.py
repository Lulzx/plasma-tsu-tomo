import itertools

import numpy as np
import pytest

from tomo.embed import (binary_bit_qubo, coupling_dynamic_range, decode, embed_bounded_degree,
                        logical_to_physical, n_copies)
from tomo.sampling import IsingProblem, check_coloring


def rand_problem(n, p, seed, hs=1.0):
    r = np.random.default_rng(seed)
    Jd = np.triu(r.normal(size=(n, n)) * (r.random((n, n)) < p), 1)
    return IsingProblem.from_dense(r.normal(size=n) * hs, Jd, offset=0.7)


def all_states(n):
    return ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(bool)


def star(n_leaf, seed=0):
    r = np.random.default_rng(seed)
    rows = np.zeros(n_leaf, int)
    return IsingProblem(r.normal(size=n_leaf + 1), rows, np.arange(1, n_leaf + 1), r.normal(size=n_leaf), 0.3)


@pytest.mark.parametrize("topology", ["tree", "chain"])
@pytest.mark.parametrize("D", [3, 4, 5, 16])
def test_max_degree_and_colouring(topology, D):
    p = rand_problem(40, 0.5, 1)
    pp, m = embed_bounded_degree(p, D, topology=topology)
    assert pp.degrees.max() <= D
    assert check_coloring(pp.n, pp.edges, m["colors"])
    assert m["n_phys"] == pp.n == m["chain_len"].sum()
    assert np.array_equal(m["chain_len"], [n_copies(d, D) for d in p.degrees])


def test_noop_when_within_D():
    p = rand_problem(8, 0.5, 2)
    pp, m = embed_bounded_degree(p, 16)
    assert pp.n == p.n and m["overhead"] == 1.0 and m["J_F"] > 0
    assert np.allclose(pp.energy(all_states(8)), p.energy(all_states(8)))


@pytest.mark.parametrize("topology", ["tree", "chain"])
def test_agree_energy_equals_logical(topology):
    p = rand_problem(10, 0.8, 3)
    pp, m = embed_bounded_degree(p, 3, topology=topology)
    assert m["n_embedded"] > 0
    S = all_states(10)
    Sp = logical_to_physical(S, m)
    assert np.allclose(pp.energy(Sp), p.energy(S))
    dec, info = decode(Sp, m)
    assert (dec == S).all() and info["broken_spin_frac"].max() == 0 and info["broken_edge_frac"].max() == 0


def test_broken_chain_reporting_and_majority():
    p = star(12)
    pp, m = embed_bounded_degree(p, 4, topology="chain")
    c = m["copies"][0]
    assert len(c) > 3
    S = logical_to_physical(np.ones(p.n, bool), m)
    S[c[0]] = False
    dec, info = decode(S, m)
    assert dec[0] and info["broken_spin_frac"] == pytest.approx(1.0)  # only logical spin 0 is embedded
    assert info["broken_edge_frac"] == pytest.approx(1 / len(m["internal_edges"]))
    assert decode(S, m, "first")[0][0] == False  # noqa: E712


@pytest.mark.parametrize("topology", ["tree", "chain"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ground_state_and_marginals(topology, seed):
    # logical <= 10 spins, physical <= ~20
    p = rand_problem(7, 0.9, seed, hs=0.3)
    pp, m = embed_bounded_degree(p, 4, topology=topology)
    assert pp.n <= 22, pp.n
    S = all_states(p.n)
    E = p.energy(S)
    Sp = all_states(pp.n)
    Ep = pp.energy(Sp)
    # ground state of embedded model decodes to a logical ground state and has all copies agreeing
    g = Ep.argmin()
    dec, info = decode(Sp[g], m)
    assert info["broken_edge_frac"] == 0
    assert p.energy(dec) == pytest.approx(E.min(), abs=1e-9)
    assert Ep.min() == pytest.approx(E.min(), abs=1e-9)
    # marginals at beta=1: large J_F -> match
    big = 40.0 * m["J_F"]
    pp2, m2 = embed_bounded_degree(p, 4, J_F=big, topology=topology)
    for beta in (0.5, 1.0):
        w = np.exp(-beta * (pp2.energy(Sp) - Ep.min() * 0))
        w /= w.sum()
        dec, _ = decode(Sp, m2)
        mag_phys = (w[:, None] * np.where(dec, 1.0, -1.0)).sum(0)
        wl = np.exp(-beta * E)
        wl /= wl.sum()
        mag = (wl[:, None] * np.where(S, 1.0, -1.0)).sum(0)
        assert np.abs(mag_phys - mag).max() < 5e-3
        # weight on broken states is negligible
        _, info = decode(Sp, m2)
        assert (w * (info["broken_edge_frac"] > 0)).sum() < 1e-6


def test_auto_JF_rules_order():
    p = rand_problem(30, 0.7, 5)
    _, mc = embed_bounded_degree(p, 5, topology="chain", jf_rule="copy")
    _, mk = embed_bounded_degree(p, 5, topology="chain", jf_rule="cut")
    assert mk["J_F"] >= mc["J_F"] > 0
    _, mt = embed_bounded_degree(p, 5, topology="tree", jf_rule="cut")
    assert mt["J_F"] <= mk["J_F"] + 1e-9     # trees have smaller cut sides than chains
    assert mt["max_path_len"] <= mk["max_path_len"]


def test_tree_shorter_than_chain():
    p = star(300)
    _, mt = embed_bounded_degree(p, 16, topology="tree")
    _, mc = embed_bounded_degree(p, 16, topology="chain")
    assert mt["max_chain_len"] == mc["max_chain_len"] == n_copies(300, 16)
    assert mt["max_path_len"] <= 4 < mc["max_path_len"]


def test_binary_encoding_matches_integer_energy():
    r = np.random.default_rng(0)
    nv, nb = 4, 3
    A = r.normal(size=(nv, nv))
    Q = A @ A.T
    cy, c0 = r.normal(size=nv), 0.4
    from tomo.sampling import bits_to_spins_qubo
    Qb, cb, cc, rowvar, off = binary_bit_qubo(Q, cy, c0, nb)
    ip = bits_to_spins_qubo(Qb, cb, cc)
    for _ in range(10):
        u = r.integers(0, 2, nv * nb).astype(bool)
        y = (u.reshape(nv, nb) * 2 ** np.arange(nb)).sum(1)
        assert ip.energy(u) == pytest.approx(0.5 * y @ Q @ y - cy @ y + c0, rel=1e-9)
    ratio, bits, lo, hi = coupling_dynamic_range(ip)
    assert bits > 0
