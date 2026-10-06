"""Degree-bounded embedding of an ``IsingProblem`` by copy nodes (ferromagnetic trees / chains).

Hardware such as Extropic's Z1 (planar, 2-colourable, degree 16) cannot host spins with hundreds of
neighbours.  The standard remedy (minor embedding with ferromagnetic chains; Choi 2008; Sajeeb et al.,
Phys. Rev. Applied 2025, arXiv:2503.01177 for p-bit Gibbs machines) replaces a logical spin ``v`` of
degree ``d > D`` by ``m`` copies joined by ferromagnetic couplings ``J_F`` along a tree, and spreads
the external edges (and the bias) over the copies so that every physical spin has degree <= D.

Counting.  A tree on m nodes has m-1 internal edges, so the copies expose
``m*D - 2(m-1) = m(D-2) + 2`` free external slots (independent of the tree shape).  The minimal
number of copies is therefore ``m = max(1, ceil((d - 2) / (D - 2)))`` for d > D.  ``topology='chain'``
uses a path (diameter m-1); ``topology='tree'`` uses a heap with branching D-1 (diameter <= 2 for
m <= 1 + (D-1) + (D-1)^2), same m but much shorter ferromagnetic paths, which matters for mixing.

Energy.  With the library convention ``E = -h.s - sum J s s + offset``::

    E_phys(S) = -sum_c h_c S_c - sum_ext J S S - J_F sum_int S_a S_b + offset + J_F * n_int

so any state where all copies of every spin agree has exactly the logical energy (the ``J_F * n_int``
offset cancels the ferromagnetic bonds), and a broken bond costs ``2 J_F``.  The bias is split evenly
over the copies of a spin (it carries no degree).

Automatic J_F (``jf_rule='cut'``, default).  Cutting one internal edge splits the copies of v into
two sides A, B.  Flipping the whole side A relative to B changes the energy by at most
``2 W_A`` where ``W_A = sum_{c in A} (|h_c| + sum_ext |J_c|)`` (the maximal gain from its external
field) and costs ``2 J_F`` in the broken bond.  So ``J_F > max_{edges} min(W_A, W_B)`` makes every
single-cut broken configuration strictly higher in energy than the repaired one (ground states are
agree states for a tree), and we use ``J_F = margin * that maximum`` (margin 1.2).  A cheaper
heuristic, ``jf_rule='copy'``, uses the largest field on a single copy (``max_c W_c``), which is the
local stability condition for a single-copy flip; it gives a smaller dynamic range but does not rule
out multi-copy domain walls.  Larger J_F means exact ground-state equivalence but slower Gibbs
mixing (copies freeze) and a larger coupling dynamic range on hardware.

Public API: ``embed_bounded_degree``, ``decode``, ``auto_J_F``, ``binary_bit_qubo``,
``coupling_dynamic_range``.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .sampling import IsingProblem, check_coloring, greedy_coloring

__all__ = ["embed_bounded_degree", "decode", "auto_J_F", "n_copies", "binary_bit_qubo",
           "coupling_dynamic_range", "embedding_summary", "coupling_quantisation",
           "bipartite_relay_estimate", "logical_to_physical"]


def n_copies(d: int, D: int) -> int:
    """Minimal number of copies for a spin of degree d under degree bound D (>=3)."""
    if d <= D:
        return 1
    return int(-(-(d - 2) // (D - 2)))


def _tree_edges(m: int, D: int, topology: str):
    """Internal edges (parent, child) of a copy tree on m nodes; internal degree <= D."""
    if m == 1:
        return []
    if topology == "chain":
        return [(i, i + 1) for i in range(m - 1)]
    b = D - 1
    return [((i - 1) // b, i) for i in range(1, m)]


def _node_weights(prob: IsingProblem):
    """W_v = |h_v| + sum_j |J_vj| per logical node."""
    w = np.abs(prob.h).copy()
    np.add.at(w, prob.rows, np.abs(prob.vals))
    np.add.at(w, prob.cols, np.abs(prob.vals))
    return w


def _slot_lists(m, D, tedges):
    """For each copy, number of free external slots D - internal degree."""
    ideg = np.zeros(m, dtype=np.int64)
    for a, b in tedges:
        ideg[a] += 1
        ideg[b] += 1
    return D - ideg


def embed_bounded_degree(prob: IsingProblem, D: int = 16, J_F=None, topology: str = "tree",
                         jf_rule: str = "cut", margin: float = 1.2):
    """Compile ``prob`` to an equivalent-in-the-large-J_F-limit Ising model of max degree <= D.

    Returns ``(prob_phys, meta)``.  ``meta`` keys: ``copies`` (list of physical index arrays per
    logical spin), ``node_of`` (physical -> logical), ``internal_edges`` ((E_int, 2) physical),
    ``J_F``, ``D``, ``topology``, ``n_logical``, ``n_phys``, ``chain_len`` (copies per spin),
    ``max_chain_len``, ``max_path_len`` (longest ferromagnetic path, edges), ``overhead``,
    ``J_F_rule``, ``max_degree`` (physical), ``colors`` (greedy colouring of the physical graph),
    ``n_colors``.  Spins of degree <= D are kept as a single copy (so a model already within D is
    returned unchanged up to ordering).
    """
    if D < 3:
        raise ValueError("D must be >= 3")
    if topology not in ("tree", "chain"):
        raise ValueError("topology must be 'tree' or 'chain'")
    n = prob.n
    deg = prob.degrees
    m_of = np.array([n_copies(int(d), D) for d in deg], dtype=np.int64)
    start = np.concatenate([[0], np.cumsum(m_of)])
    n_phys = int(start[-1])
    copies = [np.arange(start[v], start[v + 1]) for v in range(n)]
    node_of = np.repeat(np.arange(n), m_of)

    # incident logical edges per node, in neighbour order (locality-preserving slot fill)
    ends = np.concatenate([prob.rows, prob.cols])
    other = np.concatenate([prob.cols, prob.rows])
    eid = np.concatenate([np.arange(prob.n_edges)] * 2)
    order = np.lexsort((other, ends))
    ends, other, eid = ends[order], other[order], eid[order]
    ptr = np.concatenate([[0], np.cumsum(np.bincount(ends, minlength=n))])

    phys_end = np.zeros((prob.n_edges, 2), dtype=np.int64)   # physical endpoint of (row side, col side)
    int_edges, tree_info = [], []
    for v in range(n):
        m = int(m_of[v])
        te = _tree_edges(m, D, topology)
        slots = _slot_lists(m, D, te)
        # slot owner list: copy index repeated by its free slots
        owner = np.repeat(np.arange(m), slots)
        lo, hi = ptr[v], ptr[v + 1]
        k = hi - lo
        if k > len(owner):
            raise AssertionError("insufficient slots")
        if m > 1:
            # spread external edges roughly evenly over copies with slots (keeps cut weights balanced)
            cap_idx = np.flatnonzero(slots > 0)
            share = np.array_split(np.arange(k), len(cap_idx))
            assign = np.empty(k, dtype=np.int64)
            for ci, sh in zip(cap_idx, share):
                assign[sh] = ci
            # respect per-copy capacity (even split can exceed it only if k > capacity: excluded above)
            over = np.bincount(assign, minlength=m) > slots
            if over.any():                       # fall back to sequential fill
                assign = owner[:k]
        else:
            assign = np.zeros(k, dtype=np.int64)
        e = eid[lo:hi]
        is_row = prob.rows[e] == v
        pc = start[v] + assign
        phys_end[e[is_row], 0] = pc[is_row]
        phys_end[e[~is_row], 1] = pc[~is_row]
        for a, b in te:
            int_edges.append((start[v] + a, start[v] + b))
        tree_info.append((te, assign, m))

    int_edges = np.asarray(int_edges, dtype=np.int64).reshape(-1, 2)
    ext_r = np.minimum(phys_end[:, 0], phys_end[:, 1])
    ext_c = np.maximum(phys_end[:, 0], phys_end[:, 1])

    # bias split evenly over copies
    hp = np.repeat(prob.h / m_of, m_of)

    # J_F
    Wnode_ext = np.zeros(n_phys)      # per copy: |h_c| + sum |J_ext|
    Wnode_ext += np.abs(hp)
    np.add.at(Wnode_ext, phys_end[:, 0], np.abs(prob.vals))
    np.add.at(Wnode_ext, phys_end[:, 1], np.abs(prob.vals))
    jf_auto = _auto_JF_from(Wnode_ext, copies, tree_info, start, jf_rule, margin)
    JF = float(jf_auto if J_F is None else J_F)
    if JF <= 0:
        raise ValueError("J_F must be positive")

    rows = np.concatenate([ext_r, int_edges[:, 0]]) if len(int_edges) else ext_r
    cols = np.concatenate([ext_c, int_edges[:, 1]]) if len(int_edges) else ext_c
    vals = np.concatenate([prob.vals, np.full(len(int_edges), JF)])
    lo_, hi_ = np.minimum(rows, cols), np.maximum(rows, cols)
    pp = IsingProblem(hp, lo_, hi_, vals, prob.offset + JF * len(int_edges))

    colors = greedy_coloring(pp.n, pp.edges)
    assert check_coloring(pp.n, pp.edges, colors)
    pdeg = pp.degrees
    if n_phys and pdeg.max() > D:
        raise AssertionError(f"embedding failed: physical degree {pdeg.max()} > {D}")
    max_path = max((_tree_diameter(te, m) for te, _, m in tree_info), default=0)
    meta = dict(copies=copies, node_of=node_of, internal_edges=int_edges, J_F=JF, J_F_auto=jf_auto,
                J_F_rule=jf_rule, D=D, topology=topology, n_logical=n, n_phys=n_phys,
                chain_len=m_of, max_chain_len=int(m_of.max()) if n else 0, max_path_len=int(max_path),
                n_embedded=int(np.sum(m_of > 1)), overhead=n_phys / max(n, 1),
                max_degree=int(pdeg.max()) if n_phys else 0, max_degree_logical=int(deg.max()) if n else 0,
                colors=colors, n_colors=int(colors.max() + 1) if n_phys else 0, logical_offset=prob.offset)
    return pp, meta


def _tree_diameter(te, m):
    if m <= 1:
        return 0
    adj = [[] for _ in range(m)]
    for a, b in te:
        adj[a].append(b)
        adj[b].append(a)

    def far(s):
        dist = {s: 0}
        q = [s]
        for u in q:
            for w in adj[u]:
                if w not in dist:
                    dist[w] = dist[u] + 1
                    q.append(w)
        u = max(dist, key=dist.get)
        return u, dist[u]
    u, _ = far(0)
    return far(u)[1]


def _auto_JF_from(Wc, copies, tree_info, start, rule, margin):
    """J_F from per-copy external weights (see module docstring)."""
    if rule not in ("cut", "copy"):
        raise ValueError("jf_rule must be 'cut' or 'copy'")
    best = 0.0
    for v, (te, _, m) in enumerate(tree_info):
        if m == 1:
            continue
        w = Wc[start[v]:start[v] + m]
        if rule == "copy":
            best = max(best, float(w.max()))
            continue
        # subtree sums rooted at 0 (parents precede children in both topologies)
        sub = w.copy()
        for a, b in reversed(te):
            sub[a] += sub[b]
        tot = sub[0]
        for _, b in te:
            best = max(best, float(min(sub[b], tot - sub[b])))
    if best == 0.0:
        best = float(Wc.max()) if len(Wc) else 1.0
    return margin * best


def auto_J_F(prob: IsingProblem, D: int = 16, topology: str = "tree", rule: str = "cut", margin: float = 1.2):
    """The J_F that ``embed_bounded_degree(prob, D, None, topology, rule)`` would use."""
    return embed_bounded_degree(prob, D, None, topology, rule, margin)[1]["J_F"]


def decode(spins_phys, meta, method: str = "majority"):
    """Decode physical spins (..., n_phys) (bool or +/-1) to logical bool spins (..., n_logical).

    ``method``: ``'majority'`` (sum of copies; ties broken by the first copy) or ``'first'``
    (representative copy).  Returns ``(spins_logical_bool, info)`` with ``info`` holding
    ``broken_spin_frac`` (fraction of embedded logical spins whose copies disagree, per leading
    index) and ``broken_edge_frac`` (fraction of ferromagnetic bonds that are unsatisfied).
    """
    S = np.asarray(spins_phys)
    S = (S if S.dtype == bool else S > 0)
    lead = S.shape[:-1]
    S2 = S.reshape(-1, S.shape[-1])
    n = meta["n_logical"]
    m = meta["chain_len"]
    first = np.array([c[0] for c in meta["copies"]])
    cnt = (S2.astype(np.int64) @ sp.csr_matrix((np.ones(len(meta["node_of"])), (np.arange(len(meta["node_of"])), meta["node_of"])),
                                               shape=(len(meta["node_of"]), n)))
    cnt = np.asarray(cnt)
    if method == "first":
        out = S2[:, first]
    elif method == "majority":
        out = np.where(2 * cnt > m, True, np.where(2 * cnt < m, False, S2[:, first]))
    else:
        raise ValueError("method must be 'majority' or 'first'")
    emb = m > 1
    broken_spin = ((cnt > 0) & (cnt < m))[:, emb]
    ie = meta["internal_edges"]
    if len(ie):
        broken_edge = (S2[:, ie[:, 0]] != S2[:, ie[:, 1]]).mean(axis=1)
    else:
        broken_edge = np.zeros(S2.shape[0])
    bs = broken_spin.mean(axis=1) if emb.any() else np.zeros(S2.shape[0])
    info = dict(broken_spin_frac=bs.reshape(lead), broken_edge_frac=broken_edge.reshape(lead), method=method)
    return out.reshape(lead + (n,)), info


def logical_to_physical(spins, meta):
    """All-copies-agree physical state for logical spins (..., n_logical)."""
    s = np.asarray(spins)
    s = s if s.dtype == bool else s > 0
    return s[..., meta["node_of"]]


def coupling_dynamic_range(prob: IsingProblem, include_h: bool = True):
    """Dynamic range of coupling/bias magnitudes: ``(ratio, bits, min, max)`` over nonzero values.

    ``bits = log2(max/min)`` is the precision a fixed-point coupling DAC would need to represent every
    value to within one LSB of the smallest (a loose upper bound; typical hardware needs a few bits
    for the bulk of the distribution).
    """
    v = np.abs(prob.vals)
    if include_h:
        v = np.concatenate([v, np.abs(prob.h)])
    v = v[v > 0]
    if len(v) == 0:
        return 1.0, 0.0, 0.0, 0.0
    return float(v.max() / v.min()), float(np.log2(v.max() / v.min())), float(v.min()), float(v.max())


def embedding_summary(prob: IsingProblem, meta: dict) -> dict:
    """Flat dict of embedding statistics (for tables)."""
    pp_deg_max = meta["max_degree"]
    return dict(n_logical=meta["n_logical"], n_phys=meta["n_phys"], overhead=meta["overhead"],
                max_degree_logical=meta["max_degree_logical"], max_degree_phys=pp_deg_max,
                n_embedded_spins=meta["n_embedded"], max_chain_len=meta["max_chain_len"],
                max_path_len=meta["max_path_len"], n_colors=meta["n_colors"], J_F=meta["J_F"],
                topology=meta["topology"], D=meta["D"])


def binary_bit_qubo(Qy, cy, const, nbits):
    """Binary (power-of-two) encoding of integer levels ``y_v = sum_b 2^b u_vb``, u in {0,1}.

    Expands ``1/2 y^T Qy y - cy.y + const`` (the same convention as ``ebm_chain._bit_qubo``) into a
    bit QUBO.  Every bit pattern is a valid level (levels 0 .. 2^nbits - 1), so no domain-wall
    penalty is needed, but couplings carry weights ``2^(a+b)``.  ``nbits``: int or per-variable array.
    Returns ``(Qb csr zero-diagonal symmetric, cb, const, rowvar, offsets)`` for ``bits_to_spins_qubo``.
    """
    Qy = sp.csr_matrix(Qy)
    nv = Qy.shape[0]
    nbits = np.broadcast_to(np.asarray(nbits, dtype=np.int64), (nv,)).copy()
    off = np.concatenate([[0], np.cumsum(nbits)])
    nb = int(off[-1])
    rowvar = np.repeat(np.arange(nv), nbits)
    bitidx = np.arange(nb) - off[rowvar]
    R = sp.csr_matrix((2.0 ** bitidx, (np.arange(nb), rowvar)), shape=(nb, nv))
    Qfull = (0.5 * (R @ Qy @ R.T)).tocsr()
    d = Qfull.diagonal()
    Qb = (Qfull - sp.diags(d)).tocsr()
    Qb.eliminate_zeros()
    cb = -(R @ np.asarray(cy)) + d
    return Qb, cb, float(const), rowvar, off


def coupling_quantisation(prob: IsingProblem, bits=(4, 6, 8), include_h: bool = True):
    """Robust precision statistics of the magnitudes ``|J|`` (and ``|h|``).

    Plain max/min is dominated by round-off-sized entries, so this also reports percentile ranges
    and, for a b-bit sign-magnitude coupling DAC with full scale ``max|.|``, the fraction of nonzero
    entries that would round to zero (``zero_frac[b]``) and the RMS relative rounding error of the
    couplings that survive.  Returns dict(max, p50, p1, p01, bits_max_over_p1, bits_max_over_p50, zero_frac).
    """
    v = np.abs(prob.vals)
    if include_h:
        v = np.concatenate([v, np.abs(prob.h)])
    v = v[v > 1e-12 * v.max()] if len(v) else v
    if len(v) == 0:
        return {}
    mx = float(v.max())
    p1, p50, p01 = (float(np.percentile(v, q)) for q in (1, 50, 0.1))
    out = dict(max=mx, p50=p50, p1=p1, p01=p01, bits_max_over_p1=float(np.log2(mx / p1)),
               bits_max_over_p50=float(np.log2(mx / p50)))
    zf, rel = {}, {}
    for b in bits:
        lsb = mx / (2 ** b - 1)
        q = np.round(v / lsb) * lsb
        zf[b] = float(np.mean(q == 0))
        nz = q > 0
        rel[b] = float(np.sqrt(np.mean(((q[nz] - v[nz]) / v[nz]) ** 2))) if nz.any() else float("nan")
    out["zero_frac"], out["rms_rel_err_nonzero"] = zf, rel
    return out


def bipartite_relay_estimate(prob: IsingProblem, n_pass: int = 30, seed: int = 0):
    """Upper bound on extra relay spins needed to make the physical graph 2-colourable.

    Z1 is 2-colourable.  Any edge (u,v) whose endpoints get the same colour in a 2-colouring can be
    subdivided by one relay copy of u (ferromagnetically bonded to u, carrying the coupling to v);
    this keeps degrees unchanged.  We 2-colour heuristically (random start, repeated local flips of
    nodes with more same-coloured than other-coloured neighbours) and count monochromatic edges.
    Returns ``(n_relays_upper_bound, frac_edges)``; the true minimum (an edge-bipartization / max-cut
    complement) is <= this.
    """
    n = prob.n
    if prob.n_edges == 0:
        return 0, 0.0
    A = prob.J_sparse().copy()
    A.data[:] = 1.0
    rng = np.random.default_rng(seed)
    c = rng.integers(0, 2, n) * 2 - 1
    for p in range(n_pass):
        nb = A @ c                      # sum of neighbour colours; same-colour neighbours contribute c*nb > 0
        bad = (c * nb) > 0
        sel = bad & (rng.random(n) < 0.5)          # asynchronous half to avoid oscillation
        if not sel.any():
            break
        c = np.where(sel, -c, c)
    mono = int(np.sum(c[prob.rows] == c[prob.cols]))
    return mono, mono / prob.n_edges
