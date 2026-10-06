# Related work and novelty assessment (literature search, 2026-10-06)

Every reference below was checked by opening its arXiv, DOI or publisher page, except the few marked *(search only)*.
Re-check novelty claims before submitting; not finding prior work in a search does not prove there is none.

## Closest prior art to I-chain (auxiliary chains that remove dense couplings)

| Work | What it does | Overlap with I-chain |
|---|---|---|
| Sinz, CP 2005, sequential counter | Auxiliary running-count variables with local clauses for cardinality constraints | Same running-sum chain idea, but discrete, unit-weight, and in CNF |
| Eén & Sörensson, JSAT 2006 | Adder / BDD / sorting-network encodings of weighted pseudo-Boolean constraints | Weighted sums encoded with local auxiliaries, but exact and discrete |
| Camsari et al., PRX 2017 (arXiv:1610.00377) | p-bit adder chains used as invertible logic | Running sums on p-bit hardware, but exact and with no noise model |
| Suda, Naito, Hasegawa, arXiv:2601.18108 (2026) | Auxiliary networks that turn dense equality penalties into sparse QUBOs (O(N²) → O(N) edges) | Closest mechanism, but unit coefficients only and optimisation only |
| Sajeeb et al., Phys. Rev. Applied 2025 (arXiv:2503.01177) | Copy-node chains that bound degree in p-bit Ising machines | Same goal (bounded degree for parallel Gibbs), but the auxiliaries copy a spin rather than compute a partial sum |
| Lechner, Hauke, Zoller, Sci. Adv. 2015; Ender et al., Quantum 2023 | LHZ / parity architecture | Local embedding of all-to-all couplings, but O(N²) qubits and 4-body constraints |
| Choi 2008/2010; Cai, Macready, Roy 2014 | Minor embedding with ferromagnetic chains | The standard dense-to-sparse baseline |
| Chancellor, QST 2019; Berwald et al., PTRSA 2022 | Domain-wall encoding | Our encoding of levels; freezing behaviour under annealing |

## Least squares and tomography on Ising/QUBO hardware (all keep dense AᵀA)

O'Malley & Vesselinov 2016 (ToQ.jl); Borle & Lomonaco arXiv:1809.07649; Nau et al., J. Imaging 2023 (arXiv:2212.01312);
Jun, Sci. Rep. 2023 (arXiv:2207.02448); Dremel et al., IEEE TQE 2025; Lee & Jun arXiv:2504.20654; Wang et al. arXiv:2606.24561;
Ide & Ohzeki arXiv:2202.00452; Erementchouk et al. arXiv:2512.22784 (Ising-machine discrete tomography, binary images).
All of these optimise rather than sample. Connectivity is the stated bottleneck, and none of them localises the data term.

## Sampling hardware

- **Extropic, "Thermalizing Stochastic Programs", arXiv:2608.01615 (2026).** Bayesian Gaussian-field reconstruction compiled to Z1 with fixed-point bit bundles and minor embedding. The projection is 0.50 µJ and 7.5 ms per step, readout-dominated, with slow mixing for broad posteriors. **This is the most direct comparator.**
- Extropic, arXiv:2510.23972: E_cell ≈ 2 fJ (v2) and τ0 ≈ 100 ns. arXiv:2608.01615 App. B gives 7.09 fJ per p-bit per Gibbs cycle at 50 MHz, readout of 1.69 pJ per node and 25 µs per frame, and coupling writes of 153.6 pJ per node.
- Z1 (extropic.ai/hardware): 269,568 p-bits on a planar, 2-colourable, **degree-16** graph.
- Aadit et al., Nat. Electron. 2022 (arXiv:2110.02481); Niazi et al., Nat. Electron. 2024 (arXiv:2303.10728); Nikhar et al., Nat. Commun. 2024 (arXiv:2312.08748); Aadit et al. arXiv:2606.25313 (1M p-bits).
- Thermodynamic linear algebra: Aifer et al. arXiv:2308.05660; Melanson et al. arXiv:2312.04836; Aifer et al. arXiv:2410.01793 (thermodynamic Bayesian inference). This is the natural competitor for near-Gaussian posteriors.
- Margossian et al. arXiv:2110.10801: auxiliary-Gaussian block Gibbs for Ising and Potts models, with mixing bounds.

## Fusion tomography

- **Gaussian process tomography:** Svensson 2011 (JET report); Li et al., RSI 2013 *(search only)*; Wang et al., RSI 2018; Moser et al., FST 2022 (AUG bolometer); Matos et al. arXiv:2004.06429; Ueda & Nishiura arXiv:2410.11454 (log-GP with positivity).
- **MFI and Tikhonov:** Anton et al., PPCF 1996 *(search only)*; Odstrčil et al., RSI 2016; cherab-inversion.
- **Bayesian sampling:** Hamm et al. arXiv:2506.20232 (TCV; positivity enforced, ULA sampling, offline; open code).
- **Real-time:** Ferreira et al., FED 2021 (JET, a few ms, no uncertainty); Hamm et al. arXiv:2603.11856 (TCV real-time credible bounds on *region-integrated* power using a single matrix-vector product; Gaussian; no 2D map).
- **Open geometry and phantoms:** Hamm et al. arXiv:2608.03835 and their repositories (TCV bolometer, 120 channels). No open real bolometer dataset was found.
- **Classical background:** Geman & Geman 1984; Green, IEEE TMI 1990; Gouillart et al., Inverse Problems 2013 (belief-propagation discrete tomography).

## Assessment

**Likely new:**
1. A running-partial-sum chain for a *weighted, real-valued Gaussian data term* (b − Tx)².
2. Its noise accounting: effective noise σ² + nτ², variance compensation, and the rigidity window for discrete z.
3. Posterior *sampling* (rather than optimisation) of tomography on p-bit hardware.
4. Any Ising or sampling hardware applied to fusion tomography.
5. Coverage calibration of fusion tomography uncertainty, which we found nowhere.

**Not new:**
- The general mechanism: sequential counters, adder chains, and Suda et al. 2026.
- Variance compensation by itself, which is ordinary Gaussian marginalisation.

**Threats:**
- **Gaussian methods are already real-time.** Hamm 2026 delivers real-time Gaussian uncertainty with one matrix-vector product. A sampling approach therefore has to win on non-Gaussian structure (positivity, discrete levels, sparsity) or on full 2D calibrated maps.
- **Extropic's own Gaussian-field paper** reports latency similar to ours (7.5 ms).
- **I-chain does not fit Z1.** Its maximum degree of 364 exceeds Z1's degree of 16, so it would need further embedding.
