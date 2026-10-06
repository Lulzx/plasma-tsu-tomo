# Paper plan (decided 2026-10-06)

The draft lives outside this repo, in `../plasma-tsu-tomo-paper/` (LaTeX; build with `tectonic main.tex`).
The full results log is `docs/FINDINGS.md`.

**Working title:** *Sampling discrete Bayesian tomography on probabilistic hardware: what works, and the mixing barrier
that blocks the hardware-faithful route.*

The paper has one story told in two parts.
- **Part A (methods, for a fusion-diagnostics audience).** A discrete, positive posterior over emissivity levels
  (Potts / I-dense) gives accurate, calibrated 2D bolometer tomography in under a minute of CPU sampling.
- **Part B (analysis, for an unconventional-computing audience).** Local partial-sum embeddings (I-chain, I-tree) make
  the likelihood exact with bounded degree. They hit a data-augmentation mixing barrier, which we explain and predict
  quantitatively. Copy-node embedding onto degree-16 hardware freezes for the same reason.

## Claims and the evidence each needs

| # | Claim | Evidence | Status |
|---|---|---|---|
| A1 | The evidence (marginal-likelihood) λ calibrates the posterior | Gaussian posterior coverage over random fields | done: 95% coverage 0.70 → 0.95 |
| A2 | Discrete positive posterior: calibrated coverage, competitive accuracy | M3 + multi-seed run (4 phantoms × 5 seeds, ≥30 random fields) | M3 done: Potts covers 0.83–0.98 (GP 0.54–0.93); beats Tikhonov and GP on blob; GP and MFI more accurate on average. Multi-seed still to do |
| A3 | The gain comes from positivity / bound / discreteness (decomposed) | `experiments/route2_ablation.py`: truncated Gaussian, log-Laplace, Potts K sweep | 3 problems: the gain is positivity; discreteness gives none. Extend to all phantoms and K = 7, 9 |
| A4 | Holds on real geometry | TCV bolometer lines of sight (Hamm et al. open code) | geometry done (`tomo/tcv.py`); baselines run; EBMs on TCV to do |
| A5 | Practical: converged (R-hat < 1.05) in under 60 s | M3 timings with load recorded | Potts converged in 30–49 s; I-dense 57–110 s |
| B1 | Bounded-degree exact embedding: construction, σ² + nτ² accounting, compensation, τ window | `ebm_chain` / `ebm_tree` docstrings and tests | done |
| B2 | Mixing barrier explained: Gibbs spectral radius / missing-information theory predicts the measured slowdown | `docs/mixing_theory.md`, predicted vs measured autocorrelation time | done: theory within 5–20% of linear simulation, ratios within 2× of the discrete samplers, structural proposition proved |
| B3 | A calibrated prior worsens the barrier (and the noise inflation) | theory + measurements | done: slowdown ∝ 1/λ (about 8×); noise inflation 6.5× chain, 3× tree |
| B4 | Degree-16 copy embedding freezes or biases; binary encoding trades degree for coupling precision | `tomo/embed.py`, `experiments/embed_*.py` | done (binary sampling untested) |
| B5 | Hardware cost under published Extropic numbers, with honest latency | `tomo/energy.py` presets, M5 | M5 done with measured sweeps: Potts 39 nJ / 11 ms, I-dense 0.28 µJ / 75 ms, I-chain ≥ 10.8 µJ / 53 ms (not converged) |

## Required comparisons
- Hamm et al. 2025 (arXiv:2506.20232): Langevin sampling with positivity on TCV. Our truncated Gaussian is the closest re-implementation.
- Ueda & Nishiura (arXiv:2410.11454): log-GP with a Laplace approximation.
- Extropic arXiv:2608.01615: Gaussian-field reconstruction on Z1, about 0.5 µJ and 7.5 ms, using minor embedding.
- Sajeeb et al. 2025 (copy nodes) and Suda et al. 2026 (auxiliary constraint decomposition), for Part B novelty.

## Venues
- **Combined paper:** arXiv preprint, then Machine Learning: Science and Technology or Physical Review Applied.
- **Split:** Part A to Review of Scientific Instruments or PPCF; Part B to the NeurIPS ML4PS workshop or an unconventional-computing venue.
