# Findings log

This file records every result of the project, positive and negative, with the script or file each number comes from.
Unless stated otherwise:
- Problem: the default 32×32 geometry (806 pixels, 72 chords), K = 8 levels.
- λ: the evidence (marginal-likelihood) choice.
- Errors: relative L2 error of the posterior mean against the coarse truth.
- Timings: measured on a shared Apple-silicon laptop; each experiment log records the load average.

---

## 1. Geometry and data (M1)

Source: `experiments/m1_geometry.py` → `results/m1_geometry/`.

- 806 active pixels and 72 chords (3 fans × 24).
- A chord crosses 30.8 pixels on average (max 45).
- A pixel is crossed by 2.75 chords on average (max 12); 0.74% of pixels are crossed by none.
- T has rank 72 and condition number 11.6.
- Siddon ray tracing: summed chord length matches an analytic disk to −1.7e-4. The median per-chord error is 1.3e-3; the worst chords graze pixel edges.
- Data come from a 128×128 grid with its own geometry matrix (no inverse crime). The coarse truth is the 4×4 block average.

## 2. Choosing λ

Sources: `tomo/baselines.py`, `tests/test_baselines.py`, the λ study by the `ebm_common` agent, and `/tmp/evcheck.py`. The last is reproducible with `baselines.tikhonov(..., method=...)`.

**GCV overfits.** With M = 72 ≪ N = 806, GCV gives reduced χ² ≈ 0.002–0.1: the data are interpolable. Its error is fine, but its λ is meaningless as a prior precision.

**The discrepancy principle (χ²/M = 1)** was the first "tuned" choice. It makes the exact Gaussian posterior badly over-confident: 95% coverage of 0.70 on random fields.

**The evidence λ is the current default.** It maximises the Gaussian marginal likelihood p(b | λ) and comes out 6–10× smaller than the discrepancy λ. It calibrates the Gaussian posterior:

| Problems | rel-L2 (discrepancy → evidence) | 95% coverage (discrepancy → evidence) |
|---|---|---|
| 30 random fields | 0.353 → 0.336 | 0.70 → 0.95 |
| blob, 3 seeds | 0.770 → 0.777 | 0.54 → 0.95 |
| edge, 3 seeds | 0.591 → 0.605 | 0.63 → 0.95 |
| peaked, 3 seeds | 0.310 → 0.283 | 0.69 → 1.00 |
| hollow, 3 seeds | 0.653 → 0.654 | 0.37 → 0.76 |

The hollow profile is not identifiable from 72 chords at any λ tried.

**What does not cause the low coverage:**
- Forward-model mismatch: fine-grid truth against the coarse T gives χ² of about 0.05 of the noise.
- ε_max clipping: at most 3.7% of pixels are affected.

**Discrete quantile intervals** collapse onto one level when the posterior is narrow. This costs some coverage at K = 8.

## 3. Classical baselines (M2)

Source: `experiments/m2_baselines.py` → `results/m2_baselines/` (200 random fields).

Regenerated with the evidence λ:

| Baseline, 200 random fields | Mean rel-L2 | 95% coverage |
|---|---|---|
| Tikhonov, GCV | 0.336 | – |
| Tikhonov, tuned (evidence λ) | 0.335 | – |
| MFI | 0.296 | – |
| GP | 0.291 | 0.89 |

GP or MFI is the most accurate method on 195 of the 200 fields.

## 4. Model sizes and Gibbs blocking

A 2-colour checkerboard is **not** a valid Gibbs blocking once TᵀT is in the energy: every pair of pixels sharing a chord is coupled.

| Model | Spins | Max degree | Colours |
|---|---|---|---|
| Potts | 806 | 364 (pixels) | 48 |
| I-dense | 5,642 | 2,554 | 315 |
| I-sparse (threshold 0.05) | 5,642 | 951 | 140–196 |
| I-chain, K_z = 16 | 38,852 | 364 | 44 |
| I-chain, K_z = 32 (default) | 74,276 | 716 | 76 |
| I-tree, K_z = 32 | 72,044 | 468 | 114 |

## 5. Main EBM comparison (M3)

Source: `experiments/m3_ebm.py` → `results/m3_ebm/results_table.md`.

Setup: 4 phantoms plus 8 random fields, one noise seed each. Full schedule: 2,000 warm-up sweeps, then 500 samples taken every 10 sweeps. 16 chains, half started from random states.

**Relative L2 error:**

| | Peaked | Hollow | Blob | Edge | Random (8) |
|---|---|---|---|---|---|
| Tikhonov, tuned | 0.263 | 0.649 | 0.776 | 0.611 | 0.303 |
| MFI | **0.200** | 0.807 | **0.410** | 0.620 | 0.285 |
| GP | 0.226 | **0.540** | 0.798 | 0.653 | **0.263** |
| Potts | 0.257 | 0.558 | 0.593 | 0.613 | 0.308 |
| I-dense | 0.258 | 0.561 | 0.593 | 0.613 | 0.307 |
| I-chain | 0.313 | 0.557 | 0.689 | 0.613 | 0.321 |
| I-tree | 0.283 | 0.549 | 0.636 | 0.617 | 0.311 |

**95% coverage:**

| | Peaked | Hollow | Blob | Edge | Random (8) |
|---|---|---|---|---|---|
| Potts | 0.98 | 0.83 | 0.91 | 0.89 | 0.945 |
| GP | 0.54 | 0.71 | 0.87 | 0.89 | 0.93 |

**Convergence and time:**
- Potts and I-dense: max R-hat 1.00 on every problem. Potts takes 30–49 s, I-dense 57–110 s.
- I-chain and I-tree: max R-hat 2.5 to ∞ on every problem; they never converge.
- I-chain and I-tree MAP estimates are poor.
- I-chain's `time_s` includes model build and JIT compile, so it is not comparable with the others.

**Reading:**
- The discrete posterior is converged and close to calibrated everywhere except hollow.
- It clearly beats Tikhonov and GP on blob and somewhat on hollow, and ties Tikhonov elsewhere.
- GP is best on average accuracy; MFI on peaked and blob.

## 5a. Multi-seed statistics

Source: `experiments/multiseed.py` → `results/multiseed/summary.md` (default geometry) and
`results/multiseed_tcv/summary.md` (TCV). Potts and I-sparse (threshold 0.2) are compared against all baselines on
the same problems. MAP annealing is skipped. Win rates are paired comparisons of rel-L2, with an exact two-sided
sign test.

### Default geometry

50 problems: 4 phantoms × 5 noise seeds, plus 30 random fields. Values are mean rel-L2, with 95% coverage in
parentheses.

| Group | Tikhonov (tuned) | MFI | GP | Potts | I-sparse |
|---|---|---|---|---|---|
| Peaked | 0.285 | **0.210** | 0.228 (0.55) | 0.273 (0.97) | 0.281 (0.93) |
| Hollow | 0.653 | 0.790 | **0.553** (0.67) | 0.569 (0.80) | 0.630 (0.58) |
| Blob | 0.777 | **0.420** | 0.779 (0.86) | 0.620 (0.90) | 0.646 (0.54) |
| Edge | 0.604 | **0.602** | 0.732 (0.81) | 0.608 (0.88) | 0.612 (0.79) |
| Random (30) | 0.316 | 0.289 | **0.270** (0.89) | 0.319 (0.94) | 0.314 (0.90) |

Paired win rates over all 50 problems:

| Comparison | Wins / losses | p |
|---|---|---|
| Potts vs tuned Tikhonov | 29 / 21 | 0.32 (not significant) |
| Potts vs GP | 14 / 36 | 0.003 |
| Potts vs MFI | 8 / 42 | 1e-6 |
| I-sparse vs Tikhonov | 30 / 20 | 0.20 |
| I-sparse vs GP and MFI | 11 / 39 each | |

Every Potts and I-sparse run converges (max R-hat ≤ 1.004).

Time, on a heavily loaded machine (load 25–55 from other jobs): Potts 35–43 s, I-sparse 8–15 s.

### TCV geometry

18 problems: 5 phantoms × 2 seeds, plus 8 random fields. 1,148 pixels, 120 chords.

| Group | Tikhonov (tuned) | MFI | GP | Potts | I-sparse |
|---|---|---|---|---|---|
| Peaked | 0.299 | **0.094** | 0.157 (0.96) | 0.191 (0.96) | 0.309 (0.80) |
| Hollow | 0.487 | 0.391 | 0.638 (0.75) | **0.372** (0.92) | 0.518 (0.72) |
| Blob | 0.699 | **0.400** | 0.677 (0.78) | 0.558 (0.93) | 0.693 (0.68) |
| Edge | 0.685 | **0.479** | 0.668 (0.84) | 0.537 (0.90) | 0.727 (0.79) |
| Divertor | 0.403 | **0.387** | 0.399 (0.97) | 0.422 (0.94) | 0.425 (0.89) |
| Random (8) | 0.219 | 0.183 | **0.151** (0.92) | 0.211 (0.98) | 0.219 (0.97) |

Paired win rates over all 18 problems:

| Comparison | Wins / losses | p |
|---|---|---|
| Potts vs tuned Tikhonov | 13 / 5 | 0.10 |
| Potts vs GP | 6 / 12 | 0.24 |
| Potts vs MFI | 3 / 15 | 0.008 |

Every run converges. Potts takes 81–93 s here, which is over the 60 s target, on a loaded machine and with 1,148
pixels.

### Reading

- **Calibration is the robust result.** Potts covers 0.80–0.97 on the default geometry and 0.90–0.98 on TCV.
  Its worst group is hollow on the default geometry (0.80). GP is badly over-confident on default peaked (0.55) and
  hollow (0.67), and on TCV hollow and blob (0.75–0.78).
- **Accuracy: Potts ≈ tuned Tikhonov, below GP and MFI on average.** Potts beats GP on default blob and on TCV hollow,
  blob and edge. MFI is the most accurate method overall, especially on TCV.
- **I-sparse gives accurate means but miscalibrated uncertainty.** Its 95% coverage drops to 0.54–0.80 on structured
  phantoms. Dropping couplings narrows the posterior, so it is not a substitute for the exact posterior.

## 5b. Ablations (M4)

Source: `experiments/m4_ablations.py` → `results/m4_ablations/ablations_table.md`. Peaked phantom, full schedule.

**Number of levels K** (rel-L2, then 95% coverage):

| K | Potts | I-chain |
|---|---|---|
| 4 | 0.322, 0.86 | 0.361 |
| 8 | 0.258, 0.98 | 0.300 |
| 16 | 0.278, 0.98 | 0.326 (R-hat 1.11) |

**I-chain τ/Δz:**

| τ/Δz | rel-L2 | R-hat |
|---|---|---|
| 0.1 | 0.435 | ∞ |
| 0.25 | 0.395 | ∞ |
| 0.5 | 0.302 | ∞ |
| 1 | 0.337 | 3.9 |
| 2 | 0.434 | 1.09 |

At τ/Δz = 2 the chain converges but is biased: this is the tempered-likelihood regime the theory predicts.

**I-sparse threshold** (all runs converge, R-hat 1.00):

| Threshold | Colours | rel-L2 | 95% coverage | Time |
|---|---|---|---|---|
| 0.01 | 231 | 0.263 | 0.975 | 52 s |
| 0.2 | 49 | 0.261 | 0.949 | 9 s |

Dropping weak couplings costs no accuracy here and cuts colours and time about 5×. This is the most practical
variant found, but it is approximate and still not bounded-degree.

**Noise level** (rel-L2, Potts vs tuned Tikhonov):

| Noise | Potts | Tikhonov |
|---|---|---|
| 1% | 0.239 | 0.265 |
| 3% | 0.258 | 0.263 |
| 5% | 0.277 | 0.263 |
| 10% | 0.325 | 0.265 |

Potts wins at low noise and loses at high noise. Its coverage stays 0.98 throughout.

## 6. Where the gain comes from: positivity, the bound, or discreteness?

Source: `experiments/route2_ablation.py` → `results/route2_ablation/`.

**Full run: 4 phantoms + 3 random fields, mean over 7 problems:**

| Method | rel-L2 | 95% coverage |
|---|---|---|
| Gaussian (= tuned Tikhonov mean) | 0.476 | 0.93 |
| Truncated ≥ 0 (positivity only) | **0.420** | 0.91 |
| Truncated [0, ε_max] | 0.449 | 0.89 |
| Potts K = 4 | 0.461 | 0.80 (not converged) |
| Potts K = 7 / 8 / 9 | 0.430 / 0.432 / 0.436 | 0.91–0.92 |
| Potts K = 16 | 0.441 | 0.91 |

Positivity alone gives the largest gain. Discreteness adds nothing, and K = 7–9 are equivalent, so there is no
special K = 8 effect.

**Earlier 3-problem run.** Every row uses the same evidence λ and the same ε_max:

| | Peaked | Hollow | Random |
|---|---|---|---|
| Gaussian (exact) | 0.263 | 0.649 | **0.276** |
| Truncated ≥ 0 | 0.263 | 0.543 | 0.283 |
| Truncated [0, ε_max] | 0.296 | **0.536** | 0.305 |
| Potts K = 4 / 8 / 16 / 32 | 0.322 / 0.255 / 0.280 / 0.289 | 0.608 / 0.554 / 0.543 / 0.541 | 0.310 / 0.289 / 0.299 / 0.302 |
| Log-Laplace (after Ueda 2024) | 0.325 | 0.843 | 0.308 |

**Reading:**
- **Hollow:** positivity explains the whole gain.
- **Peaked:** the K = 8 advantage fades as K grows, so it looks like level alignment rather than a benefit of discreteness. The upper bound hurts because it clips the peak.
- **Random field:** no constraint beats the plain Gaussian.

**Sampler notes:**
- The truncated Gaussian is sampled by exact coordinate Gibbs and reaches R-hat ≤ 1.01 in 300 sweeps.
- `jax.random.truncated_normal` fails in the tails, so a tail-safe sampler is used instead.

## 7. I-chain: noise accounting and compensation

Sources: `tomo/ebm_chain.py` docstring; `tests/test_ebm_chain.py`.

**Effective noise.** With continuous z, integrating out the chain gives chord variance s² + Σ_k τ_k².

**Variance compensation.** Setting s² = σ² − Σ τ² makes the likelihood exact whenever Σ τ² < σ². On a toy chord, exact dynamic programming confirms the compensated variance ratio is 1.0000 (the uncompensated one is 1.45).

**Discrete-z rounding error after compensation:**

| τ/Δz | Rounding error |
|---|---|
| 0.5 | 4e-2 |
| 0.75 | 2e-4 |
| 1.0 | 2e-6 |

**Rigidity.** When τ ≪ Δz, single-bit moves cannot follow a pixel change, and the chain freezes. Below τ/Δz ≈ 0.6 the discrete chain is 10–16× slower than predicted; at 0.25 it is glassy, with 26% of pixels frozen.

**Local z windows** centred on the warm-start partial sums bring Δz/σ down to about 0.55 at K_z = 16 and 0.25 at K_z = 32.

**Compensation needs a fine z grid:**

| K_z | Σ τ²/σ² at τ/Δz = 0.75, uncompensated → compensated | Chords clamped |
|---|---|---|
| 16 | 5.8 → 4.9 | 100% |
| 32 | 1.36 → 0.47 | 78% |

**K_z needed for an exact likelihood under the evidence λ** (`experiments/kz_inflation.py` → `results/kz_inflation/`).
Inflation is the chord-variance excess after compensation, on peaked:

| K_z | I-chain inflation | I-chain spins / max degree | I-tree inflation | I-tree spins / max degree |
|---|---|---|---|---|
| 32 | 5.5 | 74k / 716 | 2.2 | 72k / 468 |
| 64 | 0.72 | 145k / 1,420 | 0.12 | 141k / 916 |
| 128 | 0.005 | 287k / 2,828 | 0 | 278k / 1,812 |

Reaching ≤ 10% inflation needs K_z ≈ 128 for the chain and ≈ 64 for the tree. At that size the degree already
exceeds that of I-dense (2,554) for the chain, and is a third of it for the tree, so the bounded-degree advantage is
largely gone. Calibration, exactness and low degree are in direct tension.

**The calibrated prior makes it worse.** The softer evidence λ widens the windows. With compensation at K_z = 32, chord variance is inflated by 6.5× for the chain and 3× for the tree, against 1.5× and 1.05× under the discrepancy λ.

## 8. I-tree

Source: `tomo/ebm_tree.py`, plus the comparison runs: 7,000 sweeps, 16 chains, half random starts, discrepancy λ.

Compared with I-chain, I-tree has:
- slightly better accuracy and coverage on all 4 cases;
- 2–3× faster burn-in;
- lower max degree (468 vs 716) but more colours (114 vs 76).

It still does **not** converge. The hoped-for change from n² to (log n)² mixing does **not** appear in the measurements.

## 9. The mixing barrier

Sources: `tomo/mixing_theory.py`, `docs/mixing_theory.md`, `experiments/mixing_theory.py` → `results/mixing_theory/`.

**Theory.** Exact linear-Gaussian theory gives the colour-block Gibbs spectral radius and the autocorrelation time of any linear quantity. On the default problem, the slowest-mode autocorrelation is:

| Model | Sweeps |
|---|---|
| Dense | 78 |
| I-chain | 11,350 |
| I-tree | 6,900 |

**Validation.**
- The theory matches an exact simulation of the linear Gibbs chain to within 5–20%.
- The discrete samplers run 5–7× *faster* than the continuous prediction at τ/Δz = 0.75, because the discrete posterior is narrower.
- The ratio of embedded to dense autocorrelation is predicted within about 2×.

**Proposition (structural bound).** Take an exact embedding with chord-private auxiliaries and a diagonal per-chord pixel precision (no pixel–pixel edges). Then

tr Λ_i ≥ ‖t_i‖₁² / σ_i² = n_eff,i × (the dense Gibbs value).

So the data-augmentation slowdown is at least about n_eff/f times dense Gibbs, where f = Σ τ²/σ² ≤ 1. I-tree violates the diagonality assumption, because sibling pixels are coupled; its measured slowdown is still 1.6× worse than the chain's.

**Further measured effects:**
- The slowdown scales as 1/λ, so the calibrated prior mixes about 8× slower than the stiff one.
- Compensation changes the rate by at most 2%.
- Relaxation time grows with chord length as n^1.94 (chain) and n^1.23 (tree).

**Remedies** (16×16 problem; dense reference T = 42 sweeps):

| Remedy | Result |
|---|---|
| Larger τ, still exact (f → 1) | Saturates at about 1.3e4 sweeps |
| τ beyond the exact limit (f > 1) | Biased; it is just a tempered likelihood. Relaxation time × noise inflation ≈ 1e4 |
| Tempering the auxiliaries | Same as changing τ |
| Non-centred auxiliaries | Reduces to dense Gibbs |
| Grouping g pixels per auxiliary | T ∝ g^-1.9, but degree grows ∝ g |
| Over-relaxation | Best target-preserving option (chain 2.5e4 → 1e3), still 25–45× dense, and needs non-binary updates |

**Conclusion:** no remedy that works with binary single-spin updates on a fixed sparse graph closes the gap.

## 10. Hardware fit

Sources: `tomo/embed.py`, `experiments/embed_report.py`, `experiments/embed_sampling.py`.

**Embedding cost.** Copy-node embedding to degree ≤ 16 multiplies the spin count by 4.6× (I-chain, K_z = 16) up to 38× (I-dense).
I-tree embeds to 190,889 spins at K_z = 16 (5.1×) and 656,301 at K_z = 32 (9.1×). I-chain at K_z = 32 needs 621k spins, more than Z1's 270k. None of the embedded graphs is 2-colourable.

**Sampling the embedded model fails** (tiny problem):

| J_F | Outcome |
|---|---|
| ≥ 8, including the exactness value (~100× a typical coupling) | Frozen at the start |
| 1–4 | 14–32% of copy bonds broken; posterior means off by 1–3 sd; autocorrelation 9–14× longer |

**Binary level encoding** cuts I-chain to about 24k spins at degree 16 (2.1× overhead). The cost is 4–8 more bits of coupling dynamic range on top of the existing 12–13 bits; at 8 bits, 32–87% of couplings round to zero. Not yet sampled.

## 11. Energy and latency

Sources: `tomo/energy.py` presets, `experiments/m5_energy.py`.

**Hardware figures in the presets:**
- Specification: 1.3 fJ per spin update, 100 ns per block update.
- Jelinčič 2025: about 2 fJ.
- Z1 (arXiv:2608.01615): 7.09 fJ per p-bit per Gibbs cycle, 20 ns sweeps on a 2-colour graph, readout 1.69 pJ per node and 25 µs per frame.

**I-chain at K_z = 16**, 16 chains, 2,500 placeholder sweeps:

| Preset and layout | Energy / frame | Latency / frame |
|---|---|---|
| Specification, logical model | 2.0 µJ | 11 ms |
| Z1, logical model, with readout | 12 µJ | 1.1 ms |
| Z1, embedded at degree 16, with readout | 55 µJ | 0.45 ms |

These numbers are optimistic. Our graphs are not 2-colourable, so the Z1 rows charge 10 ns per colour, and they ignore that the embedded models never reach convergence.

## 12. Real TCV geometry

Sources: `tomo/tcv.py`, `experiments/tcv_check.py` → `results/tcv_check/`.

**Data.** 120 lines of sight, the vessel outline and SOLPS phantom components, vendored from Hamm et al.'s MIT-licensed `real-time-tomo-prad` (see `tomo/data/tcv/NOTICE.md`). The 1,000 SOLPS phantoms are downloaded at runtime with `python -m tomo.tcv --fetch`.

**Grid.** 20×60 pixels, 1,148 of them active; 97.9% are crossed by at least one chord.

**Cross-check.** Our line-length geometry matrix correlates with the authors' etendue matrix at a median of 0.89 per chord.

**Results so far:**

| Phantom | Tikhonov (evidence) | GP |
|---|---|---|
| Peaked | 0.31 | 0.16 |
| Random | 0.19 | 0.11 |
| Divertor | 0.41 | 0.43 |

See §5a for Potts and I-sparse on TCV.

## 13. Open items

- Clean timings on an idle machine. Many of the later runs shared the CPU with unrelated jobs (load 25–55).
- Speed up Potts on TCV, which takes 81–93 s with 1,148 pixels.
- Sample the binary-encoded models.
- Measure laptop power with `powermetrics` instead of assuming 20 W.
