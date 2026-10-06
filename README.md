<div align="center">

# plasma-tsu-tomo

**Tokamak plasma tomography as Boltzmann sampling, built for a thermodynamic sampling unit**

[![tests](https://github.com/Lulzx/plasma-tsu-tomo/actions/workflows/tests.yml/badge.svg)](https://github.com/Lulzx/plasma-tsu-tomo/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.11-blue)
![jax](https://img.shields.io/badge/JAX-CPU-orange)
![thrml](https://img.shields.io/badge/thrml-0.1.4-7b3fe4)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

<img src="docs/figures/geometry_fans.png" width="92%" alt="32x32 reconstruction grid with a D-shaped plasma, 72 bolometer chords in three fans, and per-pixel chord coverage">

</div>

Tokamak control needs radiation profiles in about a millisecond. Classical inversions are fast but give no uncertainty.
Bayesian methods give uncertainty but are slow. This repository reconstructs 2D plasma emissivity from line-integrated
bolometer data by **sampling an energy-based model with [thrml](https://github.com/extropic-ai/thrml)**. It also estimates
what each reconstruction would cost in energy and latency on Extropic's p-bit hardware.

The main contribution is **I-chain**, an Ising formulation in which each node couples to a bounded number of local
neighbours. Real tomography has long-range couplings because chords cross the whole plasma, and I-chain removes them,
which is what the hardware needs.

> Everything runs on a laptop CPU with synthetic data. No TSU hardware is required. Status: research prototype, in progress.

## The problem in one picture

```
b = T ε + n,      n ~ N(0, σ²)          72 chord measurements  →  806 unknown pixels   (11× under-determined)
```

The posterior with a smoothness prior is quadratic in ε. Discretise each pixel to K levels and encode the levels as
domain-wall (thermometer) bits, and the posterior becomes **exactly an Ising model**:

```math
E(x) = \tfrac{1}{2}\lVert (b - \Delta T x)/\sigma \rVert^2 + \tfrac{\lambda\Delta^2}{2}\sum_{\langle j,k\rangle}(x_j-x_k)^2
\;\;\Longrightarrow\;\; E(s) = -\textstyle\sum_i h_i s_i - \sum_{i \lt j} J_{ij} s_i s_j
```

The catch is that $T^\top T$ couples every pair of pixels that share a chord, so each spin ends up with **hundreds to
thousands of neighbours**. TSU hardware wants sparse, local couplings.

## Four formulations, from reference to hardware-faithful

| Variant | Idea | Spins | Max degree | Gibbs blocks |
|---|---|---:|---:|---:|
| **P** Potts | `CategoricalNode` per pixel, pairwise factors carry Q | 806 (K=8 states) | ~74 pixels | 48 |
| **I-dense** | Domain-wall Ising, all couplings kept | 5,642 | 2,554 | 315 |
| **I-sparse** | Drop weak or long-range pixel couplings | 5,642 | lower | ~140–196 |
| **I-chain** ⭐ (K_z=16) | Running partial sums along each chord | 38,852 | 364 | **44** |
| **I-chain** ⭐ (K_z=32, default) | Same, finer z grid so variance compensation applies | 74,276 | 716 | 76 |

**I-chain** swaps each chord's long-range sum for a chain of auxiliary partial sums $z_{i,k}$ that runs physically
along the chord:

```math
E_\text{chain} = \sum_i \Big[\sum_k \frac{(z_{i,k}-z_{i,k-1}-T_{ij_k}\Delta x_{j_k})^2}{2\tau^2} + \frac{(b_i - z_{i,n})^2}{2\sigma_i^2}\Big]
```

After this change every term touches only adjacent chain elements and one pixel. No pixel-to-pixel $T^\top T$ edges
remain. A structured colouring (pixel checkerboard × bit index, plus chain parity × bit index) needs 2(K−1) + 2(K_z−1)
blocks: 44 at K_z = 16, which is within 7 of the clique lower bound.

The chain has two known weak points:
- **Effective noise.** Integrating out a continuous chain shows that τ inflates the chord noise to $\sigma^2 + n\tau^2$.
- **Stiffness.** With discrete $z$ the chain freezes when $\tau \ll \Delta z$.

The usable range is therefore Δz/2 ≲ τ ≲ σ/√n. Two fixes widen it:
- **Local z windows** centred on the warm-start partial sums shrink Δz.
- **Variance compensation** shrinks the endpoint variance to $\sigma^2 - \sum_k \tau_k^2$, which makes the
  continuous-z likelihood exact for any admissible τ. On a toy chord, exact dynamic programming confirms a matched
  variance to 1e-4 at τ/Δz = 0.75. The condition $\sum_k\tau_k^2 < \sigma^2$ only holds once K_z ≥ 32, which is why that
  is the default.

The derivations are in the docstring of [`tomo/ebm_chain.py`](tomo/ebm_chain.py).

## Synthetic data

The data are generated on a 128×128 grid with its own geometry matrix and reconstructed on 32×32, so the method is never
tested on data produced by its own discretisation (the "inverse crime"). Noise is 3% of the signal plus a floor of 1% of
the maximum.

<img src="docs/figures/phantoms.png" width="100%" alt="Phantom library: peaked, hollow, off-axis blob, edge band, random field; fine and coarse truth">

| Geometry check | Value |
|---|---|
| Active pixels / chords | 806 / 72 (3 fans × 24) |
| Pixels per chord (mean / max) | 30.8 / 45 |
| Pixels never crossed by a chord | 0.7% |
| Siddon ray tracing vs analytic disk (summed length) | −0.017% error |

## Results so far

### Classical baselines (M2, complete)

The baselines ran on the four phantoms and on 200 Gaussian random fields, with identical noise draws for every method.

<img src="docs/figures/reconstructions.png" width="100%" alt="Truth and baseline reconstructions (Tikhonov GCV, tuned Tikhonov, MFI, GP mean and std) for five phantoms">

<p align="center"><img src="docs/figures/random_fields_boxplot.png" width="75%" alt="Relative L2 error and SSIM over 200 random fields per baseline"></p>

| Baseline, 200 random fields | Mean rel. L2 | GP coverage (68% / 95%) |
|---|---:|---:|
| Tikhonov, GCV λ | 0.336 | – |
| Tikhonov, tuned (discrepancy principle) | 0.353 | – |
| Minimum Fisher information | 0.296 | – |
| Gaussian-process tomography | **0.291** | 0.63 / 0.89 |

With only 72 chords, every method struggles on the hollow and edge profiles. The figure shows this honestly, and it is
the regime where a better prior should pay off.

### Energy-based models (M3, preliminary)

These are single-seed runs measured on a shared, heavily loaded machine. The full M3 tables come from `make m3`.

| Method | Peaked | Hollow | Uncertainty |
|---|---:|---:|:-:|
| Tikhonov, tuned | 0.289 | 0.65 | – |
| Minimum Fisher information | 0.200 | 0.81 | – |
| Gaussian-process tomography | 0.226 | 0.54 | ✓ |
| **Potts** posterior mean | 0.277 | 0.63 | ✓ |
| I-dense posterior mean | 0.278 | 0.63 | ✓ |
| **I-chain** (K_z=32, compensated) | 0.315 | 0.61 | ✓ |
| **I-chain** (K_z=32, compensated, 2 pixels per link) | 0.290 | – | ✓ |

**Targets from the spec, and where they stand:**

- ✅ **Accuracy within 1.2× of tuned Tikhonov:** met by Potts, I-dense and I-chain on the phantoms shown. GP and MFI are still more accurate.
- ✅ **Bounded-degree Ising model, documented:** I-chain.
- ⚠️ **Under 60 s per 32×32 reconstruction:** Potts meets it at about 52 s. The batched JAX sampler projects about 62 s for the I-chain full schedule, but only at K_z=16 and under load. The dense Ising variant is limited by its 315 sequential blocks.
- ❌ **95% intervals covering 90–98% of pixels:** not met. I-chain covers about 0.55–0.60, and GP 0.89.
- ❌ **Convergence (split R-hat below 1.05):** not met for I-chain or dense Ising. Mixing along the chains, which takes about n² sweeps per chord, is the main open problem.

## TSU energy and latency model

```math
E_\text{frame} = N_\text{spins}\,N_\text{sweeps}\,N_\text{chains}\,E_\text{cell},
\qquad t_\text{frame} \approx N_\text{sweeps}\,N_\text{blocks}\,t_\text{update}
```

The spec's worked I-chain estimate uses E_cell ≈ 1.3 fJ and t_update ≈ 100 ns. It gives about **2 µJ** and **11 ms** per
posterior frame. Energy looks excellent, but latency misses the 1 ms control target unless the sweep count or block count
drops.

> **Published hardware numbers differ.** The literature search ([`docs/related_work.md`](docs/related_work.md)) found
> these figures in Extropic's papers:
> - **Energy per cell:** about 2 fJ per cell in arXiv:2510.23972 v2, and a SPICE-based 7.09 fJ per p-bit per Gibbs cycle in arXiv:2608.01615.
> - **Readout:** 1.69 pJ per node and 25 µs per frame.
> - **Z1 chip:** a degree-16 graph.
>
> With those numbers, energy per frame rises to a few µJ to tens of µJ, which is still excellent. I-chain's degree of 364
> or more would need further embedding before it fits Z1. Updating `tomo/energy.py` to these figures is on the roadmap.

`make m5` recomputes both numbers from the *measured* spin counts, block counts and sweeps to convergence. It also
reports a sensitivity range of E_cell 0.5–3× and t_update 50–500 ns.

**Fair comparison.** A precomputed Tikhonov inversion is one matrix-vector product, and no sampler beats that on a point
estimate. Real-time Gaussian uncertainty is already achieved on TCV this way (Hamm et al., arXiv:2603.11856). The TSU's
real case is non-Gaussian posteriors (positivity, discrete levels, sparsity) and full 2D maps with calibrated
uncertainty, where conventional methods need MCMC.

## Related work and novelty

A literature search (October 2026, about 60 references, listed in [`docs/related_work.md`](docs/related_work.md)) found
the following.

**Not found in prior work:**
- Running partial-sum chains applied to a *weighted Gaussian data term*.
- The noise and stiffness analysis: σ² + nτ², variance compensation, and the τ window.
- Posterior *sampling* of tomography on p-bit hardware.
- Any Ising or sampling hardware used for fusion tomography.

**Known:**
- The general mechanism of auxiliary chains that carry partial sums: SAT sequential counters, p-bit adder chains, and
  sparse-QUBO constraint decomposition.
- Bounded-degree p-bit graphs built from copy nodes (Sajeeb et al. 2025).

**Closest comparator:** Extropic's own Gaussian-field reconstruction on Z1 (arXiv:2608.01615), which uses minor
embedding.

## Quickstart

```bash
git clone https://github.com/Lulzx/plasma-tsu-tomo && cd plasma-tsu-tomo
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e ".[dev]"

pytest -q          # full test suite
make quick         # smoke-run every experiment in a few minutes
make all           # reproduce every figure and table
```

| Milestone | Command | What it produces |
|---|---|---|
| M1 Geometry and phantoms | `make m1` | chord fans, phantom library, Siddon validation |
| M2 Classical baselines | `make m2` | Tikhonov / MFI / GP on 4 phantoms + 200 random fields |
| M3 EBM reconstructions | `make m3` | Potts, I-dense, I-sparse, I-chain vs baselines, plus diagnostics |
| M4 Ablations | `make m4` | K levels, τ, sparsification threshold, noise level |
| M5 TSU estimate | `make m5` | energy and latency per frame, sensitivity analysis |
| M6 Summary | `make m6` | `report/results_table.md`, figures |

Every script takes `--config` (merged over [`configs/default.yaml`](configs/default.yaml)), `--out`, and `--quick`. The
merged config is saved next to each run's outputs, and all seeds come from the config.

## Repository layout

```
configs/default.yaml   grid, plasma shape, cameras, noise, model, schedules, experiments
tomo/
  geometry.py          grid, D-shaped mask, chord fans, Siddon ray tracing → T
  phantoms.py          peaked / hollow / blob / edge / Gaussian random fields
  forward.py           fine-grid data generation, noise model
  baselines.py         Tikhonov (GCV, discrepancy), minimum Fisher information, GP tomography
  ebm_common.py        quadratic form, domain-wall encoding, bit QUBO
  ebm_potts.py         Variant P (thrml CategoricalNode)
  ebm_ising.py         I-dense and I-sparse
  ebm_chain.py         I-chain (main contribution)
  sampling.py          thrml-backed block Gibbs, annealing, parallel tempering, fast JAX backend
  metrics.py           relative L2, SSIM, peak error, χ², coverage, split R-hat
  energy.py            TSU energy and latency model, CPU/GPU comparisons
experiments/           m1–m6, one script per milestone
tests/                 pytest suite (exact-enumeration checks of every sampler)
docs/                  spec, module interfaces, figures
```

## How correctness is checked

- **Exact enumeration.** Every sampler (thrml and the JAX backend) is checked against exact marginals on small models.
- **Energy identities.** The QUBO → Ising conversion, the domain-wall encoding and the chain energy are each checked against the spec formula, evaluated directly.
- **Geometry.** Siddon chord lengths are checked against analytic line integrals.
- **No truth leakage.** Hyperparameters (λ, ε_max, A, τ) come only from the data and the Tikhonov solution.

## Where this departs from the spec

- **Gibbs blocks.** A 2-colour pixel checkerboard is not a valid blocking once $T^\top T$ is present. Potts needs 48 blocks and dense Ising 315. Only I-chain achieves the spec's 44.
- **Choice of λ.** "Tuned Tikhonov" uses the discrepancy principle (χ²/M = 1). With M = 72 ≪ N = 806, generalised cross-validation (GCV) interpolates the data (χ²/M ≈ 0.01).
- **I-chain limit.** τ → 0 does *not* recover the exact posterior when z is discrete; the chain becomes rigid instead. See the `ebm_chain.py` docstring for the usable τ window.
- **Timings.** The wall-clock times above were measured on a shared, loaded machine and will be re-measured in isolation.

## References

- [thrml](https://github.com/extropic-ai/thrml): block Gibbs sampling of probabilistic graphical models in JAX.
- [codon_opt](https://github.com/extropic-ai/codon_opt): the Potts and domain-wall Ising template, and the E_cell and RNG timing figures.
- [An efficient probabilistic hardware architecture for diffusion-like models](https://arxiv.org/abs/2510.23972): Extropic's hardware energy model.
- The full technical specification is in [`docs/spec.md`](docs/spec.md).

## License

MIT
