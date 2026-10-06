# Technical Specification: Real-Time Plasma Tomography on a Thermodynamic Sampling Unit

Oct 6, 2026 · @Rishabh

## Overview

The project reconstructs a 2D plasma emissivity image from line-integrated sensor data by sampling an energy-based model in thrml, then estimates what that would cost on Extropic's thermodynamic sampling unit (TSU). Everything runs on a MacBook Pro with synthetic data; no TSU hardware is required.

**Motivation.** Tokamak control needs radiation profiles in about 1 ms. Classical inversions give a point estimate quickly but no uncertainty; Bayesian methods give uncertainty but are slow. Sampling hardware could, in principle, deliver posterior samples at control-loop speed.

**In scope**

- Static 2D reconstructions of a poloidal cross-section from 64 to 128 synthetic bolometer chords.
- Two formulations: a Potts model (algorithmic reference) and a binary Ising model (hardware-faithful).
- Classical baselines, uncertainty calibration, and a documented TSU energy and latency estimate.

**Out of scope**

- Running on physical TSU hardware, 3D or time-resolved tomography, and magnetic equilibrium reconstruction.
- Real tokamak data, which is a stretch goal only.

**Success criteria**

| Criterion | Target |
| --- | --- |
| Reconstruction error (relative L2) | Within 1.2× of the tuned Tikhonov baseline |
| Uncertainty calibration | 95% intervals cover 90 to 98% of true pixels |
| Laptop runtime, 32×32 grid | Under 60 s per reconstruction on CPU |
| Hardware-faithful model | Ising variant with bounded node degree, documented |
| TSU estimate | Energy and latency per frame with stated assumptions and sensitivity range |

## Background

Tomography here is an ill-posed linear inverse problem: about 100 measurements must determine about 800 unknown pixel values, so a prior is essential. Framing it as sampling a Boltzmann distribution makes it a natural fit for TSU hardware.

**The measurement model.** Each bolometer chord measures emissivity integrated along its line of sight. Stacking chords gives

```latex
b = T\,\varepsilon + n, \qquad n \sim \mathcal{N}(0, \sigma^2 I)
```

where ε is the pixel emissivity vector, T is the geometry matrix (T\_ij = length of chord i inside pixel j), and n is detector noise.

**Established methods.** Fusion groups commonly use Tikhonov regularization, minimum Fisher information (MFI) iterations, and Gaussian process tomography, which is fully Bayesian and gives uncertainty. These are the baselines this project must match.

**What a TSU does.** A TSU is an array of probabilistic bits (p-bits) built from sub-threshold analog CMOS. Each p-bit flips randomly with a bias set by its neighbors, so the array performs Gibbs sampling from

```latex
p(s) \propto \exp\big(-\beta E(s)\big), \qquad E(s) = -\sum_i h_i s_i - \sum_{i<j} J_{ij} s_i s_j
```

with couplings J that are sparse and mostly local on chip. thrml simulates this process in JAX using block Gibbs sampling.

**Why the mapping is interesting.** The tomography posterior is quadratic in ε, so with a linear bit encoding it becomes exactly an Ising energy. The hard part is that chords cross the whole plasma, creating long-range couplings the hardware does not natively support; solving that is the main technical contribution.

## Development environment

A MacBook Pro with Apple Silicon and 16 GB of RAM or more is sufficient; JAX on CPU handles every problem size in this spec. The largest model has about 6,000 spins, and the dense geometry matrix is under 5 MB.

**Setup**

```bash
brew install uv
uv venv --python 3.11 && source .venv/bin/activate
uv pip install "jax[cpu]" thrml numpy scipy matplotlib scikit-image pyyaml pytest
```

- Use the JAX CPU backend. The `jax-metal` GPU plugin is experimental and has incomplete operation coverage, so it is not worth the debugging cost at this scale.
- Run independent chains in parallel with `jax.vmap`, which uses all performance cores.
- Set `XLA_FLAGS=--xla_force_host_platform_device_count=8` if you want `pmap` across cores instead.
- Measure laptop energy with `sudo powermetrics --samplers cpu_power -i 200` during runs. This gives a real-hardware comparison point for the TSU estimate.
- Fix random seeds through `jax.random.key` and store every run's config file next to its outputs.

**Repository layout**

```
plasma-tsu-tomo/
├── configs/            # YAML: grid, cameras, noise, model, schedule
├── tomo/
│   ├── geometry.py     # grid, mask, chord fans, Siddon ray tracing -> T
│   ├── phantoms.py     # synthetic emissivity profiles
│   ├── forward.py      # b = T eps + noise, fine-grid generation
│   ├── baselines.py    # Tikhonov, MFI, Gaussian process
│   ├── ebm_potts.py    # Potts formulation in thrml
│   ├── ebm_ising.py    # domain-wall Ising: sparse and chord-chain
│   ├── sampling.py     # blocks, schedules, annealing, chains
│   ├── metrics.py      # error, SSIM, calibration, diagnostics
│   └── energy.py       # TSU energy and latency model
├── experiments/        # one script per milestone result
├── tests/
└── report/             # figures and write-up
```

## System architecture

The pipeline runs top to bottom in one Python package. Every stage is a pure function of its config and random seed, so any figure can be regenerated with a single command.

&#91;embedded content: pipeline · 7 stages, 2 side branches\]

The EBM builder is highlighted because it holds the project's main contribution; the baselines and energy estimator branch off the same data and sampler run so every comparison is like for like.

## Forward model and synthetic data

Synthetic data is generated on a 128×128 grid and reconstructed on a coarser 32×32 grid, which avoids the "inverse crime" of testing a method on data produced by its own discretization.

| Parameter | Default | Stretch |
| --- | --- | --- |
| Reconstruction grid | 32×32 over a 1 m × 1 m box | 64×64 |
| Plasma mask | D-shaped boundary, \~800 active pixels | \~3,200 active pixels |
| Cameras | 3 fans: top, outboard, divertor | 4 fans |
| Chords per camera | 24 (72 total) | 32 (128 total) |
| Noise | Gaussian, 3% of signal + 1% of max floor | Add 2% calibration offset per camera |
| Data grid | 128×128 | 256×256 |

**Phantom library.** Each phantom tests a different failure mode:

1. Peaked core: a 2D Gaussian at the magnetic axis, the easy case.
2. Hollow profile: an annulus, as seen with core impurity depletion.
3. Off-axis blob: a small bright spot, mimicking a radiating impurity or magnetic island.
4. Edge radiation: a bright band near the boundary, as in a detached divertor.
5. Random smooth fields: Gaussian random fields for statistical evaluation, 200 samples.

**Geometry matrix.** Compute T with Siddon's algorithm, which returns exact chord lengths through each pixel. Validate it by checking that Tε equals the analytic line integral for a uniform disk to within 0.5%. Store T as a dense float32 array; it also feeds the coupling computation in the next section.

## Energy-based model formulation

The posterior is written as a quadratic energy over discretized pixel levels, encoded into binary spins, and then restructured so each spin has a bounded number of neighbors. Three variants are built, from easiest to most hardware-faithful.

**Discretization.** Each pixel j takes a level x\_j in {0, …, K−1}, with ε\_j = Δ·x\_j and Δ = ε\_max / (K−1). Default K = 8; positivity is built in.

**Energy.** A Gaussian likelihood plus a nearest-neighbor smoothness prior gives

```latex
E(x) = \frac{1}{2\sigma^2}\,\lVert b - \Delta T x \rVert^2 + \frac{\lambda \Delta^2}{2} \sum_{\langle j,k \rangle} (x_j - x_k)^2
```

Expanding gives a quadratic form with matrix Q = Δ²(TᵀT/σ² + λL), where L is the grid Laplacian, and linear field c = Δ Tᵀb/σ². Sampling at β = 1 gives the posterior; annealing to large β gives the maximum a posteriori (MAP) image.

**Variant P: Potts reference.** Each pixel is a thrml `CategoricalNode` with K states, and pairwise factors hold Q. This is the algorithmic ground truth, mirroring the Potts model in Extropic's codon\_opt.

**Variant I: domain-wall Ising.** Each pixel becomes K−1 binary spins with x\_j = Σ\_m u\_{j,m}, valid only when the bits are ordered (1,1,…,0,0). A penalty enforces the order:

```latex
E_{\text{dw}} = A \sum_j \sum_{m=1}^{K-2} u_{j,m+1}\,(1 - u_{j,m})
```

Because x is linear in the bits, the total energy stays exactly quadratic and therefore Ising. Choose A larger than the biggest single-level energy change, and track the fraction of invalid states as a diagnostic. The default model has about 800 × 7 = 5,600 spins.

**The connectivity problem.** TᵀT couples every pair of pixels sharing a chord. With 72 chords, each pixel couples to roughly 100 to 200 others, or about 1,000 neighbors per spin after encoding. TSU hardware favors sparse, local couplings, so Variant I is simulated as two further forms:

- **I-sparse:** Drop couplings below a magnitude threshold or beyond a radius r, and measure the resulting reconstruction bias against Variant P. This is simple but approximate.
- **I-chain (main contribution):** Replace each chord's long-range sum with a chain of running partial sums along the chord. For chord i crossing pixels j\_1, …, j\_n, add auxiliary levels z\_{i,k} with local penalties, and put the data term only on the chain's end:

```latex
E_{\text{chain}} = \sum_i \Big[ \sum_{k=1}^{n} \frac{\big(z_{i,k} - z_{i,k-1} - T_{i j_k}\,\Delta x_{j_k}\big)^2}{2\tau^2} + \frac{(b_i - z_{i,n})^2}{2\sigma^2} \Big]
```

Every term now touches only adjacent chain elements and one pixel, and the chain lies physically along the chord. Node degree becomes independent of grid size, at the cost of about 2,300 extra auxiliary variables (roughly 35,000 spins at 16 levels each). As τ → 0 it recovers the exact posterior; τ trades accuracy against mixing speed and must be swept experimentally.

## Sampling with thrml

All variants are sampled with thrml's block Gibbs machinery: `SpinNode` or `CategoricalNode` variables, `IsingEBM` for the Ising forms, `Block` for update groups, and `sample_states` with a `SamplingSchedule`. API names were checked against the thrml repository as of October 2026 (v0.1.4); confirm details at docs.thrml.ai.

**Spin convention.** thrml spins take values ±1, while the formulation uses bits u in {0,1}. Substitute u = (s + 1)/2 and fold the resulting constants into the biases h and weights J before building `IsingEBM(nodes, edges, biases, weights, beta)`.

**Blocking.** Spins in the same block must not be coupled to each other.

| Variant | Coloring | Blocks |
| --- | --- | --- |
| P (Potts) | Pixel checkerboard | 2 |
| I-sparse | Greedy graph coloring of the sparsified graph | \~10–40, measured |
| I-chain | Pixel checkerboard × bit index, plus chain parity × bit index | 2(K−1) + 2(K\_z−1) ≈ 44 |

Bits of the same pixel are coupled to each other, which is why the bit index multiplies the color count. Report the block count, since it sets the per-sweep latency on hardware.

**Schedules**

1. Posterior sampling: β = 1, `SamplingSchedule(n_warmup=2000, n_samples=500, steps_per_sample=10)`.
2. MAP estimate: geometric annealing of β from 0.1 to 50 over 30 steps with 20 sweeps per step, scaling weights by β as codon\_opt does.
3. Hard cases (I-chain with small τ): parallel tempering with 8 replicas if single-chain mixing stalls.

Run 16 independent chains with `jax.vmap` over random keys, initialized with `hinton_init` or with the Tikhonov solution rounded to levels (a warm start).

**Readout and diagnostics.** Decode bits to levels, then to emissivity. Report the posterior mean, per-pixel standard deviation, and annealed MAP image. Track the energy trace, the split-chain R-hat statistic on pixel means (target below 1.05), and the fraction of invalid domain-wall states (target below 0.1%).

## Baselines and evaluation

Every TSU-style result is reported next to three classical baselines on the same phantoms and noise draws. The honest headline comparison is I-chain against Gaussian process tomography, since both provide uncertainty.

**Baselines**

1. **Tikhonov:** Solve (TᵀT/σ² + λL)ε = Tᵀb/σ² directly, with λ chosen by generalized cross-validation. It is fast, gives a point estimate only, and can go negative.
2. **Minimum Fisher information:** An iterative, reweighted Tikhonov scheme widely used on tokamaks. It is positive and sharper at edges.
3. **Gaussian process tomography:** A closed-form Gaussian posterior with a squared-exponential prior whose hyperparameters are fit by marginal likelihood. This gives the uncertainty reference.

**Metrics**

| Metric | Definition | Why it matters |
| --- | --- | --- |
| Relative L2 error | ‖ε̂ − ε\_true‖ / ‖ε\_true‖ | Overall accuracy |
| SSIM | Structural similarity index | Shape fidelity |
| Peak position error | Distance between true and reconstructed maxima, in cm | Locating impurities or islands |
| Total power error | Relative error of Σε, area-weighted | The quantity control systems use |
| Reduced χ² | Residual of b − Tε̂ normalized by σ² | Detects over- or under-fitting |
| Interval coverage | Fraction of true pixels inside the 68% and 95% intervals | Uncertainty calibration |
| Sweeps to converge | Gibbs sweeps until R-hat < 1.05 | Sets hardware latency |
| Laptop wall time | Seconds per reconstruction, CPU | Practicality of the simulation |

**Experiments**

- Accuracy across the phantom library and 200 random fields, for every variant and baseline.
- Ablations on K (4, 8, 16 levels), τ for I-chain, the sparsification threshold for I-sparse, and noise level (1% to 10%).
- Report failures explicitly, such as phantoms where discretization or slow mixing loses to Tikhonov.

## Hardware energy and latency estimation

A first estimate puts one I-chain posterior reconstruction at about 2 µJ but about 11 ms on a TSU, so energy looks excellent while latency misses the 1 ms target unless sweeps or block counts drop. Both numbers come from the simple model below and must be reported with their assumptions.

**Energy model**

```latex
E_{\text{frame}} = N_{\text{spins}} \times N_{\text{sweeps}} \times N_{\text{chains}} \times E_{\text{cell}}
```

E\_cell ≈ 1.3 fJ per spin per Gibbs step, the figure Extropic's codon\_opt paper uses and describes as validated against prototype measurements. It includes the random number generator (about 350 aJ), biasing, clocking, and communication.

**Latency model**

```latex
t_{\text{frame}} \approx N_{\text{sweeps}} \times N_{\text{blocks}} \times t_{\text{update}}
```

Take t\_update ≈ 100 ns, the random number generator's reported decorrelation time, and assume chains run in parallel on separate chip area.

**Worked estimate (I-chain, defaults)**

| Quantity | Value |
| --- | --- |
| Spins | \~40,600 (5,600 pixel + 35,000 chain) |
| Sweeps (warmup + sampling) | 2,500 posterior; 600 annealed MAP |
| Chains | 16 |
| Blocks | 44 |
| Energy per frame | \~2.1 µJ posterior; \~0.5 µJ MAP |
| Latency per frame | \~11 ms posterior; \~2.6 ms MAP |

Replace these placeholders with measured sweep counts and block counts, and show a sensitivity range with E\_cell from 0.5× to 3× and t\_update from 50 to 500 ns.

**Fair comparisons.** A precomputed Tikhonov inversion is a single 800×72 matrix-vector product, about 10⁵ operations, so the TSU will not beat it on point estimates. For the purely linear-Gaussian model, the Gaussian process posterior is also closed form and cheap. The TSU's real case is non-Gaussian posteriors, such as enforced positivity, discrete levels, sparsity, or flux-aligned priors, where conventional methods need MCMC. So compare against the same Gibbs sampler running on the MacBook, with energy measured by `powermetrics`, and against a GPU MCMC estimate from operation counts. Stating this boundary clearly will make the work more credible, not less.

## Milestones

The plan assumes about 10 hours a week for 10 weeks on a single MacBook Pro. Each milestone ends with a gate that must pass before the next begins.

&#91;embedded content: roadmap · 6 milestones with exit gates\]

M3 is highlighted because it carries the novel work; if time runs short, cut stretch goals and ablations, never the gates.

## Risks, open questions, and deliverables

The biggest risk is slow mixing in I-chain, because tight chain penalties create long, stiff dependencies along each chord.

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Slow mixing in I-chain at small τ | Latency estimate balloons | Sweep τ, warm-start from Tikhonov, parallel tempering |
| Discretization error with K = 8 | Accuracy below baselines | Ablate K; report the accuracy–spin-count trade-off |
| Domain-wall violations | Corrupted samples | Tune A; log invalid-state fraction every run |
| Coupling precision on hardware | Simulated results not reproducible on chip | Quantize J and h to 4–8 bits in simulation and re-measure |
| Linear-Gaussian case is too easy | TSU shows no advantage | Add a non-Gaussian prior where closed-form methods fail |
| Overclaiming energy savings | Loss of credibility | Publish assumptions, sensitivity ranges, and losing cases |

**Open questions**

- What node degree and coupling precision does current Extropic hardware actually support? Check their papers and ask the team directly.
- Can bit-level degree in I-chain be reduced further, for example by binary rather than thermometer encoding of the chain variables?
- Is there public tokamak bolometer data with a known geometry for a final real-data test?
- Could torx, Extropic's differentiable stochastic-circuit library, learn λ and τ end to end?

**Deliverables**

- [ ] Open-source repository with tests, configs, and one-command reproduction of every figure
- [ ] Results table comparing all variants and baselines across phantoms
- [ ] Energy and latency estimate with sensitivity analysis
- [ ] Short write-up of 4 to 6 pages, including failure cases
- [ ] A brief message to the Extropic team linking the repository and summarizing the I-chain idea

## Sources

- [thrml repository](https://github.com/extropic-ai/thrml): API names, installation, and the Ising chain example, checked October 2026.
- [codon\_opt repository](https://github.com/extropic-ai/codon_opt): Potts and domain-wall Ising template, plus the E\_cell ≈ 1.3 fJ and 100 ns random number generator figures from its paper.
- [An efficient probabilistic hardware architecture for diffusion-like models](https://arxiv.org/abs/2510.23972): Extropic's hardware energy model.
- Tikhonov, MFI, and Gaussian process tomography descriptions are from general knowledge, not looked up; verify against the fusion literature before citing.
