# Module interfaces (binding contract)

Every module in `tomo/` follows these signatures so modules can be written in parallel. Do not change
a signature listed here; add optional kwargs only. Pixel vectors are always over **active (masked) pixels**
in row-major order of the full grid (`np.flatnonzero(mask.ravel())`). All arrays are NumPy unless noted;
JAX is used inside samplers. Units: metres, emissivity in arbitrary units (W/m^3 nominal), chord data in units·m.

## tomo/config.py
- `load_config(path_or_dict) -> dict` loads YAML, deep-merges over `configs/default.yaml`; accepts dict overrides.
- `save_config(cfg, out_dir)` writes `config.yaml` next to outputs.

## tomo/geometry.py
- `@dataclass Grid: n:int, xmin, xmax, ymin, ymax` with properties `dx, dy, xc (n,), yc (n,), X, Y (n,n meshgrid of centers, indexing='xy', row = y index)`.
- `make_grid(n, size=1.0) -> Grid` box [0,size]^2.
- `d_shape_mask(grid, R0, Z0, a, kappa, delta) -> bool (n,n)`; boundary R = R0 + a cos(t + delta sin t), Z = Z0 + kappa a sin t. Pixel active if its center is inside.
- `@dataclass Chords: p0 (M,2), p1 (M,2), camera (M,) int`
- `make_fans(cfg_cameras: list[dict]) -> Chords`; each camera dict: `{name, origin:[x,y], n_chords, fan_center_deg, fan_width_deg}`. Chords are clipped to the box.
- `siddon(p0, p1, grid) -> (idx (k,) int flat pixel index, length (k,) float)` exact chord lengths.
- `geometry_matrix(chords, grid, mask=None) -> T` float32 (M, N_active) if mask given else (M, n*n).
- `neighbor_edges(mask) -> (E,2) int` 4-neighbour pairs (j<k) in active-pixel indexing.
- `laplacian(mask) -> L` (N_active, N_active) float64 graph Laplacian with sum over edges of (x_j-x_k)^2 = x^T L x.
- `checkerboard(mask) -> (N_active,) int in {0,1}`.

## tomo/phantoms.py
- `make_phantom(name, grid, mask, rng=None, **kw) -> (n,n) float` >= 0, zero outside mask. names: `peaked`, `hollow`, `blob`, `edge`, `random` (Gaussian random field, rng-dependent, made positive).
- `PHANTOMS = ['peaked','hollow','blob','edge']`.

## tomo/forward.py
- `make_problem(cfg, phantom, seed) -> Problem` (dataclass) with fields:
  `grid, mask, T (M,N) float32, chords, eps_true (N,) coarse truth = area average of the fine phantom,
   eps_true_img (n,n), b_clean (M,), b (M,), sigma (M,) per-chord noise std, L, edges, eps_max (float),
   fine_grid, fine_img`.
  Data are generated on the fine grid (cfg.data_grid) with its own geometry matrix (no inverse crime).
  Noise: sigma_i = rel*|b_clean_i| + floor*max(b_clean). Optional per-camera calibration offset.
- `pixel_area(problem)`.

## tomo/baselines.py   (all return dict with at least 'mean' (N,), optionally 'std' (N,), 'lam', 'time')
- `tikhonov(T, b, sigma, L, lam=None) -> dict` lam=None => choose by GCV (`gcv_lambda`).
- `gcv_lambda(T, b, sigma, L, lams=None) -> float`
- `mfi(T, b, sigma, mask, grid, n_iter=..., lam=None) -> dict` minimum Fisher information, positive.
- `gp_tomography(T, b, sigma, grid, mask) -> dict` SE-kernel GP, hyperparams by marginal likelihood; 'mean','std','cov'.

## tomo/metrics.py
- `rel_l2(est, true)`, `ssim_img(est_img, true_img, mask)`, `peak_error_cm(est, true, grid, mask)`,
  `power_error(est, true)`, `reduced_chi2(T, b, sigma, est)`,
  `coverage(samples_or_mean_std, true, level) -> float` (accepts samples (S,N) -> empirical quantile interval, or (mean,std) -> Gaussian),
  `split_rhat(chains) -> (N,)` for chains shaped (C, S, N), `sweeps_to_converge(trace (C,S,N), sweeps_per_sample, thresh=1.05)`,
  `summarize(problem, est_dict) -> dict` all scalar metrics.
- `to_img(vec, mask) -> (n,n)` with NaN/0 outside.

## tomo/sampling.py   (thrml-backed sampling infrastructure shared by all EBM variants)
- `greedy_coloring(n_nodes, edges) -> (n_nodes,) int colours` (no two coupled nodes share a colour).
- `IsingProblem` dataclass: `h (n,) , J (sparse COO: rows, cols, vals with i<j), offset` in **±1 spin convention**
  with energy `E(s) = -sum h_i s_i - sum J_ij s_i s_j + offset`. Helper `bits_to_spins_qubo(Qb, cb, const)`
  converts bit energy `E(u) = u^T Qb u + cb^T u + const` (u in {0,1}, Qb symmetric, zero diag folded) to IsingProblem.
- `run_ising(prob: IsingProblem, colors, schedule, n_chains, key, beta=1.0, init=None) -> dict(samples (C,S,n) bool, energy_trace)` using
  `thrml.IsingEBM`, `Block`, `IsingSamplingProgram`, `sample_states`, vmapped over chains.
- `anneal_ising(prob, colors, betas, sweeps_per_beta, n_chains, key, init=None) -> dict(final (C,n) bool, energy_trace)`.
- `parallel_tempering(...)` for hard cases.
- `SCHEDULES` presets from config.

## tomo/ebm_potts.py
- `quadratic_form(problem, K, lam) -> (Q (N,N), c (N,), const, Delta)` with E(x)=1/2 x^T Q x - c^T x + const, matching spec eq.
- `build_potts(problem, K, lam, beta=1.0)` -> thrml model objects; `sample_potts(problem, K, lam, cfg, key) -> result dict`
  result dict (all EBM variants): `{'mean','std','map','samples' (C,S,N) emissivity, 'rhat','energy_trace','n_blocks','n_spins','max_degree','time','invalid_frac'}`.

## tomo/ebm_ising.py
- Domain-wall encoding helpers: `dw_encode(x, K) -> u`, `dw_decode(u, K) -> x, valid`.
- `build_ising_dense(problem, K, lam, A) -> (IsingProblem, meta)`
- `build_ising_sparse(problem, K, lam, A, threshold=None, radius=None) -> (IsingProblem, meta)`
- `build_ising_chain(problem, K, lam, A, tau, Kz) -> (IsingProblem, meta)` (main contribution)
- `sample_ising_variant(problem, variant, cfg, key) -> result dict` (same keys as Potts).

## tomo/energy.py
- `tsu_estimate(n_spins, n_sweeps, n_chains, n_blocks, E_cell=None, t_update=None, *, preset=None, include_readout=False, n_readout_nodes=None, n_readouts=1, n_physical_spins=None) -> dict(energy_J, latency_s, sampling/readout parts, assumptions)`.
  - `HARDWARE` presets, each with a source string: `'spec'` (1.3 fJ, 100 ns per block), `'extropic_2510'` (about 2 fJ, 100 ns), and `'z1_2608'` (7.09 fJ per p-bit per Gibbs cycle, 20 ns per sweep on a 2-colour graph, else 10 ns per colour block; readout 1.692 pJ per node and 25 µs per frame).
  - With no preset, the call returns exactly the original specification numbers.
- `compare_presets(...)` builds the table of logical vs embedded layouts, with and without readout. `sensitivity(...)` and `worked_estimate(...)` cover the sensitivity range and the spec's worked example.
- `cpu_energy_from_powermetrics(log_path_or_watts, seconds)`, `gibbs_op_counts`, `gpu_mcmc_estimate(...)`, `laptop_gibbs_estimate(...)`.

---

# Additions after the initial contract

These modules were added during the project. Their signatures are equally stable.

## tomo/baselines.py (additions)
- `evidence_lambda(T, b, sigma, L, lams=None) -> float`: maximises the Gaussian marginal likelihood (empirical Bayes).
- `discrepancy_lambda(...)` and `lcurve_lambda(...)`.
- `tikhonov(..., method='gcv'|'discrepancy'|'lcurve'|'evidence')`.
- `tuned_tikhonov(T, b, sigma, L, method='evidence')`: the "tuned" baseline. It also supplies the EBM warm start and the λ the EBMs use.

## tomo/ebm_common.py
- `prepare(problem, K, lam=None, eps_max_factor=1.2) -> EBMSetup`:
  - computes the tuned-Tikhonov warm start, `eps_max` (from data only), `Delta`, `lam`, and `Q, c, const`;
  - sets `problem.eps_max`.
- `quadratic_form(problem, K, lam, Delta) -> (Q, c, const)` and `pixel_energy(x, Q, c, const)`.
- Level conversions: `levels_from_eps` and `eps_from_levels`.
- Domain-wall encoding:
  - `dw_encode(x, K)` and `dw_decode(u) -> (x, valid)`;
  - `invalid_fraction` and `dw_penalty_qubo`.
- `pixel_quadratic_to_bit_qubo(Q, c, const, K, A, mask_pairs=None)`, `bit_energy`, and `auto_A(Q, c, K)`.
- `make_result(...)`: builds the common result dict shared by every variant.
- `gaussian_posterior(Q, c, beta=1.0)`.

## tomo/ebm_chain.py (I-chain)
- `build_ising_chain(problem, K, lam=None, A=None, tau=None, Kz=16, *, tau_mode='dz', group=1, compensate=False, comp_floor=0.1, window='local', ...) -> (IsingProblem, meta)`.
- `sample_chain(problem, cfg=None, key=None, backend='jax', tau=None, *, init='tikhonov'|'random'|'half', compensate=None, ...) -> result dict`. Beyond the common keys, the result includes:
  - `tau`, `n_aux` and `chain_invalid_frac`;
  - `frozen_frac`, `rhat_med`, `rhat_q95` and `rhat_nonfinite_frac`;
  - `converged`, `invalid_ok` and `eff_noise_ratio`;
  - `comp_clamped_frac` and `s2`.
- Helpers: `chain_layout`, `chain_energy`, `decode_chain` and `effective_noise_variance`.

## tomo/ebm_tree.py (I-tree)
- `build_ising_tree(problem, K, lam=None, A=None, tau=None, Kz=16, *, tau_mode='dz', compensate=...) -> (IsingProblem, meta)`.
- `sample_tree(problem, cfg=None, key=None, ...) -> result dict`. It returns the same keys as `sample_chain`, plus `depth_max`, `colouring_used` and `n_pix_colours`.
- Helpers: `balanced_tree`, `tree_layout`, `tree_energy` and `decode_tree`.
- `ebm_ising.sample_ising_variant(problem, variant, cfg, key)` dispatches `'dense'`, `'sparse'`, `'chain'` and `'tree'`.

## tomo/embed.py (degree-bounded embedding)
- `embed_bounded_degree(prob, D=16, J_F=None, topology='tree'|'chain', jf_rule='cut'|'copy', margin=1.2) -> (prob_phys, meta)`:
  - replaces each high-degree spin by copies bound with ferromagnetic `J_F`;
  - an all-copies-agree state has exactly the logical energy.
- `decode(spins_phys, meta, method='majority'|'first')` returns logical spins and the broken-bond fractions.
- `auto_J_F`, `n_copies`, `logical_to_physical` and `embedding_summary`.
- Encoding and precision analysis:
  - `binary_bit_qubo(Qy, cy, const, nbits)`: power-of-two level encoding;
  - `coupling_dynamic_range` and `coupling_quantisation(prob, bits=(4,6,8))`;
  - `bipartite_relay_estimate`.

## tomo/positive.py (positivity and discreteness comparators)
- `truncated_gaussian_posterior(problem, lam, lower=0.0, upper=None, n_samples=500, n_warmup=200, n_chains=..., ...) -> result dict`: the exact continuous posterior truncated to a box, sampled by coordinate Gibbs with tail-safe univariate truncated-normal draws.
- `truncated_gaussian_sample(P, h, lower, upper, ...)`: the generic version of the same sampler.
- `gaussian_posterior_eps(problem, lam)` and `precision_system(problem, lam)`.
- `log_laplace(problem, lam_f=None, ...)`: positivity via ε = exp(f), with a Laplace approximation. This is our re-implementation of the idea in Ueda & Nishiura, arXiv:2410.11454.

## tomo/tcv.py (real TCV geometry)
- `make_tcv_problem(cfg, phantom, seed) -> Problem` (the same dataclass as `forward.make_problem`).
  - Phantoms: `peaked`, `hollow`, `blob`, `edge`, `divertor`, `random`, and `hamm:<k>` (the downloaded SOLPS phantoms).
- Geometry: `tcv_chords()` (120 lines of sight), `tcv_vessel_polygon()`, `tcv_etendues()`, `make_tcv_grid(nr=20, nz=60)`, `tcv_mask(grid)`.
- `make_tcv_phantom(...)` and `coverage_stats(T)`.
- `fetch_tcv_data()`, also available as `python -m tomo.tcv --fetch`.
- Vendored data is in `tomo/data/tcv/`, with an MIT notice.
- `geometry.Grid` takes an optional `ny` for rectangular grids.

## tomo/mixing_theory.py (linear-Gaussian mixing theory)
- `build_joint(problem, kind='dense'|'chain'|'tree', *, K=8, lam=None, tau=0.75, tau_mode='dz', Kz=32, ...)`: the joint precision over pixels and auxiliaries, together with the samplers' colour classes.
- Spectral radius:
  - `gs_rho(P, cls)`: colour-block Gibbs;
  - `sor_rho(P, cls, omega)`: over-relaxed Gibbs;
  - `da_rate(P, nx)`: the two-block data-augmentation rate.
- Autocorrelation:
  - `relaxation_time(rho)` and `iat_from_rho(rho)`;
  - `iat_functional(P, f)`, `iat_sup(P)` and `iat_pixel_sup(P, nx)`.
- Diagnostics: `da_slowdown_closed`, `chord_sum_rayleigh`, `null_space_bound` and `standard_functionals`.
- Simulation: `simulate_gs(P, cls, F, ...)`, an exact linear Gibbs simulation for validation, and `iat_series(y)`.
