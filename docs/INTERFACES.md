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
- `tsu_estimate(n_spins, n_sweeps, n_chains, n_blocks, E_cell=1.3e-15, t_update=100e-9) -> dict(energy_J, latency_s)`
- `sensitivity(n_spins, n_sweeps, n_chains, n_blocks, E_cell_factors=(0.5,1,3), t_updates=(50e-9,100e-9,500e-9)) -> table`
- `cpu_energy_from_powermetrics(log_path_or_watts, seconds)`, `gpu_mcmc_estimate(...)`.
