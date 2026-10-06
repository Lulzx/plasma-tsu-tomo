# Route-2 ablation: positivity vs upper bound vs discreteness

Same lambda (evidence) and same smoothness prior for gaussian / trunc_* / potts_K*; eps_max is the Potts bound (1.2 x max clipped Tikhonov, data only). Problems: ['peaked', 'hollow', 'random000']. Schedules: {'trunc': {'n_warmup': 300, 'n_samples': 200, 'thin': 5, 'n_chains': 8}, 'potts': {'n_warmup': 500, 'n_samples': 200, 'steps_per_sample': 5, 'n_chains': 8}}. `rhat_max` = worst over pixels (max over problems in the mean table), `rhat_med` = median over pixels; log_laplace samples are iid Laplace draws (no R-hat); log_laplace is our re-implementation of the idea of arXiv:2410.11454 with its own lambda (prior acts on log-emissivity), gp/mfi/tikhonov_tuned use their own hyperparameters.

## peaked

| method | rel_l2 | ssim | peak_err_cm | coverage68 | coverage95 | rhat_max | rhat_med | time | frac_top | frac_at_upper | frac_at_zero |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tikhonov_tuned | 0.263 | 0.785 | 0 | - | - | - | - | 0.022 | - | - | - |
| gaussian | 0.263 | 0.785 | 0 | 0.918 | 1 | - | - | 0.0124 | - | - | - |
| trunc_pos | 0.263 | 0.779 | 3.12 | 0.779 | 0.97 | 1.01 | 1 | 6.7 | - | 0 | 6.98e-06 |
| trunc_box | 0.296 | 0.752 | 0 | 0.721 | 0.95 | 1.01 | 1 | 6.48 | - | 1.01e-05 | 6.2e-06 |
| potts_K4 | 0.322 | 0.728 | 0 | 0.381 | 0.86 | 1.06 | 1.01 | 2.03 | 0.0131 | - | - |
| potts_K8 | 0.255 | 0.791 | 0 | 0.803 | 0.98 | 1.01 | 1 | 6.29 | 0.0138 | - | - |
| potts_K16 | 0.28 | 0.767 | 0 | 0.772 | 0.98 | 1.01 | 1 | 11.6 | 0.00618 | - | - |
| potts_K32 | 0.289 | 0.758 | 0 | 0.759 | 0.97 | 1.01 | 1 | 17.1 | 0.00294 | - | - |
| log_laplace | 0.325 | 0.766 | 3.12 | 0.71 | 0.933 | - | - | 4.07 | - | - | - |
| gp | 0.226 | 0.739 | 3.12 | 0.277 | 0.542 | - | - | 1.64 | - | - | - |
| mfi | 0.2 | 0.836 | 0 | - | - | - | - | 0.109 | - | - | - |

## hollow

| method | rel_l2 | ssim | peak_err_cm | coverage68 | coverage95 | rhat_max | rhat_med | time | frac_top | frac_at_upper | frac_at_zero |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tikhonov_tuned | 0.649 | 0.345 | 57 | - | - | - | - | 0.021 | - | - | - |
| gaussian | 0.649 | 0.345 | 57 | 0.48 | 0.77 | - | - | 0.0108 | - | - | - |
| trunc_pos | 0.543 | 0.48 | 53.9 | 0.419 | 0.752 | 1.01 | 1 | 6.62 | - | 0 | 3.88e-06 |
| trunc_box | 0.536 | 0.488 | 53.9 | 0.422 | 0.753 | 1.01 | 1 | 6.92 | - | 7.75e-07 | 3.1e-06 |
| potts_K4 | 0.608 | 0.413 | 57 | 0.251 | 0.685 | 1.04 | 1 | 3.03 | 0.0112 | - | - |
| potts_K8 | 0.554 | 0.474 | 57 | 0.464 | 0.826 | 1.01 | 1 | 5.24 | 0.00412 | - | - |
| potts_K16 | 0.543 | 0.485 | 53.9 | 0.454 | 0.803 | 1.01 | 1 | 11 | 0.00162 | - | - |
| potts_K32 | 0.541 | 0.483 | 53.9 | 0.433 | 0.778 | 1.01 | 1 | 17.4 | 0.000675 | - | - |
| log_laplace | 0.843 | 0.182 | 41.9 | 0.39 | 0.671 | - | - | 5.97 | - | - | - |
| gp | 0.54 | 0.504 | 53.5 | 0.455 | 0.708 | - | - | 1.93 | - | - | - |
| mfi | 0.807 | 0.207 | 57 | - | - | - | - | 0.127 | - | - | - |

## random000

| method | rel_l2 | ssim | peak_err_cm | coverage68 | coverage95 | rhat_max | rhat_med | time | frac_top | frac_at_upper | frac_at_zero |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tikhonov_tuned | 0.276 | 0.771 | 3.12 | - | - | - | - | 0.0214 | - | - | - |
| gaussian | 0.276 | 0.771 | 3.12 | 0.86 | 0.976 | - | - | 0.0103 | - | - | - |
| trunc_pos | 0.283 | 0.744 | 19 | 0.815 | 0.975 | 1.01 | 1 | 7.42 | - | 0 | 0 |
| trunc_box | 0.305 | 0.738 | 29.6 | 0.816 | 0.95 | 1.01 | 1 | 7.93 | - | 7.75e-06 | 7.75e-07 |
| potts_K4 | 0.31 | 0.707 | 25.8 | 0.52 | 0.865 | 1.42 | 1.04 | 4.55 | 0.0257 | - | - |
| potts_K8 | 0.289 | 0.759 | 25.8 | 0.78 | 0.963 | 1.01 | 1 | 5.93 | 0.0115 | - | - |
| potts_K16 | 0.299 | 0.746 | 25.8 | 0.801 | 0.955 | 1.01 | 1 | 13.3 | 0.00473 | - | - |
| potts_K32 | 0.302 | 0.741 | 25.8 | 0.816 | 0.952 | 1.01 | 1 | 16.7 | 0.00215 | - | - |
| log_laplace | 0.308 | 0.785 | 93.8 | 0.625 | 0.963 | - | - | 2.04 | - | - | - |
| gp | 0.278 | 0.779 | 93.8 | 0.773 | 0.952 | - | - | 1.67 | - | - | - |
| mfi | 0.28 | 0.771 | 93.8 | - | - | - | - | 0.105 | - | - | - |

## Mean over 3 problems

| method | n | rel_l2 | ssim | peak_err_cm | coverage68 | coverage95 | rhat_max | rhat_med | time |
|---|---|---|---|---|---|---|---|---|---|
| tikhonov_tuned | 3 | 0.396 | 0.634 | 20.1 | - | - | - | - | 0.0215 |
| gaussian | 3 | 0.396 | 0.634 | 20.1 | 0.753 | 0.916 | - | - | 0.0112 |
| trunc_pos | 3 | 0.363 | 0.668 | 25.4 | 0.671 | 0.899 | 1.01 | 1 | 6.91 |
| trunc_box | 3 | 0.379 | 0.659 | 27.9 | 0.653 | 0.885 | 1.01 | 1 | 7.11 |
| potts_K4 | 3 | 0.413 | 0.616 | 27.6 | 0.384 | 0.803 | 1.42 | 1.02 | 3.2 |
| potts_K8 | 3 | 0.366 | 0.674 | 27.6 | 0.682 | 0.923 | 1.01 | 1 | 5.82 |
| potts_K16 | 3 | 0.374 | 0.666 | 26.6 | 0.676 | 0.913 | 1.01 | 1 | 12 |
| potts_K32 | 3 | 0.377 | 0.661 | 26.6 | 0.67 | 0.9 | 1.01 | 1 | 17.1 |
| log_laplace | 3 | 0.492 | 0.578 | 46.3 | 0.575 | 0.856 | - | - | 4.02 |
| gp | 3 | 0.348 | 0.674 | 50.1 | 0.502 | 0.734 | - | - | 1.75 |
| mfi | 3 | 0.429 | 0.605 | 50.3 | - | - | - | - | 0.113 |
