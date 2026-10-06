# M4 ablations

phantoms: ['peaked']; schedule: {'n_warmup': 2000, 'n_samples': 500, 'steps_per_sample': 10}, n_chains=16

## K

| K | problem | variant | rel_l2 | map_rel_l2 | coverage95 | rhat_max | invalid_frac | n_spins | n_blocks | time_s | error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | peaked | potts | 0.322 | 0.453 | 0.857 | 1.01 | 0 | 806 | 48 | 23 | - |
| 4 | peaked | chain | 0.361 | 0.517 | 0.85 | 2.41e+14 | 0 | 71052 | 68 | 246 | - |
| 8 | peaked | potts | 0.258 | 0.276 | 0.98 | 1 | 0 | 806 | 48 | 35 | - |
| 8 | peaked | chain | 0.3 | 0.486 | 0.984 | 2.87e+14 | 0 | 74276 | 76 | 142 | - |
| 16 | peaked | potts | 0.278 | 0.23 | 0.984 | 1 | 0 | 806 | 48 | 59.6 | - |
| 16 | peaked | chain | 0.326 | 0.314 | 0.974 | 1.11 | 2.17e-06 | 80724 | 92 | 249 | - |

## tau

| tau | problem | variant | rel_l2 | map_rel_l2 | coverage95 | rhat_max | invalid_frac | n_spins | n_blocks | time_s | error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | peaked | chain | 0.435 | 0.517 | 0.954 | 6.32e+14 | 0 | 74276 | 76 | 145 | - |
| 0.25 | peaked | chain | 0.395 | 0.419 | 0.85 | 7.29e+14 | 0 | 74276 | 76 | 142 | - |
| 0.5 | peaked | chain | 0.302 | 0.487 | 0.976 | 3.52e+14 | 0 | 74276 | 76 | 139 | - |
| 1 | peaked | chain | 0.337 | 0.382 | 0.978 | 3.91 | 0 | 74276 | 76 | 135 | - |
| 2 | peaked | chain | 0.434 | 0.465 | 0.945 | 1.09 | 9.31e-07 | 74276 | 76 | 137 | - |

## sparse

| threshold | problem | variant | rel_l2 | map_rel_l2 | coverage95 | rhat_max | invalid_frac | n_spins | n_blocks | time_s | error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.01 | peaked | sparse | 0.263 | 0.293 | 0.975 | 1 | 0 | 5642 | 231 | 51.9 | - |
| 0.02 | peaked | sparse | 0.27 | 0.295 | 0.967 | 1 | 0 | 5642 | 231 | 43.9 | - |
| 0.05 | peaked | sparse | 0.264 | 0.304 | 0.958 | 1 | 0 | 5642 | 196 | 27.5 | - |
| 0.1 | peaked | sparse | 0.26 | 0.323 | 0.963 | 1 | 0 | 5642 | 133 | 17.5 | - |
| 0.2 | peaked | sparse | 0.261 | 0.302 | 0.949 | 1 | 0 | 5642 | 49 | 9.14 | - |

## noise

| noise | problem | variant | rel_l2 | map_rel_l2 | coverage95 | rhat_max | invalid_frac | n_spins | n_blocks | time_s | error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.01 | peaked | tikhonov_tuned | 0.265 | - | - | - | - | - | - | - | - |
| 0.01 | peaked | potts | 0.239 | 0.226 | 0.979 | 1 | 0 | 806 | 48 | 30.7 | - |
| 0.01 | peaked | chain | 0.287 | 0.423 | 0.984 | 2.11e+14 | 0 | 74276 | 76 | 127 | - |
| 0.03 | peaked | tikhonov_tuned | 0.263 | - | - | - | - | - | - | - | - |
| 0.03 | peaked | potts | 0.258 | 0.273 | 0.979 | 1 | 0 | 806 | 48 | 31.4 | - |
| 0.03 | peaked | chain | 0.31 | 0.44 | 0.98 | 42.6 | 0 | 74276 | 76 | 134 | - |
| 0.05 | peaked | tikhonov_tuned | 0.263 | - | - | - | - | - | - | - | - |
| 0.05 | peaked | potts | 0.277 | 0.296 | 0.985 | 1 | 0 | 806 | 48 | 30.9 | - |
| 0.05 | peaked | chain | 0.328 | 0.472 | 0.985 | 24.4 | 0 | 74276 | 76 | 146 | - |
| 0.1 | peaked | tikhonov_tuned | 0.265 | - | - | - | - | - | - | - | - |
| 0.1 | peaked | potts | 0.325 | 0.254 | 0.984 | 1 | 0 | 806 | 48 | 31 | - |
| 0.1 | peaked | chain | 0.363 | 0.534 | 0.986 | 5.33 | 0 | 74276 | 76 | 137 | - |
