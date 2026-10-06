# Multi-seed statistics (default geometry)

50 problems; phantoms x 5 noise seeds; 30 random fields. EBMs: ['potts', 'sparse'] (sparse threshold 0.2). Mean ± std over problems in each group.

| group | method | n | rel_l2 | rel_l2_median | ssim | coverage68 | coverage95 | rhat_max | time_s |
|---|---|---|---|---|---|---|---|---|---|
| peaked | tikhonov_gcv | 5 | 0.286 ± 0.014 | 0.292 | 0.760 ± 0.012 | - | - | - | - |
| peaked | tikhonov_tuned | 5 | 0.285 ± 0.013 | 0.288 | 0.761 ± 0.014 | - | - | - | - |
| peaked | mfi | 5 | 0.210 ± 0.009 | 0.209 | 0.842 ± 0.017 | - | - | - | - |
| peaked | gp | 5 | 0.228 ± 0.015 | 0.226 | 0.760 ± 0.042 | 0.290 ± 0.049 | 0.548 ± 0.086 | - | - |
| peaked | potts | 5 | 0.273 ± 0.012 | 0.272 | 0.784 ± 0.009 | 0.773 ± 0.016 | 0.972 ± 0.008 | 1.002 ± 0.000 | 35.259 ± 1.392 |
| peaked | sparse | 5 | 0.281 ± 0.011 | 0.284 | 0.800 ± 0.012 | 0.611 ± 0.025 | 0.933 ± 0.008 | 1.002 ± 0.000 | 9.907 ± 1.887 |
| hollow | tikhonov_gcv | 5 | 0.653 ± 0.007 | 0.651 | 0.349 ± 0.012 | - | - | - | - |
| hollow | tikhonov_tuned | 5 | 0.653 ± 0.006 | 0.649 | 0.346 ± 0.004 | - | - | - | - |
| hollow | mfi | 5 | 0.790 ± 0.022 | 0.790 | 0.235 ± 0.031 | - | - | - | - |
| hollow | gp | 5 | 0.553 ± 0.022 | 0.545 | 0.482 ± 0.035 | 0.374 ± 0.051 | 0.667 ± 0.043 | - | - |
| hollow | potts | 5 | 0.569 ± 0.010 | 0.567 | 0.466 ± 0.005 | 0.439 ± 0.012 | 0.797 ± 0.023 | 1.002 ± 0.000 | 36.056 ± 1.214 |
| hollow | sparse | 5 | 0.630 ± 0.008 | 0.629 | 0.365 ± 0.005 | 0.312 ± 0.008 | 0.577 ± 0.017 | 1.002 ± 0.001 | 8.149 ± 0.652 |
| blob | tikhonov_gcv | 5 | 0.780 ± 0.009 | 0.781 | 0.180 ± 0.008 | - | - | - | - |
| blob | tikhonov_tuned | 5 | 0.777 ± 0.010 | 0.776 | 0.185 ± 0.010 | - | - | - | - |
| blob | mfi | 5 | 0.420 ± 0.016 | 0.417 | 0.660 ± 0.013 | - | - | - | - |
| blob | gp | 5 | 0.779 ± 0.036 | 0.783 | 0.154 ± 0.026 | 0.519 ± 0.041 | 0.855 ± 0.015 | - | - |
| blob | potts | 5 | 0.620 ± 0.017 | 0.619 | 0.564 ± 0.011 | 0.714 ± 0.003 | 0.898 ± 0.007 | 1.002 ± 0.000 | 36.510 ± 0.811 |
| blob | sparse | 5 | 0.646 ± 0.006 | 0.647 | 0.432 ± 0.013 | 0.300 ± 0.022 | 0.537 ± 0.015 | 1.002 ± 0.000 | 7.986 ± 1.370 |
| edge | tikhonov_gcv | 5 | 0.602 ± 0.004 | 0.603 | 0.380 ± 0.004 | - | - | - | - |
| edge | tikhonov_tuned | 5 | 0.604 ± 0.004 | 0.604 | 0.380 ± 0.004 | - | - | - | - |
| edge | mfi | 5 | 0.602 ± 0.010 | 0.601 | 0.482 ± 0.017 | - | - | - | - |
| edge | gp | 5 | 0.732 ± 0.065 | 0.733 | 0.289 ± 0.052 | 0.562 ± 0.097 | 0.813 ± 0.068 | - | - |
| edge | potts | 5 | 0.608 ± 0.003 | 0.608 | 0.391 ± 0.003 | 0.682 ± 0.015 | 0.878 ± 0.010 | 1.002 ± 0.000 | 37.975 ± 3.537 |
| edge | sparse | 5 | 0.612 ± 0.005 | 0.610 | 0.422 ± 0.012 | 0.485 ± 0.022 | 0.792 ± 0.021 | 1.004 ± 0.003 | 14.829 ± 5.711 |
| random | tikhonov_gcv | 30 | 0.316 ± 0.073 | 0.302 | 0.672 ± 0.092 | - | - | - | - |
| random | tikhonov_tuned | 30 | 0.316 ± 0.073 | 0.305 | 0.672 ± 0.093 | - | - | - | - |
| random | mfi | 30 | 0.289 ± 0.061 | 0.290 | 0.726 ± 0.083 | - | - | - | - |
| random | gp | 30 | 0.270 ± 0.073 | 0.257 | 0.741 ± 0.093 | 0.648 ± 0.162 | 0.887 ± 0.107 | - | - |
| random | potts | 30 | 0.319 ± 0.063 | 0.329 | 0.668 ± 0.078 | 0.684 ± 0.100 | 0.939 ± 0.042 | 1.003 ± 0.001 | 43.043 ± 5.049 |
| random | sparse | 30 | 0.314 ± 0.071 | 0.298 | 0.680 ± 0.085 | 0.615 ± 0.107 | 0.898 ± 0.065 | 1.004 ± 0.003 | 12.142 ± 6.203 |


## Paired rel-L2 win rates (EBM better than baseline), two-sided sign test

| ebm | baseline | group | wins | losses | win_rate | sign_test_p |
|---|---|---|---|---|---|---|
| potts | tikhonov_gcv | peaked | 5 | 0 | 1.00 | 0.0625 |
| potts | tikhonov_gcv | hollow | 5 | 0 | 1.00 | 0.0625 |
| potts | tikhonov_gcv | blob | 5 | 0 | 1.00 | 0.0625 |
| potts | tikhonov_gcv | edge | 0 | 5 | 0.00 | 0.0625 |
| potts | tikhonov_gcv | random | 13 | 17 | 0.43 | 0.585 |
| potts | tikhonov_gcv | all | 28 | 22 | 0.56 | 0.48 |
| potts | tikhonov_tuned | peaked | 5 | 0 | 1.00 | 0.0625 |
| potts | tikhonov_tuned | hollow | 5 | 0 | 1.00 | 0.0625 |
| potts | tikhonov_tuned | blob | 5 | 0 | 1.00 | 0.0625 |
| potts | tikhonov_tuned | edge | 0 | 5 | 0.00 | 0.0625 |
| potts | tikhonov_tuned | random | 14 | 16 | 0.47 | 0.856 |
| potts | tikhonov_tuned | all | 29 | 21 | 0.58 | 0.322 |
| potts | mfi | peaked | 0 | 5 | 0.00 | 0.0625 |
| potts | mfi | hollow | 5 | 0 | 1.00 | 0.0625 |
| potts | mfi | blob | 0 | 5 | 0.00 | 0.0625 |
| potts | mfi | edge | 1 | 4 | 0.20 | 0.375 |
| potts | mfi | random | 2 | 28 | 0.07 | 8.68e-07 |
| potts | mfi | all | 8 | 42 | 0.16 | 1.16e-06 |
| potts | gp | peaked | 0 | 5 | 0.00 | 0.0625 |
| potts | gp | hollow | 1 | 4 | 0.20 | 0.375 |
| potts | gp | blob | 5 | 0 | 1.00 | 0.0625 |
| potts | gp | edge | 5 | 0 | 1.00 | 0.0625 |
| potts | gp | random | 3 | 27 | 0.10 | 8.43e-06 |
| potts | gp | all | 14 | 36 | 0.28 | 0.0026 |
| sparse | tikhonov_gcv | peaked | 4 | 1 | 0.80 | 0.375 |
| sparse | tikhonov_gcv | hollow | 5 | 0 | 1.00 | 0.0625 |
| sparse | tikhonov_gcv | blob | 5 | 0 | 1.00 | 0.0625 |
| sparse | tikhonov_gcv | edge | 0 | 5 | 0.00 | 0.0625 |
| sparse | tikhonov_gcv | random | 15 | 15 | 0.50 | 1 |
| sparse | tikhonov_gcv | all | 29 | 21 | 0.58 | 0.322 |
| sparse | tikhonov_tuned | peaked | 5 | 0 | 1.00 | 0.0625 |
| sparse | tikhonov_tuned | hollow | 5 | 0 | 1.00 | 0.0625 |
| sparse | tikhonov_tuned | blob | 5 | 0 | 1.00 | 0.0625 |
| sparse | tikhonov_tuned | edge | 0 | 5 | 0.00 | 0.0625 |
| sparse | tikhonov_tuned | random | 15 | 15 | 0.50 | 1 |
| sparse | tikhonov_tuned | all | 30 | 20 | 0.60 | 0.203 |
| sparse | mfi | peaked | 0 | 5 | 0.00 | 0.0625 |
| sparse | mfi | hollow | 5 | 0 | 1.00 | 0.0625 |
| sparse | mfi | blob | 0 | 5 | 0.00 | 0.0625 |
| sparse | mfi | edge | 0 | 5 | 0.00 | 0.0625 |
| sparse | mfi | random | 6 | 24 | 0.20 | 0.00143 |
| sparse | mfi | all | 11 | 39 | 0.22 | 9.02e-05 |
| sparse | gp | peaked | 0 | 5 | 0.00 | 0.0625 |
| sparse | gp | hollow | 0 | 5 | 0.00 | 0.0625 |
| sparse | gp | blob | 5 | 0 | 1.00 | 0.0625 |
| sparse | gp | edge | 5 | 0 | 1.00 | 0.0625 |
| sparse | gp | random | 1 | 29 | 0.03 | 5.77e-08 |
| sparse | gp | all | 11 | 39 | 0.22 | 9.02e-05 |

