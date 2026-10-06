# Multi-seed statistics (tcv geometry)

18 problems; phantoms x 2 noise seeds; 8 random fields. EBMs: ['potts', 'sparse'] (sparse threshold 0.2). Mean ± std over problems in each group.

| group | method | n | rel_l2 | rel_l2_median | ssim | coverage68 | coverage95 | rhat_max | time_s |
|---|---|---|---|---|---|---|---|---|---|
| peaked | tikhonov_gcv | 2 | 0.304 ± 0.016 | 0.304 | 0.539 ± 0.002 | - | - | - | - |
| peaked | tikhonov_tuned | 2 | 0.299 ± 0.012 | 0.299 | 0.526 ± 0.014 | - | - | - | - |
| peaked | mfi | 2 | 0.094 ± 0.007 | 0.094 | 0.929 ± 0.020 | - | - | - | - |
| peaked | gp | 2 | 0.157 ± 0.020 | 0.157 | 0.679 ± 0.064 | 0.757 ± 0.094 | 0.955 ± 0.004 | - | - |
| peaked | potts | 2 | 0.191 ± 0.005 | 0.191 | 0.755 ± 0.017 | 0.852 ± 0.006 | 0.960 ± 0.002 | 1.002 ± 0.000 | 81.797 ± 0.470 |
| peaked | sparse | 2 | 0.309 ± 0.015 | 0.309 | 0.790 ± 0.007 | 0.689 ± 0.002 | 0.801 ± 0.010 | 1.002 ± 0.000 | 8.529 ± 0.404 |
| hollow | tikhonov_gcv | 2 | 0.487 ± 0.003 | 0.487 | 0.352 ± 0.011 | - | - | - | - |
| hollow | tikhonov_tuned | 2 | 0.487 ± 0.003 | 0.487 | 0.346 ± 0.007 | - | - | - | - |
| hollow | mfi | 2 | 0.391 ± 0.004 | 0.391 | 0.804 ± 0.025 | - | - | - | - |
| hollow | gp | 2 | 0.638 ± 0.032 | 0.638 | 0.236 ± 0.008 | 0.420 ± 0.034 | 0.747 ± 0.028 | - | - |
| hollow | potts | 2 | 0.372 ± 0.008 | 0.372 | 0.597 ± 0.007 | 0.781 ± 0.007 | 0.916 ± 0.008 | 1.002 ± 0.000 | 81.285 ± 0.115 |
| hollow | sparse | 2 | 0.518 ± 0.004 | 0.518 | 0.688 ± 0.006 | 0.672 ± 0.002 | 0.722 ± 0.007 | 1.002 ± 0.000 | 8.184 ± 0.072 |
| blob | tikhonov_gcv | 2 | 0.701 ± 0.002 | 0.701 | 0.451 ± 0.021 | - | - | - | - |
| blob | tikhonov_tuned | 2 | 0.699 ± 0.001 | 0.699 | 0.445 ± 0.012 | - | - | - | - |
| blob | mfi | 2 | 0.400 ± 0.007 | 0.400 | 0.842 ± 0.022 | - | - | - | - |
| blob | gp | 2 | 0.677 ± 0.002 | 0.677 | 0.311 ± 0.009 | 0.519 ± 0.003 | 0.781 ± 0.001 | - | - |
| blob | potts | 2 | 0.558 ± 0.004 | 0.558 | 0.732 ± 0.014 | 0.794 ± 0.018 | 0.929 ± 0.008 | 1.002 ± 0.000 | 83.994 ± 3.772 |
| blob | sparse | 2 | 0.693 ± 0.001 | 0.693 | 0.727 ± 0.005 | 0.639 ± 0.001 | 0.683 ± 0.008 | 1.002 ± 0.000 | 8.283 ± 0.435 |
| edge | tikhonov_gcv | 2 | 0.682 ± 0.005 | 0.682 | 0.263 ± 0.003 | - | - | - | - |
| edge | tikhonov_tuned | 2 | 0.685 ± 0.003 | 0.685 | 0.266 ± 0.005 | - | - | - | - |
| edge | mfi | 2 | 0.479 ± 0.019 | 0.479 | 0.732 ± 0.005 | - | - | - | - |
| edge | gp | 2 | 0.668 ± 0.032 | 0.668 | 0.322 ± 0.020 | 0.573 ± 0.067 | 0.838 ± 0.066 | - | - |
| edge | potts | 2 | 0.537 ± 0.008 | 0.537 | 0.447 ± 0.012 | 0.765 ± 0.012 | 0.901 ± 0.006 | 1.002 ± 0.000 | 81.282 ± 0.427 |
| edge | sparse | 2 | 0.727 ± 0.003 | 0.727 | 0.493 ± 0.002 | 0.730 ± 0.005 | 0.792 ± 0.002 | 1.002 ± 0.001 | 8.399 ± 0.483 |
| divertor | tikhonov_gcv | 2 | 0.405 ± 0.009 | 0.405 | 0.515 ± 0.005 | - | - | - | - |
| divertor | tikhonov_tuned | 2 | 0.403 ± 0.007 | 0.403 | 0.514 ± 0.006 | - | - | - | - |
| divertor | mfi | 2 | 0.387 ± 0.009 | 0.387 | 0.634 ± 0.003 | - | - | - | - |
| divertor | gp | 2 | 0.399 ± 0.013 | 0.399 | 0.476 ± 0.014 | 0.808 ± 0.026 | 0.972 ± 0.009 | - | - |
| divertor | potts | 2 | 0.422 ± 0.003 | 0.422 | 0.490 ± 0.008 | 0.761 ± 0.001 | 0.939 ± 0.002 | 1.002 ± 0.000 | 92.774 ± 0.294 |
| divertor | sparse | 2 | 0.425 ± 0.010 | 0.425 | 0.495 ± 0.000 | 0.651 ± 0.021 | 0.889 ± 0.020 | 1.002 ± 0.000 | 19.915 ± 0.554 |
| random | tikhonov_gcv | 8 | 0.222 ± 0.062 | 0.214 | 0.764 ± 0.062 | - | - | - | - |
| random | tikhonov_tuned | 8 | 0.219 ± 0.063 | 0.210 | 0.763 ± 0.060 | - | - | - | - |
| random | mfi | 8 | 0.183 ± 0.033 | 0.181 | 0.834 ± 0.047 | - | - | - | - |
| random | gp | 8 | 0.151 ± 0.046 | 0.142 | 0.870 ± 0.051 | 0.671 ± 0.098 | 0.922 ± 0.039 | - | - |
| random | potts | 8 | 0.211 ± 0.050 | 0.209 | 0.771 ± 0.041 | 0.826 ± 0.035 | 0.982 ± 0.009 | 1.002 ± 0.000 | 87.415 ± 7.907 |
| random | sparse | 8 | 0.219 ± 0.059 | 0.212 | 0.770 ± 0.051 | 0.757 ± 0.044 | 0.970 ± 0.013 | 1.005 ± 0.004 | 13.995 ± 3.843 |


## Paired rel-L2 win rates (EBM better than baseline), two-sided sign test

| ebm | baseline | group | wins | losses | win_rate | sign_test_p |
|---|---|---|---|---|---|---|
| potts | tikhonov_gcv | peaked | 2 | 0 | 1.00 | 0.5 |
| potts | tikhonov_gcv | hollow | 2 | 0 | 1.00 | 0.5 |
| potts | tikhonov_gcv | blob | 2 | 0 | 1.00 | 0.5 |
| potts | tikhonov_gcv | edge | 2 | 0 | 1.00 | 0.5 |
| potts | tikhonov_gcv | divertor | 0 | 2 | 0.00 | 0.5 |
| potts | tikhonov_gcv | random | 5 | 3 | 0.62 | 0.727 |
| potts | tikhonov_gcv | all | 13 | 5 | 0.72 | 0.0963 |
| potts | tikhonov_tuned | peaked | 2 | 0 | 1.00 | 0.5 |
| potts | tikhonov_tuned | hollow | 2 | 0 | 1.00 | 0.5 |
| potts | tikhonov_tuned | blob | 2 | 0 | 1.00 | 0.5 |
| potts | tikhonov_tuned | edge | 2 | 0 | 1.00 | 0.5 |
| potts | tikhonov_tuned | divertor | 0 | 2 | 0.00 | 0.5 |
| potts | tikhonov_tuned | random | 5 | 3 | 0.62 | 0.727 |
| potts | tikhonov_tuned | all | 13 | 5 | 0.72 | 0.0963 |
| potts | mfi | peaked | 0 | 2 | 0.00 | 0.5 |
| potts | mfi | hollow | 2 | 0 | 1.00 | 0.5 |
| potts | mfi | blob | 0 | 2 | 0.00 | 0.5 |
| potts | mfi | edge | 0 | 2 | 0.00 | 0.5 |
| potts | mfi | divertor | 0 | 2 | 0.00 | 0.5 |
| potts | mfi | random | 1 | 7 | 0.12 | 0.0703 |
| potts | mfi | all | 3 | 15 | 0.17 | 0.00754 |
| potts | gp | peaked | 0 | 2 | 0.00 | 0.5 |
| potts | gp | hollow | 2 | 0 | 1.00 | 0.5 |
| potts | gp | blob | 2 | 0 | 1.00 | 0.5 |
| potts | gp | edge | 2 | 0 | 1.00 | 0.5 |
| potts | gp | divertor | 0 | 2 | 0.00 | 0.5 |
| potts | gp | random | 0 | 8 | 0.00 | 0.00781 |
| potts | gp | all | 6 | 12 | 0.33 | 0.238 |
| sparse | tikhonov_gcv | peaked | 0 | 2 | 0.00 | 0.5 |
| sparse | tikhonov_gcv | hollow | 0 | 2 | 0.00 | 0.5 |
| sparse | tikhonov_gcv | blob | 2 | 0 | 1.00 | 0.5 |
| sparse | tikhonov_gcv | edge | 0 | 2 | 0.00 | 0.5 |
| sparse | tikhonov_gcv | divertor | 0 | 2 | 0.00 | 0.5 |
| sparse | tikhonov_gcv | random | 6 | 2 | 0.75 | 0.289 |
| sparse | tikhonov_gcv | all | 8 | 10 | 0.44 | 0.815 |
| sparse | tikhonov_tuned | peaked | 0 | 2 | 0.00 | 0.5 |
| sparse | tikhonov_tuned | hollow | 0 | 2 | 0.00 | 0.5 |
| sparse | tikhonov_tuned | blob | 2 | 0 | 1.00 | 0.5 |
| sparse | tikhonov_tuned | edge | 0 | 2 | 0.00 | 0.5 |
| sparse | tikhonov_tuned | divertor | 0 | 2 | 0.00 | 0.5 |
| sparse | tikhonov_tuned | random | 3 | 5 | 0.38 | 0.727 |
| sparse | tikhonov_tuned | all | 5 | 13 | 0.28 | 0.0963 |
| sparse | mfi | peaked | 0 | 2 | 0.00 | 0.5 |
| sparse | mfi | hollow | 0 | 2 | 0.00 | 0.5 |
| sparse | mfi | blob | 0 | 2 | 0.00 | 0.5 |
| sparse | mfi | edge | 0 | 2 | 0.00 | 0.5 |
| sparse | mfi | divertor | 0 | 2 | 0.00 | 0.5 |
| sparse | mfi | random | 1 | 7 | 0.12 | 0.0703 |
| sparse | mfi | all | 1 | 17 | 0.06 | 0.000145 |
| sparse | gp | peaked | 0 | 2 | 0.00 | 0.5 |
| sparse | gp | hollow | 2 | 0 | 1.00 | 0.5 |
| sparse | gp | blob | 0 | 2 | 0.00 | 0.5 |
| sparse | gp | edge | 0 | 2 | 0.00 | 0.5 |
| sparse | gp | divertor | 0 | 2 | 0.00 | 0.5 |
| sparse | gp | random | 0 | 8 | 0.00 | 0.00781 |
| sparse | gp | all | 2 | 16 | 0.11 | 0.00131 |

