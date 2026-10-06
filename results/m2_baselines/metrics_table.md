# M2 baselines

## Named phantoms (per phantom)

### peaked

| method | rel_l2 | ssim | peak_err_cm | power_err | chi2_red | coverage68 | coverage95 | lam | time |
|---|---|---|---|---|---|---|---|---|---|
| tikhonov_gcv | 0.261 | 0.781 | 0 | 0.0115 | 0.00714 | - | - | 3.87 | 0.0108 |
| tikhonov_tuned | 0.263 | 0.785 | 0 | 0.00939 | 0.0805 | - | - | 18.3 | 0.0099 |
| mfi | 0.2 | 0.836 | 0 | 0.0611 | 0.349 | - | - | 122 | 0.0547 |
| gp | 0.226 | 0.739 | 3.12 | 0.0773 | 1.03 | 0.277 | 0.542 | - | 0.791 |

### hollow

| method | rel_l2 | ssim | peak_err_cm | power_err | chi2_red | coverage68 | coverage95 | lam | time |
|---|---|---|---|---|---|---|---|---|---|
| tikhonov_gcv | 0.649 | 0.351 | 57 | 0.163 | 0.0499 | - | - | 4.3 | 0.0101 |
| tikhonov_tuned | 0.649 | 0.345 | 57 | 0.163 | 0.125 | - | - | 8.11 | 0.01 |
| mfi | 0.807 | 0.207 | 57 | 0.213 | 0.34 | - | - | 43 | 0.0532 |
| gp | 0.54 | 0.504 | 53.5 | 0.11 | 0.793 | 0.455 | 0.708 | - | 0.797 |

### blob

| method | rel_l2 | ssim | peak_err_cm | power_err | chi2_red | coverage68 | coverage95 | lam | time |
|---|---|---|---|---|---|---|---|---|---|
| tikhonov_gcv | 0.784 | 0.178 | 3.12 | 0.163 | 0.00386 | - | - | 9.36 | 0.00946 |
| tikhonov_tuned | 0.776 | 0.188 | 3.12 | 0.175 | 0.0822 | - | - | 55.1 | 0.00931 |
| mfi | 0.41 | 0.673 | 3.12 | 0.123 | 0.376 | - | - | 638 | 0.0507 |
| gp | 0.798 | 0.161 | 4.42 | 0.11 | 0.667 | 0.581 | 0.87 | - | 0.755 |

### edge

| method | rel_l2 | ssim | peak_err_cm | power_err | chi2_red | coverage68 | coverage95 | lam | time |
|---|---|---|---|---|---|---|---|---|---|
| tikhonov_gcv | 0.604 | 0.376 | 20 | 0.0346 | 0.496 | - | - | 15.6 | 0.00976 |
| tikhonov_tuned | 0.611 | 0.375 | 20 | 0.036 | 0.14 | - | - | 5.26 | 0.00973 |
| mfi | 0.62 | 0.458 | 89.5 | 0.0738 | 0.195 | - | - | 23 | 0.0541 |
| gp | 0.653 | 0.34 | 20 | 0.0142 | 0.457 | 0.666 | 0.891 | - | 0.823 |

## Random fields (n=200): mean over fields

| method | rel_l2 | ssim | peak_err_cm | power_err | chi2_red | coverage68 | coverage95 |
|---|---|---|---|---|---|---|---|
| tikhonov_gcv | 0.336 | 0.658 | 17.8 | 0.0435 | 0.165 | - | - |
| tikhonov_tuned | 0.335 | 0.661 | 16.9 | 0.0435 | 0.172 | - | - |
| mfi | 0.296 | 0.723 | 16 | 0.0387 | 0.2 | - | - |
| gp | 0.291 | 0.718 | 14.1 | 0.0364 | 0.683 | 0.634 | 0.891 |

Median / std of rel-L2 over random fields:

| method | rel_l2_med | rel_l2_std | ssim_med | peak_err_cm_med |
|---|---|---|---|---|
| tikhonov_gcv | 0.327 | 0.0743 | 0.665 | 6.99 |
| tikhonov_tuned | 0.326 | 0.0742 | 0.667 | 6.99 |
| mfi | 0.289 | 0.0686 | 0.738 | 6.99 |
| gp | 0.279 | 0.0759 | 0.73 | 4.42 |

Best rel-L2 count over random fields: tikhonov_gcv=3, tikhonov_tuned=3, mfi=84, gp=110
