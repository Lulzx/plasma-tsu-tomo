# Vendored TCV data

Source: https://github.com/dhamm97/real-time-tomo-prad (commit 7f6cee2c244ac0c30b9267e90052ce8a9060f627, MIT License,
Copyright (c) 2026 dhamm97; see LICENSE_real-time-tomo-prad_MIT). Paper: Hamm et al., arXiv:2603.11856.

- bolo_coords_r/z.npy, etendues.npy, tcv_shape_coords.npy, tcv_mask_1_subpixels_NINO.npy: from `src/tcv_geometry/`
- magnetic_equilibrium.npy, solps_phantom_{inner_leg,outer_leg,ring_and_core}.npy, xpt_rad.npy: from
  `src/results/hyperparameter_study_results/phantoms/`

Large files (1000 phantoms, geometry_matrix_NINO.npy) are NOT vendored; `python -m tomo.tcv --fetch` downloads them to data/tcv/.
