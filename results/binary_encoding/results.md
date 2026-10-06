# Binary vs thermometer encoding (default problem, peaked seed 0, K=8, schedule {'n_warmup': 2000, 'n_samples': 500, 'steps_per_sample': 10}, 16 chains, overdispersed)

| metric | thermometer (I-dense) | binary |
|---|---|---|
| rel-L2 | 0.258 | 0.258 |
| coverage95 | 0.979 | 0.979 |
| R-hat max | 1.002 | 1.027 |
| R-hat median | 1.000 | 1.001 |
| IAT total power [sweeps] | 17.7 | 58.5 |
| IAT chord data, median [sweeps] | 9.7 | 11.3 |
| IAT chord data, max [sweeps] | 11.3 | 19.0 |
| IAT pixel level, median [sweeps] | 10.8 | 11.9 |
| n_spins | 5642 | 2418 |
| n_blocks | 315 | 144 |
| max_degree | 2554 | 1094 |
| wall time [s] | 76.5 | 14.7 |
| coupling dyn. range [bits] (max/min) | 52.9 | 62.8 |
| bits max/p1 | 20.6 | 20.1 |
| frac couplings -> 0 at 8 bit | 0.993 | 0.972 |
| frac -> 0 at 6 bit | 0.994 | 0.994 |

Posterior agreement: {'mean_rel_l2_binary_vs_thermometer': 0.00792739391383051, 'std_ratio_median': 1.0007070310232367}

## Degree-16 embedded models (tiny problem 12x12, K=8, tree embedding)

| model | embedding | n_spins | max_deg | dyn range [bits] | median |dmean|/sd | broken bonds | frozen pixels | R-hat med | wall [s] |
|---|---|---|---|---|---|---|---|---|---|---|
| binary | none | 342 | 233 | nan | 0.00 | 0.000 | 0.000 | 1.012 | 0.7 |
| binary | tree D=16 J_F=433 (auto) | 1899 | 16 | 64.8 | 2.18 | 0.000 | 0.982 | inf | 1.2 |
| binary | tree D=16 J_F=1 | 1899 | 16 | 63.2 | 1.00 | 0.348 | 0.000 | 1.122 | 1.1 |
| binary | tree D=16 J_F=4 | 1899 | 16 | 63.2 | 0.98 | 0.218 | 0.000 | 2.076 | 1.1 |
| thermometer | none | 798 | 545 | nan | 0.00 | 0.000 | 0.000 | 1.000 | 2.8 |
| thermometer | tree D=16 J_F=559 (auto) | 10024 | 16 | 53.7 | 2.20 | 0.000 | 1.000 | inf | 3.7 |
| thermometer | tree D=16 J_F=1 | 10024 | 16 | 52.7 | 0.57 | 0.137 | 0.000 | 1.018 | 4.6 |
| thermometer | tree D=16 J_F=4 | 10024 | 16 | 52.7 | 1.04 | 0.046 | 0.000 | 11.892 | 4.9 |
