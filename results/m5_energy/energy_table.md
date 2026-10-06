# TSU energy / latency estimates

Model (tomo/energy.py): E = N_spins x N_sweeps x N_chains x E_cell (E_cell = 1.3 fJ), t = N_sweeps x N_blocks x t_update (t_update = 100 ns; chains in parallel on separate chip area). These are assumptions, not measurements of TSU hardware.

> chain: not converged in all runs; using full schedule (7000 sweeps) -- not a converged-cost estimate.

> tree: not converged in all runs; using full schedule (7000 sweeps) -- not a converged-cost estimate.

| variant | task | n_spins | n_blocks | n_chains | sweeps | tsu_energy_J | tsu_latency_s | laptop_time_s | laptop_energy_J | gpu_energy_J | gpu_latency_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chain | posterior | 7.43e+04 | 76 | 16 | 7000 | 1.08e-05 | 0.0532 | 213 | 4.27e+03 | 120 | 59.7 |
| chain | MAP anneal | 7.43e+04 | 76 | 16 | 600 | 9.27e-07 | 0.00456 | 18.3 | 366 | 10.3 | 5.12 |
| dense | posterior | 5.64e+03 | 315 | 16 | 2390 | 2.8e-07 | 0.0753 | 30.7 | 614 | 11 | 5.51 |
| dense | MAP anneal | 5.64e+03 | 315 | 16 | 600 | 7.04e-08 | 0.0189 | 7.7 | 154 | 2.77 | 1.38 |
| potts | posterior | 806 | 48 | 16 | 2300 | 3.86e-08 | 0.011 | 12.5 | 251 | 0.219 | 0.109 |
| potts | MAP anneal | 806 | 48 | 16 | 600 | 1.01e-08 | 0.00288 | 3.27 | 65.4 | 0.0571 | 0.0283 |
| tree | posterior | 7.2e+04 | 114 | 16 | 7000 | 1.05e-05 | 0.0798 | 269 | 5.38e+03 | 76.3 | 37.9 |
| tree | MAP anneal | 7.2e+04 | 114 | 16 | 600 | 8.99e-07 | 0.00684 | 23.1 | 461 | 6.54 | 3.25 |


Sweeps basis per variant:

- chain: 7000 sweeps -- full configured schedule (NOT converged in 100% of runs; value is a lower bound on cost to converge)
- dense: 2390 sweeps -- measured sweeps-to-converge (R-hat<1.05, median over runs) + warm-up
- potts: 2300 sweeps -- measured sweeps-to-converge (R-hat<1.05, median over runs) + warm-up
- tree: 7000 sweeps -- full configured schedule (NOT converged in 100% of runs; value is a lower bound on cost to converge)

## Laptop and GPU comparison

Laptop energy = measured wall time x CPU package power. Default power is an ASSUMED constant (--cpu-power-w, 20 W). For a measured value run, in a separate terminal during an m3 run: `sudo powermetrics --samplers cpu_power -i 200 > pm.log` and pass `--powermetrics-log pm.log` (parsed for 'CPU Power: N mW' lines; the mean is used). This script never runs sudo itself.

Caveat: the chain variant's measured time_s includes model build, Tikhonov warm start and JIT compile (the other variants exclude compile), so its laptop time/energy is an overestimate.

GPU: tomo.energy.gpu_mcmc_estimate (op-count model: 2*degree+10 flops/update, 1e-11 J/flop, 10 TFLOP/s peak at 10% efficiency, 400 GB/s memory); energy is flop-based (lower bound when memory bound). Laptop time apportioned between posterior and MAP by sweeps; spec spins/blocks for Potts count 'categorical' sites with K levels, so TSU numbers for Potts assume a (hypothetical) categorical sampler cell and are not hardware-faithful.

## Sensitivity (chain, posterior, 7000 sweeps, 16 chains)

| E_cell_factor | E_cell_J | t_update_s | energy_J | latency_s |
|---|---|---|---|---|
| 0.5 | 6.5e-16 | 5e-08 | 5.41e-06 | 0.0266 |
| 0.5 | 6.5e-16 | 1e-07 | 5.41e-06 | 0.0532 |
| 0.5 | 6.5e-16 | 5e-07 | 5.41e-06 | 0.266 |
| 1 | 1.3e-15 | 5e-08 | 1.08e-05 | 0.0266 |
| 1 | 1.3e-15 | 1e-07 | 1.08e-05 | 0.0532 |
| 1 | 1.3e-15 | 5e-07 | 1.08e-05 | 0.266 |
| 3 | 3.9e-15 | 5e-08 | 3.24e-05 | 0.0266 |
| 3 | 3.9e-15 | 1e-07 | 3.24e-05 | 0.0532 |
| 3 | 3.9e-15 | 5e-07 | 3.24e-05 | 0.266 |


## Hardware presets (tomo.energy.HARDWARE): logical vs embedded, with / without readout

- **spec**: Project spec: Extropic codon_opt figure 1.3 fJ per spin per Gibbs step (incl. RNG ~350 aJ, biasing, clocking, comms); t_update = 100 ns RNG decorrelation time per colour block.
- **extropic_2510**: Extropic arXiv:2510.23972 v2: E_cell ~ 2 fJ per cell, tau_0 ~ 100 ns (as quoted in docs/related_work.md; approximate, not re-verified against the paper text).
- **z1_2608**: Extropic arXiv:2608.01615 App. B, Table IV (SPICE-based, 50 MHz column): Gibbs update 7.09 fJ 'per pBIT node per Gibbs cycle at 50 MHz' (Sec. on the Z1 projection restates it as '7.09 fJ per p-bit per sweep, 20 ns sweeps, 25 us readout'); read 1.692 pJ per pBIT node; write (flash couplings/biases) 153.6 pJ per pBIT node; Z1 is a planar 2-colourable graph so one 20 ns cycle updates both colours (10 ns per colour block, derived). Their projection charges energy to sweeps only (readout excluded) and the 25 us readout is per frame readout.

Sweep counts are the posterior sweeps above (placeholders until convergence is measured). 'logical' rows use the unembedded spin/colour counts (not hostable on Z1: degree > 16); 'embedded' rows use the degree-16 copy-node embedded spin count and its greedy colour count; z1_2608 latency charges 20 ns per sweep only if the graph has <= 2 colours, otherwise 10 ns per colour block (flag not_2colourable). Readout = one full read of every physical node per chain, 25 us per frame.

| variant | preset | layout | readout | n_spins | n_blocks | n_sweeps | energy_J | latency_s | not_2colourable |
|---|---|---|---|---|---|---|---|---|---|
| chain | spec | logical | False | 7.43e+04 | 76 | 7000 | 1.08e-05 | 0.0532 | False |
| chain | spec | embedded | False | 620967 | 17 | 7000 | 9.04e-05 | 0.0119 | False |
| chain | spec | logical | True | 7.43e+04 | 76 | 7000 | 1.08e-05 | 0.0532 | False |
| chain | spec | embedded | True | 620967 | 17 | 7000 | 9.04e-05 | 0.0119 | False |
| chain | extropic_2510 | logical | False | 7.43e+04 | 76 | 7000 | 1.66e-05 | 0.0532 | False |
| chain | extropic_2510 | embedded | False | 620967 | 17 | 7000 | 0.000139 | 0.0119 | False |
| chain | extropic_2510 | logical | True | 7.43e+04 | 76 | 7000 | 1.66e-05 | 0.0532 | False |
| chain | extropic_2510 | embedded | True | 620967 | 17 | 7000 | 0.000139 | 0.0119 | False |
| chain | z1_2608 | logical | False | 7.43e+04 | 76 | 7000 | 5.9e-05 | 0.00532 | True |
| chain | z1_2608 | embedded | False | 620967 | 17 | 7000 | 0.000493 | 0.00119 | True |
| chain | z1_2608 | logical | True | 7.43e+04 | 76 | 7000 | 6.1e-05 | 0.00534 | True |
| chain | z1_2608 | embedded | True | 620967 | 17 | 7000 | 0.00051 | 0.00122 | True |
| dense | spec | logical | False | 5.64e+03 | 315 | 2390 | 2.8e-07 | 0.0753 | False |
| dense | spec | embedded | False | 214207 | 17 | 2390 | 1.06e-05 | 0.00406 | False |
| dense | spec | logical | True | 5.64e+03 | 315 | 2390 | 2.8e-07 | 0.0753 | False |
| dense | spec | embedded | True | 214207 | 17 | 2390 | 1.06e-05 | 0.00406 | False |
| dense | extropic_2510 | logical | False | 5.64e+03 | 315 | 2390 | 4.32e-07 | 0.0753 | False |
| dense | extropic_2510 | embedded | False | 214207 | 17 | 2390 | 1.64e-05 | 0.00406 | False |
| dense | extropic_2510 | logical | True | 5.64e+03 | 315 | 2390 | 4.32e-07 | 0.0753 | False |
| dense | extropic_2510 | embedded | True | 214207 | 17 | 2390 | 1.64e-05 | 0.00406 | False |
| dense | z1_2608 | logical | False | 5.64e+03 | 315 | 2390 | 1.53e-06 | 0.00753 | True |
| dense | z1_2608 | embedded | False | 214207 | 17 | 2390 | 5.81e-05 | 0.000406 | True |
| dense | z1_2608 | logical | True | 5.64e+03 | 315 | 2390 | 1.68e-06 | 0.00755 | True |
| dense | z1_2608 | embedded | True | 214207 | 17 | 2390 | 6.39e-05 | 0.000431 | True |
| potts | spec | logical | False | 806 | 48 | 2300 | 3.86e-08 | 0.011 | False |
| potts | spec | logical | True | 806 | 48 | 2300 | 3.86e-08 | 0.011 | False |
| potts | extropic_2510 | logical | False | 806 | 48 | 2300 | 5.93e-08 | 0.011 | False |
| potts | extropic_2510 | logical | True | 806 | 48 | 2300 | 5.93e-08 | 0.011 | False |
| potts | z1_2608 | logical | False | 806 | 48 | 2300 | 2.1e-07 | 0.0011 | True |
| potts | z1_2608 | logical | True | 806 | 48 | 2300 | 2.32e-07 | 0.00113 | True |
| tree | spec | logical | False | 7.2e+04 | 114 | 7000 | 1.05e-05 | 0.0798 | False |
| tree | spec | embedded | False | 190889 | 17 | 7000 | 2.78e-05 | 0.0119 | False |
| tree | spec | logical | True | 7.2e+04 | 114 | 7000 | 1.05e-05 | 0.0798 | False |
| tree | spec | embedded | True | 190889 | 17 | 7000 | 2.78e-05 | 0.0119 | False |
| tree | extropic_2510 | logical | False | 7.2e+04 | 114 | 7000 | 1.61e-05 | 0.0798 | False |
| tree | extropic_2510 | embedded | False | 190889 | 17 | 7000 | 4.28e-05 | 0.0119 | False |
| tree | extropic_2510 | logical | True | 7.2e+04 | 114 | 7000 | 1.61e-05 | 0.0798 | False |
| tree | extropic_2510 | embedded | True | 190889 | 17 | 7000 | 4.28e-05 | 0.0119 | False |
| tree | z1_2608 | logical | False | 7.2e+04 | 114 | 7000 | 5.72e-05 | 0.00798 | True |
| tree | z1_2608 | embedded | False | 190889 | 17 | 7000 | 0.000152 | 0.00119 | True |
| tree | z1_2608 | logical | True | 7.2e+04 | 114 | 7000 | 5.92e-05 | 0.00801 | True |
| tree | z1_2608 | embedded | True | 190889 | 17 | 7000 | 0.000157 | 0.00122 | True |
