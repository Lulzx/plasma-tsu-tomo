"""Hardware energy and latency estimation (spec: 'Hardware energy and latency estimation').

E_frame = N_spins * N_sweeps * N_chains * E_cell
t_frame = N_sweeps * N_blocks * t_update      (chains run in parallel on separate chip area)

Defaults: E_cell = 1.3 fJ per spin per Gibbs step (Extropic codon_opt figure; includes RNG
~350 aJ, biasing, clocking, communication), t_update = 100 ns (RNG decorrelation time).
These are placeholders to be replaced by measured sweep and block counts.
Every function returns its assumptions under the 'assumptions' key.
"""
from __future__ import annotations

import re

E_CELL_DEFAULT = 1.3e-15
T_UPDATE_DEFAULT = 100e-9


def tsu_estimate(n_spins, n_sweeps, n_chains, n_blocks, E_cell=E_CELL_DEFAULT, t_update=T_UPDATE_DEFAULT):
    """TSU energy (J) and latency (s) per frame."""
    return {
        "energy_J": float(n_spins) * n_sweeps * n_chains * E_cell,
        "latency_s": float(n_sweeps) * n_blocks * t_update,
        "assumptions": {
            "model": "E = N_spins*N_sweeps*N_chains*E_cell; t = N_sweeps*N_blocks*t_update",
            "n_spins": n_spins, "n_sweeps": n_sweeps, "n_chains": n_chains, "n_blocks": n_blocks,
            "E_cell_J": E_cell, "t_update_s": t_update,
            "chains_parallel_on_separate_chip_area": True,
        },
    }


def sensitivity(n_spins, n_sweeps, n_chains, n_blocks, E_cell_factors=(0.5, 1, 3),
                t_updates=(50e-9, 100e-9, 500e-9), E_cell=E_CELL_DEFAULT):
    """Table (list of dict rows) over E_cell factors x t_update values.

    Energy depends only on E_cell and latency only on t_update; rows give the full grid.
    """
    rows = []
    for f in E_cell_factors:
        for tu in t_updates:
            r = tsu_estimate(n_spins, n_sweeps, n_chains, n_blocks, E_cell * f, tu)
            rows.append({"E_cell_factor": f, "E_cell_J": E_cell * f, "t_update_s": tu,
                         "energy_J": r["energy_J"], "latency_s": r["latency_s"]})
    return {"rows": rows, "assumptions": tsu_estimate(n_spins, n_sweeps, n_chains, n_blocks, E_cell)["assumptions"]}


def worked_estimate(n_spins=40_600, n_blocks=44, n_chains=16, sweeps_posterior=2500, sweeps_map=600, **kw):
    """The spec's worked I-chain estimate: posterior and annealed-MAP rows."""
    return {"posterior": tsu_estimate(n_spins, sweeps_posterior, n_chains, n_blocks, **kw),
            "map": tsu_estimate(n_spins, sweeps_map, n_chains, n_blocks, **kw)}


_PM_RE = re.compile(r"(?:CPU|Combined)\s+Power:\s*([0-9.]+)\s*mW", re.I)


def cpu_energy_from_powermetrics(log_path_or_watts, seconds):
    """CPU energy in joules from a powermetrics log or an average power in watts.

    A path is parsed for 'CPU Power: N mW' lines (e.g. `powermetrics --samplers cpu_power`);
    energy = mean(power) * seconds.  A number is taken as average watts.
    """
    if isinstance(log_path_or_watts, (int, float)):
        w, n, src = float(log_path_or_watts), None, "watts given"
    else:
        with open(log_path_or_watts) as f:
            vals = [float(m.group(1)) for m in _PM_RE.finditer(f.read())]
        if not vals:
            raise ValueError("no 'CPU Power: N mW' lines found")
        w, n, src = sum(vals) / len(vals) / 1e3, len(vals), str(log_path_or_watts)
    return {"energy_J": w * seconds, "avg_power_W": w, "seconds": seconds,
            "assumptions": {"source": src, "n_power_samples": n,
                            "note": "CPU package power only; excludes DRAM/GPU/idle baseline subtraction"}}


def gibbs_op_counts(n_spins, n_sweeps, n_chains, degree):
    """Operation count for single-site Gibbs: per spin update ~ degree*2 flops (field sum) + RNG/sigmoid."""
    flops_per_update = 2.0 * degree + 10.0
    updates = float(n_spins) * n_sweeps * n_chains
    return {"updates": updates, "flops_per_update": flops_per_update, "flops": updates * flops_per_update}


def gpu_mcmc_estimate(n_spins, n_sweeps, n_chains, degree, J_per_flop=1e-11,
                      peak_flops=10e12, efficiency=0.1, bytes_per_update=None, mem_bw=400e9):
    """GPU block-Gibbs estimate from operation counts.

    Assumptions: flops/update = 2*degree+10; energy = flops*J_per_flop (1e-11 J/flop default,
    ~ 100 GFLOPs/W effective); time = max(compute, memory) where compute uses peak_flops*efficiency
    and memory moves bytes_per_update (default 4*degree + 8: coupling row + state) at mem_bw.
    Gibbs is typically memory-bound, so the memory time (and energy proportional to it at
    constant power) usually dominates; the flop-based energy is therefore optimistic.
    """
    c = gibbs_op_counts(n_spins, n_sweeps, n_chains, degree)
    bpu = 4.0 * degree + 8.0 if bytes_per_update is None else bytes_per_update
    t_comp = c["flops"] / (peak_flops * efficiency)
    t_mem = c["updates"] * bpu / mem_bw
    return {"energy_J": c["flops"] * J_per_flop, "latency_s": max(t_comp, t_mem),
            "memory_bound": t_mem > t_comp, "t_compute_s": t_comp, "t_memory_s": t_mem,
            "assumptions": {**c, "degree": degree, "J_per_flop": J_per_flop, "peak_flops": peak_flops,
                            "efficiency": efficiency, "bytes_per_update": bpu, "mem_bw_Bps": mem_bw,
                            "note": "memory-bound: flop-based energy is a lower bound"}}


def laptop_gibbs_estimate(n_spins, n_sweeps, n_chains, degree, flops_rate=2e9, power_W=15.0):
    """Laptop (single-core-ish) Gibbs estimate from op counts: t = flops/flops_rate, E = power_W * t.

    Replace with a measured powermetrics value (cpu_energy_from_powermetrics) when available.
    """
    c = gibbs_op_counts(n_spins, n_sweeps, n_chains, degree)
    t = c["flops"] / flops_rate
    return {"energy_J": power_W * t, "latency_s": t,
            "assumptions": {**c, "degree": degree, "effective_flops_rate": flops_rate, "power_W": power_W}}
