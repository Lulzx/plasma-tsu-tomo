"""Hardware energy and latency estimation (spec: 'Hardware energy and latency estimation').

E_frame = N_spins * N_sweeps * N_chains * E_cell   (+ readout)
t_frame = N_sweeps * N_blocks * t_update           (chains run in parallel on separate chip area)

Default ('spec' preset, unchanged behaviour): E_cell = 1.3 fJ per spin per Gibbs step (Extropic
codon_opt figure; includes RNG ~350 aJ, biasing, clocking, communication), t_update = 100 ns per
colour block.  ``HARDWARE`` adds published presets (extropic_2510, z1_2608) with sources; see
``tsu_estimate``.  Sweep counts are placeholders to be replaced by measured values.
Every function returns its assumptions under the 'assumptions' key.
"""
from __future__ import annotations

import re

E_CELL_DEFAULT = 1.3e-15
T_UPDATE_DEFAULT = 100e-9

# Hardware presets.  'latency_basis' says what t_update_s / t_sweep_s means:
#   'per_block' : t_update_s is the time of ONE colour-block update; sweep = n_blocks * t_update_s
#   'per_sweep' : t_sweep_s is the time of one FULL sweep of a 2-colourable graph (both colour
#                 blocks, i.e. 10 ns per colour block).  Only hardware-native for n_blocks <= 2.
# E_cell_J is energy per spin per Gibbs update (per sweep for 'per_sweep' presets; each spin is
# updated once per sweep in both cases).  Readout/write numbers are per node, serial interface.
HARDWARE = {
    "spec": dict(
        E_cell_J=1.3e-15, t_update_s=100e-9, latency_basis="per_block",
        readout_J_per_node=0.0, readout_s_per_frame=0.0, write_J_per_node=0.0,
        source="Project spec: Extropic codon_opt figure 1.3 fJ per spin per Gibbs step (incl. RNG ~350 aJ, "
               "biasing, clocking, comms); t_update = 100 ns RNG decorrelation time per colour block."),
    "extropic_2510": dict(
        E_cell_J=2.0e-15, t_update_s=100e-9, latency_basis="per_block",
        readout_J_per_node=0.0, readout_s_per_frame=0.0, write_J_per_node=0.0,
        source="Extropic arXiv:2510.23972 v2: E_cell ~ 2 fJ per cell, tau_0 ~ 100 ns (as quoted in "
               "docs/related_work.md; approximate, not re-verified against the paper text)."),
    "z1_2608": dict(
        E_cell_J=7.09e-15, t_sweep_s=20e-9, t_update_s=10e-9, latency_basis="per_sweep",
        readout_J_per_node=1.692e-12, readout_s_per_frame=25e-6, write_J_per_node=153.6e-12,
        source="Extropic arXiv:2608.01615 App. B, Table IV (SPICE-based, 50 MHz column): Gibbs update 7.09 fJ "
               "'per pBIT node per Gibbs cycle at 50 MHz' (Sec. on the Z1 projection restates it as '7.09 fJ per "
               "p-bit per sweep, 20 ns sweeps, 25 us readout'); read 1.692 pJ per pBIT node; write (flash couplings/"
               "biases) 153.6 pJ per pBIT node; Z1 is a planar 2-colourable graph so one 20 ns cycle updates both "
               "colours (10 ns per colour block, derived). Their projection charges energy to sweeps only "
               "(readout excluded) and the 25 us readout is per frame readout."),
}


def _resolve(preset, E_cell, t_update):
    hw = HARDWARE[preset] if preset is not None else None
    if hw is None:
        hw = HARDWARE["spec"]
    E = hw["E_cell_J"] if E_cell is None else E_cell
    t = hw["t_update_s"] if t_update is None else t_update
    return hw, E, t


def tsu_estimate(n_spins, n_sweeps, n_chains, n_blocks, E_cell=None, t_update=None, *, preset=None,
                 include_readout=False, n_readout_nodes=None, n_readouts=1, n_physical_spins=None):
    """TSU energy (J) and latency (s) per frame.

    Default (``preset=None``) reproduces the original model with E_cell = 1.3 fJ and t_update = 100 ns::

        E = N_spins N_sweeps N_chains E_cell ;  t = N_sweeps N_blocks t_update

    ``preset`` selects an entry of ``HARDWARE`` (explicit ``E_cell`` / ``t_update`` still override).
    ``n_physical_spins`` (e.g. the copy-node embedded spin count from ``tomo.embed``) replaces
    ``n_spins`` in the energy.  Latency: ``per_block`` presets use ``n_sweeps * n_blocks * t_update``;
    ``per_sweep`` presets (z1_2608) use ``n_sweeps * t_sweep`` when ``n_blocks <= 2``; with more colour
    blocks the graph is not natively 2-colourable and we charge ``n_blocks * t_sweep / 2`` per sweep
    (i.e. 10 ns per block) and flag it in ``assumptions['not_2colourable']``.  Chains run in parallel
    on separate chip area.  ``include_readout`` adds ``readout_J_per_node * n_readout_nodes *
    n_readouts * n_chains`` and ``readout_s_per_frame * n_readouts`` (default one readout per frame;
    ``n_readout_nodes`` defaults to the spin count used for energy).  Readout and coupling write are
    zero in the 'spec' and 'extropic_2510' presets.
    """
    hw, E, t = _resolve(preset, E_cell, t_update)
    n_e = float(n_spins if n_physical_spins is None else n_physical_spins)
    flag = False
    if hw["latency_basis"] == "per_sweep" and t_update is None:
        ts = hw["t_sweep_s"]
        if n_blocks <= 2:
            lat_sweep = ts
        else:
            lat_sweep, flag = n_blocks * ts / 2.0, True
        lat = float(n_sweeps) * lat_sweep
    else:
        lat = float(n_sweeps) * n_blocks * t
    energy = n_e * n_sweeps * n_chains * E
    e_read = t_read = 0.0
    if include_readout:
        nn = n_e if n_readout_nodes is None else n_readout_nodes
        e_read = hw["readout_J_per_node"] * nn * n_readouts * n_chains
        t_read = hw["readout_s_per_frame"] * n_readouts
    return {
        "energy_J": energy + e_read,
        "latency_s": lat + t_read,
        "energy_sampling_J": energy, "energy_readout_J": e_read,
        "latency_sampling_s": lat, "latency_readout_s": t_read,
        "assumptions": {
            "model": "E = N_spins*N_sweeps*N_chains*E_cell (+ readout); t = N_sweeps*N_blocks*t_update or N_sweeps*t_sweep (+ readout)",
            "preset": preset or "spec(default)", "n_spins": n_spins, "n_physical_spins": n_physical_spins,
            "n_spins_energy": n_e, "n_sweeps": n_sweeps, "n_chains": n_chains, "n_blocks": n_blocks,
            "E_cell_J": E, "t_update_s": t, "latency_basis": hw["latency_basis"], "not_2colourable": flag,
            "include_readout": include_readout, "n_readouts": n_readouts if include_readout else 0,
            "chains_parallel_on_separate_chip_area": True, "source": hw["source"],
        },
    }


def compare_presets(n_spins, n_sweeps, n_chains, n_blocks, n_physical_spins=None, n_blocks_physical=None,
                    presets=None, include_readout=(False, True)):
    """Rows (list of dict) of energy/latency for each preset, with and without readout.

    ``n_blocks`` is the logical-model colour count (used by per_block presets); ``n_blocks_physical``
    (default ``n_blocks``) is the colour count of the embedded graph used with ``n_physical_spins``.
    The 'logical' rows use n_spins/n_blocks (a graph a real chip could not host); 'embedded' rows use
    the physical spin and block counts.
    """
    rows = []
    nbp = n_blocks if n_blocks_physical is None else n_blocks_physical
    for pr in (presets or list(HARDWARE)):
        for ro in include_readout:
            for lay, ns, nph, nb in (("logical", n_spins, None, n_blocks), ("embedded", n_spins, n_physical_spins, nbp)):
                if lay == "embedded" and n_physical_spins is None:
                    continue
                r = tsu_estimate(ns, n_sweeps, n_chains, nb, preset=pr, include_readout=ro, n_physical_spins=nph)
                rows.append(dict(preset=pr, layout=lay, readout=ro, n_spins=ns if nph is None else nph, n_blocks=nb,
                                 n_sweeps=n_sweeps, n_chains=n_chains, energy_J=r["energy_J"], latency_s=r["latency_s"],
                                 not_2colourable=r["assumptions"]["not_2colourable"]))
    return rows


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
