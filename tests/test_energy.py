import numpy as np
import pytest
from tomo import energy as E


def test_worked_estimate():
    r = E.tsu_estimate(40_600, 2500, 16, 44)
    assert r["energy_J"] == pytest.approx(2.1e-6, rel=0.02)
    assert r["latency_s"] == pytest.approx(11e-3, rel=1e-9)
    m = E.tsu_estimate(40_600, 600, 16, 44)
    assert m["energy_J"] == pytest.approx(0.5e-6, rel=0.03)
    assert m["latency_s"] == pytest.approx(2.64e-3, rel=1e-9)
    assert "assumptions" in r


def test_sensitivity():
    t = E.sensitivity(40_600, 2500, 16, 44)
    assert len(t["rows"]) == 9
    lo = min(r["energy_J"] for r in t["rows"]); hi = max(r["energy_J"] for r in t["rows"])
    assert hi / lo == pytest.approx(6)
    assert max(r["latency_s"] for r in t["rows"]) == pytest.approx(55e-3)


def test_cpu_power(tmp_path):
    p = tmp_path / "pm.log"
    p.write_text("CPU Power: 2000 mW\nGPU Power: 5 mW\nCPU Power: 4000 mW\n")
    r = E.cpu_energy_from_powermetrics(str(p), 10)
    assert r["energy_J"] == pytest.approx(30.0)
    assert E.cpu_energy_from_powermetrics(5.0, 2)["energy_J"] == 10.0


def test_gpu_laptop():
    g = E.gpu_mcmc_estimate(40_600, 2500, 16, 6)
    assert g["energy_J"] == pytest.approx(40_600 * 2500 * 16 * 22 * 1e-11)
    assert g["assumptions"]["J_per_flop"] == 1e-11
    l = E.laptop_gibbs_estimate(40_600, 2500, 16, 6)
    assert l["latency_s"] > 0 and "assumptions" in l


def test_presets_and_default_unchanged():
    base = E.tsu_estimate(40_600, 2500, 16, 44)
    spec = E.tsu_estimate(40_600, 2500, 16, 44, preset="spec")
    assert base["energy_J"] == spec["energy_J"] and base["latency_s"] == spec["latency_s"]
    assert set(E.HARDWARE) >= {"spec", "extropic_2510", "z1_2608"}
    assert all(v["source"] for v in E.HARDWARE.values())
    e2510 = E.tsu_estimate(40_600, 2500, 16, 44, preset="extropic_2510")
    assert e2510["energy_J"] == pytest.approx(base["energy_J"] * 2.0 / 1.3)
    assert e2510["latency_s"] == pytest.approx(base["latency_s"])


def test_z1_preset_sweep_latency_and_readout():
    z = E.HARDWARE["z1_2608"]
    assert z["E_cell_J"] == 7.09e-15 and z["readout_J_per_node"] == 1.692e-12
    assert z["readout_s_per_frame"] == 25e-6 and z["write_J_per_node"] == 153.6e-12
    r = E.tsu_estimate(1000, 100, 4, 2, preset="z1_2608")
    assert r["latency_s"] == pytest.approx(100 * 20e-9)          # 2-colour graph: per-sweep time
    assert r["energy_J"] == pytest.approx(1000 * 100 * 4 * 7.09e-15)
    assert not r["assumptions"]["not_2colourable"]
    r3 = E.tsu_estimate(1000, 100, 4, 17, preset="z1_2608")
    assert r3["assumptions"]["not_2colourable"] and r3["latency_s"] == pytest.approx(100 * 17 * 10e-9)
    rr = E.tsu_estimate(1000, 100, 4, 2, preset="z1_2608", include_readout=True)
    assert rr["energy_J"] == pytest.approx(r["energy_J"] + 1.692e-12 * 1000 * 4)
    assert rr["latency_s"] == pytest.approx(r["latency_s"] + 25e-6)
    # readout is zero for presets without published readout
    assert E.tsu_estimate(1000, 100, 4, 2, include_readout=True)["energy_J"] == E.tsu_estimate(1000, 100, 4, 2)["energy_J"]


def test_physical_spins_energy_and_compare():
    a = E.tsu_estimate(1000, 10, 2, 5, preset="z1_2608", n_physical_spins=5000)
    assert a["energy_J"] == pytest.approx(5000 * 10 * 2 * 7.09e-15)
    rows = E.compare_presets(1000, 10, 2, 40, n_physical_spins=5000, n_blocks_physical=2)
    assert len(rows) == 3 * 2 * 2 and {r["layout"] for r in rows} == {"logical", "embedded"}
    # explicit overrides still win
    o = E.tsu_estimate(1000, 10, 2, 5, preset="z1_2608", E_cell=1e-15, t_update=1e-9)
    assert o["energy_J"] == pytest.approx(1000 * 10 * 2 * 1e-15) and o["latency_s"] == pytest.approx(10 * 5 * 1e-9)


def test_powermetrics_window(tmp_path):
    from tomo.energy import cpu_energy_in_window, powermetrics_samples
    blk = ("*** Sampled system activity (Tue Oct  6 16:36:{s:02d} 2026 +0530) (1000.00ms elapsed) ***\n"
           "junk\nCPU Power: {mw} mW\nGPU Power: 5 mW\n\n")
    f = tmp_path / "pm.log"
    f.write_text("".join(blk.format(s=s, mw=mw) for s, mw in [(1, 1000), (2, 3000), (3, 3000), (4, 1000)]))
    smp = powermetrics_samples(f)
    assert len(smp) == 4 and smp[1][2] == 3.0 and smp[1][0] - smp[0][0] == 1.0
    r = cpu_energy_in_window(f, smp[0][0], smp[2][0], idle_W=1.0)   # covers samples 2 and 3 exactly
    assert abs(r["avg_power_W"] - 3.0) < 1e-9 and abs(r["net_energy_J"] - 4.0) < 1e-9
