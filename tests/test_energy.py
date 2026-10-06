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
