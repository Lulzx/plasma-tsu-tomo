PY ?= .venv/bin/python
.PHONY: test m1 m2 m3 m4 m5 m6 all quick clean-quick

test:
	$(PY) -m pytest -q
m1:
	$(PY) experiments/m1_geometry.py
m2:
	$(PY) experiments/m2_baselines.py
m3:
	$(PY) experiments/m3_ebm.py
m4:
	$(PY) experiments/m4_ablations.py
m5:
	$(PY) experiments/m5_energy.py
m6:
	$(PY) experiments/m6_summary.py
all: m1 m2 m3 m4 m5 m6

# smoke test: every script with tiny settings (outputs in results/*_quick/)
quick:
	$(PY) experiments/m1_geometry.py --quick
	$(PY) experiments/m2_baselines.py --quick
	$(PY) experiments/m3_ebm.py --quick --n-random 1
	$(PY) experiments/m4_ablations.py --quick
	$(PY) experiments/m5_energy.py --quick
	$(PY) experiments/m6_summary.py --quick
clean-quick:
	rm -rf results/*_quick
