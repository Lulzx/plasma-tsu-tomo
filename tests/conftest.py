import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tomo.config import load_config  # noqa: E402
from tomo.forward import make_problem  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def problem(cfg):
    """Default 32x32 problem (peaked phantom, seed 0)."""
    return make_problem(cfg, "peaked", 0)


@pytest.fixture(scope="session")
def tiny_cfg():
    """12x12 reconstruction grid, 48x48 data grid, 8 chords per camera."""
    c = load_config({"grid": {"n": 12}, "data_grid": {"n": 48}})
    for cam in c["cameras"]:
        cam["n_chords"] = 8
    return c


@pytest.fixture(scope="session")
def tiny_problem(tiny_cfg):
    return make_problem(tiny_cfg, "peaked", 0)
