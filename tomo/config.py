"""Config loading (YAML with deep merge over configs/default.yaml) and saving."""
from __future__ import annotations

import copy
import os
from typing import Any, Mapping, Union

import yaml

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "configs", "default.yaml")


def deep_merge(base: dict, over: Mapping) -> dict:
    """Return a new dict: ``over`` merged recursively over ``base``.

    Dicts merge key by key; any other value (including lists, e.g. the cameras
    list) in ``over`` replaces the one in ``base``.
    """
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path_or_dict: Union[str, os.PathLike, Mapping, None] = None) -> dict:
    """Load ``configs/default.yaml`` and deep-merge a YAML file or dict over it.

    ``None`` returns the defaults. A path loads that YAML; a dict is used directly
    as overrides.
    """
    with open(DEFAULT_PATH) as f:
        cfg = yaml.safe_load(f) or {}
    if path_or_dict is None:
        return cfg
    if isinstance(path_or_dict, Mapping):
        over = path_or_dict
    else:
        with open(path_or_dict) as f:
            over = yaml.safe_load(f) or {}
    return deep_merge(cfg, over)


def save_config(cfg: Mapping[str, Any], out_dir: Union[str, os.PathLike]) -> str:
    """Write ``config.yaml`` into ``out_dir`` (created if needed); return its path."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "config.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(dict(cfg), f, sort_keys=False)
    return path
