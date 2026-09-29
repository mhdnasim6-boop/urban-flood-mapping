"""Config loading, seeding and device selection."""
import copy
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from .bands import expand_channels


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path, overrides=()):
    """YAML with optional `base:` (relative path) inheritance and key.sub=value overrides."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    name = cfg.pop("name", path.stem)
    if "base" in cfg:
        cfg = _merge(load_config(path.parent / cfg.pop("base")), cfg)
    for o in overrides:
        key, val = o.split("=", 1)
        node = cfg
        *parents, last = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[last] = yaml.safe_load(val)
    cfg["name"] = name
    cfg["channels"] = expand_channels(cfg["channels"])
    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(name=None):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
