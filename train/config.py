"""Confection config loading with dotted overrides and global seeding."""

import json
import random
from pathlib import Path

import numpy as np
from confection import Config


def load_config(path: Path, overrides: dict[str, str] | None = None) -> Config:
    cfg = Config().from_disk(path)
    for dotted, value in (overrides or {}).items():
        section, key = dotted.split(".", 1)
        try:
            cfg[section][key] = json.loads(value)
        except json.JSONDecodeError:
            cfg[section][key] = value
    return cfg


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
