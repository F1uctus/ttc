from pathlib import Path

import pytest

pytestmark = pytest.mark.train

CFG = Path(__file__).parents[2] / "train" / "configs" / "base.cfg"


def test_base_config_loads_and_has_required_sections():
    from train.config import load_config

    cfg = load_config(CFG)
    assert cfg["encoder"]["window"] > 0
    assert cfg["encoder"]["checkpoint"]
    assert 0 < cfg["mixing"]["temperature"] <= 1
    assert cfg["training"]["seed"] == 20260704
    assert isinstance(cfg["stages"]["order"], list)


def test_overrides():
    from train.config import load_config

    cfg = load_config(CFG, {"training.seed": "7"})
    assert cfg["training"]["seed"] == 7


def test_set_seeds_deterministic():
    import torch

    from train.config import set_seeds

    set_seeds(1)
    a = torch.rand(3)
    set_seeds(1)
    assert torch.equal(a, torch.rand(3))
