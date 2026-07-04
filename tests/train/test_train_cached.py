import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.train

FIXTURES = Path(__file__).parent / "fixtures"
TINY = "hf-internal-testing/tiny-random-BertModel"


def test_cached_heads_load_into_model():
    from train.model import AttributionModel, CachedHeads

    m = AttributionModel(TINY, dim=None, hidden=16)
    heads = CachedHeads(dim=m.dim, hidden=16)
    _, unexpected = m.load_state_dict(heads.state_dict(), strict=False)
    assert unexpected == []


def test_train_heads_cached_runs(tmp_path: Path):
    from tests.train.test_data import _make_fixture
    from train.cache import build_cache
    from train.config import load_config
    from train.data import build_examples
    from train.train import train_heads_cached

    src = FIXTURES / "tiny_corpus.jsonl"
    if not src.exists():
        _make_fixture(src)
    build_examples([src], tmp_path / "ex")
    build_cache(tmp_path / "ex", tmp_path / "cache", TINY)
    cfg = load_config(
        Path("train/configs/base.cfg"),
        {"training.steps": "3", "training.batch_size": "2", "encoder.dim": "null"},
    )
    out = train_heads_cached(
        tmp_path / "cache", cfg, tmp_path / "runs", "pretrain_nonru"
    )
    assert out.exists() and out.name == "heads.pt"
    meta = json.loads((out.parent / "run.json").read_text())
    assert meta["stage"] == "pretrain_nonru" and "mix" in meta
    from train.model import AttributionModel

    m = AttributionModel(TINY, dim=None, hidden=cfg["heads"]["hidden"])
    m.load_state_dict(torch.load(out), strict=False)


def test_run_routes_to_cached_by_default(tmp_path: Path):
    from tests.train.test_data import _make_fixture
    from train.cache import build_cache
    from train.data import build_examples
    from train.train import run

    src = FIXTURES / "tiny_corpus.jsonl"
    if not src.exists():
        _make_fixture(src)
    build = tmp_path / "build"
    build_examples([src], build / "examples")
    build_cache(build / "examples", build / "cache", TINY)
    out = run(
        Path("train/configs/base.cfg"),
        stage="pretrain_nonru",
        out_dir=tmp_path / "runs",
        overrides={
            "training.steps": "2",
            "training.batch_size": "2",
            "encoder.dim": "null",
            "data.examples_dir": str(build / "examples"),
        },
    )
    assert out.name == "heads.pt" and out.exists()  # default mode is "cached"
