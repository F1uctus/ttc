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
    overrides = {
        "training.steps": "3",
        "training.batch_size": "2",
        "encoder.dim": "null",
    }
    cfg = load_config(Path("train/configs/base.cfg"), overrides)
    out = train_heads_cached(
        tmp_path / "cache", cfg, tmp_path / "runs", "pretrain_nonru", overrides
    )
    assert out.exists() and out.name == "heads.pt"
    meta = json.loads((out.parent / "run.json").read_text())
    assert meta["stage"] == "pretrain_nonru" and "mix" in meta
    from train.model import AttributionModel

    m = AttributionModel(TINY, dim=None, hidden=cfg["heads"]["hidden"])
    m.load_state_dict(torch.load(out), strict=False)


REPO = Path(__file__).resolve().parents[2]


def _tiny_cache(tmp_path: Path) -> Path:
    from tests.train.test_data import _make_fixture
    from train.cache import build_cache
    from train.data import build_examples

    src = FIXTURES / "tiny_corpus.jsonl"
    if not src.exists():
        _make_fixture(src)
    build = tmp_path / "build"
    build_examples([src], build / "examples")
    build_cache(build / "examples", build / "cache", TINY)
    return build / "examples"


def _overrides(examples: Path, **extra: str) -> dict:
    return {
        "training.steps": "2",
        "training.batch_size": "2",
        "encoder.dim": "null",
        "data.examples_dir": str(examples),
        **extra,
    }


def test_run_log_stays_in_out_dir(tmp_path: Path, monkeypatch):
    from train.train import run

    examples = _tiny_cache(tmp_path)
    work = tmp_path / "cwd"
    work.mkdir()
    monkeypatch.chdir(work)
    run(
        REPO / "train/configs/base.cfg",
        "pretrain_nonru",
        tmp_path / "runs",
        _overrides(examples),
    )
    assert (tmp_path / "runs" / "eval-log.md").exists()
    assert list(work.iterdir()) == []


def test_cached_stage_uses_only_its_sources(tmp_path: Path):
    from train.train import run

    examples = _tiny_cache(tmp_path)
    stage = json.dumps({"sources": ["pdnc"], "steps": 2})
    out = run(
        REPO / "train/configs/base.cfg",
        "finetune_ru",
        tmp_path / "runs",
        _overrides(examples, **{"stages.finetune_ru": stage}),
    )
    mix = json.loads((out.parent / "run.json").read_text())["mix"]
    assert {src for task_mix in mix.values() for src in task_mix} == {"pdnc"}


def test_cached_stage_without_examples_fails(tmp_path: Path):
    from train.train import run

    examples = _tiny_cache(tmp_path)
    with pytest.raises(RuntimeError, match="no cached examples"):
        # the fixture has only pdnc and rusdracor
        run(
            REPO / "train/configs/base.cfg",
            "finetune_ru",
            tmp_path / "runs",
            _overrides(examples),
        )


def test_cached_stage_starts_from_previous_heads(tmp_path: Path):
    from train.train import run

    examples = _tiny_cache(tmp_path)
    cfg_path = REPO / "train/configs/base.cfg"
    prev = run(cfg_path, "pretrain_nonru", tmp_path / "runs", _overrides(examples))
    # zero steps: heads must equal the previous stage
    stage = json.dumps({"sources": ["pdnc", "rusdracor"], "steps": 0})
    out = run(
        cfg_path,
        "distant_ru",
        tmp_path / "runs",
        _overrides(examples, **{"training.steps": "0", "stages.distant_ru": stage}),
    )
    before, after = torch.load(prev), torch.load(out)
    assert before.keys() == after.keys()
    assert all(torch.equal(before[k], after[k]) for k in before)


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
