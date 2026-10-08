import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.train

FIXTURES = Path(__file__).parent / "fixtures"


def test_two_step_training_run(tmp_path: Path):
    from tests.train.test_data import _make_fixture
    from train.data import build_examples
    from train.train import run

    src = FIXTURES / "tiny_corpus.jsonl"
    if not src.exists():
        _make_fixture(src)
    build_examples([src], tmp_path / "examples")
    ckpt = run(
        Path("train/configs/base.cfg"),
        stage="pretrain_nonru",
        out_dir=tmp_path / "runs",
        overrides={
            "encoder.checkpoint": "hf-internal-testing/tiny-random-BertModel",
            "encoder.dim": "null",
            "training.steps": "2",
            "training.batch_size": "2",
            "training.mode": "live",
            "data.examples_dir": str(tmp_path / "examples"),
        },
    )
    assert ckpt.exists()
    meta = json.loads((ckpt.parent / "run.json").read_text())
    assert meta["stage"] == "pretrain_nonru" and "mix" in meta
