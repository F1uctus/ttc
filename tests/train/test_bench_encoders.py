from pathlib import Path

import pytest

pytestmark = pytest.mark.train

FIXTURES = Path(__file__).parent / "fixtures"


def test_bench_smoke(tmp_path: Path):
    from tests.train.test_data import _make_fixture
    from train.bench_encoders import bench
    from train.data import build_examples

    src = FIXTURES / "tiny_corpus.jsonl"
    if not src.exists():
        _make_fixture(src)
    build_examples([src], tmp_path)
    rows = bench(
        ["hf-internal-testing/tiny-random-BertModel"],
        tmp_path / "ranker.jsonl",
        sample=4,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.params_m > 0 and row.latency_ms > 0
    assert 0.0 <= row.probe_acc <= 1.0
