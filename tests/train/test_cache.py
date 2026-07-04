from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.train

FIXTURES = Path(__file__).parent / "fixtures"
TINY = "hf-internal-testing/tiny-random-BertModel"


def test_build_cache_shapes(tmp_path: Path):
    from tests.train.test_data import _make_fixture
    from train.cache import build_cache, load_task
    from train.data import build_examples

    src = FIXTURES / "tiny_corpus.jsonl"
    if not src.exists():
        _make_fixture(src)
    build_examples([src], tmp_path / "ex")
    counts = build_cache(tmp_path / "ex", tmp_path / "cache", TINY)

    cand = load_task(tmp_path / "cache", "candidate")
    dim = (cand["X"].shape[1] - 2) // 2
    assert cand["X"].shape[1] == 2 * dim + 2
    assert cand["X"].dtype == np.float32

    rank = load_task(tmp_path / "cache", "ranker")
    assert rank["X"].shape[1] == 2 * dim + 2
    assert int(rank["groups"].sum()) == rank["X"].shape[0]
    assert len(rank["gold"]) == counts["ranker"]

    pair = load_task(tmp_path / "cache", "pair")
    assert pair["X"].shape[1] == 3 * dim

    cue = load_task(tmp_path / "cache", "cue")
    assert cue["emb"].shape[1] == dim
    assert int(cue["lengths"].sum()) == cue["emb"].shape[0] == len(cue["bio"])
