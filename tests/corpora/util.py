import os
from pathlib import Path
from typing import List

from ttc.corpora.schema import CorpusDoc, read_jsonl, to_dict, write_jsonl


def assert_matches_golden(docs: List[CorpusDoc], golden: Path) -> None:
    if os.environ.get("TTC_REGEN_GOLDENS"):
        write_jsonl(docs, golden)
    assert golden.exists(), f"golden {golden} missing; run with TTC_REGEN_GOLDENS=1"
    expected = list(read_jsonl(golden))
    assert [to_dict(d) for d in docs] == [to_dict(d) for d in expected]
