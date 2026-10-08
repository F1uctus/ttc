"""Aggregate accuracy floors over the annotated corpus."""

from pathlib import Path
from typing import Final

import pytest

import ttc
from ttc.eval import aggregate, evaluate_paths, format_report

TEXTS_PATH: Final = Path(__file__).parent / "texts"

TUNE_FLOOR: Final = 0.97
HELDOUT_FLOOR: Final = 0.60


@pytest.fixture(scope="module")
def cc():
    yield ttc.load("ru")


@pytest.mark.parametrize(
    "split, floor",
    [("tune", TUNE_FLOOR), ("heldout", HELDOUT_FLOOR)],
)
def test_attribution_floor(cc, split, floor):
    reports = evaluate_paths(cc, [TEXTS_PATH / split])
    assert reports, f"no corpus files in {split}"
    print(f"\n== {split}\n{format_report(reports, by_file=True)}")
    accuracy = aggregate(reports).end_to_end_accuracy
    assert accuracy is not None and accuracy >= floor, (
        f"{split} end-to-end accuracy {accuracy:.1%} fell below the"
        f" {floor:.0%} floor"
    )
