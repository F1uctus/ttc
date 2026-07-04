from pathlib import Path

import pytest

import ttc
from ttc.corpora.native import convert
from ttc.eval import evaluate_interchange_doc, format_report

FIXTURES = Path(__file__).parent / "fixtures" / "native"


@pytest.fixture(scope="module")
def cc():
    yield ttc.load("ru")


def test_evaluate_interchange_doc(cc):
    doc = next(convert(FIXTURES / "sample.txt"))
    # give the doc qtypes so the breakdown has something to group
    doc.replicas[0].qtype = "explicit"
    doc.replicas[1].qtype = "explicit"
    report = evaluate_interchange_doc(cc, doc)
    assert report.n_gold == 2
    assert report.lang == "ru"
    assert "explicit" in report.qtype_counters
    assert report.qtype_counters["explicit"].n_gold == 2
    text = format_report([report])
    assert "explicit" in text
