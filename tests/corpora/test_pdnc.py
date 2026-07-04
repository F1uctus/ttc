from pathlib import Path

from ttc.corpora.pdnc import convert
from ttc.corpora.schema import validate
from tests.corpora.util import assert_matches_golden

FIXTURES = Path(__file__).parent / "fixtures" / "pdnc"


def test_convert_mini_novel():
    docs = list(convert(FIXTURES))
    assert len(docs) == 1
    doc = docs[0]
    assert doc.lang == "en" and doc.domain == "prose" and doc.source == "pdnc"
    assert doc.license == "CC-BY-NC-4.0"
    assert validate(doc) == []
    assert len(doc.replicas) == 2
    r1, r2 = doc.replicas
    assert doc.text[r1.start : r1.end] == '"Come here,"'
    assert r1.qtype == "explicit" and r2.qtype == "anaphoric"
    names = {c.id: c.name for c in doc.characters}
    assert names[r1.speaker] == "Emma"
    assert r1.addressee == r2.speaker  # Harriet
    # referring expression text located near the quote -> cue span
    assert doc.text[r1.cue.start : r1.cue.end] == "said Emma"
    assert r2.cue is None  # "nan" referring expression
    assert any(doc.text[m.start : m.end] == "she" for m in doc.mentions)
    assert_matches_golden(docs, FIXTURES / "golden.jsonl")
