from pathlib import Path

from tests.corpora.util import assert_matches_golden
from ttc.corpora.riqua import convert
from ttc.corpora.schema import validate

FIXTURES = Path(__file__).parent / "fixtures" / "riqua"


def test_convert_mini_work():
    docs = list(convert(FIXTURES))
    assert len(docs) == 1
    doc = docs[0]
    assert doc.lang == "en" and doc.source == "riqua"
    assert validate(doc) == []
    assert len(doc.replicas) == 3
    r1, r2, r3 = doc.replicas
    assert doc.text[r1.start : r1.end] == '"Stop,"'
    assert doc.text[r1.cue.start : r1.cue.end] == "said"
    # each distinct entity span string becomes a character
    names = {c.id: c.name for c in doc.characters}
    assert names[r1.speaker] == "Mr. Bennet"
    assert names[r1.addressee] == "his wife"
    assert names[r2.speaker] == "she"
    assert (r1.mode, r2.mode, r3.mode) == ("speech", "speech", "indirect")
    assert_matches_golden(docs, FIXTURES / "golden.jsonl")
