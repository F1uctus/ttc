from pathlib import Path

from tests.corpora.util import assert_matches_golden
from ttc.corpora.quoteli3 import convert
from ttc.corpora.schema import validate

FIXTURES = Path(__file__).parent / "fixtures" / "quoteli3"


def test_convert_mini():
    docs = list(convert(FIXTURES))
    assert len(docs) == 1
    doc = docs[0]
    assert doc.lang == "en" and doc.source == "quoteli3"
    assert validate(doc) == []
    assert len(doc.replicas) == 1
    r = doc.replicas[0]
    assert doc.text[r.start : r.end] == '"Uncle, where are we going?"'
    names = {c.id: c.name for c in doc.characters}
    assert names[r.speaker] == "Yegorushka"
    assert len(doc.mentions) == 2
    assert doc.characters[0].gender == "m"
    assert "Egorushka" in [a for c in doc.characters for a in c.aliases]
    assert_matches_golden(docs, FIXTURES / "golden.jsonl")
