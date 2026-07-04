from pathlib import Path

from ttc.corpora.droc import convert
from ttc.corpora.schema import validate
from tests.corpora.util import assert_matches_golden

FIXTURES = Path(__file__).parent / "fixtures" / "droc"


def test_convert_mini():
    docs = list(convert(FIXTURES))
    assert len(docs) == 1
    doc = docs[0]
    assert doc.lang == "de" and doc.source == "droc"
    assert validate(doc) == []
    # two directspeech spans + one thought; the bare "name" span is skipped
    assert len(doc.replicas) == 3
    r1, r2, r3 = doc.replicas
    assert doc.text[r1.start : r1.end] == "»Ich fürchte mich nicht.«"
    names = {c.id: c.name for c in doc.characters}
    # Speaker xmi:id -> NamedEntity -> cluster ID -> character
    assert names[r1.speaker] == "Effi"
    assert names[r2.speaker] == "Briest"
    assert r1.addressee == r2.speaker  # SpokenTo resolves to Briest
    assert r1.mode == "speech" and r2.mode == "speech"
    # a thought is a speaker-owned utterance: kept, attributed, tagged
    assert doc.text[r3.start : r3.end] == "nur Mut."
    assert r3.mode == "thought" and names[r3.speaker] == "Effi"
    assert len(doc.mentions) == 2
    assert_matches_golden(docs, FIXTURES / "golden.jsonl")
