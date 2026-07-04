from pathlib import Path

from ttc.corpora.rusdracor import convert, parse_tei
from ttc.corpora.schema import validate
from tests.corpora.util import assert_matches_golden

FIXTURES = Path(__file__).parent / "fixtures" / "rusdracor"


def test_parse_mini_play():
    doc = parse_tei(
        (FIXTURES / "mini-play.xml").read_text(encoding="utf-8"),
        doc_id="rusdracor/mini-play",
    )
    assert doc.lang == "ru" and doc.domain == "drama"
    assert validate(doc) == []
    assert [c.id for c in doc.characters] == ["gorodnichij", "anna"]
    assert doc.characters[0].gender == "m" and doc.characters[1].gender == "f"
    assert len(doc.replicas) == 2
    assert doc.replicas[0].speaker == "gorodnichij"
    assert doc.text[doc.replicas[1].start : doc.replicas[1].end] == "Как ревизор?"
    # speaker labels are in text and emitted as mentions
    assert doc.mentions[0].char == "gorodnichij"
    assert doc.text[doc.mentions[0].start : doc.mentions[0].end].startswith(
        "Городничий"
    )


def test_convert_dir():
    docs = list(convert(FIXTURES))
    assert len(docs) == 1
    assert_matches_golden(docs, FIXTURES / "golden.jsonl")
