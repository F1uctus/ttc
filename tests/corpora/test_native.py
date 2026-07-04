from pathlib import Path

from ttc.corpora.native import convert
from ttc.corpora.schema import validate
from tests.corpora.util import assert_matches_golden

FIXTURES = Path(__file__).parent / "fixtures" / "native"


def test_convert_sample():
    docs = list(convert(FIXTURES / "sample.txt"))
    assert len(docs) == 1
    doc = docs[0]
    assert doc.lang == "ru" and doc.domain == "prose" and doc.source == "native"
    assert validate(doc) == []
    # both replicas located in raw text and attributed
    assert [doc.text[r.start : r.end] for r in doc.replicas] == [
        "Привет,",
        "Здравствуй, светлость,",
    ]
    ids = {c.name: c.id for c in doc.characters}
    assert doc.replicas[0].speaker == ids["ясна"]
    assert doc.replicas[1].speaker == ids["тозбек"]
    # alias table becomes Character.aliases
    yasna = next(c for c in doc.characters if c.name == "ясна")
    assert set(yasna.aliases) == {"принцесса", "светлость"}
    assert_matches_golden(docs, FIXTURES / "golden.jsonl")
