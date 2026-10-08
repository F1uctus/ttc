from pathlib import Path

from tests.corpora.util import assert_matches_golden
from ttc.corpora.jy_quoteplus import convert
from ttc.corpora.schema import validate

FIXTURES = Path(__file__).parent / "fixtures" / "jy_quoteplus"


def test_convert_mini_items():
    docs = list(convert(FIXTURES))
    assert len(docs) == 2  # one doc per quote item
    d1, d2 = docs
    assert d1.lang == "zh" and d1.source == "jy_quoteplus"
    assert validate(d1) == [] and validate(d2) == []
    (r1,), (r2,) = d1.replicas, d2.replicas
    assert d1.text[r1.start : r1.end] == "“师父，我们走吧。”"
    names1 = {c.id: c.name for c in d1.characters}
    names2 = {c.id: c.name for c in d2.characters}
    assert names1[r1.speaker] == "郭靖" and names2[r2.speaker] == "黄蓉"
    assert names1[r1.addressee] == "黄蓉"
    assert d1.text[r1.cue.start : r1.cue.end] == "说道"
    # speaker mention located before the quote
    assert d1.text[d1.mentions[0].start : d1.mentions[0].end] == "郭靖"
    assert_matches_golden(docs, FIXTURES / "golden.jsonl")
