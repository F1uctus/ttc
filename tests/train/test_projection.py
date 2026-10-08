from pathlib import Path

import pytest

pytestmark = pytest.mark.train


def _identity_aligner():
    # maps each span to itself at conf 1
    class A:
        def align(self, src, tgt):
            return []

    return A()


def test_project_doc_identity(tmp_path: Path):
    from train.projection import project_doc
    from ttc.corpora.schema import Character, CorpusDoc, Mention, Replica

    text = "Anna said hello. Boris said bye."
    doc = CorpusDoc(
        doc_id="pdnc/x/0",
        lang="en",
        domain="prose",
        source="pdnc",
        license="CC-BY-NC-4.0",
        text=text,
        replicas=[Replica(text.index("hello"), text.index("hello") + 5, "a")],
        characters=[Character("a", "Anna"), Character("b", "Boris")],
        mentions=[
            Mention(0, 4, "a"),
            Mention(text.index("Boris"), text.index("Boris") + 5, "b"),
        ],
    )

    class IdAligner:
        def align(self, src, tgt):
            return [((0, len(src)), (0, len(tgt)), 1.0)]

    projected, _ = project_doc(doc, lambda s: s, "ru", IdAligner())
    assert projected.lang == "ru" and projected.source == "projected"
    r = projected.replicas[0]
    assert projected.text[r.start : r.end] == "hello"
    assert projected.replicas[0].speaker == "a"


def test_low_confidence_dropped_and_flagged():
    from train.projection import project_doc
    from ttc.corpora.schema import Character, CorpusDoc, Mention, Replica

    text = "Anna said hi."
    doc = CorpusDoc(
        doc_id="pdnc/x/1",
        lang="en",
        domain="prose",
        source="pdnc",
        license="CC-BY-NC-4.0",
        text=text,
        replicas=[Replica(text.index("hi"), text.index("hi") + 2, "a")],
        characters=[Character("a", "Anna")],
        mentions=[Mention(0, 4, "a")],
    )

    class LowAligner:
        def align(self, src, tgt):
            return [((0, len(src)), (0, len(tgt)), 0.1)]

    projected, flags = project_doc(doc, lambda s: s, "ru", LowAligner(), min_conf=0.5)
    assert projected.replicas == []
    assert flags and flags[0]["kind"] == "replica"


def test_project_span_preserves_contiguous_multiword():
    from train.projection import _project_span

    alignment = [((0, 5), (0, 5), 0.9), ((6, 11), (6, 11), 0.9)]
    (ts, te), conf = _project_span((0, 11), alignment)
    assert (ts, te) == (0, 11)
    assert conf > 0.8


def test_project_span_flags_reordering_balloon():
    from train.projection import _project_span

    # reordering: the two words land at opposite ends of the target
    alignment = [((0, 5), (20, 25), 1.0), ((8, 14), (0, 6), 1.0)]
    _, conf = _project_span((0, 14), alignment)
    assert conf < 0.5


def test_project_span_boundary_pair_does_not_dominate():
    from train.projection import _project_span

    # a 1-char overlap with a far token must not balloon the span
    good = [((i * 6, i * 6 + 5), (i * 6, i * 6 + 5), 0.9) for i in range(4)]
    bad = [((23, 25), (100, 120), 0.05)]
    _, conf = _project_span((0, 24), good + bad)
    assert conf < 0.5
