from pathlib import Path

import pytest

pytestmark = pytest.mark.train

FIXTURES = Path(__file__).parent / "fixtures"


def _make_fixture(path: Path) -> None:
    from ttc.corpora.schema import (
        Character,
        CorpusDoc,
        Cue,
        Mention,
        Replica,
        write_jsonl,
    )

    text = '"Come here," said Emma. Harriet came in. "I am glad," she replied.'
    en = CorpusDoc(
        doc_id="pdnc/mini/0",
        lang="en",
        domain="prose",
        source="pdnc",
        license="CC-BY-NC-4.0",
        text=text,
        replicas=[
            Replica(
                text.index('"Come'),
                text.index("said") - 1,
                "c0",
                qtype="explicit",
                cue=Cue(text.index("said"), text.index("said") + 4),
            ),
            Replica(
                text.index('"I am'), text.index("she") - 1, "c1", qtype="anaphoric"
            ),
        ],
        characters=[Character("c0", "Emma"), Character("c1", "Harriet")],
        mentions=[
            Mention(text.index("Emma"), text.index("Emma") + 4, "c0"),
            Mention(text.index("Harriet"), text.index("Harriet") + 7, "c1"),
            Mention(text.index("she"), text.index("she") + 3, "c1"),
        ],
    )
    ru_text = "Городничий\nЯ пригласил вас, господа.\nАнна\nКак ревизор?\n"
    ru = CorpusDoc(
        doc_id="rusdracor/mini/0",
        lang="ru",
        domain="drama",
        source="rusdracor",
        license="CC0-1.0",
        text=ru_text,
        replicas=[
            Replica(ru_text.index("Я "), ru_text.index("господа.") + 8, "g"),
            Replica(ru_text.index("Как"), ru_text.index("?") + 1, "a"),
        ],
        characters=[Character("g", "Городничий"), Character("a", "Анна")],
        mentions=[
            Mention(0, 10, "g"),
            Mention(ru_text.index("Анна"), ru_text.index("Анна") + 4, "a"),
        ],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl([en, ru], path)


def test_build_examples(tmp_path: Path):
    import json

    from train.data import build_examples

    src = FIXTURES / "tiny_corpus.jsonl"
    if not src.exists():
        _make_fixture(src)
    counts = build_examples([src], tmp_path)
    assert counts["ranker"] == 4
    assert counts["cue"] >= 1
    assert counts["candidate"] >= 4
    assert counts["pair"] >= 1

    ranker = [
        json.loads(l) for l in (tmp_path / "ranker.jsonl").read_text().splitlines()
    ]
    ex = ranker[0]
    assert ex["text"].startswith("[LANG=en] [DOMAIN=prose] ")
    prefix = len("[LANG=en] [DOMAIN=prose] ")
    s, e = ex["candidates"][ex["gold"]]
    assert s >= prefix  # spans are window-relative incl. prefix
    assert ex["text"][s:e] in ("Emma", "Harriet", "she")


def test_audit_gate_blocks_unaudited_native(tmp_path: Path):
    from train.data import build_examples
    from ttc.corpora.schema import read_jsonl, write_jsonl

    src = FIXTURES / "tiny_corpus.jsonl"
    if not src.exists():
        _make_fixture(src)
    doc = next(read_jsonl(src))
    doc.source = "native"
    native = tmp_path / "native.jsonl"
    write_jsonl([doc], native)
    with pytest.raises(RuntimeError, match="audit"):
        build_examples([native], tmp_path / "out", audit_report=tmp_path / "nope.md")
    assert (
        build_examples(
            [native],
            tmp_path / "out",
            audit_report=tmp_path / "nope.md",
            allow_unaudited=True,
        )["ranker"]
        >= 1
    )
