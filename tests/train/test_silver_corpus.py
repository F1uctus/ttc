from pathlib import Path

import pytest

pytestmark = pytest.mark.train


def test_dialogue_chunks_keep_dialogue_and_break_on_paragraphs():
    from train.silver_corpus import dialogue_chunks

    text = "\n\n".join(
        [
            "Наррация без диалога. " * 15,
            "— Привет, — сказал Иван.\n— Здравствуй, — ответила Маша.",
            "Ещё длинная наррация тут. " * 15,
            "— Как дела? — спросил Иван.\n— Хорошо, — сказала Маша.",
        ]
    )
    chunks = list(dialogue_chunks(text, target_chars=200, min_dialogue_lines=2))
    assert chunks
    assert all("—" in c for c in chunks)
    assert all(c.count("\n\n") >= 0 for c in chunks)


def test_build_silver_corpus_writes_valid_jsonl(tmp_path: Path):
    import ttc
    from train.silver_corpus import build_silver_corpus
    from ttc.corpora.schema import read_jsonl

    cc = ttc.load("ru", pipeline="rules")
    text = (
        "— Привет, — сказала Ясна.\n"
        "— И тебе привет, — ответил Тозбек.\n"
        "— Идём дальше, — сказала Ясна.\n"
    )

    def fake_llm(prompt: str):
        return [
            {"replica_index": 0, "speaker": "Ясна"},
            {"replica_index": 1, "speaker": "Тозбек"},
            {"replica_index": 2, "speaker": "Ясна"},
        ]

    stats = build_silver_corpus(
        [("test/0", text)],
        tmp_path / "silver.jsonl",
        fake_llm,
        cc,
        max_docs=5,
        target_chars=50,
        min_attributed=1,
        chunks_per_work=5,
    )
    assert stats["docs"] >= 1
    assert stats["attributed"] >= 1
    assert stats["issues"] == 0

    docs = list(read_jsonl(tmp_path / "silver.jsonl"))
    assert docs and docs[0].source == "silver_llm"
    assert {c.name for c in docs[0].characters} <= {"Ясна", "Тозбек"}
