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


def test_dialogue_chunks_caps_oversized_paragraph():
    from train.silver_corpus import dialogue_chunks

    # one oversized paragraph of single-newline dialogue
    big = "\n".join(f"— Реплика номер {i}, — сказал кто-то." for i in range(200))
    chunks = list(
        dialogue_chunks(big, target_chars=500, min_dialogue_lines=2, max_chars=800)
    )
    assert len(chunks) > 1
    assert all(len(c) <= 800 for c in chunks)
    assert all("—" in c for c in chunks)


def test_load_ru_prose_round_robin_interleaves_authors(monkeypatch):
    import datasets

    from train.silver_corpus import load_ru_prose

    fake = [
        {"type": "prose", "author": "A", "text": "a0"},
        {"type": "prose", "author": "A", "text": "a1"},
        {"type": "poems", "author": "A", "text": "poem"},
        {"type": "prose", "author": "B", "text": "b0"},
    ]
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: fake)

    got = [doc_id for doc_id, _ in load_ru_prose(round_robin=True)]
    assert [g.split("/")[0] for g in got] == ["a", "b", "a"]

    grouped = [doc_id for doc_id, _ in load_ru_prose(round_robin=False)]
    assert [g.split("/")[0] for g in grouped] == ["a", "a", "b"]


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
