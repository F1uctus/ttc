import pytest

pytestmark = pytest.mark.train


def test_label_fiction_yields_docs():
    import ttc
    from train.distant import label_fiction

    cc = ttc.load("ru", pipeline="rules")
    text = "— Привет, — сказала Ясна. — Как дела?\n— Хорошо, — ответил Тозбек.\n"
    docs = list(label_fiction([text], cc))
    assert len(docs) == 1
    d = docs[0]
    assert d.source == "distant" and d.lang == "ru"
    assert len(d.replicas) >= 1
    ids = {c.id for c in d.characters}
    assert all(r.speaker in ids for r in d.replicas if r.speaker)
