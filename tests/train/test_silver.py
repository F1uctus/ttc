import pytest

pytestmark = pytest.mark.train


def test_silver_label_with_stub_llm():
    import ttc
    from train.silver import silver_label

    cc = ttc.load("ru", pipeline="rules")
    text = "— Привет, — сказала Ясна.\n— Как дела?\n"

    def stub_llm(prompt_text):
        return [{"replica_index": 0, "speaker": "Ясна"}]

    docs = list(silver_label([text], stub_llm, cc))
    assert len(docs) == 1
    d = docs[0]
    assert d.source == "silver_llm"
    assert any(r.speaker for r in d.replicas)


def test_make_llm_backend_parses_and_enumerates():
    import ttc
    from train.silver import make_llm, silver_label

    cc = ttc.load("ru", pipeline="rules")
    text = "— Привет, — сказала Ясна.\n— И тебе привет, — ответил Тозбек.\n"
    seen = {}

    def fake_respond(prompt: str) -> str:
        seen["prompt"] = prompt
        # JSON wrapped in prose and a code fence
        return (
            "Here you go:\n```json\n"
            '[{"replica_index":0,"speaker":"Ясна"},'
            '{"replica_index":1,"speaker":"Тозбек"},'
            '{"replica_index":2,"speaker":null}]\n```'
        )

    llm = make_llm(cc, fake_respond)
    labels = llm(text)
    assert labels == [
        {"replica_index": 0, "speaker": "Ясна"},
        {"replica_index": 1, "speaker": "Тозбек"},
    ]
    assert "Replicas (index: text):" in seen["prompt"]
    assert "0:" in seen["prompt"]

    docs = list(silver_label([text], make_llm(cc, fake_respond), cc))
    names = {c.name for c in docs[0].characters}
    assert {"Ясна", "Тозбек"} <= names


def test_openrouter_llm_requires_key(monkeypatch):
    import ttc
    from train.silver import openrouter_llm

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    cc = ttc.load("ru", pipeline="rules")
    with pytest.raises(RuntimeError, match="OpenRouter API key"):
        openrouter_llm(cc)
