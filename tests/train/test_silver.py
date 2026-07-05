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


def test_rotating_respond_rotates_past_nonretryable():
    from train.silver import _ProviderError, _rotating_respond

    seen = []

    def bad(_p):
        seen.append("bad")
        raise _ProviderError("down", retryable=False)

    def good(_p):
        seen.append("good")
        return "[ok]"

    respond = _rotating_respond([("bad", bad), ("good", good)], max_retries=3)
    assert respond("x") == "[ok]"
    assert seen == ["bad", "good"]


def test_rotating_respond_retries_then_rotates():
    from train.silver import _ProviderError, _rotating_respond

    calls = {"n": 0}

    def flaky(_p):
        calls["n"] += 1
        raise _ProviderError("limited", retryable=True, wait=0.0)

    def good(_p):
        return "[]"

    respond = _rotating_respond([("flaky", flaky), ("good", good)], max_retries=3)
    assert respond("x") == "[]"
    assert calls["n"] == 3


def test_rotating_respond_raises_when_all_exhausted():
    from train.silver import _ProviderError, _rotating_respond

    def bad(_p):
        raise _ProviderError("nope", retryable=False)

    respond = _rotating_respond([("a", bad), ("b", bad)], max_retries=1)
    with pytest.raises(_ProviderError):
        respond("x")


def test_agent_provider_sniffs_quota_marker(monkeypatch):
    import subprocess

    from train.silver import _agent_provider, _ProviderError

    class R:  # quota error on stdout with exit code 0
        returncode = 0
        stdout = "ActionRequiredError: You've hit your usage limit"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: R())
    with pytest.raises(_ProviderError, match="agent"):
        _agent_provider("auto", 10)("prompt")


def test_agent_provider_returns_json_output(monkeypatch):
    import subprocess

    from train.silver import _agent_provider

    class R:
        returncode = 0
        stdout = '[{"replica_index":0,"speaker":"Ясна"}]'

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: R())
    assert "Ясна" in _agent_provider("auto", 10)("prompt")


def test_mixed_llm_agent_only_labels(monkeypatch):
    import subprocess

    import ttc
    from train.silver import mixed_llm

    class R:
        returncode = 0
        stdout = (
            '[{"replica_index":0,"speaker":"Ясна"},'
            '{"replica_index":1,"speaker":"Тозбек"}]'
        )

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: R())
    cc = ttc.load("ru", pipeline="rules")
    llm = mixed_llm(cc, use_openrouter=False, agent_model="auto")
    labels = llm("— Привет, — сказала Ясна.\n— И тебе, — ответил Тозбек.\n")
    assert {d["speaker"] for d in labels} == {"Ясна", "Тозбек"}


def test_mixed_llm_requires_a_provider(monkeypatch):
    import ttc
    from train.silver import mixed_llm

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    cc = ttc.load("ru", pipeline="rules")
    with pytest.raises(RuntimeError, match="no providers"):
        mixed_llm(cc, use_agent=False)  # no key and agent disabled
