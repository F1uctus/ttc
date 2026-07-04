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
