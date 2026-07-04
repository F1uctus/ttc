from pathlib import Path

import pytest

import ttc

MINI = Path(__file__).parent / "fixtures" / "mini_package"
TEXT = (
    "Ясна вошла в комнату. Тозбек поднял голову.\n" "– Мы отплываем? – спросила Ясна.\n"
)


@pytest.fixture()
def hybrid_cc(monkeypatch):
    monkeypatch.setenv("TTC_MODEL_DIR", str(MINI))
    yield ttc.load("ru")  # auto -> hybrid


def test_hybrid_connect_play_returns_play(hybrid_cc):
    assert hybrid_cc.pipeline_mode == "hybrid"
    dialogue = hybrid_cc.extract_dialogue(TEXT)
    play = hybrid_cc.connect_play(dialogue)
    assert len(play) == len(dialogue.replicas)
    # the untrained ranker may abstain; the rule twin fills the gap
    actor = play.last_actor
    assert actor is not None and actor.text


def test_learned_mode_allows_unknown(monkeypatch):
    monkeypatch.setenv("TTC_MODEL_DIR", str(MINI))
    cc = ttc.load("ru", pipeline="learned")
    dialogue = cc.extract_dialogue(TEXT)
    play = cc.connect_play(dialogue)
    assert len(play) == len(dialogue.replicas)


def test_rules_mode_unaffected_by_package(monkeypatch):
    monkeypatch.setenv("TTC_MODEL_DIR", str(MINI))
    rules_cc = ttc.load("ru", pipeline="rules")
    monkeypatch.delenv("TTC_MODEL_DIR")
    bare_cc = ttc.load("ru")
    d1 = rules_cc.extract_dialogue(TEXT)
    d2 = bare_cc.extract_dialogue(TEXT)
    p1, p2 = rules_cc.connect_play(d1), bare_cc.connect_play(d2)
    assert [(str(r), str(a)) for r, a in p1.lines] == [
        (str(r), str(a)) for r, a in p2.lines
    ]
