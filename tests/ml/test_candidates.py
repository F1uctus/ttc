from pathlib import Path

import pytest

import ttc
from ttc.ml.candidates import generate
from ttc.ml.encoder import Encoder
from ttc.ml.entities import resolve_rule
from ttc.ml.packages import ModelPackage
from ttc.ml.session import OnnxSession

MINI = Path(__file__).parent / "fixtures" / "mini_package"
TEXT = (
    "Ясна вошла в комнату. Тозбек поднял голову.\n" "– Мы отплываем? – спросила Ясна.\n"
)


@pytest.fixture(scope="module")
def doc_and_replica():
    cc = ttc.load("ru", pipeline="rules")
    dialogue = cc.extract_dialogue(TEXT)
    assert dialogue.replicas, "replicizer must find the replica"
    yield dialogue.doc, dialogue.replicas[0]


def test_rule_provider_recalls_both_names(doc_and_replica):
    doc, replica = doc_and_replica
    entities = resolve_rule(doc)
    cands = generate(doc, replica, entities, mode="rule")
    names = " ".join(c.name.lower() for c in cands)
    assert "ясна" in names and "тозбек" in names
    assert replica._.speaker_candidates == cands


def test_union_contains_rule_candidates(doc_and_replica):
    doc, replica = doc_and_replica
    entities = resolve_rule(doc)
    pkg = ModelPackage.from_dir(MINI)
    rule = generate(doc, replica, entities, mode="rule")
    union = generate(
        doc,
        replica,
        entities,
        mode="union",
        encoder=Encoder(pkg),
        scorer=OnnxSession(pkg.graph_path("candidate")),
    )
    assert {c.id for c in rule} <= {c.id for c in union}


def test_top_k_respected(doc_and_replica):
    doc, replica = doc_and_replica
    entities = resolve_rule(doc)
    assert len(generate(doc, replica, entities, mode="rule", top_k=1)) == 1
