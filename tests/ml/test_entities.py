from pathlib import Path

import pytest

import ttc
from ttc.ml.encoder import Encoder
from ttc.ml.entities import person_mentions, resolve_learned, resolve_rule
from ttc.ml.packages import ModelPackage
from ttc.ml.session import OnnxSession

MINI = Path(__file__).parent / "fixtures" / "mini_package"


@pytest.fixture(scope="module")
def ru_doc():
    cc = ttc.load("ru", pipeline="rules")
    # names must lemmatize consistently under both sm and lg models
    doc = cc.language("Маша посмотрела на Ивана. Иван молчал. Она ждала.")
    yield doc


def test_person_mentions_finds_names(ru_doc):
    texts = {m.text for m in person_mentions(ru_doc)}
    assert any("Маша" in t for t in texts)
    assert any("Иван" in t for t in texts)


def test_resolve_rule_clusters_repeated_name(ru_doc):
    entities = resolve_rule(ru_doc)
    ivan = [e for e in entities if "иван" in e.name.lower()]
    assert len(ivan) == 1
    assert len(ivan[0].mentions) == 2
    assert ru_doc._.characters == entities


def test_resolve_learned_runs_and_clusters(ru_doc):
    pkg = ModelPackage.from_dir(MINI)
    entities = resolve_learned(
        ru_doc, Encoder(pkg), OnnxSession(pkg.graph_path("pair"))
    )
    assert entities  # untrained weights: shape only
    assert all(e.mentions for e in entities)
