from pathlib import Path

import pytest

import ttc
from ttc.language.russian.dependency_patterns import (
    ACTION_VERB_CONJUNCT_ACTOR,
    ACTION_VERB_TO_ACTOR,
)
from ttc.ml.cues import detect_learned, detect_rule
from ttc.ml.encoder import Encoder
from ttc.ml.packages import ModelPackage
from ttc.ml.session import OnnxSession

MINI = Path(__file__).parent / "fixtures" / "mini_package"
RU_PATTERNS = {
    "verb_to_actor": ACTION_VERB_TO_ACTOR,
    "verb_conjunct_actor": ACTION_VERB_CONJUNCT_ACTOR,
}


@pytest.fixture(scope="module")
def ru_doc():
    cc = ttc.load("ru", pipeline="rules")
    yield cc.language("– Привет, – сказала Ясна и посмотрела в окно.")


def test_detect_rule_finds_speech_verb_and_actor(ru_doc):
    cues = detect_rule(ru_doc, RU_PATTERNS)
    assert cues
    verb, actor = cues[0]
    assert verb.text == "сказала"
    assert actor is not None and "Ясна" in actor.text
    assert ru_doc._.ttc_cues == cues


def test_detect_learned_shape(ru_doc):
    pkg = ModelPackage.from_dir(MINI)
    cues = detect_learned(ru_doc, Encoder(pkg), OnnxSession(pkg.graph_path("cue")))
    # untrained weights: structure only
    for verb, actor in cues:
        assert verb.doc is ru_doc
        assert actor is None or actor.doc is ru_doc
