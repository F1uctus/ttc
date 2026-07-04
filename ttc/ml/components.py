"""spaCy factory registrations for the learned components."""

from pathlib import Path

from spacy.language import Language
from spacy.tokens import Doc

from ttc.ml import extensions
from ttc.ml.encoder import Encoder
from ttc.ml.packages import ModelPackage


@Language.factory("ttc_encoder", default_config={"package_dir": ""})
def make_ttc_encoder(nlp: Language, name: str, package_dir: str):
    extensions.register()
    encoder = Encoder(ModelPackage.from_dir(Path(package_dir)))

    def component(doc: Doc) -> Doc:
        encoder.encode_doc(doc)
        return doc

    return component


@Language.factory(
    "ttc_char_resolver",
    default_config={"package_dir": "", "mode": "learned", "threshold": 0.5},
)
def make_ttc_char_resolver(
    nlp: Language, name: str, package_dir: str, mode: str, threshold: float
):
    from ttc.ml import entities
    from ttc.ml.session import OnnxSession

    extensions.register()
    if mode == "rule":
        return lambda doc: (entities.resolve_rule(doc), doc)[1]
    package = ModelPackage.from_dir(Path(package_dir))
    encoder = Encoder(package)
    pair = OnnxSession(package.graph_path("pair"))

    def component(doc: Doc) -> Doc:
        entities.resolve_learned(doc, encoder, pair, threshold)
        return doc

    return component


@Language.factory(
    "ttc_cue_detector", default_config={"package_dir": "", "mode": "learned"}
)
def make_ttc_cue_detector(nlp: Language, name: str, package_dir: str, mode: str):
    from ttc.ml import cues as cues_mod
    from ttc.ml.session import OnnxSession

    extensions.register()
    if mode == "rule":
        from ttc.language.russian.dependency_patterns import (
            ACTION_VERB_CONJUNCT_ACTOR,
            ACTION_VERB_TO_ACTOR,
        )

        patterns = {
            "verb_to_actor": ACTION_VERB_TO_ACTOR,
            "verb_conjunct_actor": ACTION_VERB_CONJUNCT_ACTOR,
        }
        return lambda doc: (cues_mod.detect_rule(doc, patterns), doc)[1]
    package = ModelPackage.from_dir(Path(package_dir))
    encoder = Encoder(package)
    session = OnnxSession(package.graph_path("cue"))

    def component(doc: Doc) -> Doc:
        cues_mod.detect_learned(doc, encoder, session)
        return doc

    return component


@Language.factory(
    "ttc_candidate_gen",
    default_config={
        "package_dir": "",
        "mode": "union",
        "window_chars": 1200,
        "top_k": 8,
    },
)
def make_ttc_candidate_gen(
    nlp: Language,
    name: str,
    package_dir: str,
    mode: str,
    window_chars: int,
    top_k: int,
):
    from ttc.ml import candidates as cand_mod
    from ttc.ml.session import OnnxSession

    extensions.register()
    encoder = scorer = None
    if mode in ("learned", "union") and package_dir:
        package = ModelPackage.from_dir(Path(package_dir))
        encoder = Encoder(package)
        scorer = OnnxSession(package.graph_path("candidate"))

    def component(doc: Doc) -> Doc:
        for replica in doc._.replicas:
            cand_mod.generate(
                doc,
                replica,
                doc._.characters,
                mode,
                encoder,
                scorer,
                window_chars,
                top_k,
            )
        return doc

    return component
