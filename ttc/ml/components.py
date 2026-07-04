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
