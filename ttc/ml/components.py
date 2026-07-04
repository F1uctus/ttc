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
