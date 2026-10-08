from dataclasses import dataclass

from spacy import Language
from spacy.tokens import Doc, Span


@dataclass
class Dialogue:
    language: Language
    doc: Doc
    replicas: list[Span]
