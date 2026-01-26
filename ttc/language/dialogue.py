from dataclasses import dataclass
from typing import List

from spacy import Language
from spacy.tokens import Span, Doc


@dataclass
class Dialogue:
    language: Language
    doc: Doc
    replicas: List[Span]
