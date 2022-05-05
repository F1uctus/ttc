from dataclasses import dataclass
from typing import List

from spacy import Language
from spacy.tokens import Token, Span, Doc


@dataclass
class Dialogue:
    language: Language
    doc: Doc
    replicas: List[Span]

    def __contains__(self, item: Token):
        return any(item in r for r in self.replicas)

    def __repr__(self) -> str:
        return "- " + "\n- ".join(repr(r) for r in self.replicas)
