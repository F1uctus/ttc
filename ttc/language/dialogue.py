from typing import List

from attr import attrs
from spacy import Language
from spacy.tokens import Token, Span, Doc


@attrs(init=False, repr=False)
class Dialogue:
    def __init__(self, language: Language, replicas: List[Span]):
        self.language = language
        self.replicas = replicas
        self.doc: Doc = self.replicas[0].doc

    def __contains__(self, item: Token):
        return any(item in r for r in self.replicas)

    def __repr__(self) -> str:
        return "- " + "\n- ".join(repr(r) for r in self.replicas)
