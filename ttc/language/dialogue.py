from dataclasses import dataclass

from spacy import Language
from spacy.tokens import Doc

from .replica import Replica


@dataclass(slots=True)
class Dialogue:
    language: Language
    replicas: list[Replica]

    @property
    def doc(self) -> Doc:
        return self.replicas[0].doc

    def __repr__(self) -> str:
        return "\n\n".join(repr(r) for r in self.replicas)
