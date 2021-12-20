from attr import attrs, attrib
from spacy import Language
from spacy.tokens import Doc

from .replica import Replica


@attrs(slots=True, repr=False)
class Dialogue:
    nlp: Language = attrib()
    replicas: list[Replica] = attrib()

    @property
    def doc(self) -> Doc:
        return self.replicas[0].doc

    def __repr__(self) -> str:
        return "\n\n".join(repr(r) for r in self.replicas)
