from typing import Any

from attr import attrs, attrib
from spacy.tokens import Token, Doc


def _check_tokens(instance: "Replica", attribute: Any, value: list[Token]) -> None:
    if len(value) == 0:
        raise ValueError("Cannot construct an empty Replica")


@attrs(slots=True, cmp=False, repr=False)
class Replica:
    tokens: list[Token] = attrib(validator=_check_tokens)

    @property
    def doc(self) -> Doc:
        return self.tokens[0].doc

    @property
    def start_index(self) -> int:
        return self.tokens[0].idx

    @property
    def end_index(self) -> int:
        return self.tokens[-1].idx + len(self.tokens[-1])

    def __eq__(self, other):
        return repr(self) == repr(other)

    def __hash__(self):
        return hash(repr(self))

    def __repr__(self) -> str:
        return str.strip("".join(t.text_with_ws for t in self.tokens))
