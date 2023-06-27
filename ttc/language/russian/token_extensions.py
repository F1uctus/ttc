from typing import Union

from spacy.tokens import Token, Span

from ttc.language.common.constants import HYPHENS
from ttc.language.common.token_extensions import (
    morph_equals,
)
from ttc.language.russian.constants import SPEAKING_VERBS


def is_hyphen(self: Token):
    return self.text in HYPHENS


def is_speaking_verb(self: Token):
    """Used in dependency matching"""
    return any(v in self.lemma_ for v in SPEAKING_VERBS)


def is_not_second_person(self: Token):
    return "Person=Second" not in self.morph


def ref_matches(ref: Union[Token, Span], target: Span) -> bool:
    if isinstance(ref, Span):
        ref = ref.root
    return morph_equals(target.root, ref, "Gender", "Number")


TOKEN_EXTENSIONS = {
    name: {"getter": f}
    for name, f in locals().items()
    if callable(f) and f.__module__ == __name__
}
