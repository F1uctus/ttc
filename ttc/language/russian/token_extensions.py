from typing import Optional, Union

from spacy.symbols import (  # type: ignore
    VERB,
    AUX,
    NOUN,
    PROPN,
    PRON,
    DET,
    ADJ,
    NUM,
    nsubj,
    obj,
    obl,
)
from spacy.tokens import Token, Span

from ttc.language.common.constants import HYPHENS
from ttc.language.common.token_extensions import (
    morph_equals,
    as_span,
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


def is_speaker_noun(self: Token) -> bool:
    return bool(
        self.pos in {PROPN, PRON, NOUN, ADJ, NUM, DET}
        and set(self.morph.get("Case")).intersection({"Nom", "Acc", "Dat", "Gen"})
        and is_not_second_person(self)
    )


def lowest_linked_verbal(t: Token) -> Optional[Span]:
    p = [t.head]
    while p[0] and p[0] != p[0].head:
        p = [p[0].head] + p
    for child in p[0].children:
        if child.dep_ == "cop":
            return t.doc[child.i : p[0].i + 1]
    return next((as_span(t) for t in p if t.pos in {VERB, AUX}), None)


TOKEN_EXTENSIONS = {
    name: {"getter": f}
    for name, f in locals().items()
    if callable(f) and f.__module__ == __name__
}
