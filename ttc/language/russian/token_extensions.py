from typing import Callable

from spacy.symbols import VERB, nsubj, obj, obl  # type: ignore
from spacy.tokens import Token, Span

from ttc.language.russian.constants import (
    HYPHENS,
    OPEN_QUOTES,
    CLOSE_QUOTES,
    SPEAKING_VERBS,
)


def is_sent_end(self: Token):
    return self.is_sent_end


def is_hyphen(self: Token):
    return self.text in HYPHENS


def is_open_quote(self: Token):
    return self.text in OPEN_QUOTES


def is_close_quote(self: Token):
    return self.text in CLOSE_QUOTES


def is_speaking_verb(self: Token):
    return any(v in self.lemma_ for v in SPEAKING_VERBS)


def is_not_second_person(self: Token):
    return "Person=Second" not in self.morph


def has_newline(self: Token):
    start = self.idx + len(self.text)
    return self.i == len(self.doc) - 1 or any(
        start + i in self.doc._.nl_indices
        for i in range(len(self.whitespace_) - 1, -1, -1)
    )


def morph_equals(
    self: Token,
    other: Token,
    *morphs: str,  # Morph
) -> bool:
    if self.morph is None or other.morph is None:
        return False
    return all(self.morph.get(k) == other.morph.get(k) for k in morphs)


def has_linked_verb(self: Token):
    head: Token = self.head
    while head and head.pos != VERB:
        if head == head.head:
            return False
        head = head.head
    else:
        return True


def as_span(self: Token) -> Span:
    return self.doc[self.i : self.i + 1]


def non_word(self: Token) -> bool:
    return self.is_punct or has_newline(self)


def expand_to_matching_noun_chunk(self: Token) -> Span:
    for nc in self.sent.noun_chunks:
        if self in nc:
            return nc
    # cannot expand noun, use token as-is
    return as_span(self)


def contains_near(self: Token, radius: int, predicate: Callable[[Token], bool]) -> bool:
    return any(
        predicate(t)
        for t in self.doc[max(0, self.i - radius) : min(len(self.doc), self.i + radius)]
    )


# from least to most important
SPEAKER_DEP_ORDER = [obl, obj, nsubj]


def speaker_dep_order(self: Token) -> int:
    try:
        return SPEAKER_DEP_ORDER.index(self.dep)
    except ValueError:
        return -1


def ref_match(ref: Token, target: Token) -> bool:
    return morph_equals(target, ref, "Gender", "Number")


TOKEN_EXTENSIONS = {
    name: {"getter": f}
    for name, f in locals().items()
    if callable(f) and f.__module__ == __name__
}
