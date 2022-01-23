from typing import Callable

from spacy.symbols import VERB  # type: ignore
from spacy.tokens import Token, Span

from ttc.language.russian.constants import (
    HYPHENS,
    OPEN_QUOTES,
    CLOSE_QUOTES,
    SPEAKING_VERBS,
)

PREDICATIVE_TOKEN_EXTENSIONS = {
    "is_sent_end": lambda t: t.is_sent_end,
    "is_hyphen": lambda t: t.text in HYPHENS,
    "is_newline": lambda t: "\n" in t.text,
    "is_open_quote": lambda t: t.text in OPEN_QUOTES,
    "is_close_quote": lambda t: t.text in CLOSE_QUOTES,
    "is_speaking_verb": lambda t: any(v in t.lemma_ for v in SPEAKING_VERBS),
    "is_not_second_person": lambda t: "Person=Second" not in t.morph,
}


def morph_equals(
    a: Token,
    b: Token,
    *morphs: str,  # Morph
) -> bool:
    if a.morph is None or b.morph is None:
        return False
    return all(a.morph.get(k) == b.morph.get(k) for k in morphs)


def has_linked_verb(t: Token):
    head: Token = t.head
    while head and head.pos != VERB:
        if head == head.head:
            return False
        head = head.head
    else:
        return True


def token_as_span(t: Token) -> Span:
    return t.doc[t.i : t.i + 1]


def expand_to_matching_noun_chunk(t: Token) -> Span:
    for nc in t.sent.noun_chunks:
        if t in nc:
            return nc
    else:
        # cannot expand noun, use token as-is
        return token_as_span(t)


def contains_near(t: Token, radius: int, predicate: Callable[[Token], bool]):
    return any(
        predicate(t)
        for t in t.doc[max(0, t.i - radius) : min(len(t.doc), t.i + radius)]
    )
