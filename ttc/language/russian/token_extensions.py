from typing import Callable, Optional, Union

from spacy.symbols import (  # type: ignore
    VERB,
    AUX,
    NOUN,
    PROPN,
    PRON,
    ADJ,
    NUM,
    nsubj,
    obj,
    obl,
)
from spacy.tokens import Token, Span

from ttc.language.russian.constants import (
    HYPHENS,
    OPEN_QUOTES,
    CLOSE_QUOTES,
    SPEAKING_VERBS,
)


def is_hyphen(self: Token):
    return self.text in HYPHENS


def is_open_quote(self: Token):
    return self.text in OPEN_QUOTES


def is_close_quote(self: Token):
    return self.text in CLOSE_QUOTES


def is_speaking_verb(self: Token):
    """Used in dependency matching"""
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


def as_span(self: Token) -> Span:
    return self.doc[self.i : self.i + 1]


def non_word(self: Token) -> bool:
    return self.is_punct or has_newline(self)


def expand_to_noun_chunk(self: Union[Token, Span]) -> Span:
    span = self if isinstance(self, Span) else as_span(self)
    for nc in span.sent.noun_chunks:
        if all(t in nc for t in span):
            return nc.ents[0] if len(nc) > 2 and len(nc.ents) == 1 else nc
    # cannot expand noun, use token as-is
    return span


def contains_near(self: Token, radius: int, predicate: Callable[[Token], bool]) -> bool:
    return any(
        predicate(t)
        for t in self.doc[max(0, self.i - radius) : min(len(self.doc), self.i + radius)]
    )


def ref_matches(ref: Union[Token, Span], target: Span) -> bool:
    if isinstance(ref, Span):
        ref = ref.root
    return morph_equals(target.root, ref, "Gender", "Number")


def is_speaker_noun(self: Token) -> bool:
    return bool(
        self.pos in {NOUN, PROPN, PRON, ADJ, NUM}
        and set(self.morph.get("Case")).intersection({"Nom", "Acc", "Dat"})
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


def dep_dist_up_to(self: Token, parent: Token):
    dist = 0
    while self != parent:
        self = self.head
        dist += 1
    return dist


TOKEN_EXTENSIONS = {
    name: {"getter": f}
    for name, f in locals().items()
    if callable(f) and f.__module__ == __name__
}
