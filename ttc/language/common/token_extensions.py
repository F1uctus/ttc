from typing import Union, Callable

from spacy.tokens import Token, Span

from ttc.language.common.constants import OPEN_QUOTES, CLOSE_QUOTES


def is_open_quote(self: Token):
    return self.text in OPEN_QUOTES


def is_close_quote(self: Token):
    return self.text in CLOSE_QUOTES


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


def as_span(self: Union[Token, Span]) -> Span:
    if isinstance(self, Span):
        return self
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


def dep_dist_up_to(self: Token, parent: Token):
    dist = 0
    while self != parent and self != self.head:
        self = self.head
        dist += 1
    return dist if self == parent else -1


TOKEN_EXTENSIONS = {
    name: {"getter": f}
    for name, f in locals().items()
    if callable(f) and f.__module__ == __name__
}
