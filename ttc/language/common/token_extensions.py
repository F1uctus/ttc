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


def morph_distance(self: Token, other: Token, *morphs: str) -> int:
    if self.morph is None or other.morph is None:
        return len(morphs)
    return len(morphs) - sum(
        self.morph.get(m, None) == other.morph.get(m, None) for m in morphs
    )


def morph_equals(self: Token, other: Token, *morphs: str) -> bool:
    return morph_distance(self, other, *morphs) == 0


def as_span(self: Union[Token, Span]) -> Span:
    if isinstance(self, Span):
        return self
    return self.doc[self.i : self.i + 1]


def non_word(self: Token) -> bool:
    return self.is_punct or has_newline(self)


def noun_chunk(self: Union[Token, Span]) -> Span:
    from ttc.language.common.span_extensions import is_inside

    span = self if isinstance(self, Span) else as_span(self)
    for nc in span.sent.noun_chunks:
        if is_inside(span, nc):
            # tighten the chunk using bounds from NER
            return nc.ents[0] if len(nc) > 2 and len(nc.ents) == 1 else nc
    # cannot expand noun, use token as-is
    return span


def contains_near(self: Token, radius: int, predicate: Callable[[Token], bool]) -> bool:
    return any(
        predicate(t)
        for t in self.doc[max(0, self.i - radius) : min(len(self.doc), self.i + radius)]
    )


TOKEN_EXTENSIONS = {
    name: {"getter": f}
    for name, f in locals().items()
    if callable(f) and f.__module__ == __name__
}
