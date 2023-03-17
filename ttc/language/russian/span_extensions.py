from typing import Callable

from spacy.symbols import NOUN, PROPN, PRON, DET, NUM, VERB, AUX, ADP, CCONJ, punct  # type: ignore
from spacy.tokens import Token, Span
from ttc.language.russian.token_extensions import non_word, has_newline, contains_near


def trim(self: Span, should_trim: Callable[[Token], bool]) -> Span:
    doc = self.doc
    start = self.start
    while start < self.end and should_trim(doc[start]):
        start += 1
    end = self.end - 1
    while end > self.start and should_trim(doc[end]):
        end -= 1
    return doc[start : end + 1]


def trim_non_word(self: Span) -> Span:
    return trim(self, non_word)


def is_inside(self: Span, outer: Span):
    return self.start >= outer.start and self.end <= outer.end


def non_overlapping_span_len(self: Span, inner: Span) -> int:
    self, inner = trim(self, non_word), trim(inner, non_word)
    if is_inside(self, inner):
        return 0  # outer must be a strict superset of inner
    return abs(self.start - inner.start) + abs(self.end - inner.end)


def sent_resembles_replica(self: Span, replica: Span) -> bool:
    return non_overlapping_span_len(self, replica) <= 3


def is_parenthesized(self: Span):
    return contains_near(self[0], 3, lambda t: t.text == "(") and contains_near(
        self[-1], 3, lambda t: t.text == ")"
    )


def replica_fills_line(replica: Span) -> bool:
    doc = replica.doc
    return (
        replica.start - 3 >= 0
        and replica.end + 3 < len(doc)  # TODO: Rewrite and check end-of-doc case
        and any(has_newline(t) for t in doc[replica.start - 3 : replica.start])
        # colon means that the author still annotates the replica, just on previous line
        and not any(t.text == ":" for t in doc[replica.start - 3 : replica.start])
        and any(has_newline(t) for t in doc[replica.end - 1 : replica.end + 3])
    )


SPAN_EXTENSIONS = {
    name: {"getter": f}
    for name, f in locals().items()
    if callable(f) and f.__module__ == __name__
}
