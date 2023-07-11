from functools import wraps
from typing import Dict, Callable, Literal, Any, Union, TypeVar, cast

from spacy.tokens import Token, Span

from ttc.language.common.token_extensions import (
    has_newline,
    non_word,
    contains_near,
    as_span,
)

ExtensionKind = Literal["method", "getter", "default"]
Ext = TypeVar("Ext", bound=Callable[..., Any])

EXTENSIONS: Dict[Callable, ExtensionKind] = {}


def span_extension(kind: ExtensionKind, default_value: Any = None):
    def ext(f: Ext) -> Ext:
        @wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)

        EXTENSIONS[wrapper] = kind
        if default_value is not None:
            wrapper.default_value = default_value  # type: ignore
        return cast(Ext, wrapper)

    return ext


############
# EXTENSIONS


@span_extension("method")
def line_breaks_between(a: Union[Span, Token], b: Union[Span, Token]) -> int:
    if isinstance(a, Token):
        a = as_span(a)
    if isinstance(b, Token):
        b = as_span(b)
    return int(
        b._.start_line_no - a._.end_line_no
        if b.start > a.end
        else a._.start_line_no - b._.end_line_no
    )


@span_extension("method")
def expand_line_start(self: Union[Span, Token]):
    if isinstance(self, Token):
        self = as_span(self)
    if not self:
        return self
    t = self[0]
    while t.i > 0 and not t._.has_newline:
        t = t.nbor(-1)
    return self.doc[t.i : self.end]


@span_extension("method")
def expand_line_end(self: Span):
    t = self[-1]
    while t.i < len(self.doc) and not t._.has_newline:
        t = t.nbor()
    return self.doc[self.start : t.i + 1]


@span_extension("method")
def trim(self: Span, should_trim: Callable[[Token], bool]) -> Span:
    doc = self.doc
    start = self.start
    while start < self.end and should_trim(doc[start]):
        start += 1
    end = self.end - 1
    while end > self.start and should_trim(doc[end]):
        end -= 1
    return doc[start : end + 1]


@span_extension("method")
def trim_non_word(self: Span) -> Span:
    return trim(self, non_word)  # type: ignore


@span_extension("method")
def contiguous(a: Span, b: Span):
    """Check if given spans are separated only by non-words and do not overlap."""
    return not trim_non_word(
        a.doc[b.end : a.start] if b.end < a.start else a.doc[a.end : b.start]
    )


@span_extension("method")
def is_inside(self: Span, outer: Span):
    return self.start >= outer.start and self.end <= outer.end


@span_extension("getter")
def is_parenthesized(self: Span):
    return contains_near(self[0], 3, lambda t: t.text == "(") and contains_near(
        self[-1], 3, lambda t: t.text == ")"
    )


@span_extension("getter")
def fills_line(self: Span) -> bool:
    threshold = 3
    doc = self.doc
    l = max(self.start - threshold, 0)
    r = min(self.end + threshold, len(doc))
    return (
        any(has_newline(t) for t in doc[l : self.start])
        # colon means that the author still annotates the replica, just on previous line
        and not any(t.text == ":" for t in doc[l : self.start])
        and any(has_newline(t) for t in doc[self.end - 1 : r])
    )


@span_extension("getter")
def line_above(doclike: Union[Span, Token]):
    return trim_non_word(
        expand_line_start(doclike.doc[expand_line_start(doclike).start - 1])
    )


@span_extension("default", False)
def is_unannotated_alternation(self: Span) -> bool:
    return self._.is_unannotated_alternation or False


@span_extension("default", False)
def is_before_author_insertion(self: Span) -> bool:
    return self._.is_before_author_insertion or False


@span_extension("default", False)
def is_after_author_starting(self: Span) -> bool:
    return self._.is_after_author_starting or False


@span_extension("default", False)
def is_before_author_ending(self: Span) -> bool:
    return self._.is_before_author_ending or False


SPAN_EXTENSIONS = {
    name: {EXTENSIONS[f]: f.default_value if EXTENSIONS[f] == "default" else f}
    for name, f in locals().items()
    if callable(f) and hasattr(f, "__wrapped__") and f.__module__ == __name__
}
