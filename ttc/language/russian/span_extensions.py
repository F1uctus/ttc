from typing import Dict, Callable, Literal, Any, TypeVar, cast
from functools import wraps

from spacy.tokens import Token, Span

from ttc.language.russian.token_extensions import non_word, has_newline, contains_near

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
def is_inside(self: Span, outer: Span):
    return self.start >= outer.start and self.end <= outer.end


@span_extension("method")
def non_overlapping_span_len(self: Span, inner: Span) -> int:
    self, inner = trim(self, non_word), trim(inner, non_word)
    if is_inside(self, inner):
        return 0  # outer must be a strict superset of inner
    return abs(self.start - inner.start) + abs(self.end - inner.end)


@span_extension("method")
def sent_resembles_replica(self: Span, replica: Span) -> bool:
    return non_overlapping_span_len(self, replica) <= 3  # type: ignore


@span_extension("getter")
def is_parenthesized(self: Span):
    return contains_near(self[0], 3, lambda t: t.text == "(") and contains_near(
        self[-1], 3, lambda t: t.text == ")"
    )


@span_extension("getter")
def fills_line(self: Span) -> bool:
    doc = self.doc
    return (
        self.start - 3 >= 0
        and self.end + 3 < len(doc)  # TODO: Rewrite and check end-of-doc case
        and any(has_newline(t) for t in doc[self.start - 3 : self.start])
        # colon means that the author still annotates the replica, just on previous line
        and not any(t.text == ":" for t in doc[self.start - 3 : self.start])
        and any(has_newline(t) for t in doc[self.end - 1 : self.end + 3])
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
