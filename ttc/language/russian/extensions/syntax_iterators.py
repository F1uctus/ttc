from typing import Iterator, Callable, Iterable, Union, Tuple

from spacy import Errors
from spacy.symbols import NOUN, PROPN, PRON, nmod, amod, NUM, acl, PART  # type: ignore
from spacy.tokens import Doc, Span, Token


def _traverse_children(
    word: Token,
    children_selector: Callable[[Token], Iterable[Token]],
    dep_predicate: Callable[[Token], bool],
):
    children = list(children_selector(word))
    max_i = max((i for i, c in enumerate(children) if dep_predicate(c)), default=-1)
    return (
        _traverse_children(
            children[max_i],
            children_selector,
            dep_predicate,
        )
        if max_i > -1
        else word
    )


def noun_chunks(doclike: Union[Doc, Span]) -> Iterator[Tuple[int, int, int]]:
    """Detect base noun phrases from a dependency parse. Works on Doc and Span."""
    doc = doclike.doc  # Ensure works on both Doc and Span.

    if not doc.has_annotation("DEP"):
        raise ValueError(Errors.E029)

    np_label = doc.vocab.strings.add("NP")
    labels = ["nsubj", "nsubj:pass", "obj", "iobj", "appos", "ROOT", "obl"]
    np_deps = [doc.vocab.strings.add(label) for label in labels]

    prev_end = -1

    for i, word in enumerate(doclike):
        # Prevent nested chunks from being produced
        if word.left_edge.i <= prev_end:
            continue

        if word.pos not in (NOUN, PROPN, PRON, NUM):
            continue

        if word.dep not in np_deps:
            continue

        start = _traverse_children(
            word,
            lambda t: t.lefts,
            lambda t: t.dep_ in ("nmod", "amod", "obl", "appos"),
        )

        end = _traverse_children(
            word,
            lambda t: t.rights,
            lambda t: (
                t.dep_ in ("nmod", "amod", "acl", "obl", "case", "appos")
                or t.dep_.startswith("acl")
            ),
        )

        prev_end = end.i
        yield start.i, end.i + 1, np_label
