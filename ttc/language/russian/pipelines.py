from typing import Iterator, Callable, Iterable, Union

from spacy import Errors
from spacy.language import Language
from spacy.symbols import NOUN, PROPN, PRON, nmod, amod, NUM, acl  # type: ignore
from spacy.tokens import Doc, Span, Token

from ttc.language.russian.constants import MISPREDICTED_VERBS


@Language.component("correct_mispredictions")
def _correct_mispredictions(doc):
    for token in doc:
        if token.text in MISPREDICTED_VERBS:
            token.pos_ = "VERB"
    return doc


# noinspection PyProtectedMember
# (spaCy "._." extensions)
@Language.component("mark_line_numbers")
def _mark_line_numbers(doc):
    line_no = 1
    for token in doc:
        token._.line_no = line_no
        if "\n" in token.text_with_ws:
            line_no += 1
    return doc


def traverse_children(
    word: Token,
    children_selector: Callable[[Token], Iterable[Token]],
    dep_predicate: Callable[[Token], bool],
):
    children = list(children_selector(word))
    max_i = max((i for i, c in enumerate(children) if dep_predicate(c)), default=-1)
    return (
        traverse_children(
            children[max_i],
            children_selector,
            dep_predicate,
        )
        if max_i > -1
        else word
    )


def _noun_chunks(doclike: Union[Doc, Span]) -> Iterator[tuple[int, int, int]]:
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

        start = traverse_children(
            word,
            lambda t: t.lefts,
            lambda t: t.dep_ in ("nmod", "amod", "obl", "appos"),
        )

        end = traverse_children(
            word,
            lambda t: t.rights,
            lambda t: (
                t.dep_ in ("nmod", "amod", "acl", "obl", "case", "appos")
                or t.dep_.startswith("acl")
            ),
        )

        prev_end = end.i
        yield start.i, end.i + 1, np_label


def setup_language(lang: Language):
    """
    Registers some custom correction and extension
    pipelines and vocabs for spaCy Language object.
    """
    if not lang.has_pipe("correct_mispredictions"):
        lang.add_pipe("correct_mispredictions", after="morphologizer")

    if not lang.has_pipe("mark_line_numbers"):
        lang.add_pipe("mark_line_numbers", after="senter")

    # Type error is CPython <-> Python incompatibility
    lang.vocab.get_noun_chunks = _noun_chunks  # type: ignore
