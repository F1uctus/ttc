from typing import Iterator, Union, Tuple, Optional

from spacy import Errors
from spacy.symbols import (  # type: ignore
    NOUN,
    PROPN,
    PRON,
    DET,
    NUM,
    VERB,
    AUX,
    ADP,
    CCONJ,
    punct,
    nmod,
    appos,
)
from spacy.tokens import Doc, Span, Token

from ttc.language.russian.constants import HYPHENS
from ttc.language.russian.token_extensions import morph_equals, is_speaker_noun


def can_mark_chunk(t: Token):
    return t.pos in {PROPN, NOUN, PRON, NUM}


LEFT_DEP_LABELS = [
    "det",
    # common
    "fixed",
    "nmod:poss",
    "nmod",
    "amod",
    "flat",
    "goeswith",
    "nummod",
    "obl",
    "appos",
]
RIGHT_DEP_LABELS = [
    # common
    "fixed",
    "nmod:poss",
    "nmod",
    "amod",
    "flat",
    "goeswith",
    "nummod",
    "obl",
    "appos",
]


def get_right_adp(t: Token) -> Optional[Token]:
    """e.g: t=человек + [с ножом]"""
    if t.i + 2 >= len(t.doc):
        return None
    nt, nnt = t.nbor(+1), t.nbor(+2)
    return nnt if nt.pos == ADP and (nt.dep_ == "case" and can_mark_chunk(nnt)) else None


def noun_chunks(doclike: Union[Doc, Span]) -> Iterator[Tuple[int, int, int]]:
    def are_uniform(t1: Token, t2: Token) -> bool:
        return morph_equals(t1, t2, "Number", "Case", "Voice")

    def get_left_bound(root: Token) -> Token:
        l_bound = root
        old_l_bound = None
        while old_l_bound != l_bound:
            old_l_bound = l_bound

            for t in root.lefts:
                if t.dep in np_left_deps and t.i < l_bound.i:
                    l_bound = t

            # immediately preceding uniform words
            if l_bound.i and (
                (prev_t := l_bound.nbor(-1))
                and prev_t.dep in np_left_deps
                and l_bound.pos != PROPN
                and are_uniform(prev_t, l_bound)
            ):
                l_bound = prev_t

        return l_bound

    def get_right_bound(root: Token) -> Token:
        r_bound = root
        old_r_bound = None
        while old_r_bound != r_bound:
            old_r_bound = r_bound

            for t in r_bound.rights:
                if t.dep in np_right_deps:
                    r_bound = t

            # immediately following uniform words
            try:
                nt = r_bound.nbor(+1)
                if r_bound.pos != PROPN and are_uniform(r_bound, nt):
                    r_bound = nt
                elif adp := get_right_adp(r_bound):
                    r_bound = adp
                # should support things like "все, кроме A"
                # but should ignore things like "A и B"
                elif (nt.pos == CCONJ or nt.text in HYPHENS) and are_uniform(
                    r_bound, nnt := nt.nbor(+1)
                ):
                    # noun-noun, e.g: коротышка-сержант
                    # or conj-ed parts, e.g: <noun> смешной и бесполезный
                    r_bound = nnt
            except IndexError:
                pass

        return r_bound

    def get_bounds(root):
        return get_left_bound(root), get_right_bound(root)

    doc = doclike.doc  # Ensure works on both Doc and Span.

    if not doc.has_annotation("DEP"):
        raise ValueError(Errors.E029)

    if not len(doc):
        return

    np_label = doc.vocab.strings.add("NP")
    np_left_deps = {doc.vocab.strings.add(label) for label in LEFT_DEP_LABELS}
    np_right_deps = {doc.vocab.strings.add(label) for label in RIGHT_DEP_LABELS}

    prev_right = -1
    for token in doc:
        if token.i < prev_right:
            continue
        if can_mark_chunk(token):
            left, right = get_bounds(token)
            if left.i <= prev_right:
                continue
            yield left.i, right.i + 1, np_label
            prev_right = right.i
