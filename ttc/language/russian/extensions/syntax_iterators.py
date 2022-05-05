from typing import Iterator, Union, Tuple, Optional, Collection

from spacy import Errors
from spacy.symbols import NOUN, PROPN, PRON, NUM, VERB, AUX, ADP, punct  # type: ignore
from spacy.tokens import Doc, Span, Token


def next_with_pos(
    *,
    from_: Token,
    max_distance: int = 2**32,
    poss: Collection[int],
) -> Optional[Token]:
    max_distance = min(max_distance, len(from_.doc) - from_.i)
    if from_.dep == punct:
        try:
            from_ = from_.nbor()
        except IndexError:
            return None
    i = 1
    while i < max_distance:
        from_ = from_.nbor()
        if from_.pos in poss:
            return from_
        i += 1
    return None


def noun_chunks(doclike: Union[Doc, Span]) -> Iterator[Tuple[int, int, int]]:
    def is_verb_token(tok):
        return tok.pos in [VERB, AUX]

    def get_left_bound(root):
        left_bound = root
        for tok in reversed(list(root.lefts)):
            if tok.dep in np_left_deps:
                left_bound = tok
        return left_bound

    def get_right_bound(root):
        right_bound = root
        for tok in root.rights:
            if tok.dep in np_right_deps:
                right = get_right_bound(tok)
                if list(filter(lambda t: is_verb_token(t), doc[root.i : right.i])):
                    break
                else:
                    right_bound = right
        complement_np_adp = next_with_pos(from_=right_bound, max_distance=2, poss={ADP})
        if complement_np_adp:
            complement_np_noun = next_with_pos(from_=complement_np_adp, poss=nn_poss)
            if complement_np_noun and complement_np_noun.is_ancestor(complement_np_adp):
                right_bound = get_right_bound(complement_np_noun)
        return right_bound

    def get_bounds(root):
        return get_left_bound(root), get_right_bound(root)

    doc = doclike.doc  # Ensure works on both Doc and Span.

    if not doc.has_annotation("DEP"):
        raise ValueError(Errors.E029)

    if not len(doc):
        return

    left_labels = [
        "det",
        "case",
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
    right_labels = [
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

    np_label = doc.vocab.strings.add("NP")
    np_left_deps = {doc.vocab.strings.add(label) for label in left_labels}
    np_right_deps = {doc.vocab.strings.add(label) for label in right_labels}
    nn_poss = {PROPN, NOUN, PRON, NUM}

    prev_right = -1
    for token in doclike:
        if token.pos in nn_poss:
            left, right = get_bounds(token)
            if left.i <= prev_right:
                continue
            if token.pos == NUM and left.i == right.i:
                continue
            yield left.i, right.i + 1, np_label
            prev_right = right.i
