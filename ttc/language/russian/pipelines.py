from typing import Iterator

from spacy import Errors
from spacy.language import Language
from spacy.symbols import NOUN, PROPN, PRON  # type: ignore
from spacy.tokens import Doc, Span

from ttc.language.russian.constants import MISPREDICTED_VERBS


@Language.component("correct_mispredictions")
def _correct_mispredictions(doc):
    for token in doc:
        if token.text in MISPREDICTED_VERBS:
            token.pos_ = "VERB"
    return doc


# TODO: Adapt to the Russian lang (Copied from Greek pipeline)
def _noun_chunks(doclike: Doc | Span) -> Iterator[tuple[int, int, int]]:
    """Detect base noun phrases from a dependency parse. Works on Doc and Span."""
    # It follows the logic of the noun chunks finder of English language,
    # adjusted to some Greek language special characteristics.
    # obj tag corrects some DEP tagger mistakes.
    # Further improvement of the models will eliminate the need for this tag.
    labels = ["nsubj", "obj", "iobj", "appos", "ROOT", "obl"]
    doc = doclike.doc  # Ensure works on both Doc and Span.
    if not doc.has_annotation("DEP"):
        raise ValueError(Errors.E029)
    np_deps = [doc.vocab.strings.add(label) for label in labels]
    conj = doc.vocab.strings.add("conj")
    nmod = doc.vocab.strings.add("nmod")
    np_label = doc.vocab.strings.add("NP")
    prev_end = -1
    for i, word in enumerate(doclike):
        if word.pos not in (NOUN, PROPN, PRON):
            continue
        # Prevent nested chunks from being produced
        if word.left_edge.i <= prev_end:
            continue
        if word.dep in np_deps:
            flag = False
            if word.pos == NOUN:
                #  check for patterns such as γραμμή παραγωγής
                for potential_nmod in word.rights:
                    if potential_nmod.dep == nmod:
                        prev_end = potential_nmod.i
                        yield word.left_edge.i, potential_nmod.i + 1, np_label
                        flag = True
                        break
            if flag is False:
                prev_end = word.i
                yield word.left_edge.i, word.i + 1, np_label
        elif word.dep == conj:
            # covers the case: έχει όμορφα και έξυπνα παιδιά
            head = word.head
            while head.dep == conj and head.head.i < head.i:
                head = head.head
            # If the head is an NP, and we're coordinated to it, we're an NP
            if head.dep in np_deps:
                prev_end = word.i
                yield word.left_edge.i, word.i + 1, np_label


def setup_language(lang: Language):
    """
    Registers some custom correction and extension
    pipelines and vocabs for spaCy Language object.
    """
    if not lang.has_pipe("correct_mispredictions"):
        lang.add_pipe("correct_mispredictions", after="morphologizer")
    # Type error is CPython <-> Python incompatibility
    lang.vocab.get_noun_chunks = _noun_chunks  # type: ignore
