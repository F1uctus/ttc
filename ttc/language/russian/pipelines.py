from spacy.language import Language

from ttc.language.russian.constants import MISPREDICTED_VERBS


@Language.component("correct_mispredictions")
def _correct_mispredictions(doc):
    for token in doc:
        if token.text in MISPREDICTED_VERBS:
            token.pos_ = "VERB"
    return doc


def setup_language(lang: Language):
    if not lang.has_pipe("correct_mispredictions"):
        lang.add_pipe("correct_mispredictions", after="morphologizer")
