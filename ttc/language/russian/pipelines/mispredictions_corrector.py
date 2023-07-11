from spacy import Language
from spacy.tokens import Doc

NAME = "correct_mispredictions"

MISPREDICTED_VERBS = {
    "бормочет",
}


@Language.component(NAME)
def _correct_mispredictions(doc: Doc):
    for token in doc:
        norm_text = token.text.strip().lower()

        if norm_text in MISPREDICTED_VERBS:
            token.pos_ = "VERB"

    return doc
