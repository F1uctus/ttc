from spacy.tokens import Token, MorphAnalysis

from language.russian.common.constants import SPECIAL_VERBS


def is_name_weak(t: Token):
    return (
        t.pos_ in ["NOUN", "PROPN", "ADJ", "PRON"]
        and t.dep_ not in ["obl"]
        and "mod" not in t.dep_
    )


def is_name(t: Token, linked_verb_morph: MorphAnalysis = None):
    return (
        t.pos_ in ["NOUN", "PROPN", "ADJ", "PRON"]
        and t.text[0].isupper()
        and (
            linked_verb_morph is None
            or linked_verb_morph.get("Number") == t.morph.get("Number")
        )
    )


def is_special_verb(t: Token):
    return t.lemma_ in SPECIAL_VERBS and t.morph.get("VerbForm")[0] not in [
        "Conv",
        "Part",
    ]


def is_names_verb(t: Token, linked_name_morph: MorphAnalysis = None):
    return (
        t.pos_ == "VERB"
        and t.morph.get("VerbForm")[0] not in ["Conv", "Part"]
        and (
            linked_name_morph is None
            or linked_name_morph.get("Number") == t.morph.get("Number")
        )
    )
