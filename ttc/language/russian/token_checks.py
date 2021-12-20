from typing import Union, Optional

from spacy.tokens import Token

from ttc.language import Speaker
from ttc.language.russian.constants import SPECIAL_VERBS


def morph_equals(
    morph_name: str,
    a: Union[Token, Speaker],
    b: Union[Token, Speaker],
) -> bool:
    if a.morph is None or b.morph is None:
        return False
    return a.morph.get(morph_name) == b.morph.get(morph_name)


def maybe_name(t: Token) -> bool:
    return t.pos_ in ["NOUN", "PROPN", "ADJ", "PRON"] and "mod" not in t.dep_


def is_name(t: Token, linked_verb: Optional[Token] = None) -> bool:
    return (
        t.pos_ in ["NOUN", "PROPN", "ADJ", "PRON"]
        and t.text[0].isupper()
        and (linked_verb is None or morph_equals("Number", linked_verb, t))
    )


def is_special_verb(t: Token) -> bool:
    return t.lemma_ in SPECIAL_VERBS and (
        t.morph.get("VerbForm")[0] not in ["Conv", "Part"]
    )


def is_names_verb(t: Token, linked_name: Optional[Token] = None) -> bool:
    return (
        t.pos_ == "VERB"
        and t.morph.get("VerbForm")[0] not in ["Conv", "Part"]
        and (linked_name is None or morph_equals("Number", linked_name, t))
    )
