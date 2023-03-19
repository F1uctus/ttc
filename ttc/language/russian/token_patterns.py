from enum import Enum
from typing import Any, Literal, Set, List, Dict, Tuple

from ttc.iterables import merge
from ttc.language.russian.constants import (
    HYPHENS as HYPHENS_STR,
    ELLIPSES as ELLIPSES_STR,
)

HYPHENS, ELLIPSES = list(HYPHENS_STR), list(ELLIPSES_STR)


def text(*texts: str) -> Dict:
    return {"TEXT": {"IN": texts}}


def some(pattern: Dict) -> Dict:
    return merge(pattern, {"OP": "*"})


def one_or_more(pattern: Dict) -> Dict:
    return merge(pattern, {"OP": "+"})


def in_same_sentence(pattern: Dict) -> Dict:
    return merge(pattern, {"_": {"is_sent_end": False}})


WORDS_ON_SAME_LINE = {
    "_": {"has_newline": False},
}

IN_SAME_SENTENCE = {
    "_": {"is_sent_end": False},
}

HYPHEN = {"TEXT": {"IN": HYPHENS}}
NOT_HYPHEN = {"TEXT": {"NOT_IN": HYPHENS}}

SPEAKER_WORD = {
    "POS": {"IN": ["NOUN", "PROPN", "PRON", "ADJ", "NUM"]},
    "MORPH": {"INTERSECTS": ["Case=Nom", "Case=Acc", "Case=Dat"]},
    "_": {"is_not_second_person": True},
}

VERB = {"POS": "VERB"}

TokenMatcherClass = Literal[
    "AUTHOR_INSERTION",
    "AUTHOR_ENDING",
]

TOKEN_MATCHER_CLASSES: Set[TokenMatcherClass] = {
    "AUTHOR_INSERTION",
    "AUTHOR_ENDING",
}


class TokenPattern(Enum):
    # p, — <AUTHOR>.
    AUTHOR_ENDING_1 = [
        text(*".,!?", *ELLIPSES),
        HYPHEN,
        some(WORDS_ON_SAME_LINE),
        merge(VERB, {"_": {"is_sent_end": False}}),
        some(WORDS_ON_SAME_LINE),
        one_or_more(SPEAKER_WORD),
        some(WORDS_ON_SAME_LINE),
        merge(text(*".!?", *ELLIPSES), {"_": {"has_newline": True}}),
    ]

    # p, — <AUTHOR>.
    AUTHOR_ENDING_2 = [
        text(*".,!?", *ELLIPSES),
        HYPHEN,
        some(WORDS_ON_SAME_LINE),
        merge(one_or_more(SPEAKER_WORD), {"_": {"is_sent_end": False}}),
        some(WORDS_ON_SAME_LINE),
        VERB,
        some(WORDS_ON_SAME_LINE),
        merge(text(*".!?", *ELLIPSES), {"_": {"has_newline": True}}),
    ]

    # p[.,…!?] — <INSERTION>. — A
    AUTHOR_INSERTION_1_1 = [
        text(*".,!?", *ELLIPSES),
        HYPHEN,
        some(WORDS_ON_SAME_LINE),
        in_same_sentence(VERB),
        some(WORDS_ON_SAME_LINE),
        one_or_more(SPEAKER_WORD),
        some(WORDS_ON_SAME_LINE),
        text(*".;:", *ELLIPSES),
        HYPHEN,
        {"IS_TITLE": True},
    ]

    # p[.,…!?] — <INSERTION>. — A
    AUTHOR_INSERTION_1_2 = [
        text(*".,!?", *ELLIPSES),
        HYPHEN,
        some(WORDS_ON_SAME_LINE),
        in_same_sentence(one_or_more(SPEAKER_WORD)),
        some(WORDS_ON_SAME_LINE),
        VERB,
        some(WORDS_ON_SAME_LINE),
        text(*".;:", *ELLIPSES),
        HYPHEN,
        {"IS_TITLE": True},
    ]

    # p, — <INSERTION>, — a
    # p... — <INSERTION>, — a
    AUTHOR_INSERTION_2_1 = [
        text(",", *ELLIPSES),
        HYPHEN,
        some(WORDS_ON_SAME_LINE),
        in_same_sentence(VERB),
        some(WORDS_ON_SAME_LINE),
        one_or_more(SPEAKER_WORD),
        some(WORDS_ON_SAME_LINE),
        text(","),
        HYPHEN,
        {"IS_TITLE": False},
    ]

    # p, — <INSERTION>, — a
    # p... — <INSERTION>, — a
    AUTHOR_INSERTION_2_2 = [
        text(",", *ELLIPSES),
        HYPHEN,
        some(WORDS_ON_SAME_LINE),
        in_same_sentence(one_or_more(SPEAKER_WORD)),
        some(WORDS_ON_SAME_LINE),
        VERB,
        some(WORDS_ON_SAME_LINE),
        text(","),
        HYPHEN,
        {"IS_TITLE": False},
    ]

    @classmethod
    def entries(cls) -> List[Tuple[str, List[Dict[str, Any]]]]:
        return [(item.name, item.value) for item in cls]  # type: ignore
