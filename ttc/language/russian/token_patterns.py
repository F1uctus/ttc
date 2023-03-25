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


LINE_FILLER = some({"_": {"has_newline": False}})
HYPHEN = {"TEXT": {"IN": HYPHENS}}
NOT_HYPHEN = {"TEXT": {"NOT_IN": HYPHENS}}
SPEAKER_WORD = {
    "POS": {"IN": ["NOUN", "PROPN", "PRON", "ADJ", "NUM"]},
    "MORPH": {"INTERSECTS": ["Case=Nom", "Case=Acc", "Case=Dat"]},
    "_": {"is_not_second_person": True},
}
VERB = {"POS": {"IN": ["VERB", "AUX"]}}

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
        LINE_FILLER,
        VERB,
        LINE_FILLER,
        one_or_more(SPEAKER_WORD),
        LINE_FILLER,
        merge(text(*".!?", *ELLIPSES), {"_": {"has_newline": True}}),
    ]

    # p, — <AUTHOR>.
    AUTHOR_ENDING_2 = [
        text(*".,!?", *ELLIPSES),
        HYPHEN,
        LINE_FILLER,
        one_or_more(SPEAKER_WORD),
        LINE_FILLER,
        VERB,
        LINE_FILLER,
        merge(text(*".!?", *ELLIPSES), {"_": {"has_newline": True}}),
    ]

    # p[.,…!?] — <INSERTION>. — A
    AUTHOR_INSERTION_1_1 = [
        text(*".,!?", *ELLIPSES),
        HYPHEN,
        LINE_FILLER,
        VERB,
        LINE_FILLER,
        one_or_more(SPEAKER_WORD),
        LINE_FILLER,
        text(*".;:", *ELLIPSES),
        some(text(")")),
        HYPHEN,
        {"IS_TITLE": True},
    ]

    # p[.,…!?] — <INSERTION>. — A
    AUTHOR_INSERTION_1_2 = [
        text(*".,!?", *ELLIPSES),
        HYPHEN,
        LINE_FILLER,
        one_or_more(SPEAKER_WORD),
        LINE_FILLER,
        VERB,
        LINE_FILLER,
        text(*".;:", *ELLIPSES),
        some(text(")")),
        HYPHEN,
        {"IS_TITLE": True},
    ]

    # p, — <INSERTION>, — a
    # p... — <INSERTION>, — a
    AUTHOR_INSERTION_2_1 = [
        text(",", *ELLIPSES),
        HYPHEN,
        LINE_FILLER,
        VERB,
        LINE_FILLER,
        one_or_more(SPEAKER_WORD),
        LINE_FILLER,
        text(*",;"),
        HYPHEN,
        {"IS_TITLE": False},
    ]

    # p, — <INSERTION>, — a
    # p... — <INSERTION>, — a
    AUTHOR_INSERTION_2_2 = [
        text(",", *ELLIPSES),
        HYPHEN,
        LINE_FILLER,
        one_or_more(SPEAKER_WORD),
        LINE_FILLER,
        VERB,
        LINE_FILLER,
        text(*",;"),
        HYPHEN,
        {"IS_TITLE": False},
    ]

    @classmethod
    def entries(cls) -> List[Tuple[str, List[Dict[str, Any]]]]:
        return [(item.name, item.value) for item in cls]
