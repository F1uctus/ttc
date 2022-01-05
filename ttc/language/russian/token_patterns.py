from enum import Enum
from typing import Any, Literal

from ttc.language.russian.constants import DASHES, M_DOTS

INSERTION_1 = [
    {"_": {"is_sent_end": False}, "OP": "*"},
    {
        "POS": "VERB",
        "_": {"is_sent_end": False, "is_author_verb": True},
    },
    {"_": {"is_sent_end": False}, "OP": "*"},
    {
        "POS": {"IN": ["NOUN", "PROPN", "PRON", "ADJ", "NUM"]},
        "MORPH": {"INTERSECTS": ["Case=Nom", "Case=Acc"]},
        "OP": "+",
    },
    {"OP": "*"},
]

INSERTION_2 = [
    {"_": {"is_sent_end": False}, "OP": "*"},
    {
        "POS": {"IN": ["NOUN", "PROPN", "PRON", "ADJ", "NUM"]},
        "MORPH": {"INTERSECTS": ["Case=Nom", "Case=Acc"]},
        "_": {"is_sent_end": False},
        "OP": "+",
    },
    {"_": {"is_sent_end": False}, "OP": "*"},
    {
        "POS": "VERB",
        "_": {"is_author_verb": True},
    },
    {"OP": "*"},
]

MatcherClass = Literal[
    "AUTHOR_INSERTION",
    "AUTHOR_ENDING",
]

MATCHER_CLASSES: set[MatcherClass] = {
    "AUTHOR_INSERTION",
    "AUTHOR_ENDING",
}


class TokenPatterns(Enum):
    DASH_VERB_NOUN = [
        {"TEXT": {"IN": list(DASHES)}},
        {"POS": "VERB"},
        {"POS": {"IN": ["NOUN", "PROPN", "PRON", "ADJ"]}, "OP": "+"},
    ]

    AUTHOR_ENDING_1 = [
        # p, — <AUTHOR>.
        {"TEXT": ","},
        {"TEXT": {"IN": list(DASHES)}},
        *INSERTION_1,
        {"TEXT": {"IN": [".", *M_DOTS, "!", "?"]}},
    ]

    AUTHOR_ENDING_2 = [
        # p, — <AUTHOR>.
        {"TEXT": ","},
        {"TEXT": {"IN": list(DASHES)}},
        *INSERTION_2,
        {"TEXT": {"IN": [".", *M_DOTS, "!", "?"]}},
    ]

    AUTHOR_INSERTION_1_1 = [
        # p[.…!?] — <INSERTION>. — A
        {"TEXT": {"IN": [".", *M_DOTS, "!", "?"]}},
        {"TEXT": {"IN": list(DASHES)}},
        *INSERTION_1,
        {"TEXT": {"IN": [".", ";", ":", *M_DOTS, "!", "?"]}},
        {"TEXT": {"IN": list(DASHES)}},
        {"IS_TITLE": True},
    ]

    AUTHOR_INSERTION_1_2 = [
        # p[.…!?] — <INSERTION>. — A
        {"TEXT": {"IN": [".", *M_DOTS, "!", "?"]}},
        {"TEXT": {"IN": list(DASHES)}},
        *INSERTION_2,
        {"TEXT": {"IN": [".", ";", ":", *M_DOTS, "!", "?"]}},
        {"TEXT": {"IN": list(DASHES)}},
        {"IS_TITLE": True},
    ]

    AUTHOR_INSERTION_2_1 = [
        # p, — <INSERTION>. — A
        {"TEXT": ","},
        {"TEXT": {"IN": list(DASHES)}},
        *INSERTION_1,
        {"TEXT": {"IN": [".", ";", ":", *M_DOTS, "!", "?"]}},
        {"TEXT": {"IN": list(DASHES)}},
        {"IS_TITLE": True},
    ]

    AUTHOR_INSERTION_2_2 = [
        # p, — <INSERTION>. — A
        {"TEXT": ","},
        {"TEXT": {"IN": list(DASHES)}},
        *INSERTION_2,
        {"TEXT": {"IN": [".", ";", ":", *M_DOTS, "!", "?"]}},
        {"TEXT": {"IN": list(DASHES)}},
        {"IS_TITLE": True},
    ]

    AUTHOR_INSERTION_3_1 = [
        # p, — <INSERTION>, — a
        # p... — <INSERTION>, — a
        {"TEXT": {"IN": [",", *M_DOTS]}},
        {"TEXT": {"IN": list(DASHES)}},
        *INSERTION_1,
        {"TEXT": {"IN": [",", ";", ":"]}},
        {"TEXT": {"IN": list(DASHES)}},
        {"IS_TITLE": False},
    ]

    AUTHOR_INSERTION_3_2 = [
        # p, — <INSERTION>, — a
        # p... — <INSERTION>, — a
        {"TEXT": {"IN": [",", *M_DOTS]}},
        {"TEXT": {"IN": list(DASHES)}},
        *INSERTION_2,
        {"TEXT": {"IN": [",", ";", ":"]}},
        {"TEXT": {"IN": list(DASHES)}},
        {"IS_TITLE": False},
    ]

    @classmethod
    def entries(cls) -> list[tuple[str, list[dict[str, Any]]]]:
        return [(item.name, item.value) for item in cls]  # type: ignore