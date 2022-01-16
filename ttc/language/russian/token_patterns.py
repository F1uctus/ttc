from enum import Enum
from typing import Any, Literal, Set, List, Dict, Tuple

from ttc.language.russian.constants import DASHES, M_DOTS

INSERTION_1 = [
    {"_": {"is_sent_end": False}, "OP": "*"},
    {
        "POS": "VERB",
        "_": {"is_sent_end": False, "is_speaking_verb": True},
    },
    {"_": {"is_sent_end": False}, "OP": "*"},
    {
        "POS": {"IN": ["NOUN", "PROPN", "PRON", "ADJ", "NUM"]},
        "MORPH": {"INTERSECTS": ["Case=Nom", "Case=Acc"]},
        "_": {"is_not_second_person": True},
        "OP": "+",
    },
    {"OP": "*"},
]

INSERTION_2 = [
    {"_": {"is_sent_end": False}, "OP": "*"},
    {
        "POS": {"IN": ["NOUN", "PROPN", "PRON", "ADJ", "NUM"]},
        "MORPH": {"INTERSECTS": ["Case=Nom", "Case=Acc"]},
        "_": {"is_sent_end": False, "is_not_second_person": True},
        "OP": "+",
    },
    {"_": {"is_sent_end": False}, "OP": "*"},
    {
        "POS": "VERB",
        "_": {"is_speaking_verb": True},
    },
    {"OP": "*"},
]

TokenMatcherClass = Literal[
    "AUTHOR_INSERTION",
    "AUTHOR_ENDING",
]

TOKEN_MATCHER_CLASSES: Set[TokenMatcherClass] = {
    "AUTHOR_INSERTION",
    "AUTHOR_ENDING",
}


class TokenPattern(Enum):
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
    def entries(cls) -> List[Tuple[str, List[Dict[str, Any]]]]:
        return [(item.name, item.value) for item in cls]  # type: ignore
