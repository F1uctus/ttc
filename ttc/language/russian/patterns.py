from enum import Enum
from typing import Any

from ttc.language.russian.constants import DASHES


class Patterns(Enum):
    DASH_VERB_NOUN = [
        {"TEXT": {"IN": list(DASHES)}},
        {"POS": "VERB"},
        {"POS": {"IN": ["NOUN", "PROPN", "PRON", "ADJ"]}, "OP": "+"},
    ]

    @classmethod
    def entries(cls) -> list[tuple[str, list[dict[str, Any]]]]:
        return [(item.name, item.value) for item in cls]  # type: ignore
