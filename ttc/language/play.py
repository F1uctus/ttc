from dataclasses import dataclass

from spacy import Language

from ttc.language.replica import Replica
from ttc.language.speaker import Speaker


@dataclass
class Play:
    language: Language
    content: dict[Replica, Speaker | None]
