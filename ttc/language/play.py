from dataclasses import dataclass
from typing import Optional

from spacy import Language

from .replica import Replica
from .speaker import Speaker


@dataclass
class Play:
    language: Language
    content: dict[Replica, Optional[Speaker]]
