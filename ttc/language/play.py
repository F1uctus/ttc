from dataclasses import dataclass
from typing import Optional, Dict

from spacy import Language
from spacy.tokens import Span

from ttc.language.speaker import Speaker


@dataclass
class Play:
    language: Language
    content: Dict[Span, Optional[Speaker]]
