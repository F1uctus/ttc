from dataclasses import dataclass
from typing import Optional, Dict

from spacy import Language
from spacy.tokens import Span


@dataclass
class Play:
    language: Language
    content: Dict[Span, Optional[Span]]
