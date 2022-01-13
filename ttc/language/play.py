from dataclasses import dataclass

from spacy import Language
from spacy.tokens import Span

from ttc.language.speaker import Speaker


@dataclass
class Play:
    language: Language
    content: dict[Span, Speaker | None]
