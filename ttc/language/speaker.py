from dataclasses import dataclass
from typing import Optional

from spacy.tokens import Token, MorphAnalysis


@dataclass(slots=True)
class Speaker:
    tokens: list[Token]
    text: str
    lemma: str
    first_token: Optional[Token]
    morph: Optional[MorphAnalysis]
    gender: Optional[str]

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.text = "".join(t.text_with_ws for t in self.tokens)
        self.lemma = " ".join(t.lemma_ for t in self.tokens)
        self.first_token = None if len(self.tokens) == 0 else self.tokens[0]
        self.morph = self.first_token.morph if self.first_token else None
        self.gender = self.morph.get("Gender")[0] if self.morph else None

    def __repr__(self):
        return self.text
