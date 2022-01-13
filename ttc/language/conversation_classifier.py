from abc import ABC, abstractmethod
from dataclasses import dataclass

from spacy import Language

from ttc.language.dialogue import Dialogue
from ttc.language.play import Play


@dataclass(slots=True)  # type: ignore
class ConversationClassifier(ABC):
    """
    Topmost abstraction over low-level conversation classification logic done by NLP libs.
    """

    language: Language

    @abstractmethod
    def extract_dialogue(self, text: str) -> Dialogue:
        ...

    @abstractmethod
    def connect_play(self, dialogue: Dialogue) -> Play:
        ...
