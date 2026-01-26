from abc import ABC, abstractmethod

from spacy import Language

from ttc.language.dialogue import Dialogue
from ttc.language.play import Play


class ConversationClassifier(ABC):
    """
    Abstraction over conversation classification logic done using NLP libs.
    """

    language: Language

    @abstractmethod
    def extract_dialogue(self, text: str) -> Dialogue: ...

    @abstractmethod
    def connect_play(self, dialogue: Dialogue) -> Play: ...
