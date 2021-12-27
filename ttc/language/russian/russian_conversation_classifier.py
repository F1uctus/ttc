from dataclasses import dataclass

from spacy.tokens import Token

from ttc.language.conversation_classifier import ConversationClassifier
from ttc.language.dialogue import Dialogue
from ttc.language.play import Play
from ttc.language.russian import extract_replicas, classify_speakers
from ttc.language.russian.constants import DASHES, OPEN_QUOTES, CLOSE_QUOTES, SENT_ENDS


@dataclass(slots=True)
class RussianConversationClassifier(ConversationClassifier):
    token_predicates = {
        "is_sent_end": lambda t: any(c in SENT_ENDS for c in t.text),
        "is_dash": lambda t: t.text in DASHES,
        "is_newline": lambda t: "\n" in t.text,
        "is_open_quote": lambda t: t.text in OPEN_QUOTES,
        "is_close_quote": lambda t: t.text in CLOSE_QUOTES,
    }

    def extract_dialogue(self, text: str) -> Dialogue:
        for name, pred in RussianConversationClassifier.token_predicates.items():
            if not Token.has_extension(name):
                Token.set_extension(name, getter=pred)

        return Dialogue(self.language, extract_replicas(self.language(text)))

    def connect_play(self, dialogue: Dialogue) -> Play:
        return Play(
            self.language,
            classify_speakers(self.language, dialogue),
        )
