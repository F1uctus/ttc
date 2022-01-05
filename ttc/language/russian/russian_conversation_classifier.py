import dataclasses
from dataclasses import dataclass

from spacy.matcher import Matcher
from spacy.tokens import Token

from ttc.language.conversation_classifier import ConversationClassifier
from ttc.language.dialogue import Dialogue
from ttc.language.play import Play
from ttc.language.russian.constants import (
    DASHES,
    OPEN_QUOTES,
    CLOSE_QUOTES,
    SPECIAL_VERBS,
)
from ttc.language.russian.pipelines import setup_language
from ttc.language.russian.regular import extract_replicas, classify_speakers
from ttc.language.russian.token_patterns import (
    TokenPatterns,
    MATCHER_CLASSES,
    MatcherClass,
)


@dataclass(slots=True)
class RussianConversationClassifier(ConversationClassifier):
    token_predicates = {
        "is_sent_end": lambda t: t.is_sent_end,
        "is_dash": lambda t: t.text in DASHES,
        "is_newline": lambda t: "\n" in t.text,
        "is_open_quote": lambda t: t.text in OPEN_QUOTES,
        "is_close_quote": lambda t: t.text in CLOSE_QUOTES,
        "is_author_verb": lambda t: any(v in t.lemma_ for v in SPECIAL_VERBS),
        "is_not_second_person": lambda t: "Person=Second" not in t.morph,
    }
    matchers: dict[MatcherClass, Matcher] = dataclasses.field(default_factory=dict)

    def extract_dialogue(self, text: str) -> Dialogue:
        setup_language(self.language)

        if len(self.matchers) == 0:
            for cls in MATCHER_CLASSES:
                matcher = Matcher(self.language.vocab)
                for name, value in TokenPatterns.entries():
                    if name.startswith(cls):
                        matcher.add(name, [value])
                self.matchers[cls] = matcher

        for name, pred in RussianConversationClassifier.token_predicates.items():
            if not Token.has_extension(name):
                Token.set_extension(name, getter=pred)

        return Dialogue(
            self.language,
            extract_replicas(self.language(text), self.matchers),
        )

    def connect_play(self, dialogue: Dialogue) -> Play:
        return Play(
            self.language,
            classify_speakers(self.language, dialogue),
        )
