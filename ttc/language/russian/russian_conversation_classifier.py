from dataclasses import dataclass, field
from typing import Dict

import spacy
from spacy import Language
from spacy.matcher import Matcher
from spacy.tokens import Token

import ttc.language.russian.pipelines as russian_pipelines
from ttc.language import ConversationClassifier, Dialogue, Play
from ttc.language.russian.extensions.syntax_iterators import noun_chunks
from ttc.language.russian.pipelines.replicizer import extract_replicas
from ttc.language.russian.pipelines.speaker_classifier import classify_speakers
from ttc.language.russian.token_extensions import PREDICATIVE_TOKEN_EXTENSIONS
from ttc.language.russian.token_patterns import (
    TokenMatcherClass,
    TOKEN_MATCHER_CLASSES,
    TokenPattern,
)


@dataclass
class RussianConversationClassifier(ConversationClassifier):
    language: Language = None  # type: ignore
    token_matchers: Dict[TokenMatcherClass, Matcher] = field(default_factory=dict)

    def extract_dialogue(self, text: str) -> Dialogue:
        self.prepare_spacy()
        doc = self.language(text)
        return Dialogue(
            self.language,
            doc,
            extract_replicas(doc, self.token_matchers),
        )

    def connect_play(self, dialogue: Dialogue) -> Play:
        self.prepare_spacy()
        return Play(
            self.language,
            classify_speakers(self.language, dialogue),
        )

    def prepare_spacy(self):
        if self.language:
            return

        self.language = spacy.load("ru_core_news_sm", exclude=["senter"])

        russian_pipelines.register_for(self.language)

        # Type error is Cython <-> Python incompatibility
        self.language.vocab.get_noun_chunks = noun_chunks

        if len(self.token_matchers) == 0:
            for cls in TOKEN_MATCHER_CLASSES:
                matcher = Matcher(self.language.vocab)
                for name, value in TokenPattern.entries():
                    if name.startswith(cls):
                        matcher.add(name, [value])
                self.token_matchers[cls] = matcher

        for name, pred in PREDICATIVE_TOKEN_EXTENSIONS.items():
            if not Token.has_extension(name):
                Token.set_extension(name, getter=pred)
