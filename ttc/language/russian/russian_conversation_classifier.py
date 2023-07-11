from dataclasses import dataclass
from typing import Dict

import spacy
from spacy import Language
from spacy.matcher import Matcher
from spacy.tokens import Doc, Token, Span

import ttc.language.russian.pipelines as russian_pipelines
from ttc.language import ConversationClassifier, Dialogue, Play
from ttc.language.common.span_extensions import SPAN_EXTENSIONS
from ttc.language.common.token_extensions import TOKEN_EXTENSIONS as TOKEN_EXTS
from ttc.language.russian.extensions.syntax_iterators import noun_chunks
from ttc.language.russian.pipelines.replicizer import extract_replicas
from ttc.language.russian.pipelines.actor_classifier import classify_actors
from ttc.language.russian.token_extensions import TOKEN_EXTENSIONS as RU_TOKEN_EXTS
from ttc.language.russian.token_patterns import (
    TokenMatcherClass,
    TOKEN_MATCHER_CLASSES,
    TokenPattern,
)


@dataclass
class RussianConversationClassifier(ConversationClassifier):
    language: Language
    token_matchers: Dict[TokenMatcherClass, Matcher]

    def __init__(self):
        super().__init__()
        self.language = spacy.load("ru_core_news_sm", exclude=["senter"])

        russian_pipelines.register_for(self.language)

        self.language.vocab.get_noun_chunks = noun_chunks

        if not Doc.has_extension("nl_indices"):
            Doc.set_extension("nl_indices", default=frozenset())

        self.token_matchers = {}
        for cls in TOKEN_MATCHER_CLASSES:
            matcher = Matcher(self.language.vocab)
            for name, value in TokenPattern.entries():
                if name.startswith(cls):
                    matcher.add(name, [value])
            self.token_matchers[cls] = matcher

        for name, ext in {**TOKEN_EXTS, **RU_TOKEN_EXTS}.items():
            if not Token.has_extension(name):
                Token.set_extension(name, **ext)

        for name, ext in SPAN_EXTENSIONS.items():
            if not Span.has_extension(name):
                Span.set_extension(name, **ext)

    def extract_dialogue(self, text: str) -> Dialogue:
        # 1. store newline indices in the separate text metadata
        # 2. pass the text to spacy with newlines completely removed/replaced with space
        #    (the latter is preferred if it does not create the separate SPACE tokens)
        doc = self.language.make_doc(text.replace("\n", " "))
        doc._.nl_indices = frozenset(i for i, c in enumerate(text) if c == "\n")
        doc = self.language(doc)
        return Dialogue(
            self.language,
            doc,
            extract_replicas(doc, self.language, self.token_matchers),
        )

    def connect_play(self, dialogue: Dialogue) -> Play:
        return classify_actors(self.language, dialogue)
