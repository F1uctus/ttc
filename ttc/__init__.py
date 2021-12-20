__version__ = "0.1.0"

from typing import Literal, Optional

import spacy

from ttc.language import ConversationClassifier


def load(lang_code: Literal["ru", "en"]) -> Optional[ConversationClassifier]:
    normalized_lang_code = lang_code.strip().lower()
    if normalized_lang_code == "ru":
        nlp = spacy.load("ru_core_news_md")
        from ttc.language.russian import RussianConversationClassifier

        return RussianConversationClassifier(nlp)
    return None
