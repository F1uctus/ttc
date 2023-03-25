__version__ = "0.1.0"

from typing import Optional

from ttc.language import ConversationClassifier, LanguageCode


def load(lang_code: LanguageCode) -> Optional[ConversationClassifier]:
    normalized_lang_code = lang_code.strip().lower()
    if normalized_lang_code == "ru":
        from ttc.language.russian import RussianConversationClassifier

        return RussianConversationClassifier()
    return None
