from typing import Callable, Optional, Any

from language.span import Span


def get_functions(language: str, exact: bool) -> tuple[
    Callable[[str], Optional[Any]],
    Callable[[Any], list[Span]],
    Callable[[Any, list[Span]], list[Span]],
]:
    language = language.strip().lower()
    if language in ['ru', 'russian']:
        if exact:
            from language.russian.exact import (
                process_doc,
                extract_replicas,
                classify_speakers
            )
        else:
            from language.russian.regular import (
                process_doc,
                extract_replicas,
                classify_speakers
            )
    else:
        # TODO: fallback converter
        if exact:
            from language.universal.exact import (
                process_doc,
                extract_replicas,
                classify_speakers
            )
        else:
            from language.universal.regular import (
                process_doc,
                extract_replicas,
                classify_speakers
            )

    return process_doc, extract_replicas, classify_speakers


def extract_replicas(
    text: str,
    language: str,
    exact: bool = False
) -> list[Span]:
    process_doc, extract_replicas, classify_speakers = get_functions(
        language,
        exact
    )
    document = process_doc(text)
    replicas = extract_replicas(document)
    replicas = classify_speakers(document, replicas)
    return replicas


def extract_speakers(
    text: str,
    language: str,
    exact: bool = False
) -> list[Span]:
    process_doc, extract_replicas, classify_speakers = get_functions(
        language,
        exact
    )
    document = process_doc(text)
    replicas = extract_replicas(document)
    replicas = classify_speakers(document, replicas)
    return replicas
