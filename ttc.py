from typing import Callable, Optional, Any, Tuple

from language.replica import Replica


def get_functions(
    language: str,
) -> Tuple[
    Callable[[str], Optional[Any]],
    Callable[[Any], list[Replica]],
    Callable[[Any, list[Replica]], list[Replica]],
]:
    language = language.strip().lower()
    if language in ["ru", "russian"]:
        from language.russian.regular import (
            process_doc,
            extract_replicas,
            classify_speakers,
        )
    else:
        # TODO: fallback converter
        from language.universal.regular import (
            process_doc,
            extract_replicas,
            classify_speakers,
        )

    return process_doc, extract_replicas, classify_speakers


def extract_replicas(text: str, language: str) -> list[Replica]:
    _process_doc, _extract_replicas, _ = get_functions(language)
    processor = _process_doc(text)
    return _extract_replicas(processor)


def extract_speakers(text: str, language: str) -> list[Replica]:
    _process_doc, _extract_replicas, _classify_speakers = get_functions(language)
    processor = _process_doc(text)
    replicas = _extract_replicas(processor)
    replicas = _classify_speakers(processor, replicas)
    return replicas
