"""Adapter for the native ttc annotated-corpus format (see ttc/corpus.py).

This is the `ttc annotate` output format and the RU gold storage format,
so it doubles as the INCEpTION-replacement ingestion path.
"""

import re
import warnings
from collections.abc import Iterator
from pathlib import Path

from ttc.corpora.schema import Character, CorpusDoc, Replica
from ttc.corpus import (
    CorpusFile,
    canonical_actor,
    find_corpus_files,
    load_corpus_file,
)

LICENSE = "annotator-owned"


def _find_span(text: str, needle: str, from_pos: int) -> tuple[int, int] | None:
    """Whitespace-insensitive ordered search of ``needle`` in ``text``."""
    pattern = r"\s+".join(re.escape(w) for w in needle.split())
    m = re.compile(pattern).search(text, from_pos)
    return (m.start(), m.end()) if m else None


def doc_from_corpus_file(cf: CorpusFile, doc_id: str) -> CorpusDoc:
    char_ids: dict[str, str] = {}  # canonical name -> char id
    characters: list[Character] = []

    def char_id(canonical: str) -> str | None:
        if canonical == "none":
            return None
        if canonical not in char_ids:
            char_ids[canonical] = f"char_{len(char_ids)}"
            characters.append(Character(char_ids[canonical], canonical))
        return char_ids[canonical]

    # alias table first, so aliases attach to their canonical character
    per_canonical: dict[str, list[str]] = {}
    for alias, canonical in cf.aliases.items():
        per_canonical.setdefault(canonical, []).append(alias)
    for canonical, aliases in per_canonical.items():
        cid = char_id(canonical)
        if cid:
            characters[-1].aliases = sorted(aliases)

    replicas: list[Replica] = []
    pos = 0
    for actor, replica_text in cf.pairs:
        span = _find_span(cf.text, replica_text, pos)
        if span is None:
            warnings.warn(f"{doc_id}: replica not found in text: {replica_text[:60]!r}")
            continue
        replicas.append(
            Replica(span[0], span[1], char_id(canonical_actor(actor, cf.aliases)))
        )
        pos = span[1]

    return CorpusDoc(
        doc_id=doc_id,
        lang="ru",
        domain="prose",
        source="native",
        license=LICENSE,
        text=cf.text,
        replicas=replicas,
        characters=characters,
    )


def convert(path: Path) -> Iterator[CorpusDoc]:
    files = find_corpus_files(path) if path.is_dir() else [path]
    for f in files:
        yield doc_from_corpus_file(load_corpus_file(f), doc_id=f"native/{f.stem}")
