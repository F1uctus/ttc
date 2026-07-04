"""Deterministic tune/heldout assignment for non-native corpora.

Native RU gold keeps its directory-based split (tests/russian/texts/*);
everything else is split by a stable hash of doc_id so that re-running
a conversion never migrates a document across splits.
"""

import hashlib


def split_of(doc_id: str, heldout_fraction: float = 0.2) -> str:
    digest = hashlib.sha1(doc_id.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:4], "big") / 2**32
    return "heldout" if bucket < heldout_fraction else "tune"
