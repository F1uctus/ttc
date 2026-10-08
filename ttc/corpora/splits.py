"""Deterministic tune/heldout split of non-native corpora by doc_id hash."""

import hashlib


def split_of(doc_id: str, heldout_fraction: float = 0.2) -> str:
    digest = hashlib.sha1(doc_id.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:4], "big") / 2**32
    return "heldout" if bucket < heldout_fraction else "tune"
