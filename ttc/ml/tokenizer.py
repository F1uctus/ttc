"""Tokenizers for ONNX components: a hashing stand-in and HF tokenizer.json."""

import re
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

WORD = re.compile(r"\S+")


@dataclass
class Encoding:
    ids: list[int]
    offsets: list[tuple[int, int]]


class Tokenizer(Protocol):
    def encode(self, text: str) -> Encoding: ...


class HashTokenizer:
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    def encode(self, text: str) -> Encoding:
        ids, offsets = [], []
        for m in WORD.finditer(text):
            # hash() is salted per process
            ids.append(
                zlib.crc32(m.group().encode("utf-8")) % (self.vocab_size - 1) + 1
            )
            offsets.append((m.start(), m.end()))
        return Encoding(ids, offsets)


class HFTokenizer:
    def __init__(self, tokenizer_file: Path) -> None:
        from tokenizers import Tokenizer as _HFTok

        self._tok = _HFTok.from_file(str(tokenizer_file))

    def encode(self, text: str) -> Encoding:
        enc = self._tok.encode(text)
        pairs = [
            (i, off) for i, off in zip(enc.ids, enc.offsets) if off[1] > off[0]
        ]  # drop specials with (0,0) offsets
        return Encoding([p[0] for p in pairs], [tuple(p[1]) for p in pairs])


def make_tokenizer(config: dict, package_dir: Path) -> Tokenizer:
    kind = config.get("type", "hf")
    if kind == "hash":
        return HashTokenizer(int(config["vocab_size"]))
    if kind == "hf":
        return HFTokenizer(package_dir / config.get("file", "tokenizer.json"))
    raise ValueError(f"unknown tokenizer type {kind!r}")
