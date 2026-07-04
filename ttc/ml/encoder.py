"""Windowed ONNX encoding pooled onto spaCy tokens, cached on Doc._.ttc_emb."""

from collections.abc import Sequence

import numpy as np
from spacy.tokens import Doc

from ttc.ml.packages import ModelPackage
from ttc.ml.session import OnnxSession


class Encoder:
    def __init__(
        self, package: ModelPackage, providers: Sequence[str] | None = None
    ) -> None:
        cfg = package.meta["encoder"]
        self.dim = int(cfg["dim"])
        self.window = int(cfg["window"])
        self.stride = int(cfg["stride"])
        self.tokenizer = package.make_tokenizer()
        self.session = OnnxSession(package.graph_path("encoder"), providers)

    def encode_doc(self, doc: Doc) -> np.ndarray:
        if doc._.ttc_emb is not None:
            return doc._.ttc_emb
        enc = self.tokenizer.encode(doc.text)
        sub_emb = np.zeros((len(enc.ids), self.dim), dtype=np.float32)
        counts = np.zeros(len(enc.ids), dtype=np.int32)
        for start in range(0, max(1, len(enc.ids)), self.stride):
            ids = enc.ids[start : start + self.window]
            if not ids:
                break
            (window_emb,) = self.session.run(
                {"input_ids": np.asarray([ids], dtype=np.int64)}
            )
            sub_emb[start : start + len(ids)] += window_emb[0]
            counts[start : start + len(ids)] += 1
            if start + self.window >= len(enc.ids):
                break
        sub_emb /= np.maximum(counts, 1)[:, None]

        # mean-pool sub-token vectors into spaCy tokens by char overlap
        token_emb = np.zeros((len(doc), self.dim), dtype=np.float32)
        token_n = np.zeros(len(doc), dtype=np.int32)
        starts = np.array([t.idx for t in doc])
        ends = starts + np.array([len(t) for t in doc])
        ti = 0
        for (s, e), vec in zip(enc.offsets, sub_emb):
            while ti < len(doc) and ends[ti] <= s:
                ti += 1
            tj = ti
            while tj < len(doc) and starts[tj] < e:
                token_emb[tj] += vec
                token_n[tj] += 1
                tj += 1
        token_emb /= np.maximum(token_n, 1)[:, None]
        doc._.ttc_emb = token_emb
        return token_emb


def pool(emb: np.ndarray, start: int, end: int) -> np.ndarray:
    """Mean vector over token range [start, end) with an empty-range guard."""
    if end <= start:
        return np.zeros(emb.shape[1], dtype=np.float32)
    return emb[start:end].mean(axis=0)
