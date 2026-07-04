"""Frozen-encoder feature cache for cached head training."""

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from train.train import _token_range

TASKS = ("cue", "pair", "candidate", "ranker")


def _pool(hidden: np.ndarray, rng: tuple[int, int]) -> np.ndarray:
    s, e = rng
    return hidden[s:e].mean(0) if e > s else np.zeros(hidden.shape[1], np.float32)


def _extra(dist: float, same_line: int) -> np.ndarray:
    return np.array([np.log1p(dist) / 10.0, float(same_line)], np.float32)


class _Encoder:
    def __init__(self, checkpoint: str, device: str, max_len: int) -> None:
        self.tok = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModel.from_pretrained(checkpoint).to(device).eval()
        self.device, self.max_len = device, max_len
        self.dim = self.model.config.hidden_size
        self._memo: dict[str, tuple[np.ndarray, list]] = {}

    def __call__(self, text: str) -> tuple[np.ndarray, list]:
        if text not in self._memo:
            enc = self.tok(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            )
            with torch.no_grad():
                hidden = (
                    self.model(
                        input_ids=enc["input_ids"].to(self.device),
                        attention_mask=enc["attention_mask"].to(self.device),
                    )
                    .last_hidden_state[0]
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
            self._memo[text] = (hidden, enc["offset_mapping"][0].tolist())
        return self._memo[text]


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text("utf-8").splitlines()]


def build_cache(
    examples_dir: Path,
    out_dir: Path,
    checkpoint: str,
    device: str = "cpu",
    max_len: int = 256,
    max_cue: int = 20000,
) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    enc = _Encoder(checkpoint, device, max_len)
    counts: dict[str, int] = {}

    X, y, src = [], [], []
    for ex in _read(examples_dir / "pair.jsonl"):
        hidden, off = enc(ex["text"])
        a = _pool(hidden, _token_range(off, ex["a"]))
        b = _pool(hidden, _token_range(off, ex["b"]))
        X.append(np.concatenate([a, b, a * b]))
        y.append(ex["label"])
        src.append(ex["source"])
    np.savez(
        out_dir / "pair.npz",
        X=np.array(X, np.float32).reshape(len(X), -1),
        y=np.array(y, np.int64),
        source=np.array(src),
    )
    counts["pair"] = len(X)

    X, y, src = [], [], []
    for ex in _read(examples_dir / "candidate.jsonl"):
        hidden, off = enc(ex["text"])
        r = _pool(hidden, _token_range(off, ex["replica"]))
        m = _pool(hidden, _token_range(off, ex["mention"]))
        X.append(np.concatenate([r, m, _extra(ex["dist"], ex["same_line"])]))
        y.append(ex["label"])
        src.append(ex["source"])
    np.savez(
        out_dir / "candidate.npz",
        X=np.array(X, np.float32).reshape(len(X), -1),
        y=np.array(y, np.int64),
        source=np.array(src),
    )
    counts["candidate"] = len(X)

    # ranker: ragged K per replica, concatenated rows plus group lengths
    rows, groups, gold, src = [], [], [], []
    for ex in _read(examples_dir / "ranker.jsonl"):
        hidden, off = enc(ex["text"])
        r = _pool(hidden, _token_range(off, ex["replica"]))
        k = 0
        for cand, d, sl in zip(ex["candidates"], ex["dists"], ex["same_lines"]):
            m = _pool(hidden, _token_range(off, cand))
            rows.append(np.concatenate([r, m, _extra(d, sl)]))
            k += 1
        groups.append(k)
        gold.append(ex["gold"])
        src.append(ex["source"])
    np.savez(
        out_dir / "ranker.npz",
        X=np.array(rows, np.float32).reshape(len(rows), -1),
        groups=np.array(groups, np.int64),
        gold=np.array(gold, np.int64),
        source=np.array(src),
    )
    counts["ranker"] = len(groups)

    # cue: ragged S per window, concatenated token rows plus BIO labels
    all_cue = _read(examples_dir / "cue.jsonl")
    if len(all_cue) > max_cue:
        print(f"cue cache capped at {max_cue} of {len(all_cue)} windows")
    embs, lengths, bio, src = [], [], [], []
    for ex in all_cue[:max_cue]:
        hidden, off = enc(ex["text"])
        labels = np.zeros(len(off), np.int64)
        for spans, b, i in ((ex["cue_spans"], 1, 2), (ex["spk_spans"], 3, 4)):
            for span in spans:
                ts, te = _token_range(off, span)
                labels[ts] = b
                labels[ts + 1 : te] = i
        embs.append(hidden)
        lengths.append(len(off))
        bio.append(labels)
        src.append(ex["source"])
    np.savez(
        out_dir / "cue.npz",
        emb=np.concatenate(embs) if embs else np.zeros((0, enc.dim), np.float32),
        lengths=np.array(lengths, np.int64),
        bio=np.concatenate(bio) if bio else np.zeros(0, np.int64),
        source=np.array(src),
    )
    counts["cue"] = len(lengths)
    return counts


def load_task(out_dir: Path, task: str) -> dict:
    with np.load(out_dir / f"{task}.npz", allow_pickle=True) as z:
        return {k: z[k] for k in z.files}
