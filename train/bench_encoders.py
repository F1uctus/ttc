"""Encoder bench: size, latency, probe quality and int8 drop per checkpoint."""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

CANDIDATES = [
    "microsoft/Multilingual-MiniLM-L12-H384",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "intfloat/multilingual-e5-small",
    "distilbert-base-multilingual-cased",
]


@dataclass
class BenchRow:
    checkpoint: str
    params_m: float
    latency_ms: float
    int8_latency_ms: float
    probe_acc: float
    int8_probe_acc: float


def _pool(hidden: torch.Tensor, span, offsets) -> torch.Tensor:
    s, e = span
    idx = [i for i, (a, b) in enumerate(offsets) if b > s and a < e and b > a]
    return hidden[idx].mean(0) if idx else hidden.mean(0)


def _probe_features(model, tok, examples, device):
    xs, ys = [], []
    with torch.no_grad():
        for ex in examples:
            enc = tok(
                ex["text"],
                return_offsets_mapping=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            hidden = (
                model(
                    input_ids=enc["input_ids"].to(device),
                    attention_mask=enc["attention_mask"].to(device),
                )
                .last_hidden_state[0]
                .cpu()
            )
            offsets = enc["offset_mapping"][0].tolist()
            r = _pool(hidden, ex["replica"], offsets)
            for ci, cand in enumerate(ex["candidates"]):
                m = _pool(hidden, cand, offsets)
                feats = torch.cat(
                    [
                        r,
                        m,
                        torch.tensor(
                            [
                                np.log1p(ex["dists"][ci]) / 10.0,
                                float(ex["same_lines"][ci]),
                            ]
                        ),
                    ]
                )
                xs.append(feats.numpy())
                ys.append(int(ci == ex["gold"]))
    return np.array(xs, dtype=np.float32), np.array(ys)


def _fit_probe(x: np.ndarray, y: np.ndarray) -> float:
    w = np.zeros(x.shape[1], dtype=np.float32)
    b = 0.0
    for _ in range(300):
        p = 1 / (1 + np.exp(-(x @ w + b)))
        g = p - y
        w -= 0.1 * (x.T @ g) / len(y)
        b -= 0.1 * g.mean()
    return float((((x @ w + b) > 0) == y).mean())


def bench(
    checkpoints: list[str], ranker_jsonl: Path, sample: int = 500
) -> list[BenchRow]:
    examples = [json.loads(l) for l in ranker_jsonl.read_text("utf-8").splitlines()]
    examples = examples[:sample]
    rows = []
    for ckpt in checkpoints:
        tok = AutoTokenizer.from_pretrained(ckpt)
        model = AutoModel.from_pretrained(ckpt).eval()
        params_m = sum(p.numel() for p in model.parameters()) / 1e6
        ids = torch.randint(100, 1000, (1, 256))
        mask = torch.ones_like(ids)

        def timed(m, ids=ids, mask=mask):
            with torch.no_grad():
                m(input_ids=ids, attention_mask=mask)  # warmup
                t0 = time.perf_counter()
                for _ in range(5):
                    m(input_ids=ids, attention_mask=mask)
                return (time.perf_counter() - t0) / 5 * 1000

        latency = timed(model)
        qmodel = torch.ao.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        int8_latency = timed(qmodel)
        x, y = _probe_features(model, tok, examples, "cpu")
        xq, yq = _probe_features(qmodel, tok, examples, "cpu")
        rows.append(
            BenchRow(
                ckpt,
                params_m,
                latency,
                int8_latency,
                _fit_probe(x, y),
                _fit_probe(xq, yq),
            )
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", type=Path, required=True)
    ap.add_argument("--sample", type=int, default=500)
    ap.add_argument("--checkpoints", nargs="*", default=CANDIDATES)
    args = ap.parse_args()
    print(f"{'checkpoint':<55}{'Mp':>6}{'ms':>8}{'i8ms':>8}{'probe':>7}{'i8probe':>9}")
    for r in bench(args.checkpoints, args.examples, args.sample):
        print(
            f"{r.checkpoint:<55}{r.params_m:>6.0f}{r.latency_ms:>8.1f}"
            f"{r.int8_latency_ms:>8.1f}{r.probe_acc:>7.2f}{r.int8_probe_acc:>9.2f}"
        )


if __name__ == "__main__":
    main()
