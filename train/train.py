"""Config-driven multi-task training: one curriculum stage per invocation."""

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer

from train.config import load_config, set_seeds
from train.mix import MixtureSampler, temperature_weights
from train.model import AttributionModel

TASKS = ("cue", "pair", "candidate", "ranker")
BIO = {"O": 0, "B-CUE": 1, "I-CUE": 2, "B-SPK": 3, "I-SPK": 4}


def _load_examples(examples_dir: Path, sources: list[str]) -> dict[str, dict]:
    by_task: dict[str, dict[str, list[dict]]] = {t: defaultdict(list) for t in TASKS}
    for task in TASKS:
        path = examples_dir / f"{task}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text("utf-8").splitlines():
            ex = json.loads(line)
            if ex["source"] in sources or "*" in sources:
                by_task[task][ex["source"]].append(ex)
    return by_task


def _token_range(offsets, span):
    s, e = span
    idx = [i for i, (a, b) in enumerate(offsets) if b > s and a < e and b > a]
    return (idx[0], idx[-1] + 1) if idx else (0, 1)


def _step(model, tok, task: str, ex: dict, device) -> torch.Tensor:
    enc = tok(
        ex["text"],
        return_offsets_mapping=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    offsets = enc["offset_mapping"][0].tolist()
    hidden = model.encode(enc["input_ids"].to(device), enc["attention_mask"].to(device))
    h0 = hidden[0]
    if task == "cue":
        bio = torch.zeros(len(offsets), dtype=torch.long, device=device)
        for spans, b, i in (
            (ex["cue_spans"], BIO["B-CUE"], BIO["I-CUE"]),
            (ex["spk_spans"], BIO["B-SPK"], BIO["I-SPK"]),
        ):
            for span in spans:
                ts, te = _token_range(offsets, span)
                bio[ts] = b
                bio[ts + 1 : te] = i
        return model.cue_loss(hidden, bio[None])
    if task == "pair":
        return model.pair_loss(
            h0,
            [_token_range(offsets, ex["a"])],
            [_token_range(offsets, ex["b"])],
            torch.tensor([ex["label"]], device=device),
        )

    def extra_of(dists, lines):
        return torch.tensor(
            [[np.log1p(d) / 10.0, float(sl)] for d, sl in zip(dists, lines)],
            dtype=torch.float32,
            device=device,
        )

    if task == "candidate":
        return model.scorer_loss(
            h0,
            _token_range(offsets, ex["replica"]),
            [_token_range(offsets, ex["mention"])],
            extra_of([ex["dist"]], [ex["same_line"]]),
            torch.tensor([ex["label"]], device=device),
        )
    return model.ranker_loss(
        h0,
        _token_range(offsets, ex["replica"]),
        [_token_range(offsets, c) for c in ex["candidates"]],
        extra_of(ex["dists"], ex["same_lines"]),
        ex["gold"],
    )


def run(
    config_path: Path, stage: str, out_dir: Path, overrides: dict | None = None
) -> Path:
    overrides = overrides or {}
    cfg = load_config(config_path, overrides)
    if cfg["training"].get("mode", "cached") == "cached":
        return train_heads_cached(
            Path(cfg["data"]["examples_dir"]).parent / "cache", cfg, out_dir, stage
        )
    set_seeds(cfg["training"]["seed"])
    device = cfg["training"]["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    stage_cfg = cfg["stages"][stage]
    examples_dir = Path(cfg["data"]["examples_dir"])
    by_task = _load_examples(examples_dir, stage_cfg["sources"])
    samplers = {
        t: MixtureSampler(
            src_map, cfg["mixing"]["temperature"], cfg["training"]["seed"]
        )
        for t, src_map in by_task.items()
        if src_map
    }
    if not samplers:
        raise RuntimeError(f"no examples for stage {stage!r} in {examples_dir}")

    model = AttributionModel(
        cfg["encoder"]["checkpoint"], cfg["encoder"]["dim"], cfg["heads"]["hidden"]
    ).to(device)
    order = cfg["stages"]["order"]
    # previous stage only: order[-1] would wrap to the last stage
    if (prev_i := order.index(stage)) > 0:
        prev_dir = out_dir / order[prev_i - 1]
        if (prev_model := prev_dir / "model.pt").exists():
            model.load_state_dict(torch.load(prev_model, map_location=device))
        if (prev_heads := prev_dir / "heads.pt").exists():
            model.load_state_dict(
                torch.load(prev_heads, map_location=device), strict=False
            )
    n_unfreeze = int(cfg["encoder"].get("unfreeze_layers", 0))
    if n_unfreeze >= 0:
        for p in model.encoder.parameters():
            p.requires_grad = False
        layers = model.encoder.encoder.layer  # BERT-family layer stack
        for layer in layers[len(layers) - n_unfreeze :] if n_unfreeze else []:
            for p in layer.parameters():
                p.requires_grad = True

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"])
    steps = int(overrides.get("training.steps", stage_cfg["steps"]))
    tok = AutoTokenizer.from_pretrained(cfg["encoder"]["checkpoint"])
    task_cycle = [t for t in TASKS if t in samplers]
    losses: dict[str, float] = defaultdict(float)
    model.train()
    for step in range(steps):
        task = task_cycle[step % len(task_cycle)]
        opt.zero_grad()
        loss = (
            sum(
                _step(model, tok, task, next(samplers[task]), device)
                for _ in range(cfg["training"]["batch_size"])
            )
            / cfg["training"]["batch_size"]
        )
        loss.backward()
        opt.step()
        losses[task] = loss.item()

    stage_dir = out_dir / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), stage_dir / "model.pt")
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    mix = {
        t: temperature_weights(
            {k: len(v) for k, v in by_task[t].items()},
            cfg["mixing"]["temperature"],
        )
        for t in samplers
    }
    (stage_dir / "run.json").write_text(
        json.dumps(
            {
                "stage": stage,
                "commit": commit,
                "mix": mix,
                "final_losses": losses,
                "config": dict(cfg),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    with open("docs/eval-log.md", "a", encoding="utf-8") as f:
        f.write(
            f"\n- train {stage} @ {commit}: "
            f"losses {json.dumps(dict(losses))}, mix {json.dumps(mix)}\n"
        )
    return stage_dir / "model.pt"


def _infer_dim(cache_dir: Path) -> int:
    from train.cache import load_task

    return (load_task(cache_dir, "candidate")["X"].shape[1] - 2) // 2


def train_heads_cached(cache_dir: Path, cfg, out_dir: Path, stage: str) -> Path:
    from train.cache import load_task
    from train.mix import temperature_weights
    from train.model import CachedHeads

    set_seeds(cfg["training"]["seed"])
    dim = cfg["encoder"]["dim"] or _infer_dim(cache_dir)
    heads = CachedHeads(dim, cfg["heads"]["hidden"])
    opt = torch.optim.AdamW(heads.parameters(), lr=cfg["training"]["lr"])
    t = cfg["mixing"]["temperature"]
    rng = np.random.default_rng(cfg["training"]["seed"])

    data = {task: load_task(cache_dir, task) for task in TASKS}
    idx = {}
    for task in TASKS:
        by_src = defaultdict(list)
        for i, s in enumerate(data[task]["source"]):
            by_src[str(s)].append(i)
        idx[task] = by_src
    weights = {
        task: temperature_weights({s: len(v) for s, v in idx[task].items()}, t)
        for task in TASKS
        if idx[task]
    }

    def sample(task: str) -> int:
        srcs = list(weights[task])
        s = srcs[int(rng.choice(len(srcs), p=[weights[task][k] for k in srcs]))]
        return int(rng.choice(idx[task][s]))

    task_cycle = [t_ for t_ in TASKS if weights.get(t_)]
    bs = cfg["training"]["batch_size"]
    losses: dict[str, float] = defaultdict(float)
    heads.train()
    for step in range(int(cfg["training"]["steps"])):
        task = task_cycle[step % len(task_cycle)]
        opt.zero_grad()
        loss = _cached_loss(heads, task, data[task], [sample(task) for _ in range(bs)])
        loss.backward()
        opt.step()
        losses[task] = loss.detach().item()

    stage_dir = out_dir / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    torch.save(heads.state_dict(), stage_dir / "heads.pt")
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    mix = {task: weights.get(task, {}) for task in TASKS}
    (stage_dir / "run.json").write_text(
        json.dumps(
            {
                "stage": stage,
                "mode": "cached",
                "commit": commit,
                "mix": mix,
                "final_losses": losses,
                "dim": dim,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    with open("docs/eval-log.md", "a", encoding="utf-8") as f:
        f.write(f"\n- train {stage} (cached) @ {commit}: losses {dict(losses)}\n")
    return stage_dir / "heads.pt"


def _cached_loss(heads, task: str, arrs: dict, rows: list[int]) -> torch.Tensor:
    if task == "cue":
        # rows index windows; rebuild each window's [S,dim] slice from lengths
        starts = np.concatenate([[0], np.cumsum(arrs["lengths"])])
        total = torch.zeros((), dtype=torch.float32)
        for r in rows:
            s, e = int(starts[r]), int(starts[r + 1])
            emb = torch.from_numpy(arrs["emb"][s:e])
            bio = torch.from_numpy(arrs["bio"][s:e])
            total = total + nn.functional.cross_entropy(heads.cue_head(emb), bio)
        return total / len(rows)
    if task == "pair":
        x = torch.from_numpy(arrs["X"][rows])
        y = torch.from_numpy(arrs["y"][rows]).float()
        return nn.functional.binary_cross_entropy_with_logits(
            heads.pair_head(x)[:, 0], y
        )
    if task == "candidate":
        x = torch.from_numpy(arrs["X"][rows])
        y = torch.from_numpy(arrs["y"][rows]).float()
        return nn.functional.binary_cross_entropy_with_logits(
            heads.scorer_head(x)[:, 0], y
        )
    # ranker: rows index groups; slice each group's rows and softmax-CE vs gold
    starts = np.concatenate([[0], np.cumsum(arrs["groups"])])
    total = torch.zeros((), dtype=torch.float32)
    for r in rows:
        s, e = int(starts[r]), int(starts[r + 1])
        scores = heads.scorer_head(torch.from_numpy(arrs["X"][s:e]))[:, 0]
        total = total + nn.functional.cross_entropy(
            scores[None], torch.tensor([int(arrs["gold"][r])])
        )
    return total / len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()
    overrides = dict(o.split("=", 1) for o in args.override)
    print(run(args.config, args.stage, args.out, overrides))


if __name__ == "__main__":
    main()
