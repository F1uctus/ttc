"""Config-driven multi-task training: one curriculum stage per invocation."""

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
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
    if (prev_i := order.index(stage)) > 0:
        prev = out_dir / order[prev_i - 1] / "model.pt"
        if prev.exists():
            model.load_state_dict(torch.load(prev, map_location=device))

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
        losses[task] = float(loss)

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
            f"\n- train {stage} @ {commit}: losses {dict(losses)},"
            f" mix {json.dumps(mix)[:200]}\n"
        )
    return stage_dir / "model.pt"


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
