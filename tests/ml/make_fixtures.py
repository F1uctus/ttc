"""Generate the seeded mini ONNX package in tests/ml/fixtures/mini_package."""

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT = Path(__file__).parent / "fixtures" / "mini_package"
DIM, VOCAB = 8, 64


def save(graph: onnx.GraphProto, path: Path) -> None:
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10
    )
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def tensor(name, arr):
    return numpy_helper.from_array(arr.astype(np.float32), name=name)


def encoder(rng) -> None:
    table = tensor("table", rng.normal(0, 0.5, (VOCAB, DIM)))
    graph = helper.make_graph(
        [helper.make_node("Gather", ["table", "input_ids"], ["emb"], axis=0)],
        "mini_encoder",
        [helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["b", "s"])],
        [helper.make_tensor_value_info("emb", TensorProto.FLOAT, ["b", "s", DIM])],
        [table],
    )
    save(graph, OUT / "encoder.onnx")


def mlp(name, in_dim, out_dim, rng, hidden=16) -> None:
    w1 = tensor("w1", rng.normal(0, 0.4, (in_dim, hidden)))
    w2 = tensor("w2", rng.normal(0, 0.4, (hidden, out_dim)))
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["x", "w1"], ["h"]),
            helper.make_node("Relu", ["h"], ["hr"]),
            helper.make_node("MatMul", ["hr", "w2"], ["y"]),
        ],
        name,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["b", in_dim])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["b", out_dim])],
        [w1, w2],
    )
    save(graph, OUT / f"{name}.onnx")


def cue_head(rng) -> None:
    w = tensor("w", rng.normal(0, 0.4, (DIM, 5)))
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["emb", "w"], ["logits"])],
        "cue",
        [helper.make_tensor_value_info("emb", TensorProto.FLOAT, ["b", "s", DIM])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["b", "s", 5])],
        [w],
    )
    save(graph, OUT / "cue.onnx")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260704)
    encoder(rng)
    cue_head(rng)
    mlp("pair", 3 * DIM, 1, rng)
    mlp("scorer", 2 * DIM + 2, 1, rng)
    (OUT / "meta.json").write_text(
        json.dumps(
            {
                "name": "ttc_attrib_mini_test",
                "version": "0.0.1",
                "langs": ["ru", "en"],
                "encoder": {
                    "graph": "encoder.onnx",
                    "dim": DIM,
                    "window": 32,
                    "stride": 24,
                    "tokenizer": {"type": "hash", "vocab_size": VOCAB},
                },
                "heads": {
                    "cue": "cue.onnx",
                    "pair": "pair.onnx",
                    "candidate": "scorer.onnx",
                    "ranker": "scorer.onnx",
                },
                "inference": {
                    "unknown_threshold": 0.05,
                    "top_k": 8,
                    "window_chars": 1200,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"fixture package -> {OUT}")


if __name__ == "__main__":
    main()
