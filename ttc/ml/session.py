"""onnxruntime session wrapper with an execution-provider preference chain."""

import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np

EP_PREFERENCE = (
    "QNNExecutionProvider",
    "CoreMLExecutionProvider",
    "OpenVINOExecutionProvider",
    "DmlExecutionProvider",
    "NnapiExecutionProvider",
    "CPUExecutionProvider",
)


def _ort():
    try:
        import onnxruntime
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "onnxruntime is required for learned ttc components;"
            " install a ttc model package or `onnxruntime` itself."
        ) from e
    return onnxruntime


def available_providers() -> list[str]:
    return list(_ort().get_available_providers())


def resolve_providers(preferred: Sequence[str] | None = None) -> list[str]:
    if env := os.environ.get("TTC_ONNX_EP"):
        return [p.strip() for p in env.split(",") if p.strip()]
    wanted = list(preferred) if preferred else list(EP_PREFERENCE)
    installed = set(available_providers())
    chain = [p for p in wanted if p in installed]
    if "CPUExecutionProvider" not in chain:
        chain.append("CPUExecutionProvider")
    return chain


class OnnxSession:
    def __init__(
        self, model_path: Path, providers: Sequence[str] | None = None
    ) -> None:
        if not Path(model_path).exists():
            raise FileNotFoundError(model_path)
        self.providers = resolve_providers(providers)
        self._session = _ort().InferenceSession(
            str(model_path), providers=self.providers
        )

    def run(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        return self._session.run(None, inputs)
