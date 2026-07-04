from pathlib import Path

import numpy as np
import pytest

from ttc.ml.session import EP_PREFERENCE, OnnxSession, resolve_providers

MINI = Path(__file__).parent / "fixtures" / "mini_package"


def test_resolve_providers_ends_with_cpu():
    providers = resolve_providers()
    assert providers[-1] == "CPUExecutionProvider"
    assert all(p in EP_PREFERENCE for p in providers)


def test_resolve_providers_env_override(monkeypatch):
    monkeypatch.setenv("TTC_ONNX_EP", "CPUExecutionProvider")
    assert resolve_providers() == ["CPUExecutionProvider"]


def test_session_runs_mini_encoder():
    session = OnnxSession(MINI / "encoder.onnx")
    ids = np.array([[1, 2, 3, 0]], dtype=np.int64)
    (emb,) = session.run({"input_ids": ids})
    assert emb.shape == (1, 4, 8)
    assert emb.dtype == np.float32
    (emb2,) = session.run({"input_ids": ids})
    np.testing.assert_array_equal(emb, emb2)


def test_missing_model_file_raises():
    with pytest.raises(FileNotFoundError):
        OnnxSession(MINI / "nope.onnx")
