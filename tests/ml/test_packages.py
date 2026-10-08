from pathlib import Path

from ttc.ml.packages import ModelPackage, find_model_package

MINI = Path(__file__).parent / "fixtures" / "mini_package"


def test_from_dir():
    pkg = ModelPackage.from_dir(MINI)
    assert pkg.name == "ttc_attrib_mini_test"
    assert "ru" in pkg.langs
    assert pkg.graph_path("encoder").exists()
    assert pkg.graph_path("ranker").name == "scorer.onnx"
    assert pkg.make_tokenizer().encode("hi there").ids


def test_find_via_env(monkeypatch):
    monkeypatch.setenv("TTC_MODEL_DIR", str(MINI))
    pkg = find_model_package("ru")
    assert pkg is not None and pkg.name == "ttc_attrib_mini_test"
    assert find_model_package("zh") is None  # mini package: ru/en only


def test_find_without_anything(monkeypatch):
    monkeypatch.delenv("TTC_MODEL_DIR", raising=False)
    # no entry points in the test env
    assert find_model_package("ru") is None
