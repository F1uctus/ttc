import ttc


def test_no_package_means_pure_rules(monkeypatch):
    monkeypatch.delenv("TTC_MODEL_DIR", raising=False)
    cc = ttc.load("ru")
    assert cc.package is None
    assert cc.pipeline_mode == "rules"


def test_explicit_rules_ignores_package(monkeypatch, tmp_path):
    from pathlib import Path

    mini = Path(__file__).parent / "fixtures" / "mini_package"
    monkeypatch.setenv("TTC_MODEL_DIR", str(mini))
    cc = ttc.load("ru", pipeline="rules")
    assert cc.pipeline_mode == "rules"


def test_auto_uses_package_when_present(monkeypatch):
    from pathlib import Path

    mini = Path(__file__).parent / "fixtures" / "mini_package"
    monkeypatch.setenv("TTC_MODEL_DIR", str(mini))
    cc = ttc.load("ru")
    assert cc.package is not None
    assert cc.pipeline_mode == "hybrid"
