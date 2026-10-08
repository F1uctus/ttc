from pathlib import Path

from click.testing import CliRunner

from ttc.cli import cli

MINI = Path(__file__).parent / "fixtures" / "mini_package"
TUNE_FILE = "tests/russian/texts/tune/sanderson-wayofkings-shallan-tozbek.txt"


def test_eval_pipeline_flag_rules():
    res = CliRunner().invoke(cli, ["eval", TUNE_FILE, "--pipeline", "rules"])
    assert res.exit_code == 0, res.output
    assert "TOTAL" in res.output


def test_eval_pipeline_flag_hybrid(monkeypatch):
    monkeypatch.setenv("TTC_MODEL_DIR", str(MINI))
    res = CliRunner().invoke(cli, ["eval", TUNE_FILE, "--pipeline", "hybrid"])
    assert res.exit_code == 0, res.output
    assert "TOTAL" in res.output


def test_eval_candidates_recall():
    res = CliRunner().invoke(cli, ["eval", TUNE_FILE, "--candidates", "rule"])
    assert res.exit_code == 0, res.output
    assert "recall" in res.output.lower()
