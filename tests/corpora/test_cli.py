import json
from pathlib import Path

from click.testing import CliRunner

from ttc.cli import cli

FIXTURES = Path(__file__).parent / "fixtures"


def test_corpus_convert_and_stats(tmp_path: Path):
    out = tmp_path / "native.jsonl"
    runner = CliRunner()
    res = runner.invoke(
        cli,
        [
            "corpus",
            "convert",
            "native",
            str(FIXTURES / "native" / "sample.txt"),
            "--out",
            str(out),
            "--split",
            "all",
        ],
    )
    assert res.exit_code == 0, res.output
    assert "1 doc" in res.output
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["source"] == "native"

    res = runner.invoke(cli, ["corpus", "stats", str(out)])
    assert res.exit_code == 0, res.output
    assert "native" in res.output and "ru" in res.output


def test_corpus_convert_unknown_source():
    res = CliRunner().invoke(cli, ["corpus", "convert", "nope", ".", "--out", "x"])
    assert res.exit_code != 0
    assert "Unknown corpus source" in res.output
