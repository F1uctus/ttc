from pathlib import Path

from ttc.corpora.audit import AuditReport, Disagreement, audit_native, format_report

FIXTURES = Path(__file__).parent / "fixtures" / "native"


def test_mechanical_pass_without_pipeline():
    report = audit_native([FIXTURES / "sample.txt"], cc=None)
    assert report.mechanical == []
    assert report.disagreements == []
    assert report.clean


def test_mechanical_catches_broken_file(tmp_path: Path):
    bad = tmp_path / "bad.txt"
    bad.write_text(
        "Текст без второй реплики.\n"
        "--------------------\n"
        "Кто-то::Этой реплики нет в тексте.\n",
        encoding="utf-8",
    )
    report = audit_native([bad], cc=None)
    assert not report.clean
    assert any(
        "not found" in issue or "unlocatable" in issue for issue in report.mechanical
    )


def test_format_report_lists_disagreements():
    report = AuditReport(
        mechanical=["doc: replica 0 out of bounds"],
        disagreements=[
            Disagreement(Path("a.txt"), "– Привет.", "ясна", "тозбек", True)
        ],
    )
    text = format_report(report)
    assert "out of bounds" in text
    assert "ясна" in text and "тозбек" in text and "HIGH" in text
