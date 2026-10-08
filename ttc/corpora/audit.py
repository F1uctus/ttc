"""Gold audit gate: the RU gold set must pass before training uses it.

Two layers:
- mechanical: schema validation of the converted docs + unlocatable-replica
  detection (the native adapter warns; here they become hard findings);
- disagreement mining: run the rule pipeline over each gold file and flag
  prediction/annotation mismatches, ranked (high confidence = the pipeline
  predicted a concrete actor that contradicts gold, not merely None).
"""

import warnings
from dataclasses import dataclass, field
from pathlib import Path

from ttc.corpora.native import doc_from_corpus_file
from ttc.corpora.schema import validate
from ttc.corpus import UNATTRIBUTED, find_corpus_files, load_corpus_file


@dataclass
class Disagreement:
    path: Path
    replica: str
    gold: str
    pred: str
    high_confidence: bool


@dataclass
class AuditReport:
    mechanical: list[str] = field(default_factory=list)
    disagreements: list[Disagreement] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.mechanical and not any(
            d.high_confidence for d in self.disagreements
        )


def audit_native(paths: list[Path], cc=None) -> AuditReport:
    report = AuditReport()
    files: list[Path] = []
    for path in paths:
        files += find_corpus_files(path) if path.is_dir() else [path]

    for f in files:
        cf = load_corpus_file(f)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            doc = doc_from_corpus_file(cf, doc_id=f"native/{f.stem}")
        report.mechanical += [f"{f.name}: {w.message}" for w in caught]
        report.mechanical += validate(doc)
        if len(doc.replicas) < len(cf.pairs):
            report.mechanical.append(
                f"{f.name}: {len(cf.pairs) - len(doc.replicas)} unlocatable replica(s)"
            )
        if cc is not None:
            from ttc.eval import evaluate_file

            for err in evaluate_file(cc, cf).errors:
                report.disagreements.append(
                    Disagreement(
                        f,
                        err.replica,
                        err.gold,
                        err.pred,
                        high_confidence=err.pred != UNATTRIBUTED,
                    )
                )
    report.disagreements.sort(key=lambda d: not d.high_confidence)
    return report


def format_report(report: AuditReport) -> str:
    lines = ["# Gold audit report", "", "## Mechanical issues", ""]
    lines += [f"- {issue}" for issue in report.mechanical] or ["- none"]
    lines += ["", "## Rule-pipeline disagreements (review in `ttc annotate`)", ""]
    for d in report.disagreements:
        rank = "HIGH" if d.high_confidence else "low "
        lines.append(
            f"- [{rank}] {d.path.name}: gold {d.gold!r} vs pred {d.pred!r}"
            f" | {d.replica[:80]}"
        )
    if not report.disagreements:
        lines.append("- none")
    lines += ["", f"**Verdict: {'CLEAN' if report.clean else 'BLOCKED'}**", ""]
    return "\n".join(lines)
