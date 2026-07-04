"""Accuracy evaluation of the pipeline against the annotated corpus.

Metrics per file and micro-averaged:

- extraction precision / recall — how well predicted replica texts match
  the gold replica sequence (order-preserving exact-text alignment);
- attribution accuracy — share of *matched* replicas whose predicted
  actor equals the gold actor (after alias canonicalization);
- end-to-end accuracy — correctly attributed replicas / all gold replicas.
"""

import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from spacy.tokens import Span

from ttc.corpus import (
    CorpusFile,
    UNATTRIBUTED,
    canonical_actor,
    find_corpus_files,
    load_corpus_file,
    normalize_name,
)


@dataclass
class AttrError:
    replica: str
    gold: str
    pred: str


@dataclass
class Counters:
    n_gold: int = 0
    n_pred: int = 0
    n_matched: int = 0
    n_attr_correct: int = 0

    @property
    def extraction_precision(self) -> Optional[float]:
        return self.n_matched / self.n_pred if self.n_pred else None

    @property
    def extraction_recall(self) -> Optional[float]:
        return self.n_matched / self.n_gold if self.n_gold else None

    @property
    def attribution_accuracy(self) -> Optional[float]:
        return self.n_attr_correct / self.n_matched if self.n_matched else None

    @property
    def end_to_end_accuracy(self) -> Optional[float]:
        return self.n_attr_correct / self.n_gold if self.n_gold else None

    def add(self, other: "Counters") -> None:
        self.n_gold += other.n_gold
        self.n_pred += other.n_pred
        self.n_matched += other.n_matched
        self.n_attr_correct += other.n_attr_correct


@dataclass
class FileReport(Counters):
    path: Optional[Path] = None
    errors: List[AttrError] = field(default_factory=list)
    seconds: float = 0.0
    lang: str = "ru"
    qtype_counters: Dict[str, Counters] = field(default_factory=dict)


def pred_actor_key(actor: Optional[Span], aliases: Dict[str, str]) -> str:
    """Canonicalize a predicted actor span.

    Predictions are often inflected surface forms («Ясну»), so when the
    surface form has no alias entry, the span lemma is also tried before
    giving up.
    """
    if actor is None or not len(actor):
        return UNATTRIBUTED
    surface = normalize_name(str(actor))
    if surface in aliases:
        return aliases[surface]
    lemma = normalize_name(actor.lemma_)
    if lemma in aliases:
        return aliases[lemma]
    return surface


def align_replicas(gold: List[str], pred: List[str]) -> List[Tuple[int, int]]:
    matcher = SequenceMatcher(a=gold, b=pred, autojunk=False)
    return [
        (block.a + k, block.b + k)
        for block in matcher.get_matching_blocks()
        for k in range(block.size)
    ]


def evaluate_file(cc, cf: CorpusFile) -> FileReport:
    started = time.perf_counter()
    dialogue = cc.extract_dialogue(cf.text)
    play = cc.connect_play(dialogue)
    seconds = time.perf_counter() - started

    gold = [
        (replica, canonical_actor(actor, cf.aliases)) for actor, replica in cf.pairs
    ]
    pred = [(str(r), pred_actor_key(a, cf.aliases)) for r, a in play.lines]

    report = FileReport(path=cf.path, seconds=seconds)
    report.n_gold = len(gold)
    report.n_pred = len(pred)
    for gi, pi in align_replicas([g[0] for g in gold], [p[0] for p in pred]):
        report.n_matched += 1
        if gold[gi][1] == pred[pi][1]:
            report.n_attr_correct += 1
        else:
            report.errors.append(AttrError(gold[gi][0], gold[gi][1], pred[pi][1]))
    return report


def evaluate_interchange_doc(cc, doc) -> FileReport:
    """Evaluate attribution on one interchange doc (gold = doc.replicas).

    ``doc`` is a :class:`ttc.corpora.schema.CorpusDoc`. Gold speakers are
    canonicalized through the doc's own character/alias table; results are
    additionally broken down per PDNC-style quotation type (qtype).
    """
    started = time.perf_counter()
    dialogue = cc.extract_dialogue(doc.text)
    play = cc.connect_play(dialogue)
    seconds = time.perf_counter() - started

    names = {c.id: normalize_name(c.name) for c in doc.characters}
    aliases: Dict[str, str] = {}
    for c in doc.characters:
        aliases[normalize_name(c.name)] = names[c.id]
        for alias in c.aliases:
            aliases[normalize_name(alias)] = names[c.id]

    gold = [
        (
            doc.text[r.start : r.end],
            names.get(r.speaker, UNATTRIBUTED) if r.speaker else UNATTRIBUTED,
            r.qtype,
        )
        for r in doc.replicas
    ]
    pred = [(str(r), pred_actor_key(a, aliases)) for r, a in play.lines]

    report = FileReport(path=Path(doc.doc_id), lang=doc.lang, seconds=seconds)
    report.n_gold = len(gold)
    report.n_pred = len(pred)
    for qtype in {g[2] for g in gold if g[2]}:
        report.qtype_counters[qtype] = Counters(
            n_gold=sum(1 for g in gold if g[2] == qtype)
        )
    for gi, pi in align_replicas([g[0] for g in gold], [p[0] for p in pred]):
        report.n_matched += 1
        qt = gold[gi][2]
        if qt:
            report.qtype_counters[qt].n_matched += 1
        if gold[gi][1] == pred[pi][1]:
            report.n_attr_correct += 1
            if qt:
                report.qtype_counters[qt].n_attr_correct += 1
        else:
            report.errors.append(AttrError(gold[gi][0], gold[gi][1], pred[pi][1]))
    return report


def evaluate_paths(cc, paths: List[Path]) -> List[FileReport]:
    files: List[Path] = []
    for path in paths:
        files += find_corpus_files(path) if path.is_dir() else [path]
    return [evaluate_file(cc, load_corpus_file(f)) for f in files]


def aggregate(reports: List[FileReport]) -> Counters:
    total = Counters()
    for report in reports:
        total.add(report)
    return total


def _percent(value: Optional[float]) -> str:
    return f"{value:7.1%}" if value is not None else "      -"


def format_report(
    reports: List[FileReport],
    by_file: bool = False,
    show_errors: bool = False,
) -> str:
    lines = []
    if by_file or show_errors:
        for r in reports:
            name = r.path.name if r.path else "<content>"
            lines.append(
                f"{name:<44} e2e {_percent(r.end_to_end_accuracy)}"
                f"  attr {_percent(r.attribution_accuracy)}"
                f"  extr P {_percent(r.extraction_precision)}"
                f" R {_percent(r.extraction_recall)}"
                f"  ({r.n_attr_correct}/{r.n_gold}, {r.seconds:.1f}s)"
            )
            if show_errors:
                for e in r.errors:
                    lines.append(
                        f"    gold {e.gold!r} != pred {e.pred!r} | {e.replica[:80]}"
                    )
    total = aggregate(reports)
    lines.append(
        f"{'TOTAL (' + str(len(reports)) + ' files)':<44}"
        f" e2e {_percent(total.end_to_end_accuracy)}"
        f"  attr {_percent(total.attribution_accuracy)}"
        f"  extr P {_percent(total.extraction_precision)}"
        f" R {_percent(total.extraction_recall)}"
        f"  ({total.n_attr_correct}/{total.n_gold})"
    )
    by_qtype: Dict[str, Counters] = {}
    for r in reports:
        for qt, c in r.qtype_counters.items():
            by_qtype.setdefault(qt, Counters()).add(c)
    for qt in sorted(by_qtype):
        c = by_qtype[qt]
        lines.append(
            f"  qtype {qt:<10} e2e {_percent(c.end_to_end_accuracy)}"
            f"  attr {_percent(c.attribution_accuracy)}"
            f"  ({c.n_attr_correct}/{c.n_gold})"
        )
    return "\n".join(lines)
