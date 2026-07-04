"""Interchange JSONL to per-head training examples."""

import json
import random
from pathlib import Path

from ttc.corpora.schema import CorpusDoc, read_jsonl

PREFIX = "[LANG={lang}] [DOMAIN={domain}] "
NEG_PER_POS = 3  # pair head only


def _require_audit(doc: CorpusDoc, report: Path | None, allow: bool) -> None:
    if doc.source != "native" or allow:
        return
    if (
        report is None
        or not report.exists()
        or "Verdict: CLEAN" not in report.read_text(encoding="utf-8")
    ):
        raise RuntimeError(
            f"{doc.doc_id}: native gold requires a clean `ttc corpus audit`"
            f" report ({report}); pass allow_unaudited=True to waive."
        )


def _window(doc: CorpusDoc, start: int, end: int, before: int, after: int):
    w_start, w_end = max(0, start - before), min(len(doc.text), end + after)
    prefix = PREFIX.format(lang=doc.lang, domain=doc.domain)
    text = prefix + doc.text[w_start:w_end]

    def rel(s: int, e: int) -> tuple[int, int] | None:
        if s < w_start or e > w_end:
            return None
        return s - w_start + len(prefix), e - w_start + len(prefix)

    return text, rel


def build_examples(
    jsonl_paths: list[Path],
    out_dir: Path,
    window_before: int = 1000,
    window_after: int = 200,
    allow_unaudited: bool = False,
    audit_report: Path | None = Path("docs/corpus-audit-tune.md"),
    seed: int = 20260704,
) -> dict[str, int]:
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        task: (out_dir / f"{task}.jsonl").open("w", encoding="utf-8")
        for task in ("cue", "pair", "candidate", "ranker")
    }
    counts = dict.fromkeys(files, 0)

    def emit(task: str, obj: dict, doc: CorpusDoc) -> None:
        obj.update(lang=doc.lang, domain=doc.domain, source=doc.source)
        files[task].write(json.dumps(obj, ensure_ascii=False) + "\n")
        counts[task] += 1

    for path in jsonl_paths:
        for doc in read_jsonl(path):
            _require_audit(doc, audit_report, allow_unaudited)
            mentions = sorted(doc.mentions, key=lambda m: m.start)

            # pair head: positive iff both mentions share a character
            for i, a in enumerate(mentions):
                same = [b for b in mentions[i + 1 :] if b.char == a.char]
                diff = [b for b in mentions[i + 1 :] if b.char != a.char]
                for b in same[:1] + rng.sample(diff, min(NEG_PER_POS, len(diff))):
                    text, rel = _window(
                        doc, a.start, b.end, window_before, window_after
                    )
                    ra, rb = rel(a.start, a.end), rel(b.start, b.end)
                    if ra and rb:
                        emit(
                            "pair",
                            {
                                "text": text,
                                "a": ra,
                                "b": rb,
                                "label": int(a.char == b.char),
                            },
                            doc,
                        )

            for r in doc.replicas:
                text, rel = _window(doc, r.start, r.end, window_before, window_after)
                rr = rel(r.start, r.end)
                if rr is None:
                    continue

                # cue head
                cue_spans = [
                    c for c in [rel(r.cue.start, r.cue.end) if r.cue else None] if c
                ]
                spk_spans = [
                    s
                    for m in mentions
                    if m.char == r.speaker and (s := rel(m.start, m.end))
                ]
                if cue_spans or spk_spans:
                    emit(
                        "cue",
                        {"text": text, "cue_spans": cue_spans, "spk_spans": spk_spans},
                        doc,
                    )

                if r.speaker is None:
                    continue
                # candidate and ranker heads
                in_win: list[tuple[tuple[int, int], str, float, int]] = []
                for m in mentions:
                    if rm := rel(m.start, m.end):
                        dist = float(min(abs(m.start - r.end), abs(r.start - m.end)))
                        # same line iff both starts follow the same newline
                        same_line = int(
                            doc.text.rfind("\n", 0, r.start)
                            == doc.text.rfind("\n", 0, m.start)
                        )
                        in_win.append((rm, m.char, dist, same_line))
                gold_ids = [
                    i for i, (_, c, _, _) in enumerate(in_win) if c == r.speaker
                ]
                if not gold_ids:
                    continue
                for rm, char, dist, sl in in_win:
                    emit(
                        "candidate",
                        {
                            "text": text,
                            "replica": rr,
                            "mention": rm,
                            "label": int(char == r.speaker),
                            "dist": dist,
                            "same_line": sl,
                        },
                        doc,
                    )
                emit(
                    "ranker",
                    {
                        "text": text,
                        "replica": rr,
                        "candidates": [x[0] for x in in_win],
                        "dists": [x[2] for x in in_win],
                        "same_lines": [x[3] for x in in_win],
                        "gold": gold_ids[0],
                    },
                    doc,
                )

    for f in files.values():
        f.close()
    return counts
