"""Translation projection: MT plus word-alignment span transfer."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from ttc.corpora.schema import (
    CorpusDoc,
    Cue,
    Mention,
    Replica,
    read_jsonl,
    write_jsonl,
)

Span = tuple[int, int]


class Aligner(Protocol):
    def align(self, src: str, tgt: str) -> list[tuple[Span, Span, float]]: ...


def _merged_length(spans: list[Span]) -> int:
    """Total length covered by (possibly overlapping) spans after merging."""
    if not spans:
        return 0
    ordered = sorted(spans)
    total = 0
    cur_s, cur_e = ordered[0]
    for s, e in ordered[1:]:
        if s > cur_e:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    total += cur_e - cur_s
    return total


def _project_span(
    span: Span, alignment: list[tuple[Span, Span, float]]
) -> tuple[Span, float] | None:
    """Project a source char span onto the target, or None without overlap.

    Confidence is overlap-weighted and scaled by target coverage density.
    """
    s, e = span
    subspans: list[tuple[int, int, float, int]] = []
    for (ss, se), (ts, te), conf in alignment:
        if se <= s or ss >= e:
            continue
        width = se - ss
        if width <= 0:
            ov_ts, ov_te = ts, te
        else:
            ov_s, ov_e = max(s, ss), min(e, se)
            ov_ts = ts + round((ov_s - ss) * (te - ts) / width)
            ov_te = ts + round((ov_e - ss) * (te - ts) / width)
        overlap = max(1, min(e, se) - max(s, ss))
        subspans.append((min(ov_ts, ov_te), max(ov_ts, ov_te), conf, overlap))
    if not subspans:
        return None
    t_start = min(x[0] for x in subspans)
    t_end = max(x[1] for x in subspans)
    wsum = sum(x[3] for x in subspans)
    conf = sum(x[2] * x[3] for x in subspans) / wsum
    covered = _merged_length([(x[0], x[1]) for x in subspans])
    extent = max(1, t_end - t_start)
    return (t_start, t_end), conf * covered / extent


def project_doc(
    doc: CorpusDoc,
    translate: Callable[[str], str],
    target_lang: str,
    aligner: Aligner,
    min_conf: float = 0.5,
) -> tuple[CorpusDoc, list[dict]]:
    tgt_text = translate(doc.text)
    alignment = aligner.align(doc.text, tgt_text)
    flags: list[dict] = []

    def project(span: Span, kind: str, ref: str) -> Span | None:
        got = _project_span(span, alignment)
        if got is None or got[1] < min_conf:
            flags.append(
                {
                    "kind": kind,
                    "ref": ref,
                    "src_span": list(span),
                    "conf": None if got is None else got[1],
                    "doc_id": doc.doc_id,
                }
            )
            return None
        return got[0]

    replicas: list[Replica] = []
    for i, r in enumerate(doc.replicas):
        ts = project((r.start, r.end), "replica", str(i))
        if ts is None:
            continue
        cue = None
        if r.cue and (tc := project((r.cue.start, r.cue.end), "cue", str(i))):
            cue = Cue(tc[0], tc[1])
        replicas.append(
            Replica(ts[0], ts[1], r.speaker, r.addressee, r.qtype, cue, r.mode)
        )

    mentions: list[Mention] = []
    for j, m in enumerate(doc.mentions):
        if tm := project((m.start, m.end), "mention", str(j)):
            mentions.append(Mention(tm[0], tm[1], m.char))

    projected = CorpusDoc(
        doc_id=f"projected:{target_lang}:{doc.doc_id}",
        lang=target_lang,
        domain=doc.domain,
        source="projected",
        license=doc.license,
        text=tgt_text,
        replicas=replicas,
        characters=list(doc.characters),
        mentions=mentions,
    )
    return projected, flags


def project_corpus(
    jsonl_paths: list[Path],
    out_path: Path,
    translate: Callable[[str], str],
    target_lang: str,
    aligner: Aligner,
    min_conf: float = 0.5,
    flags_path: Path | None = None,
) -> dict[str, int]:
    docs, all_flags, n_repl, n_drop = [], [], 0, 0
    for path in jsonl_paths:
        for doc in read_jsonl(path):
            projected, flags = project_doc(
                doc, translate, target_lang, aligner, min_conf
            )
            docs.append(projected)
            all_flags.extend(flags)
            n_repl += len(projected.replicas)
            n_drop += sum(1 for f in flags if f["kind"] == "replica")
    write_jsonl(docs, out_path)
    if flags_path:
        flags_path.write_text(
            "\n".join(json.dumps(f, ensure_ascii=False) for f in all_flags),
            encoding="utf-8",
        )
    return {"docs": len(docs), "replicas": n_repl, "dropped": n_drop}


def hf_translator(model_name: str) -> Callable[[str], str]:
    from transformers import pipeline

    mt = pipeline("translation", model=model_name)
    return lambda text: mt(text, max_length=512)[0]["translation_text"]


class SimAligner:
    def __init__(self, model: str = "bert-base-multilingual-cased") -> None:
        from simalign import SentenceAligner

        self._a = SentenceAligner(model=model, matching_methods="i")

    def align(self, src: str, tgt: str) -> list[tuple[Span, Span, float]]:
        src_toks, src_spans = _whitespace_spans(src)
        tgt_toks, tgt_spans = _whitespace_spans(tgt)
        result = self._a.get_word_aligns(src_toks, tgt_toks)["itermax"]
        return [(src_spans[i], tgt_spans[j], 1.0) for i, j in result]


def _whitespace_spans(text: str) -> tuple[list[str], list[Span]]:
    import re

    toks, spans = [], []
    for m in re.finditer(r"\S+", text):
        toks.append(m.group())
        spans.append((m.start(), m.end()))
    return toks, spans
