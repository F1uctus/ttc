"""RU silver corpus from public-domain prose in RafaelUI/russian_literature."""

import re
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

from train.silver import silver_label
from ttc.corpora.schema import CorpusDoc, validate, write_jsonl

DATASET = "RafaelUI/russian_literature"
DASH = ("\N{EM DASH}", "\N{EN DASH}")


def load_ru_prose(
    authors: Iterable[str] | None = None,
) -> Iterator[tuple[str, str]]:
    """Yield (doc_id, text) for prose works, one author per pass if round_robin."""
    from datasets import load_dataset

    keep = {a.lower() for a in authors} if authors else None
    ds = load_dataset(DATASET, split="train")
    for i, row in enumerate(ds):
        if row.get("type") != "prose":
            continue
        author = (row.get("author") or "unknown").lower()
        if keep and author not in keep:
            continue
        text = row.get("text") or ""
        if text.strip():
            yield f"{author}/{i}", text


def _dialogue_lines(chunk: str) -> int:
    return sum(1 for ln in chunk.splitlines() if ln.lstrip()[:1] in DASH)


def dialogue_chunks(
    text: str, target_chars: int = 2000, min_dialogue_lines: int = 2
) -> Iterator[str]:
    """Dialogue chunks of about target_chars, at most max_chars, never mid-line."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    buf: list[str] = []
    size = 0
    for para in paras:
        buf.append(para)
        size += len(para)
        if size >= target_chars:
            chunk = "\n\n".join(buf)
            if _dialogue_lines(chunk) >= min_dialogue_lines:
                yield chunk
            buf, size = [], 0
    if buf:
        chunk = "\n\n".join(buf)
        if _dialogue_lines(chunk) >= min_dialogue_lines:
            yield chunk


def build_silver_corpus(
    sources: Iterable[tuple[str, str]],
    out_path: Path,
    llm: Callable[[str], list[dict]],
    cc,
    max_docs: int = 50,
    target_chars: int = 2000,
    min_attributed: int = 2,
    chunks_per_work: int = 3,
    max_consecutive_failures: int = 6,
    on_doc: Callable[[CorpusDoc], None] | None = None,
) -> dict[str, int]:
    """Silver-label dialogue chunks into interchange JSONL and return run stats."""
    docs: list[CorpusDoc] = []
    issues = 0
    failures = 0
    consecutive = 0
    for doc_id, text in sources:
        taken = 0
        for ci, chunk in enumerate(
            dialogue_chunks(_normalize_newlines(text), target_chars)
        ):
            if taken >= chunks_per_work or len(docs) >= max_docs:
                break
            try:
                doc = next(
                    silver_label(
                        [chunk], llm, cc, doc_id_prefix=f"silver/{doc_id}/{ci}"
                    )
                )
            except Exception:  # noqa: BLE001  # provider or network failure
                failures += 1
                consecutive += 1
                if consecutive >= max_consecutive_failures:
                    break
                continue
            consecutive = 0
            attributed = sum(1 for r in doc.replicas if r.speaker)
            if attributed < min_attributed:
                continue
            issues += len(validate(doc))
            docs.append(doc)
            taken += 1
            if on_doc:
                on_doc(doc)
        if len(docs) >= max_docs or consecutive >= max_consecutive_failures:
            break
    n = write_jsonl(docs, out_path)
    return {
        "docs": n,
        "replicas": sum(len(d.replicas) for d in docs),
        "attributed": sum(sum(1 for r in d.replicas if r.speaker) for d in docs),
        "issues": issues,
        "failures": failures,
    }


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def main() -> None:
    import argparse

    import ttc
    from train.silver import agent_cli_llm, openrouter_llm

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--backend", choices=["openrouter", "agent"], default="openrouter")
    ap.add_argument("--model", default=None)
    ap.add_argument("--authors", nargs="*", default=None)
    ap.add_argument("--max-docs", type=int, default=50)
    ap.add_argument("--target-chars", type=int, default=2000)
    ap.add_argument("--chunks-per-work", type=int, default=3)
    args = ap.parse_args()

    cc = ttc.load("ru", pipeline="rules")
    if args.backend == "openrouter":
        llm = openrouter_llm(cc, model=args.model) if args.model else openrouter_llm(cc)
    else:
        llm = agent_cli_llm(cc, model=args.model or "composer-2.5")

    def progress(doc: CorpusDoc) -> None:
        a = sum(1 for r in doc.replicas if r.speaker)
        print(f"  + {doc.doc_id}: {a}/{len(doc.replicas)} attributed", flush=True)

    stats = build_silver_corpus(
        load_ru_prose(args.authors),
        args.out,
        llm,
        cc,
        max_docs=args.max_docs,
        target_chars=args.target_chars,
        chunks_per_work=args.chunks_per_work,
        on_doc=progress,
    )
    print(f"silver corpus -> {args.out}: {stats}")


if __name__ == "__main__":
    main()
