"""PDNC adapter: per-novel quotation/character CSVs -> interchange docs.

https://github.com/Priya22/project-dialogism-novel-corpus — annotations are
CC-BY-NC-4.0 (research/eval OK; flag before any commercial model release).

Real-release schema (verified 2026-07-04): ``quotation_info.csv`` columns
quoteID, quoteText, subQuotationList, quoteByteSpans, speaker, addressees,
quoteType, referringExpression (text, may be "nan"), mentionTextsList,
mentionSpansList, mentionEntitiesList (both nested per sub-quotation).
All shipped novel texts are pure ASCII, so byte offsets == char offsets.
"""

import ast
import csv
import warnings
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from ttc.corpora.schema import Character, CorpusDoc, Cue, Mention, Replica

QTYPE_MAP = {"explicit": "explicit", "anaphoric": "anaphoric", "implicit": "implicit"}
GENDERS = {"F": "f", "M": "m"}


def _lit(cell: Optional[str], default):
    """ast.literal_eval with 'nan'/empty/garbage tolerance."""
    if cell is None or cell == "" or cell == "nan":
        return default
    try:
        return ast.literal_eval(cell)
    except (ValueError, SyntaxError):
        return default


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _find_near(text: str, needle: str, around: int, radius: int = 300) -> Optional[Cue]:
    lo = max(0, around - radius)
    pos = text.find(needle, lo, around + radius)
    return Cue(pos, pos + len(needle)) if pos >= 0 else None


def parse_novel(novel_dir: Path) -> CorpusDoc:
    text_path = next(
        p
        for name in ("novel_text.txt", "text.txt")
        if (p := novel_dir / name).exists()
    )
    text = text_path.read_text(encoding="utf-8")

    characters: List[Character] = []
    by_name: Dict[str, str] = {}  # any alias/name -> char id
    for row in _read_csv(novel_dir / "character_info.csv"):
        cid = f"char_{row['Character ID']}"
        name = row["Main Name"] or cid
        parsed = _lit(row.get("Aliases"), [])
        aliases = sorted(parsed) if isinstance(parsed, set) else list(parsed)
        aliases = [a for a in aliases if a != name]
        characters.append(
            Character(
                cid,
                name,
                aliases=aliases,
                gender=GENDERS.get((row.get("Gender") or "").upper()),
            )
        )
        for key in [name, *aliases]:
            by_name.setdefault(key, cid)

    replicas: List[Replica] = []
    mentions: List[Mention] = []
    for row in _read_csv(novel_dir / "quotation_info.csv"):
        speaker = by_name.get(row.get("speaker") or "")
        qtype = QTYPE_MAP.get((row.get("quoteType") or "").strip().lower())
        addressees = _lit(row.get("addressees"), [])
        addressee = by_name.get(addressees[0]) if addressees else None
        spans = [tuple(map(int, s)) for s in _lit(row.get("quoteByteSpans"), [])]
        if not spans:
            continue
        cue = None
        if ref_exp := (row.get("referringExpression") or ""):
            if ref_exp != "nan":
                cue = _find_near(text, ref_exp, spans[0][0])
        for start, end in spans:
            replicas.append(Replica(start, end, speaker, addressee, qtype, cue))
            cue = None  # attach the cue to the first span only
        for sub_spans, sub_ents in zip(
            _lit(row.get("mentionSpansList"), []),
            _lit(row.get("mentionEntitiesList"), []),
        ):
            for span, ent_names in zip(sub_spans, sub_ents):
                for ent_name in ent_names:
                    if cid := by_name.get(ent_name):
                        mentions.append(Mention(int(span[0]), int(span[1]), cid))
                        break  # one Mention per span; first resolvable entity

    replicas.sort(key=lambda r: r.start)
    return CorpusDoc(
        doc_id=f"pdnc/{novel_dir.name}",
        lang="en",
        domain="prose",
        source="pdnc",
        license="CC-BY-NC-4.0",
        text=text,
        replicas=replicas,
        characters=characters,
        mentions=mentions,
    )


def convert(path: Path) -> Iterator[CorpusDoc]:
    novel_dirs = (
        [path]
        if (path / "quotation_info.csv").exists()
        else sorted(p for p in path.iterdir() if (p / "quotation_info.csv").exists())
    )
    if not novel_dirs:
        warnings.warn(f"{path}: no PDNC novel folders found")
    for novel_dir in novel_dirs:
        yield parse_novel(novel_dir)
