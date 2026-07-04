"""JY-QuotePlus adapter: per-quote context JSON (zh, prose).

Real-release format (verified 2026-07-04): one JSON list of 8,144 items,
each carrying the quote, a context window, and labels
说话人-mention/说话人-entity (speaker), 听者-entity (addressees),
线索 (cue text), 方式 (mode). There is no global chapter text, so every
item becomes its own CorpusDoc with ``text = context``.
"""

import json
import warnings
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from ttc.corpora.schema import Character, CorpusDoc, Cue, Mention, Replica


def _item_doc(item: dict, doc_id: str) -> Optional[CorpusDoc]:
    text: str = item.get("context") or ""
    quote: str = item.get("quote") or ""
    labels: Dict = item.get("labels") or {}
    start = text.find(quote)
    if not text or not quote or start < 0:
        warnings.warn(f"{doc_id}: quote not found in context: {quote[:30]!r}")
        return None
    end = start + len(quote)

    by_name: Dict[str, str] = {}
    characters: List[Character] = []

    def char_id(name: Optional[str]) -> Optional[str]:
        if not name:
            return None
        if name not in by_name:
            by_name[name] = f"char_{len(by_name)}"
            characters.append(Character(by_name[name], name))
        return by_name[name]

    speaker = char_id(labels.get("说话人-entity"))
    addressees = labels.get("听者-entity") or []
    addressee = char_id(addressees[0]) if addressees else None

    cue = None
    if cue_text := labels.get("线索"):
        cue_start = text.rfind(cue_text, max(0, start - 30), start)
        if cue_start < 0:
            cue_start = text.find(cue_text, end, end + 30)
        if cue_start >= 0:
            cue = Cue(cue_start, cue_start + len(cue_text))

    mentions: List[Mention] = []
    if (m_text := labels.get("说话人-mention")) and speaker:
        m_start = text.rfind(m_text, 0, start)
        if m_start < 0:
            m_start = text.find(m_text, end)
        if m_start >= 0:
            mentions.append(Mention(m_start, m_start + len(m_text), speaker))

    return CorpusDoc(
        doc_id=doc_id,
        lang="zh",
        domain="prose",
        source="jy_quoteplus",
        license="research release (GitHub LimboChen/JY-QuotePlus)",
        text=text,
        replicas=[Replica(start, end, speaker, addressee, None, cue)],
        characters=characters,
        mentions=mentions,
    )


def convert(path: Path) -> Iterator[CorpusDoc]:
    files = [path] if path.is_file() else sorted(path.glob("*.json"))
    for f in files:
        items = json.loads(f.read_text(encoding="utf-8"))
        for i, item in enumerate(items):
            doc = _item_doc(item, doc_id=f"jy_quoteplus/{f.stem}/{i:05d}")
            if doc is not None:
                yield doc
