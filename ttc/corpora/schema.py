"""Unified multilingual interchange schema: one JSON object per document.

All offsets are character offsets into ``text``. ``speaker``/``Mention.char``
values reference ``Character.id`` entries; ``speaker is None`` means the
replica has no identifiable speaker (narrator noise, crowd, etc.).
"""

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

QTYPES = ("explicit", "anaphoric", "implicit")
DOMAINS = ("prose", "drama")


@dataclass
class Cue:
    start: int
    end: int


@dataclass
class Replica:
    start: int
    end: int
    speaker: str | None
    addressee: str | None = None
    qtype: str | None = None
    cue: Cue | None = None
    mode: str | None = None
    """Utterance category — how it is voiced, not who says it.

    ``"speech"`` (spoken dialogue) or ``"thought"`` (internal monologue) so
    far; ``None`` when the source does not distinguish. A thought is a
    speaker-owned utterance a TTS engine still voices (possibly with a
    different engine/voice than speech), so it is kept, never dropped.
    """


@dataclass
class Character:
    id: str
    name: str
    aliases: list[str] = field(default_factory=list)
    gender: str | None = None


@dataclass
class Mention:
    start: int
    end: int
    char: str


@dataclass
class CorpusDoc:
    doc_id: str
    lang: str
    domain: str
    source: str
    license: str
    text: str
    replicas: list[Replica] = field(default_factory=list)
    characters: list[Character] = field(default_factory=list)
    mentions: list[Mention] = field(default_factory=list)


def to_dict(doc: CorpusDoc) -> dict:
    return asdict(doc)


def doc_from_dict(d: dict) -> CorpusDoc:
    return CorpusDoc(
        doc_id=d["doc_id"],
        lang=d["lang"],
        domain=d["domain"],
        source=d["source"],
        license=d["license"],
        text=d["text"],
        replicas=[
            Replica(
                start=r["start"],
                end=r["end"],
                speaker=r.get("speaker"),
                addressee=r.get("addressee"),
                qtype=r.get("qtype"),
                cue=Cue(**r["cue"]) if r.get("cue") else None,
                mode=r.get("mode"),
            )
            for r in d.get("replicas", [])
        ],
        characters=[
            Character(
                id=c["id"],
                name=c["name"],
                aliases=list(c.get("aliases", [])),
                gender=c.get("gender"),
            )
            for c in d.get("characters", [])
        ],
        mentions=[
            Mention(start=m["start"], end=m["end"], char=m["char"])
            for m in d.get("mentions", [])
        ],
    )


def write_jsonl(docs: Iterable[CorpusDoc], path: Path) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(to_dict(doc), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> Iterator[CorpusDoc]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield doc_from_dict(json.loads(line))


def validate(doc: CorpusDoc) -> list[str]:
    """Mechanical consistency checks. Returns [] when the doc is clean."""
    issues: list[str] = []
    n = len(doc.text)
    char_ids = [c.id for c in doc.characters]
    known = set(char_ids)
    if len(known) != len(char_ids):
        dups = sorted({c for c in char_ids if char_ids.count(c) > 1})
        issues.append(f"{doc.doc_id}: duplicate character ids {dups}")
    if doc.domain not in DOMAINS:
        issues.append(f"{doc.doc_id}: bad domain {doc.domain!r}")

    def check_span(what: str, start: int, end: int) -> bool:
        if not (0 <= start < end <= n):
            issues.append(f"{doc.doc_id}: {what} [{start}:{end}] out of bounds")
            return False
        return True

    prev_end = -1
    for i, r in enumerate(sorted(doc.replicas, key=lambda r: r.start)):
        check_span(f"replica {i}", r.start, r.end)
        if r.start < prev_end:
            issues.append(f"{doc.doc_id}: replica {i} overlaps previous")
        prev_end = max(prev_end, r.end)
        if r.speaker is not None and r.speaker not in known:
            issues.append(f"{doc.doc_id}: replica {i} unknown speaker {r.speaker!r}")
        if r.addressee is not None and r.addressee not in known:
            issues.append(
                f"{doc.doc_id}: replica {i} unknown addressee {r.addressee!r}"
            )
        if r.qtype is not None and r.qtype not in QTYPES:
            issues.append(f"{doc.doc_id}: replica {i} bad qtype {r.qtype!r}")
        if r.cue is not None:
            check_span(f"replica {i} cue", r.cue.start, r.cue.end)
    for i, m in enumerate(doc.mentions):
        check_span(f"mention {i}", m.start, m.end)
        if m.char not in known:
            issues.append(f"{doc.doc_id}: mention {i} unknown char {m.char!r}")
    return issues
