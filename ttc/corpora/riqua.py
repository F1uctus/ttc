"""RiQuA adapter: brat-style standoff quotes/cues/entities (en, prose).

RiQuA has no canonical character table — speaker entities are text spans.
Characters are synthesized per distinct entity surface string; mention
spans are emitted for every entity occurrence linked to a quote.
"""

from pathlib import Path
from typing import Dict, Iterator, List, Tuple

from ttc.corpora.schema import Character, CorpusDoc, Cue, Mention, Replica


def _parse_ann(ann_text: str):
    spans: Dict[str, Tuple[str, int, int]] = {}
    relations: List[Tuple[str, str, str]] = []
    for line in ann_text.splitlines():
        if not (line := line.strip()):
            continue
        tag, body = line.split("\t", 1)[0], line.split("\t")[1]
        if tag.startswith("T"):
            kind, start, end = body.split()[:3]
            spans[tag] = (kind, int(start), int(end))
        elif tag.startswith("R"):
            kind, arg1, arg2 = body.split()
            relations.append((kind, arg1.split(":", 1)[1], arg2.split(":", 1)[1]))
    return spans, relations


def parse_work(txt: Path, ann: Path) -> CorpusDoc:
    text = txt.read_text(encoding="utf-8")
    spans, relations = _parse_ann(ann.read_text(encoding="utf-8"))

    char_by_name: Dict[str, str] = {}
    characters: List[Character] = []
    mentions: List[Mention] = []

    def char_for(entity_id: str) -> str:
        kind, start, end = spans[entity_id]
        name = text[start:end]
        if name not in char_by_name:
            char_by_name[name] = f"char_{len(char_by_name)}"
            characters.append(Character(char_by_name[name], name))
        mentions.append(Mention(start, end, char_by_name[name]))
        return char_by_name[name]

    speaker_of: Dict[str, str] = {}
    addressee_of: Dict[str, str] = {}
    cue_of: Dict[str, Cue] = {}
    for kind, quote_id, arg_id in relations:
        if kind.lower() in ("speaker", "speakerof"):
            speaker_of[quote_id] = char_for(arg_id)
        elif kind.lower() in ("addressee", "addresseeof"):
            addressee_of[quote_id] = char_for(arg_id)
        elif kind.lower() in ("cue", "cueof"):
            _, start, end = spans[arg_id]
            cue_of[quote_id] = Cue(start, end)

    replicas = [
        Replica(
            start,
            end,
            speaker_of.get(tid),
            addressee_of.get(tid),
            None,
            cue_of.get(tid),
        )
        for tid, (kind, start, end) in sorted(spans.items(), key=lambda kv: kv[1][1])
        if kind == "Quote"
    ]
    return CorpusDoc(
        doc_id=f"riqua/{txt.stem}",
        lang="en",
        domain="prose",
        source="riqua",
        license="research release (see RiQuA distribution terms)",
        text=text,
        replicas=replicas,
        characters=characters,
        mentions=mentions,
    )


def convert(path: Path) -> Iterator[CorpusDoc]:
    for txt in sorted(path.glob("*.txt")):
        ann = txt.with_suffix(".ann")
        if ann.exists():
            yield parse_work(txt, ann)
