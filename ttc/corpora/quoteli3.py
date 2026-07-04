"""QuoteLi3 adapter: inline quote/mention XML (Muzny et al. 2017)."""

from pathlib import Path
from typing import Dict, Iterator, List, Tuple
from xml.etree import ElementTree

from ttc.corpora.schema import Character, CorpusDoc, Mention, Replica

GENDERS = {"male": "m", "female": "f"}


def parse_xml(xml_text: str, doc_id: str) -> CorpusDoc:
    root = ElementTree.fromstring(xml_text)

    characters: List[Character] = []
    by_name: Dict[str, str] = {}
    for ch in root.iter("character"):
        name = ch.get("name") or ""
        cid = f"char_{len(characters)}"
        aliases = [a for a in (ch.get("aliases") or "").split(";") if a and a != name]
        characters.append(
            Character(cid, name, aliases, GENDERS.get(ch.get("gender") or ""))
        )
        for key in [name, *aliases]:
            by_name.setdefault(key, cid)

    parts: List[str] = []
    pos = 0
    replicas: List[Replica] = []
    mentions: List[Mention] = []

    def emit(chunk: str) -> Tuple[int, int]:
        nonlocal pos
        start = pos
        parts.append(chunk)
        pos += len(chunk)
        return start, pos

    def walk(el: ElementTree.Element) -> None:
        if el.text:
            if el.tag == "quote":
                start, end = emit(el.text)
                replicas.append(
                    Replica(start, end, by_name.get(el.get("speaker") or ""))
                )
            elif el.tag == "mention":
                start, end = emit(el.text)
                if (cid := by_name.get(el.get("speaker") or "")) is not None:
                    mentions.append(Mention(start, end, cid))
            else:
                emit(el.text)
        for child in el:
            walk(child)
            if child.tail:
                emit(child.tail)

    text_el = root.find("text")
    if text_el is not None:
        walk(text_el)

    return CorpusDoc(
        doc_id=doc_id,
        lang="en",
        domain="prose",
        source="quoteli3",
        license="research release (Stanford)",
        text="".join(parts),
        replicas=replicas,
        characters=characters,
        mentions=mentions,
    )


def convert(path: Path) -> Iterator[CorpusDoc]:
    for f in sorted(path.glob("*.xml")):
        yield parse_xml(f.read_text(encoding="utf-8"), doc_id=f"quoteli3/{f.stem}")
