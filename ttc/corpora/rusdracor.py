"""RusDraCor adapter: TEI-P5 plays -> interchange docs (ru, drama).

Text layout: for every <sp>, the speaker label line (if present) is kept,
followed by the utterance paragraphs; the replica span covers the spoken
text only, and the label becomes a Mention of the speaking character.
Cast metadata (annotations) is CC0; play texts are mostly public domain.
"""

import json
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from xml.etree import ElementTree

from ttc.corpora.schema import Character, CorpusDoc, Mention, Replica

TEI = "{http://www.tei-c.org/ns/1.0}"
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"
API = "https://dracor.org/api/v1/corpora/rus"
GENDERS = {"MALE": "m", "FEMALE": "f"}


def _text_of(el: ElementTree.Element) -> str:
    return " ".join("".join(el.itertext()).split())


def parse_tei(xml_text: str, doc_id: str) -> CorpusDoc:
    root = ElementTree.fromstring(xml_text)
    characters: list[Character] = []
    for person in root.iter(f"{TEI}person"):
        pid = person.get(XML_ID)
        name_el = person.find(f"{TEI}persName")
        if pid and name_el is not None:
            characters.append(
                Character(
                    pid,
                    _text_of(name_el),
                    gender=GENDERS.get(person.get("sex") or ""),
                )
            )
    known = {c.id for c in characters}

    parts: list[str] = []
    replicas: list[Replica] = []
    mentions: list[Mention] = []
    pos = 0

    def append(chunk: str) -> tuple[int, int]:
        nonlocal pos
        start = pos
        parts.append(chunk + "\n")
        pos += len(chunk) + 1
        return start, start + len(chunk)

    for sp in root.iter(f"{TEI}sp"):
        who: str | None = (sp.get("who") or "").lstrip("#") or None
        speaker_el = sp.find(f"{TEI}speaker")
        if speaker_el is not None and (label := _text_of(speaker_el)):
            m_start, m_end = append(label)
            if who in known:
                mentions.append(Mention(m_start, m_end, who))
        utterance = " ".join(
            t for child in sp if child.tag != f"{TEI}speaker" and (t := _text_of(child))
        )
        if utterance:
            r_start, r_end = append(utterance)
            replicas.append(Replica(r_start, r_end, who if who in known else None))

    return CorpusDoc(
        doc_id=doc_id,
        lang="ru",
        domain="drama",
        source="rusdracor",
        license="CC0-1.0 (annotations); texts mostly public domain",
        text="".join(parts),
        replicas=replicas,
        characters=characters,
        mentions=mentions,
    )


def convert(path: Path) -> Iterator[CorpusDoc]:
    for f in sorted(path.glob("*.xml")):
        yield parse_tei(f.read_text(encoding="utf-8"), doc_id=f"rusdracor/{f.stem}")


def download(out_dir: Path) -> None:
    """Fetch all RusDraCor TEI files via the DraCor API (network!)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(API) as resp:
        plays = json.load(resp)["plays"]
    for play in plays:
        name = play["name"]
        target = out_dir / f"{name}.xml"
        if target.exists():
            continue
        with urllib.request.urlopen(f"{API}/plays/{name}/tei") as resp:
            data = resp.read()
        if data[:2] == b"\x1f\x8b":  # server gzips regardless of Accept-Encoding
            import gzip

            data = gzip.decompress(data)
        target.write_bytes(data)
        print(f"fetched {name}")
