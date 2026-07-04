"""DROC adapter: UIMA CAS XMI -> interchange docs (de, prose).

Real-release format (Würzburg DROC-Release, verified 2026-07-04): each
fragment is an XMI file with a ``cas:Sofa`` holding the text and standoff
annotations. ``type:NamedEntity`` mentions carry ``Name`` and a coref
cluster ``ID``; ``type:DirectSpeech`` spans carry ``Speaker``/``SpokenTo``
that reference a NamedEntity by its ``xmi:id`` (→ cluster → character).
Every speaker-owned utterance becomes a replica carrying a ``mode``:
spoken dialogue is ``"speech"`` and internal monologue is ``"thought"`` —
both are voiced by TTS (a thought is owned by its speaker), so neither is
dropped. Bare ``name`` mentions are the only skipped category. 90
canonical fragments live under ``droc/DROC-xmi/``.
"""

from pathlib import Path
from typing import Dict, Iterator, List, Optional
from xml.etree import ElementTree

CAS = "{http:///uima/cas.ecore}"
TYPE = "{http:///de/uniwue/kalimachos/coref/type.ecore}"
XMI_ID = "{http://www.omg.org/XMI}id"

# DROC Category -> interchange Replica.mode. Owned utterances only; a bare
# "name" mention is not an utterance and is skipped.
CATEGORY_MODE = {
    "directspeech": "speech",
    "fictionalspeech": "speech",
    "fictional speech": "speech",
    "citation": "speech",
    "thought": "thought",
    "other": "speech",
}

from ttc.corpora.schema import Character, CorpusDoc, Mention, Replica


def parse_xmi(xml_text: str, doc_id: str) -> CorpusDoc:
    root = ElementTree.fromstring(xml_text)

    sofa = root.find(f"{CAS}Sofa")
    text = sofa.get("sofaString") if sofa is not None else ""
    text = text or ""

    # NamedEntity: xmi:id -> (begin, end, name, cluster_id)
    ne_by_xmi: Dict[str, tuple] = {}
    cluster_names: Dict[str, List[str]] = {}
    mentions: List[Mention] = []
    for ne in root.iter(f"{TYPE}NamedEntity"):
        xmi_id = ne.get(XMI_ID)
        cluster = ne.get("ID")
        if xmi_id is None or cluster is None:
            continue
        begin, end = int(ne.get("begin")), int(ne.get("end"))
        name = ne.get("Name") or ""
        cid = f"char_{cluster}"
        ne_by_xmi[xmi_id] = (begin, end, name, cid)
        cluster_names.setdefault(cid, []).append(name)
        mentions.append(Mention(begin, end, cid))

    def representative(names: List[str]) -> str:
        # longest non-pronominal surface form is the readable canonical name
        real = [n for n in names if n and n.lower() not in ("er", "sie", "es")]
        return max(real or names or [""], key=len)

    characters = [
        Character(cid, representative(names))
        for cid, names in cluster_names.items()
    ]

    def speaker_char(ref: Optional[str]) -> Optional[str]:
        return ne_by_xmi[ref][3] if ref and ref in ne_by_xmi else None

    replicas: List[Replica] = []
    for ds in root.iter(f"{TYPE}DirectSpeech"):
        category = (ds.get("Category") or "directspeech").lower()
        mode = CATEGORY_MODE.get(category)
        if mode is None:  # e.g. "name" — not a spoken/thought utterance
            continue
        begin, end = int(ds.get("begin")), int(ds.get("end"))
        replicas.append(
            Replica(
                begin,
                end,
                speaker_char(ds.get("Speaker")),
                speaker_char(ds.get("SpokenTo")),
                mode=mode,
            )
        )

    replicas.sort(key=lambda r: r.start)
    mentions.sort(key=lambda m: m.start)
    return CorpusDoc(
        doc_id=doc_id,
        lang="de",
        domain="prose",
        source="droc",
        license="research release (Würzburg DROC)",
        text=text,
        replicas=replicas,
        characters=characters,
        mentions=mentions,
    )


def convert(path: Path) -> Iterator[CorpusDoc]:
    files = [path] if path.is_file() else sorted(path.glob("*.xmi"))
    for f in files:
        yield parse_xmi(f.read_text(encoding="utf-8"), doc_id=f"droc/{f.stem}")
