"""Parsing and serialization of the annotated-text corpus format.

A corpus file consists of two or three sections separated by a line of dashes:

    <raw text>
    --------------------
    <Actor>::<Replica text>          (one line per replica, in document order)
    --------------------             (optional third section)
    <Canonical> = <alias> | <alias>  (one line per character; '#' starts a comment)

The alias section lets gold annotations use several surface forms for the
same character (e.g. "Ясна = принцесса | светлость") while being scored
as one identity.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DELIMITER = "-" * 20

UNATTRIBUTED = "none"
"""Canonical actor value for a replica with no identifiable speaker.
Matches both a gold ``None::`` annotation and a ``None`` prediction."""


@dataclass
class CorpusFile:
    text: str
    """Raw input text (the part fed to the pipeline)."""

    pairs: List[Tuple[str, str]] = field(default_factory=list)
    """Ordered (actor, replica text) gold annotations."""

    aliases: Dict[str, str] = field(default_factory=dict)
    """Normalized alias -> normalized canonical actor name."""

    path: Optional[Path] = None


def normalize_name(name: str) -> str:
    return " ".join(name.split()).lower().replace("ё", "е")


def canonical_actor(name: Optional[str], aliases: Dict[str, str]) -> str:
    if name is None:
        return UNATTRIBUTED
    n = normalize_name(name)
    return aliases.get(n, n)


def parse_corpus_content(content: str, path: Optional[Path] = None) -> CorpusFile:
    sections = content.split(DELIMITER)
    if len(sections) > 3:
        raise ValueError(f"{path or 'corpus content'}: too many section delimiters")

    text = sections[0].strip()
    pairs: List[Tuple[str, str]] = []
    aliases: Dict[str, str] = {}

    if len(sections) > 1:
        for line in sections[1].strip().split("\n"):
            if not (line := line.strip()):
                continue
            if "::" not in line:
                raise ValueError(f"{path or 'corpus content'}: malformed pair {line!r}")
            actor, replica = line.split("::", 1)
            pairs.append((actor.strip(), replica.strip()))

    if len(sections) > 2:
        for line in sections[2].strip().split("\n"):
            if not (line := line.strip()) or line.startswith("#"):
                continue
            if "=" not in line:
                raise ValueError(f"{path or 'corpus content'}: malformed alias {line!r}")
            canonical, alts = line.split("=", 1)
            canonical = normalize_name(canonical)
            for alt in alts.split("|"):
                if alt := normalize_name(alt):
                    aliases[alt] = canonical

    return CorpusFile(text=text, pairs=pairs, aliases=aliases, path=path)


def load_corpus_file(path: Path) -> CorpusFile:
    return parse_corpus_content(path.read_text(encoding="utf-8"), path)


def serialize_corpus_file(
    text: str,
    pairs: List[Tuple[str, str]],
    aliases: Optional[Dict[str, List[str]]] = None,
) -> str:
    """Inverse of :func:`parse_corpus_content`.

    ``aliases`` maps a canonical name to its alias list (the readable
    one-line-per-character form, not the flat lookup dict).
    """
    parts = [text.strip(), DELIMITER]
    parts += [f"{actor}::{replica}" for actor, replica in pairs]
    if aliases:
        parts.append(DELIMITER)
        parts += [f"{canonical} = {' | '.join(alts)}" for canonical, alts in aliases.items()]
    return "\n".join(parts) + "\n"


def find_corpus_files(root: Path, recursive: bool = True) -> List[Path]:
    pattern = "**/*.txt" if recursive else "*.txt"
    return sorted(p for p in root.glob(pattern) if "raw" not in p.parts)
