"""Corpus adapters converting public corpora into the interchange schema."""

from typing import Callable, Dict, Iterator, TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path
    from ttc.corpora.schema import CorpusDoc

ADAPTERS: Dict[str, str] = {
    # source name -> module path; modules expose convert(path) -> Iterator[CorpusDoc]
    "native": "ttc.corpora.native",
    "rusdracor": "ttc.corpora.rusdracor",
    "pdnc": "ttc.corpora.pdnc",
    "riqua": "ttc.corpora.riqua",
    "quoteli3": "ttc.corpora.quoteli3",
    "jy_quoteplus": "ttc.corpora.jy_quoteplus",
    "droc": "ttc.corpora.droc",
}


def get_adapter(name: str) -> "Callable[[Path], Iterator[CorpusDoc]]":
    import importlib

    if name not in ADAPTERS:
        raise KeyError(
            f"Unknown corpus source {name!r}; known: {sorted(ADAPTERS)}"
        )
    return importlib.import_module(ADAPTERS[name]).convert
