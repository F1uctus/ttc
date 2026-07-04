"""Custom extensions shared by the learned components."""

from spacy.tokens import Doc, Span

EXTENSIONS = {
    Doc: {"ttc_emb": None, "replicas": [], "characters": [], "ttc_cues": []},
    Span: {"speaker_candidates": None},
}


def register() -> None:
    for cls, names in EXTENSIONS.items():
        for name, default in names.items():
            if not cls.has_extension(name):
                cls.set_extension(name, default=default)
