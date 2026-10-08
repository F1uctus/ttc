"""Speaker candidates per replica: rule, learned or union."""

import numpy as np
from spacy.tokens import Doc, Span

from ttc.ml import extensions
from ttc.ml.encoder import Encoder, pool
from ttc.ml.entities import CharacterEntity
from ttc.ml.session import OnnxSession


def _in_window(entity: CharacterEntity, replica: Span, window_chars: int) -> bool:
    return any(
        abs(m.start_char - replica.end_char) <= window_chars
        or abs(replica.start_char - m.end_char) <= window_chars
        for m in entity.mentions
    )


def _distance(entity: CharacterEntity, replica: Span) -> int:
    m = entity.nearest_mention(replica)
    return min(
        abs(m.start_char - replica.end_char), abs(replica.start_char - m.end_char)
    )


def generate_rule(
    doc: Doc,
    replica: Span,
    entities: list[CharacterEntity],
    window_chars: int = 1200,
    top_k: int = 8,
) -> list[CharacterEntity]:
    scoped = [e for e in entities if _in_window(e, replica, window_chars)]
    scoped.sort(key=lambda e: _distance(e, replica))
    # cue-detected speakers go first
    cue_ranges = [(a.start, a.end) for _, a in (doc._.ttc_cues or []) if a is not None]

    def cued(e: CharacterEntity) -> bool:
        return any(
            m.start < ce and m.end > cs for m in e.mentions for cs, ce in cue_ranges
        )

    scoped.sort(key=lambda e: not cued(e))  # stable: keeps distance order
    return scoped[:top_k]


def features(emb: np.ndarray, replica: Span, entity: CharacterEntity) -> np.ndarray:
    m = entity.nearest_mention(replica)
    dist = np.float32(np.log1p(_distance(entity, replica)) / 10.0)
    same_line = np.float32(m.sent == replica.sent)
    return np.concatenate(
        [
            pool(emb, replica.start, replica.end),
            pool(emb, m.start, m.end),
            [dist, same_line],
        ]
    ).astype(np.float32)


def generate_learned(
    doc: Doc,
    replica: Span,
    entities: list[CharacterEntity],
    encoder: Encoder,
    scorer: OnnxSession,
    window_chars: int = 1200,
    top_k: int = 8,
) -> list[CharacterEntity]:
    scoped = [e for e in entities if _in_window(e, replica, window_chars)]
    if not scoped:
        return []
    emb = encoder.encode_doc(doc)
    x = np.stack([features(emb, replica, e) for e in scoped])
    (scores,) = scorer.run({"x": x})
    order = np.argsort(-scores[:, 0])
    return [scoped[i] for i in order[:top_k]]


def generate(
    doc: Doc,
    replica: Span,
    entities: list[CharacterEntity],
    mode: str = "union",
    encoder: Encoder | None = None,
    scorer: OnnxSession | None = None,
    window_chars: int = 1200,
    top_k: int = 8,
) -> list[CharacterEntity]:
    extensions.register()
    rule = (
        generate_rule(doc, replica, entities, window_chars, top_k)
        if mode in ("rule", "union")
        else []
    )
    learned = (
        generate_learned(doc, replica, entities, encoder, scorer, window_chars, top_k)
        if mode in ("learned", "union") and encoder is not None and scorer is not None
        else []
    )
    seen, out = set(), []
    for e in [*rule, *learned]:
        if e.id not in seen:
            seen.add(e.id)
            out.append(e)
    out = out[: top_k if mode != "union" else max(top_k, len(rule))]
    replica._.speaker_candidates = out
    return out
