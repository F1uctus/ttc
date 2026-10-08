"""Listwise actor ranking over speaker candidates and play assembly."""

import numpy as np
from spacy.tokens import Doc, Span

from ttc.language import Dialogue, Play
from ttc.ml import extensions
from ttc.ml.candidates import features, generate
from ttc.ml.encoder import Encoder
from ttc.ml.entities import CharacterEntity, resolve_learned, resolve_rule
from ttc.ml.session import OnnxSession


def rank(
    doc: Doc,
    replica: Span,
    encoder: Encoder,
    scorer: OnnxSession,
    threshold: float = 0.5,
) -> tuple[CharacterEntity | None, float]:
    cands = replica._.speaker_candidates or []
    if not cands:
        return None, 0.0
    emb = encoder.encode_doc(doc)
    x = np.stack([features(emb, replica, e) for e in cands])
    (scores,) = scorer.run({"x": x})
    z = scores[:, 0] - scores[:, 0].max()
    probs = np.exp(z) / np.exp(z).sum()
    best = int(np.argmax(probs))
    if probs[best] < threshold:
        return None, float(probs[best])
    return cands[best], float(probs[best])


def connect_play_learned(cc, dialogue: Dialogue, mode: str) -> Play:
    """Learned chain; mode "hybrid" backfills unknowns from the rule twin."""
    extensions.register()
    doc = dialogue.doc
    package = cc.package
    inference = package.meta.get("inference", {})
    threshold = float(inference.get("unknown_threshold", 0.5))
    window_chars = int(inference.get("window_chars", 1200))
    top_k = int(inference.get("top_k", 8))

    encoder = Encoder(package)
    pair = OnnxSession(package.graph_path("pair"))
    cand_scorer = OnnxSession(package.graph_path("candidate"))
    rank_scorer = OnnxSession(package.graph_path("ranker"))

    entities = resolve_learned(doc, encoder, pair)
    if not entities:
        entities = resolve_rule(doc)
    doc._.replicas = list(dialogue.replicas)

    rule_play = None
    if mode == "hybrid":
        from ttc.language.russian.pipelines.actor_classifier import classify_actors

        rule_play = classify_actors(cc.language, dialogue)

    play = Play(cc.language)
    for replica in dialogue.replicas:
        generate(
            doc,
            replica,
            entities,
            "union",
            encoder,
            cand_scorer,
            window_chars,
            top_k,
        )
        entity, _prob = rank(doc, replica, encoder, rank_scorer, threshold)
        if entity is not None:
            play[replica] = entity.nearest_mention(replica)
        elif rule_play is not None and replica in rule_play:
            play[replica] = rule_play[replica]
        else:
            play[replica] = None
    return play
