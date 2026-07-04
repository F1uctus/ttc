"""Character entities: person mentions clustered by rule or learned scorer."""

from dataclasses import dataclass, field

import numpy as np
from spacy.tokens import Doc, Span

from ttc.corpus import normalize_name
from ttc.language.common.token_extensions import noun_chunk
from ttc.ml import extensions
from ttc.ml.encoder import Encoder, pool
from ttc.ml.session import OnnxSession


@dataclass
class CharacterEntity:
    id: str
    mentions: list[Span] = field(default_factory=list)

    @property
    def name(self) -> str:
        return max(self.mentions, key=len).text

    def nearest_mention(self, replica: Span) -> Span:
        return min(
            self.mentions,
            key=lambda m: min(abs(m.start - replica.end), abs(replica.start - m.end)),
        )


def person_mentions(doc: Doc) -> list[Span]:
    seen = set()
    mentions: list[Span] = []
    for token in doc:
        is_person = token.pos_ == "PROPN" or token.ent_type_ == "PER"
        is_animate = "Animacy=Anim" in token.morph and token.pos_ == "NOUN"
        if not (is_person or is_animate):
            continue
        span = noun_chunk(token)
        key = (span.start, span.end)
        if key not in seen:
            seen.add(key)
            mentions.append(span)
    return mentions


def _key(span: Span) -> str:
    propn = " ".join(t.lemma_ for t in span if t.pos_ == "PROPN")
    return normalize_name(propn or span.root.lemma_)


def resolve_rule(doc: Doc) -> list[CharacterEntity]:
    extensions.register()
    by_key = {}
    for mention in person_mentions(doc):
        by_key.setdefault(_key(mention), []).append(mention)
    entities = [
        CharacterEntity(f"ent_{i}", mentions)
        for i, (_, mentions) in enumerate(sorted(by_key.items()))
    ]
    doc._.characters = entities
    return entities


def resolve_learned(
    doc: Doc,
    encoder: Encoder,
    pair_session: OnnxSession,
    threshold: float = 0.5,
) -> list[CharacterEntity]:
    extensions.register()
    mentions = person_mentions(doc)
    if not mentions:
        doc._.characters = []
        return []
    emb = encoder.encode_doc(doc)
    vecs = np.stack([pool(emb, m.start, m.end) for m in mentions])
    cluster_of = list(range(len(mentions)))  # greedy: link to best earlier mention
    for j in range(1, len(mentions)):
        pairs = np.concatenate(
            [np.repeat(vecs[j][None], j, axis=0), vecs[:j], vecs[:j] * vecs[j]],
            axis=1,
        ).astype(np.float32)
        (scores,) = pair_session.run({"x": pairs})
        best = int(np.argmax(scores[:, 0]))
        if 1 / (1 + np.exp(-scores[best, 0])) >= threshold:
            cluster_of[j] = cluster_of[best]
    grouped = {}
    for mi, ci in enumerate(cluster_of):
        grouped.setdefault(ci, []).append(mentions[mi])
    entities = [
        CharacterEntity(f"ent_{i}", ms) for i, ms in enumerate(grouped.values())
    ]
    doc._.characters = entities
    return entities
