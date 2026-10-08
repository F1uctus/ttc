"""Attribution cues: cue verbs and speaker NP spans."""

import numpy as np
from spacy.matcher import DependencyMatcher
from spacy.tokens import Doc, Span

from ttc.language.common.token_extensions import noun_chunk
from ttc.ml import extensions
from ttc.ml.encoder import Encoder
from ttc.ml.session import OnnxSession

# BIO label ids of the learned head
O, B_CUE, I_CUE, B_SPK, I_SPK = range(5)


def detect_rule(doc: Doc, patterns: dict[str, list]) -> list[tuple[Span, Span | None]]:
    extensions.register()
    matcher = DependencyMatcher(doc.vocab)
    for name, pattern in patterns.items():
        matcher.add(name, [pattern])
    cues: list[tuple[Span, Span | None]] = []
    seen = set()
    for _, token_ids in matcher(doc):
        # patterns list the actor first and the verb last
        actor_t, verb_t = doc[token_ids[0]], doc[token_ids[-1]]
        if verb_t.i in seen:
            continue
        seen.add(verb_t.i)
        cues.append((doc[verb_t.i : verb_t.i + 1], noun_chunk(actor_t)))
    cues.sort(key=lambda c: c[0].start)
    doc._.ttc_cues = cues
    return cues


def _bio_spans(tags: np.ndarray, b: int, i: int) -> list[tuple[int, int]]:
    spans, start = [], None
    for idx, tag in enumerate(tags):
        if tag == b:
            if start is not None:
                spans.append((start, idx))
            start = idx
        elif tag != i and start is not None:
            spans.append((start, idx))
            start = None
    if start is not None:
        spans.append((start, len(tags)))
    return spans


def detect_learned(
    doc: Doc, encoder: Encoder, cue_session: OnnxSession
) -> list[tuple[Span, Span | None]]:
    extensions.register()
    emb = encoder.encode_doc(doc)
    (logits,) = cue_session.run({"emb": emb[None].astype(np.float32)})
    tags = logits[0].argmax(axis=-1)
    cue_spans = _bio_spans(tags, B_CUE, I_CUE)
    spk_spans = _bio_spans(tags, B_SPK, I_SPK)
    cues: list[tuple[Span, Span | None]] = []
    for cs, ce in cue_spans:
        verb = doc[cs:ce]
        same_sent = [(ss, se) for ss, se in spk_spans if doc[ss].sent == verb.sent]
        actor = None
        if same_sent:
            ss, se = min(same_sent, key=lambda s: abs(s[0] - cs))
            actor = doc[ss:se]
        cues.append((verb, actor))
    doc._.ttc_cues = cues
    return cues
