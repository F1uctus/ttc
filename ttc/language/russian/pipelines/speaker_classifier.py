import math
from typing import Collection, Optional, List, Dict, Callable

from spacy import Language
from spacy.matcher import DependencyMatcher
from spacy.symbols import PROPN  # type: ignore
from spacy.tokens import Token, Span

from ttc.iterables import iter_by_triples, canonical_int_enumeration
from ttc.language import Dialogue
from ttc.language.russian.constants import REFERRAL_PRON
from ttc.language.russian.token_extensions import (
    morph_equals,
    token_as_span,
    has_linked_verb,
    contains_near,
    expand_to_matching_noun_chunk,
)

SPEAKER_TO_SPEAKING_VERB = [
    {  # (anchor) speaker
        "RIGHT_ID": "speaker",
        "RIGHT_ATTRS": {
            "DEP": "nsubj",
            "POS": {"NOT_IN": ["SPACE", "PUNCT"]},
        },
    },
    {  # speaker <--- speaking verb
        "LEFT_ID": "speaker",
        "REL_OP": "<",
        "RIGHT_ID": "speaking_verb",
        "RIGHT_ATTRS": {
            "POS": "VERB",
            "_": {"is_speaking_verb": True},
        },
    },
]

SPEAKER_CONJUNCT_SPEAKING_VERB = [
    {  # (anchor) speaker
        "RIGHT_ID": "speaker",
        "RIGHT_ATTRS": {
            "DEP": "nsubj",
            "POS": {"NOT_IN": ["SPACE", "PUNCT"]},
        },
    },
    {  # speaker <--- conjunct
        "LEFT_ID": "speaker",
        "REL_OP": "<",
        "RIGHT_ID": "conjunct",
        "RIGHT_ATTRS": {},
    },
    {  # conjunct ---> speaking verb
        "LEFT_ID": "conjunct",
        "REL_OP": ">>",  # TODO: Replace by search in Token.conjuncts?
        "RIGHT_ID": "speaking_verb",
        "RIGHT_ATTRS": {
            "POS": "VERB",
            "_": {"is_speaking_verb": True},
        },
    },
]


def replica_fills_line(replica: Span) -> bool:
    doc = replica.doc
    return (
        replica.start - 3 >= 0
        and replica.end + 3 < len(doc)  # TODO: Check end-of-doc case
        and any(t._.has_newline for t in doc[replica.start - 3 : replica.start])
        # colon means that the author still annotates the replica, just on previous line
        and not any(t.text == ":" for t in doc[replica.start - 3 : replica.start])
        and any(t._.has_newline for t in doc[replica.end : replica.end + 3])
    )


def find_by_reference(spans: List[Span], reference: Token, misses=0) -> Optional[Span]:
    """
    Find a noun chunk that might be a definition of
    the given reference token inside the given span.

    Parameters:
        spans: A list of sentences or other doc-like spans to search definition in.
        reference: A reference to some speaker (e.g. он, она, оно, ...)
    """
    for nc in spans[-1].noun_chunks:
        for t in nc:
            if morph_equals(t, reference, "Gender", "Number"):
                if nc.lemma_ in REFERRAL_PRON:
                    return find_by_reference(spans[:-1], t, misses)
                return nc
    else:
        if len(spans) > 1 and misses < 4:
            return find_by_reference(spans[:-1], reference, misses + 1)
    return None


def from_queue_with_verb(sent: Span, queue: Collection[str]) -> Optional[Span]:
    """Select a speaker from queue if it is mentioned in the given sentence."""
    if len(queue) == 0:
        return None

    for t in sent:
        if t.lemma_ in queue and has_linked_verb(t):
            return expand_to_matching_noun_chunk(t)

    return None


def from_named_ent_with_verb(sent: Span) -> Optional[Span]:
    """Select a speaker from named entities if it is mentioned in the given sentence."""
    for ent in sent.ents:
        for t in ent:
            if has_linked_verb(t):
                return ent

    return None


def from_noun_chunks_with_verb(sent: Span) -> Optional[Span]:
    """Select the first speaker with verb from noun chunks of given sentence."""
    for noun_chunk in sent.noun_chunks:
        for t in noun_chunk:
            if has_linked_verb(t):
                return noun_chunk
    # TODO: If > 1, prefer the nominative noun chunk
    return None


def from_exact_propn_with_verb(sent: Span) -> Optional[Span]:
    propns: List[Token] = [t for t in sent if t.pos == PROPN]
    for t in propns:
        if has_linked_verb(t):
            return token_as_span(t)
        break

    return None


def trim_span(span: Span, should_trim: Callable[[Token], bool]):
    doc = span.doc
    start = span.start
    while start < span.end and should_trim(doc[start]):
        start += 1
    end = span.end - 1
    while end > span.start and should_trim(doc[end]):
        end -= 1
    return doc[start:end]


def non_overlapping_span_len(outer: Span, inner: Span) -> int:
    outer = trim_span(outer, lambda t: t.is_punct or t._.has_newline)
    inner = trim_span(inner, lambda t: t.is_punct or t._.has_newline)
    if outer.start >= inner.start and outer.end <= inner.end:
        return 0
    return abs(outer.start - inner.start) + abs(outer.end - inner.end)


def is_parenthesized(span: Span):
    return contains_near(span[0], 3, lambda t: t.text == "(") and contains_near(
        span[-1], 3, lambda t: t.text == ")"
    )


def classify_speakers(
    language: Language,
    dialogue: Dialogue,
) -> Dict[Span, Optional[Span]]:
    relations: Dict[Span, Optional[Span]] = {}

    if len(dialogue.replicas) == 0:
        return relations

    doc = dialogue.doc
    sents = list(doc.sents)

    dep_matcher = DependencyMatcher(language.vocab)
    dep_matcher.add("*", [SPEAKER_TO_SPEAKING_VERB])
    dep_matcher.add("**", [SPEAKER_CONJUNCT_SPEAKING_VERB])

    # Handles speakers alteration & continuation
    speakers_queue: Dict[str, Span] = {}

    def mark_speaker(*, span: Optional[Span] = None, queue_i: Optional[int] = None):
        if queue_i:
            try:
                speaker = list(speakers_queue.values())[queue_i]
            except IndexError:
                return
        elif span:
            speaker = span
        else:
            return
        lemma = speaker.lemma_
        if lemma in speakers_queue:
            # Move repeated speaker to the end of queue
            speakers_queue[lemma] = relations[replica] = speakers_queue.pop(lemma)
        else:
            speakers_queue[lemma] = relations[replica] = speaker

    for prev_replica, replica, next_replica in iter_by_triples(dialogue.replicas):
        # "p_" - previous; "n_" - next; "r_" - replica
        r_start_sent_i = sents.index(replica[0].sent)
        r_start_sent = sents[r_start_sent_i]
        r_end_sent_i = sents.index(replica[-1].sent)
        p_r_sent_i = sents.index(prev_replica[-1].sent) if prev_replica else None
        n_r_sent_i = sents.index(next_replica[0].sent) if next_replica else None

        skip_sign_change = False
        back_first = replica.start - 3 >= 0 and any(
            t.text == ":" for t in doc[replica.start - 3 : replica.start]
        )  # Replica is after colon - means back-first speaker search

        back_done = forw_done = False

        for prev_offset, offset, _ in iter_by_triples(canonical_int_enumeration()):
            # Iterate over sentences near the current replica.
            # They have "up-down" indices
            # (e.g i, i + 1, i - 1, i + 2, i - 2, ... if forward-first);
            # (and i, i - 1, i + 1, i - 2, i + 2, ... if backward-first).
            if skip_sign_change and prev_offset:
                offset = int(math.copysign(offset, prev_offset))
                skip_sign_change = False

            sent_i = r_start_sent_i - offset if back_first else r_end_sent_i + offset

            # Got up to the previous replica - stop going back
            back_done = back_done or (p_r_sent_i is not None and sent_i < p_r_sent_i)

            # Got up to the next replica - stop going forward
            forw_done = forw_done or (n_r_sent_i is not None and sent_i > n_r_sent_i)

            if back_done and forw_done:
                # We've got up to the sentences containing
                # the previous AND next replicas -> cannot find
                # the speaker this way, use alternative method.
                break

            if back_done and offset < 0:
                continue

            if forw_done and offset > 0:
                continue

            sent = sents[sent_i]
            prev_sent = sents[sent_i - 1] if sent_i > 0 else None
            next_sent = sents[sent_i + 1] if sent_i < len(sents) - 1 else None

            if prev_replica and prev_replica._.end_line_no == replica._.start_line_no:
                # Replica is on the same line - probably separated by author speech
                # Assign it to the previous speaker
                mark_speaker(queue_i=-1)
                break  # speaker found

            if prev_replica and (replica_fills_line(replica) or is_parenthesized(sent)):
                # If the author's sentence is enclosed in parentheses,
                # there is unlikely a speaker definition,
                # but some description of the situation

                if 1 < abs(replica._.start_line_no - prev_replica._.end_line_no) < 4:
                    # If there were 1-2 lines filled with author speech,
                    # it is more probable a continuation than alteration,
                    # because otherwise a reader will likely lose context
                    # if this replica is not annotated by speaker name
                    # and follows after many lines of author text
                    mark_speaker(queue_i=-1)
                    break  # speaker found, skip to the next replica

                if len(speakers_queue) > 1:
                    # Line has no author speech -> speakers alteration
                    # Assign replica to the penultimate speaker
                    mark_speaker(queue_i=-2)
                    break  # speaker found, skip to the next replica

            sent_resembles_replica = non_overlapping_span_len(sent, replica) <= 3

            if sent_resembles_replica:
                skip_sign_change = True
                continue

            for match_id, token_ids in dep_matcher(sent):
                # For this matcher speaker is the first token in a pattern
                token: Token = sent[token_ids[0]]
                if prev_sent and token.lemma_ in REFERRAL_PRON:
                    dep_span = find_by_reference(sents[:sent_i], token)
                    if dep_span:
                        dep_span = expand_to_matching_noun_chunk(dep_span[0])
                else:
                    dep_span = expand_to_matching_noun_chunk(token)

                mark_speaker(span=dep_span if dep_span else token_as_span(token))
                break  # dependency matching

            if replica in relations:
                break  # speaker found

            repeated_span = from_queue_with_verb(sent, speakers_queue.keys())
            if repeated_span:
                mark_speaker(span=repeated_span)
                break  # speaker found

            named_entity_span = from_named_ent_with_verb(sent)
            if named_entity_span:
                mark_speaker(span=named_entity_span)
                break  # speaker found

            propn_span = from_exact_propn_with_verb(sent)
            if propn_span:
                mark_speaker(span=propn_span)
                continue  # speaker can be overridden

            noun_chunk_span = from_noun_chunks_with_verb(sent)
            if noun_chunk_span:
                mark_speaker(span=noun_chunk_span)
                continue  # speaker can be overridden

        # end of sentences traversal

    # end of replicas traversal

    return relations
