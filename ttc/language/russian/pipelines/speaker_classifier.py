import functools
import math
from typing import Collection, Optional, List, Dict, Callable

from spacy import Language
from spacy.matcher import DependencyMatcher
from spacy.symbols import PROPN, nsubj, obj, obl  # type: ignore
from spacy.tokens import Token, Span

from ttc.iterables import iter_by_triples, canonical_int_enumeration, flatmap
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


# from least to most important
SPEAKER_DEP_ORDER = [obl, obj, nsubj]


def speaker_dep_order(t: Token) -> int:
    try:
        return SPEAKER_DEP_ORDER.index(t.dep)
    except ValueError:
        return -1


def ref_match(ref: Token, target: Token) -> bool:
    return morph_equals(target, ref, "Gender", "Number")


def find_by_reference(spans: List[Span], reference: Token, _misses=0):
    """
    Find a noun chunk that might be a definition of the given reference token
    inside the given span.

    Parameters
    ----------
    spans
        A list of sentences or other doc-like spans to search definition in.
    reference
        A reference to some speaker (e.g. он, она, оно, ...).
    _misses
        [DO NOT SET]: Specifies a number of sentences already checked
        for matching noun chunks. Used to limit search to some meaningful area.
    """
    if len(spans) == 0:
        return
    # Order noun chunks from most to least probably denoting the actor.
    ncs = sorted(
        spans[-1].noun_chunks,
        key=lambda nc: min(
            (i for i in map(speaker_dep_order, nc) if i != -1), default=0
        ),
        reverse=True,
    )
    for nc in ncs:
        for t in nc:
            if ref_match(t, reference):
                if nc.lemma_ in REFERRAL_PRON:
                    yield from find_by_reference(spans[:-1], t, _misses)
                yield nc
    else:
        if len(spans) > 1 and _misses < 4:
            yield from find_by_reference(spans[:-1], reference, _misses + 1)


def from_queue_with_verb(sent: Span, queue: Collection[str]) -> Optional[Span]:
    """
    Select a speaker from queue if it is mentioned in the given sentence.
    """
    if len(queue) == 0:
        return None

    for t in sent:
        if t.lemma_ in queue and has_linked_verb(t):
            return expand_to_matching_noun_chunk(t)

    return None


def from_named_ent_with_verb(sent: Span) -> Optional[Span]:
    """
    Select a speaker from named entities if it is mentioned in the given sentence.
    """
    for ent in sent.ents:
        for t in ent:
            if has_linked_verb(t):
                return ent

    return None


def from_noun_chunks_with_verb(sent: Span) -> Optional[Span]:
    """
    Select the first speaker with verb from noun chunks of given sentence.
    """
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


def trim(span: Span, should_trim: Callable[[Token], bool]):
    doc = span.doc
    start = span.start
    while start < span.end and should_trim(doc[start]):
        start += 1
    end = span.end - 1
    while end > span.start and should_trim(doc[end]):
        end -= 1
    return doc[start : end + 1]


def is_inside(inner: Span, outer: Span):
    return inner.start >= outer.start and inner.end <= outer.end


def non_overlapping_span_len(outer: Span, inner: Span) -> int:
    outer, inner = trim(outer, non_word), trim(inner, non_word)
    if is_inside(outer, inner):
        return 0  # outer must be a strict superset of inner
    return abs(outer.start - inner.start) + abs(outer.end - inner.end)


def sent_resembles_replica(sent: Span, replica: Span):
    return non_overlapping_span_len(sent, replica) <= 3


def is_parenthesized(span: Span):
    return contains_near(span[0], 3, lambda t: t.text == "(") and contains_near(
        span[-1], 3, lambda t: t.text == ")"
    )


def non_word(t: Token) -> bool:
    return t.is_punct or t._.has_newline


def reference_search_context(
    *,
    relations: Dict[Span, Optional[Span]],
    replica_preceding_reference: Span,
    reference: Token,
    exclude_previous_speakers: bool = True,
):
    doc = reference.doc
    replicas = list(relations.keys()) + [replica_preceding_reference]
    replicas_to_search_between = replicas[max(0, len(replicas) - 5) :]
    search_context = []

    # add context piece between doc start and referral pronoun indices
    if len(relations) == 0 and replica_preceding_reference.start > reference.i:
        nearest_l_context = trim(doc[0 : reference.i], non_word)
        if len(nearest_l_context) > 1:
            search_context.append(nearest_l_context)

    ref_matches = functools.partial(ref_match, reference)

    # add context pieces between some replicas
    # laying before the referral pronoun
    for l, r in zip(replicas_to_search_between, replicas_to_search_between[1:]):
        between_reps = trim(doc[l[-1].i + 1 : r[0].i], non_word)
        # if that span contains an already annotated matching speaker
        if (
            exclude_previous_speakers
            and (
                previous_spk := next(
                    (s for s in relations.values() if s and is_inside(s, between_reps)),
                    None,
                )
            )
            and previous_spk
            and any(map(ref_matches, previous_spk))
        ):
            # exclude the speaker and all matching nouns
            # of its subtree from the search context
            ll = doc[between_reps.start : previous_spk.start]
            rr = doc[previous_spk.end : between_reps.end + 1]
            matching_nouns_on_same_lvl: List[Token] = sorted(
                (
                    t
                    for t in flatmap(lambda tt: tt.head.children, previous_spk)
                    if ref_matches(t)
                ),
                key=lambda t: t.i,
            )
            if matching_nouns_on_same_lvl:
                ll = doc[between_reps.start : matching_nouns_on_same_lvl[0].i]
                rr = doc[matching_nouns_on_same_lvl[-1].i + 1 : between_reps.end]
            if len(ll) > 1:
                search_context.append(ll)
            if len(rr) > 1:
                search_context.append(rr)
        else:
            if len(between_reps) > 1:
                search_context.append(between_reps)

    # add context piece between the last replica to the left
    # of the referral pronoun and the pronoun itself
    nearest_r_context = trim(
        doc[replica_preceding_reference.end : reference.i], non_word
    )
    if len(nearest_r_context) > 1:
        search_context.append(nearest_r_context)

    return search_context


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

            if sent_resembles_replica(sent, replica):
                skip_sign_change = True
                continue

            for match_id, token_ids in dep_matcher(sent):
                # For this matcher speaker is the first token in a pattern
                token: Token = sent[token_ids[0]]
                if prev_sent and token.lemma_ in REFERRAL_PRON and (ref := token):
                    # First, exclude previously annotated speakers from search
                    ref_search_context = reference_search_context(
                        relations=relations,
                        replica_preceding_reference=replica,
                        reference=ref,
                    )
                    try:
                        dep_span = next(find_by_reference(ref_search_context, ref))
                    except StopIteration:
                        # If no ref targets were found, include
                        # previously annotated speakers in search
                        ref_search_context = reference_search_context(
                            relations=relations,
                            replica_preceding_reference=replica,
                            reference=ref,
                            exclude_previous_speakers=False,
                        )
                        try:
                            dep_span = next(find_by_reference(ref_search_context, ref))
                        except StopIteration:
                            dep_span = token_as_span(token)
                else:
                    dep_span = expand_to_matching_noun_chunk(token)

                mark_speaker(span=dep_span)
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
