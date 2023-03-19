import functools
import math
from collections import deque
from typing import Collection, Optional, List, Dict

import numpy as np
from spacy import Language
from spacy.matcher import DependencyMatcher
from spacy.symbols import PROPN, nsubj, obj, obl  # type: ignore
from spacy.tokens import Token, Span

from ttc.iterables import iter_by_triples, canonical_int_enumeration, flatmap
from ttc.language import Dialogue
from ttc.language.russian.constants import REFERRAL_PRON
from ttc.language.russian.dependency_patterns import (
    SPEAKER_TO_SPEAKING_VERB,
    SPEAKER_CONJUNCT_SPEAKING_VERB,
)
from ttc.language.russian.span_extensions import (
    is_parenthesized,
    trim_non_word,
    is_inside,
    fills_line,
    sent_resembles_replica,
)
from ttc.language.russian.token_extensions import (
    speaker_dep_order,
    ref_match,
    linked_verb,
    expand_to_matching_noun_chunk,
    as_span,
)


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
        key=lambda chunk: min(
            (i for tt in chunk if (i := speaker_dep_order(tt)) != -1), default=0
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
        if t.lemma_ in queue and linked_verb(t):
            return expand_to_matching_noun_chunk(t)

    return None


def from_named_ent_with_verb(sent: Span) -> Optional[Span]:
    """
    Select a speaker from named entities if it is mentioned in the given sentence.
    """
    for ent in sent.ents:
        for t in ent:  # TODO check only the root token of a chunk
            if (verb := linked_verb(t)) and not (
                True  # "Case=Dat" in t.morph
                and {"Number=Sing", "Tense=Past", "Voice=Act"}.issubset(verb.morph)
            ):
                return ent

    return None


def from_noun_chunks_with_verb(sent: Span) -> Optional[Span]:
    """
    Select the first speaker with verb from noun chunks of given sentence.
    """
    for noun_chunk in sent.noun_chunks:
        for t in noun_chunk:
            if (verb := linked_verb(t)) and not (
                "Case=Dat" in t.morph
                and {"Number=Sing", "Tense=Past", "Voice=Act"}.issubset(verb.morph)
            ):
                return noun_chunk
    # TODO: If > 1, prefer the nominative noun chunk
    return None


def from_exact_propn_with_verb(sent: Span) -> Optional[Span]:
    propns: List[Span] = [
        expand_to_matching_noun_chunk(t) for t in sent if t.pos == PROPN
    ]
    for propn in propns:
        for t in propn:  # TODO check only the root token of a chunk
            if (verb := linked_verb(t)) and not (
                True  # "Case=Dat" in t.morph
                and {"Number=Sing", "Tense=Past", "Voice=Act"}.issubset(verb.morph)
            ):
                return propn
            else:
                break

    return None


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
        nearest_l_context = trim_non_word(doc[0 : reference.i])
        if len(nearest_l_context) > 1:
            search_context.append(nearest_l_context)

    ref_matches = functools.partial(ref_match, reference)

    # add context pieces between some replicas
    # laying before the referral pronoun
    for l, r in zip(replicas_to_search_between, replicas_to_search_between[1:]):
        between_reps = trim_non_word(doc[l[-1].i + 1 : r[0].i])
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
    nearest_r_context = trim_non_word(
        doc[replica_preceding_reference.end : reference.i]
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
    speaker_queue: Dict[str, Span] = {}

    def mark_speaker(*, span: Optional[Span] = None, queue_i: Optional[int] = None):
        if queue_i:
            try:
                speaker = list(speaker_queue.values())[queue_i]
            except IndexError:
                return
        elif span:
            speaker = span
        else:
            return
        lemma = speaker.lemma_
        if lemma in speaker_queue:
            # Move repeated speaker to the end of queue
            speaker_queue[lemma] = relations[replica] = speaker_queue.pop(lemma)
        else:
            speaker_queue[lemma] = relations[replica] = speaker

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
                # the speaker this way, will try alteration
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

            if prev_replica and (fills_line(replica) or is_parenthesized(sent)):
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

                if len(speaker_queue) > 1:
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
                if token in replica:
                    # We're not interested in "verb-noun"s inside of replicas yet
                    continue
                if prev_sent and token.lemma_ in REFERRAL_PRON:
                    ref = token
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
                            dep_span = as_span(token)
                else:
                    dep_span = expand_to_matching_noun_chunk(token)

                mark_speaker(span=dep_span)
                break  # dependency matching

            if replica in relations:
                break  # speaker found

            repeated_span = from_queue_with_verb(sent, speaker_queue.keys())
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
