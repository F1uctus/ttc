from collections import OrderedDict
from typing import Collection, Optional, List, Dict

from spacy import Language
from spacy.matcher import DependencyMatcher
from spacy.symbols import PROPN, nsubj, obj, obl  # type: ignore
from spacy.tokens import Token, Span

from ttc.iterables import iter_by_triples, flatmap
from ttc.language import Dialogue
from ttc.language.russian.constants import REFERRAL_PRON
from ttc.language.russian.dependency_patterns import (
    SPEAKER_TO_SPEAKING_VERB,
    SPEAKER_CONJUNCT_SPEAKING_VERB,
)
from ttc.language.russian.span_extensions import (
    trim_non_word,
    is_inside,
    is_parenthesized,
    fills_line,
    expand_to_prev_line,
    expand_to_next_line,
)
from ttc.language.russian.token_extensions import (
    speaker_dep_order,
    ref_match,
    ref_match_any,
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


def from_named_ent_with_verb(sent: Span) -> Optional[Span]:
    """
    Select a speaker from named entities if it is mentioned in the given sentence.
    """
    for ent in sent.ents:
        for t in ent:  # TODO check only the root token of a chunk
            if (verb := linked_verb(t)) and not (
                "Case=Dat" in t.morph
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
                True  # "Case=Dat" in t.morph
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

    # add context pieces between some replicas
    # laying before the referral pronoun
    for l, r in zip(replicas_to_search_between, replicas_to_search_between[1:]):
        between_reps = trim_non_word(doc[l[-1].i + 1 : r[0].i])
        # if that span contains an already annotated matching speaker
        previous_spk: Optional[Span] = next(
            (s for s in relations.values() if s and is_inside(s, between_reps)),
            None,
        )
        if (
            exclude_previous_speakers
            and previous_spk
            and ref_match_any(reference, previous_spk)
        ):
            # exclude the speaker and all matching nouns
            # of its subtree from the search context
            ll = doc[between_reps.start : previous_spk.start]
            rr = doc[previous_spk.end : between_reps.end + 1]
            matching_nouns_on_same_lvl: List[Token] = sorted(
                (
                    t
                    for t in flatmap(lambda tt: tt.head.children, previous_spk)
                    if ref_match(reference, t)
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


def search_for_speaker(
    search_region: Span,
    replica: Span,
    relations: Dict[Span, Optional[Span]],
    dep_matcher: DependencyMatcher,
):
    for match_id, token_ids in dep_matcher(search_region):
        # For this matcher speaker is the first token in a pattern
        token: Token = search_region[token_ids[0]]
        if token in replica:
            # We're not interested in "verb-noun"s inside of replicas yet
            continue
        if token.lemma_ in REFERRAL_PRON:
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

        return dep_span

    if relations and (prev_speaker := relations[next(reversed(relations.keys()))]):
        for ref_t in [
            r[0] for r in search_region.noun_chunks if r[0].lemma_ in REFERRAL_PRON
        ]:
            if len(search_region.ents) == 1 and search_region.ents[0].end < ref_t.i:
                return search_region.ents[0]
            if ref_match_any(ref_t, prev_speaker):
                return prev_speaker

    if named_entity_span := from_named_ent_with_verb(search_region):
        return named_entity_span

    if propn_span := from_exact_propn_with_verb(search_region):
        return propn_span

    if noun_chunk_span := from_noun_chunks_with_verb(search_region):
        return noun_chunk_span

    return None


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

    for prev_replica, replica, next_replica in iter_by_triples(dialogue.replicas):
        if prev_replica and prev_replica._.end_line_no == replica._.start_line_no:
            # Replica is on the same line - probably separated by author speech
            # Assign it to the previous speaker
            relations[replica] = relations[prev_replica]

        elif (
            prev_replica
            and replica._.start_line_no - prev_replica._.end_line_no == 1
            and fills_line(replica)
            and len(
                speakers := list(
                    OrderedDict.fromkeys(
                        [s.lemma_ for s in relations.values() if s][::-1]
                    )
                )[::-1]
            )
            > 1
        ):
            # Line has no author speech => speakers alteration
            # Assign replica to the penultimate speaker
            relations[replica] = next(
                s for s in relations.values() if s and s.lemma_ == speakers[-2]
            )

        elif replica._.is_after_author_starting:
            leading = doc[min(replica.start, replica.sent.start) : replica.start]
            if len(leading) == 0:
                if leading.start > 0:
                    leading = doc[leading.start - 1 : leading.end]
                else:
                    breakpoint()
            if (
                len(leading) < 2 < leading.start
                and doc[leading.start - 1]._.has_newline
            ):
                # Check if colon was on a previous line
                # That indicates that speaker definition may be on that line.
                leading = doc[leading.start - 2 : leading.end]
            search_region = expand_to_prev_line(leading)
            relations[replica] = search_for_speaker(
                search_region, replica, relations, dep_matcher
            )

        elif replica._.is_before_author_ending:
            trailing = doc[replica.end : max(replica.end, replica.sent.end)]
            if len(trailing) == 0:
                if trailing.end + 1 < len(doc):
                    trailing = doc[trailing.start : trailing.end + 1]
                else:
                    breakpoint()
            search_region = expand_to_next_line(trailing)
            relations[replica] = search_for_speaker(
                search_region, replica, relations, dep_matcher
            )
            # TODO: Extract common alternation logic
            if not relations[replica] and (
                prev_replica
                and replica._.start_line_no - prev_replica._.end_line_no == 1
                and len(
                    speakers := list(
                        OrderedDict.fromkeys(
                            [s.lemma_ for s in relations.values() if s][::-1]
                        )
                    )[::-1]
                )
                > 1
            ):
                # Author speech is present, but
                # it has no reference to the speaker => speakers alteration
                # Assign replica to the penultimate speaker
                relations[replica] = next(
                    s for s in relations.values() if s and s.lemma_ == speakers[-2]
                )

        elif replica._.is_before_author_insertion and next_replica:
            search_region = doc[replica.end : next_replica.start]
            if is_parenthesized(search_region):
                # author is commenting on a situation, there should be
                # no reference to the speaker.
                search_region = doc[replica.end : replica.end]
            relations[replica] = search_for_speaker(
                search_region, replica, relations, dep_matcher
            )
            # TODO: Extract common alternation logic
            if not relations[replica] and (
                prev_replica
                and replica._.start_line_no - prev_replica._.end_line_no == 1
                and len(
                    speakers := list(
                        OrderedDict.fromkeys(
                            # TODO: Do not erase None-s from the speaker queue
                            [s.lemma_ for s in relations.values() if s][::-1]
                        )
                    )[::-1]
                )
                > 1
            ):
                # Author speech is present, but
                # it has no reference to the speaker => speakers alteration
                # Assign replica to the penultimate speaker
                relations[replica] = next(
                    s for s in relations.values() if s and s.lemma_ == speakers[-2]
                )

        elif (
            (i := dialogue.replicas.index(replica)) > 1
            and (pr := dialogue.replicas[i - 1])
            and pr in relations
        ):
            # Assign to the previous speaker
            relations[replica] = relations[pr]

        else:
            print("MISS", replica)  # TODO: handle

    # end of replicas traversal

    prev_replica = None
    for replica, speaker in relations.items():
        if speaker and speaker.lemma_ in REFERRAL_PRON and prev_replica:
            relations[replica] = relations[prev_replica]
        prev_replica = replica

    return relations
