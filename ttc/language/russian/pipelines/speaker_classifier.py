from collections import OrderedDict
from typing import Optional, Dict, Iterable, Generator

from spacy import Language
from spacy.matcher import DependencyMatcher
from spacy.symbols import VERB, nsubj, obj, obl  # type: ignore
from spacy.tokens import Token, Span

from ttc.iterables import iter_by_triples
from ttc.language import Dialogue
from ttc.language.russian.constants import REFERRAL_PRON
from ttc.language.russian.dependency_patterns import (
    SPEAKER_TO_SPEAKING_VERB,
    SPEAKER_CONJUNCT_SPEAKING_VERB,
)
from ttc.language.russian.span_extensions import (
    is_parenthesized,
    fills_line,
    expand_to_prev_line,
    expand_to_next_line,
)
from ttc.language.russian.token_extensions import (
    ref_match_any,
    is_speaker_noun,
    expand_to_noun_chunk,
)


def lowest_linked_verb(t: Token) -> Optional[Token]:
    p = [t.head]
    while p[0] and p[0] != p[0].head:
        p = [p[0].head] + p
    return next((t for t in p if t.pos == VERB), None)


def filter_by_verbs(subjects: Iterable[Span]) -> Generator[Span, None, None]:
    for noun in subjects:
        t = noun.root
        if is_speaker_noun(t) and (
            (verb := lowest_linked_verb(t)) and t in verb.children and t.dep in {nsubj}
        ):
            yield noun


def nearest_of_filtered_by_verbs(subjects: Iterable[Span]) -> Optional[Span]:
    return next(reversed(list(filter_by_verbs(subjects))), None)


def search_for_speaker(
    s_region: Span,
    relations: Dict[Span, Optional[Span]],
    dep_matcher: DependencyMatcher,
):
    for match_id, token_ids in dep_matcher(s_region):
        # For this matcher speaker is the first token in a pattern
        token: Token = s_region[token_ids[0]]
        # if token in replica:
        #     # We're not interested in "verb-noun"s inside of replicas yet
        #     continue
        if token.lemma_ in REFERRAL_PRON:
            ref = token
            if (
                len(
                    speakers := list(
                        OrderedDict.fromkeys(
                            [s.lemma_ for s in relations.values() if s][::-1]
                        )
                    )[::-1]
                )
                > 1
            ):
                for s in relations.values():
                    if s and s.lemma_ == speakers[-2] and ref_match_any(ref, s):
                        return s
                else:
                    dep = token
            else:
                dep = token
        else:
            dep = token

        return expand_to_noun_chunk(dep)

    if relations and (prev_speaker := relations[next(reversed(relations.keys()))]):
        for ref_t in [
            r[0] for r in s_region.noun_chunks if r[0].lemma_ in REFERRAL_PRON
        ]:
            if len(s_region.ents) == 1 and s_region.ents[0].end < ref_t.i:
                return s_region.ents[0]
            if ref_match_any(ref_t, prev_speaker):
                return prev_speaker

    if named_entity_span := nearest_of_filtered_by_verbs(s_region.ents):
        return named_entity_span

    if propn_span := nearest_of_filtered_by_verbs(
        expand_to_noun_chunk(t) for t in s_region if is_speaker_noun(t)
    ):
        return propn_span

    if noun_chunk_span := nearest_of_filtered_by_verbs(s_region.noun_chunks):
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
            # Line has no author speech => speakers alternation
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
                search_region, relations, dep_matcher
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
                search_region, relations, dep_matcher
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
                # it has no reference to the speaker => speakers alternation
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
                search_region, relations, dep_matcher
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
                # it has no reference to the speaker => speakers alternation
                # Assign replica to the penultimate speaker
                relations[replica] = next(
                    s for s in relations.values() if s and s.lemma_ == speakers[-2]
                )

        elif (i := dialogue.replicas.index(replica)) > 0 and (
            (pr := dialogue.replicas[i - 1]) and pr in relations
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
