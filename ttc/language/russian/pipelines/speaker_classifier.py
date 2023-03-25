from collections import OrderedDict
from typing import Optional, Dict, List, Iterable, Generator, Callable, Union

from spacy import Language
from spacy.matcher import DependencyMatcher
from spacy.symbols import VERB, nsubj  # type: ignore
from spacy.tokens import Token, Span

from ttc.iterables import iter_by_triples, flatmap
from ttc.language import Dialogue
from ttc.language.russian.dependency_patterns import (
    SPEAKING_VERB_TO_SPEAKER,
    SPEAKING_VERB_CONJUNCT_SPEAKER,
)
from ttc.language.russian.span_extensions import (
    is_parenthesized,
    fills_line,
    expand_to_prev_line,
    expand_to_next_line,
    trim_non_word,
    is_inside,
    is_referential,
)
from ttc.language.russian.token_extensions import (
    ref_matches,
    is_speaker_noun,
    expand_to_noun_chunk,
    as_span,
    lowest_linked_verbal,
    dep_dist_up_to,
)


def where_with_verbs(subjects: Iterable[Span]) -> List[Span]:
    verb_subjects = []
    for noun in subjects:
        if is_speaker_noun(t := noun.root) and (
            (verbal := lowest_linked_verbal(t))
            and (t.dep_ in {"ROOT", "nsubj", "nmod"})
        ):
            verb_subjects.append((dep_dist_up_to(noun.root, verbal.root), noun))
    verb_subjects.sort(key=lambda dist_and_subj: dist_and_subj[0])
    return [vs for dist, vs in verb_subjects]


def nearest_with_verbs(subjects: Iterable[Span]) -> Optional[Span]:
    # TODO: Rank subjects by frequency in rels
    subjects = where_with_verbs(subjects)
    for subj in subjects:
        # If subject is in the sentence immediately preceding a replica with a colon,
        # Then it is more likely to be the speaker than subjects from earlier sentences.
        if subj.sent.text.endswith(":") and subj.sent[-1]._.has_newline:
            return subj
    return subjects[0] if subjects else None


def unique_speakers(relations: Dict[Span, Optional[Span]]) -> List[str]:
    return list(
        OrderedDict.fromkeys([s.lemma_ if s else "" for s in relations.values()][::-1])
    )[::-1]


def ref_search_ctx(
    *,
    rels: Dict[Span, Optional[Span]],
    last_replica: Optional[Span] = None,
    ref: Union[Token, Span],
):
    if isinstance(ref, Span):
        ref = ref.root
    doc = ref.doc
    reps = list(rels.keys())
    if last_replica:
        reps.append(last_replica)
    reps_to_search_between = reps[max(0, len(reps) - 4) :]
    search_ctx = []

    # add context piece between doc start and referral pronoun indices
    if (
        len(rels) == 0
        and last_replica
        and last_replica.start > ref.i
        and len(nearest_l_ctx := trim_non_word(doc[: ref.i])) > 1
    ):
        search_ctx.append(nearest_l_ctx)

    # add context pieces between some replicas
    # laying before the referral pronoun
    for l, r in zip(reps_to_search_between, reps_to_search_between[1:]):
        between_reps = trim_non_word(doc[l.end : r.start])
        if len(between_reps) > 1:
            search_ctx.append(between_reps)

    # add context piece between the last replica to the left
    # of the referral pronoun and the pronoun itself
    if (
        last_replica
        and len(nearest_r_ctx := trim_non_word(doc[last_replica.end : ref.i])) > 1
    ):
        search_ctx.append(nearest_r_ctx)

    return search_ctx[::-1]


def search_for_speaker(
    s_region: Span,
    rels: Dict[Span, Optional[Span]],
    dep_matcher: DependencyMatcher,
    *,
    replica: Optional[Span] = None,
    speaker_pred: Optional[Callable[[Span], bool]] = None,
):
    select: Callable[[Iterable[Span]], Iterable[Span]] = lambda sps: filter(
        speaker_pred or (lambda sp: True), sps
    )
    for match_id, token_ids in dep_matcher(s_region):
        # For this matcher speaker is the first token in a pattern
        t = s_region[token_ids[0]]
        if is_referential(as_span(t)):
            speakers = unique_speakers(rels)
            if replica:
                spans = ref_search_ctx(rels=rels, last_replica=replica, ref=t)
                for span in spans:
                    if s := search_for_speaker(
                        span,
                        rels,
                        dep_matcher,
                        speaker_pred=lambda sp: (
                            ref_matches(t, sp)
                            and (not speakers or sp.lemma_ != speakers[-1])
                        ),
                    ):
                        return s
            if len(speakers) > 1 and (
                s := next(
                    (
                        s
                        for s in rels.values()
                        if s and s.lemma_ == speakers[-2] and ref_matches(t, s)
                    ),
                    None,
                )
            ):
                return s
            if len(rels) == 2 and len(speakers) == 1:
                return rels[list(rels.keys())[-1]]

        if any(select([as_span(t)])):
            return expand_to_noun_chunk(t)

    if rels and (prev_speaker := rels[next(reversed(rels.keys()))]):
        for chunk in s_region.noun_chunks:
            if not is_referential(ref := chunk):
                continue
            if len(s_region.ents) == 1 and (
                (ent := s_region.ents[0]).end < ref.start
                and ref_matches(ref, ent)
                and (not speaker_pred or speaker_pred(ent))
            ):
                return s_region.ents[0]
            if ref_matches(ref, prev_speaker):
                return prev_speaker

            preceding_rels = {r: s for r, s in rels.items() if r.end <= ref.start}
            spans = ref_search_ctx(rels=preceding_rels, ref=ref)
            for span in spans:
                if s := search_for_speaker(
                    span,
                    preceding_rels,
                    dep_matcher,
                    speaker_pred=lambda spk: ref_matches(ref, spk),
                ):
                    return s

    if named_entity := nearest_with_verbs(select(s_region.ents)):
        return named_entity

    if speaker := nearest_with_verbs(
        map(
            expand_to_noun_chunk,
            select(map(as_span, filter(is_speaker_noun, s_region))),
        )
    ):
        return speaker

    if noun_chunk := nearest_with_verbs(select(s_region.noun_chunks)):
        return noun_chunk

    if speaker_pred:
        return next(
            (
                expand_to_noun_chunk(t)
                for t in s_region
                if is_speaker_noun(t) and speaker_pred(as_span(t))
            ),
            None,
        )

    return None


def alternated(
    relations: Dict[Span, Optional[Span]],
    p_replica: Optional[Span],
    replica: Span,
):
    speakers = unique_speakers(relations)
    return (
        next((s for s in relations.values() if s and s.lemma_ == speakers[-2]), None)
        if (
            p_replica
            and replica._.start_line_no - p_replica._.end_line_no == 1
            and len(speakers) > 1
        )
        else None
    )


def classify_speakers(
    language: Language,
    dialogue: Dialogue,
) -> Dict[Span, Optional[Span]]:
    rels: Dict[Span, Optional[Span]] = {}

    if len(dialogue.replicas) == 0:
        return rels

    doc = dialogue.doc

    dep_matcher = DependencyMatcher(language.vocab)
    dep_matcher.add("*", [SPEAKING_VERB_TO_SPEAKER])
    dep_matcher.add("**", [SPEAKING_VERB_CONJUNCT_SPEAKER])

    for p_replica, replica, n_replica in iter_by_triples(dialogue.replicas):

        # On the same line as prev replica
        if (
            p_replica
            and p_replica._.end_line_no == replica._.start_line_no
            and p_replica in rels
        ):
            # Replica is on the same line - probably separated by author speech
            rels[replica] = rels[p_replica]  # <=> previous speaker

        # Non-first replica fills line
        elif fills_line(replica) and (alt := alternated(rels, p_replica, replica)):
            # Line has no author speech => speakers alternation
            rels[replica] = alt  # <=> penultimate speaker

        # After author starting ( ... [:])
        elif replica._.is_after_author_starting:
            leading = doc[min(replica.start, replica.sent.start) : replica.start]
            if (
                len(leading) < 2 < leading.start
                and doc[leading.start - 1]._.has_newline
            ):
                # Check if colon was on a previous line
                # That indicates that speaker definition may be on that line.
                leading = doc[leading.start - 2 : leading.end]
            search_start_region = expand_to_prev_line(leading)

            rels[replica] = search_for_speaker(
                search_start_region, rels, dep_matcher, replica=replica
            )

        # Before author ending ([-] ... [\n])
        elif replica._.is_before_author_ending:
            trailing = doc[replica.end : max(replica.end, replica.sent.end)]
            if len(trailing) == 0 and trailing.end + 1 < len(doc):
                trailing = doc[trailing.start : trailing.end + 1]
            search_start_region = expand_to_next_line(trailing)

            rels[replica] = search_for_speaker(
                search_start_region, rels, dep_matcher, replica=replica
            )
            if not rels[replica] and (alt := alternated(rels, p_replica, replica)):
                # Author speech is present, but it has
                # no reference to the speaker => speakers alternation
                rels[replica] = alt  # <=> penultimate speaker

        # Author insertion
        elif replica._.is_before_author_insertion and n_replica:
            search_start_region = doc[replica.end : n_replica.start]
            if is_parenthesized(search_start_region):
                # Author is commenting on a situation, there should be
                # no reference to the speaker.
                search_start_region = doc[replica.end : replica.end]

            rels[replica] = search_for_speaker(
                search_start_region, rels, dep_matcher, replica=replica
            )
            if not rels[replica]:
                del rels[replica]
                if alt := alternated(rels, p_replica, replica):
                    # Author speech is present, but it has
                    # no reference to the speaker => speakers alternation
                    rels[replica] = alt  # <=> penultimate speaker

        # Fallback - repeat speaker from prev replica
        elif (
            fills_line(replica)
            # and p_replica TODO
            # and p_replica._.end_line_no - replica._.start_line_no >= 1
            and (
                (i := dialogue.replicas.index(replica)) > 0
                and ((pr := dialogue.replicas[i - 1]) and pr in rels)
            )
        ):
            rels[replica] = rels[pr]  # <=> previous speaker

        else:
            rels[replica] = None
            print("MISS", replica)  # TODO: handle

        # TODO: handle "голос" speaker by the nearest context [with gender-based filtering at least]

    return rels
