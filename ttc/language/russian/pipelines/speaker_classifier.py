from operator import itemgetter
from typing import Optional, List, Iterable, Callable, Union

from spacy import Language
from spacy.matcher import DependencyMatcher
from spacy.symbols import VERB, root, nsubj, nmod  # type: ignore
from spacy.tokens import Token, Span

from ttc.iterables import iter_by_triples
from ttc.language import Dialogue, Play
from ttc.language.common.span_extensions import (
    is_parenthesized,
    fills_line,
    expand_to_prev_line,
    expand_to_next_line,
    trim_non_word,
)
from ttc.language.common.token_extensions import (
    expand_to_noun_chunk,
    as_span,
    dep_dist_up_to,
)
from ttc.language.russian.constants import REFERRAL_PRON
from ttc.language.russian.dependency_patterns import (
    SPEAKING_VERB_TO_SPEAKER,
    SPEAKING_VERB_CONJUNCT_SPEAKER,
    VOICE_TO_AMOD,
)
from ttc.language.russian.token_extensions import (
    ref_matches,
    is_speaker_noun,
    lowest_linked_verbal,
)


def where_with_verbs(subjects: Iterable[Span]) -> List[Span]:
    verb_subjects = []
    for noun in subjects:
        if is_speaker_noun(t := noun.root) and (
            (verbal := lowest_linked_verbal(t)) and (t.dep in {root, nsubj, nmod})
        ):
            verb_subjects.append((dep_dist_up_to(noun.root, verbal.root), noun))
    verb_subjects.sort(key=itemgetter(0))
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


def is_referential(noun: Union[Span, Token]):
    if isinstance(noun, Token):
        noun = as_span(noun)
    if any(t.lemma_ in REFERRAL_PRON for t in noun):
        return True
    dm = DependencyMatcher(noun.vocab)
    dm.add(0, [VOICE_TO_AMOD])
    return bool(dm(expand_to_noun_chunk(noun)))


def ref_search_ctx(
    play: Play,
    *,
    last_replica: Optional[Span] = None,
    ref: Union[Token, Span],
):
    if isinstance(ref, Span):
        ref = ref.root
    doc = ref.doc
    reps = list(play.replicas)
    if last_replica:
        reps.append(last_replica)
    reps_to_search_between = reps[max(0, len(reps) - 4) :]
    search_ctx = []

    # add context piece between doc start and referral pronoun indices
    if (
        len(play) == 0
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
    play: Play,
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
            speakers = play.unique_speaker_lemmas()
            if replica:
                spans = ref_search_ctx(play, last_replica=replica, ref=t)
                for span in spans:
                    if s := search_for_speaker(
                        span,
                        play,
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
                        for s in play.speakers
                        if s and s.lemma_ == speakers[-2] and ref_matches(t, s)
                    ),
                    None,
                )
            ):
                return s
            if len(play) == 2 and len(speakers) == 1:
                return play.last_speaker

        if any(select([as_span(t)])):
            return expand_to_noun_chunk(t)

    if len(play) and (prev_speaker := play.last_speaker):
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

            preceding_play = play.slice(lambda r, _: r.end <= ref.start)
            spans = ref_search_ctx(preceding_play, ref=ref)
            for span in spans:
                if s := search_for_speaker(
                    span,
                    preceding_play,
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


def classify_speakers(
    language: Language,
    dialogue: Dialogue,
) -> Play:
    play = Play(language, {})

    if len(dialogue.replicas) == 0:
        return play

    doc = dialogue.doc

    dep_matcher = DependencyMatcher(language.vocab)
    dep_matcher.add("*", [SPEAKING_VERB_TO_SPEAKER])
    dep_matcher.add("**", [SPEAKING_VERB_CONJUNCT_SPEAKER])

    for p_replica, replica, n_replica in iter_by_triples(dialogue.replicas):
        # On the same line as prev replica
        if (
            p_replica
            and p_replica._.end_line_no == replica._.start_line_no
            and p_replica in play
        ):
            # Replica is on the same line - probably separated by author speech
            play[replica] = play[p_replica]  # <=> previous speaker

        # Non-first replica fills line
        elif fills_line(replica) and (alt := play.alternated(p_replica, replica)):
            # Line has no author speech => speakers alternation
            play[replica] = alt  # <=> penultimate speaker

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

            play[replica] = search_for_speaker(
                search_start_region, play, dep_matcher, replica=replica
            )

        # Before author ending ([-] ... [\n])
        elif replica._.is_before_author_ending:
            trailing = doc[replica.end : max(replica.end, replica.sent.end)]
            if len(trailing) == 0 and trailing.end + 1 < len(doc):
                trailing = doc[trailing.start : trailing.end + 1]
            search_start_region = expand_to_next_line(trailing)

            play[replica] = search_for_speaker(
                search_start_region, play, dep_matcher, replica=replica
            )
            if not play[replica] and (alt := play.alternated(p_replica, replica)):
                # Author speech is present, but it has
                # no reference to the speaker => speakers alternation
                play[replica] = alt  # <=> penultimate speaker

        # Author insertion
        elif replica._.is_before_author_insertion and n_replica:
            search_start_region = doc[replica.end : n_replica.start]
            if is_parenthesized(search_start_region):
                # Author is commenting on a situation, there should be
                # no reference to the speaker.
                search_start_region = doc[replica.end : replica.end]

            play[replica] = search_for_speaker(
                search_start_region, play, dep_matcher, replica=replica
            )
            if not play[replica]:
                del play[replica]
                if alt := play.alternated(p_replica, replica):
                    # Author speech is present, but it has
                    # no reference to the speaker => speakers alternation
                    play[replica] = alt  # <=> penultimate speaker

        # Fallback - repeat speaker from prev replica
        elif (
            fills_line(replica)
            # and p_replica TODO
            # and p_replica._.end_line_no - replica._.start_line_no >= 1
            and (
                (i := dialogue.replicas.index(replica)) > 0
                and ((pr := dialogue.replicas[i - 1]) and pr in play)
            )
        ):
            play[replica] = play[pr]  # <=> previous speaker

        else:
            play[replica] = None
            print("MISS", replica)  # TODO: handle

        # TODO: Handle "-кий голос" speaker by the nearest context [with gender-based filtering at least]
        # TODO: Setup linked list based on next-prev replica span indices as spaCy extension props.
        #       Then do full-list improvements by finding phrases such as "Я - X", etc.
        # TODO: Save verbs associated with speakers, such as "продолжил" to correct predictions.

    return play
