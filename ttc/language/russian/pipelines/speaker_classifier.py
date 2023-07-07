from typing import Optional, List, Callable, Union, DefaultDict

from spacy import Language
from spacy.matcher import DependencyMatcher
from spacy.symbols import (  # type: ignore
    VERB,
    AUX,
    PROPN,
    nsubj,
    obj,
    obl,
    amod,
    advcl,
    cop,
)
from spacy.tokens import Token, Span

from ttc.iterables import iter_by_triples, flatten
from ttc.language import Dialogue, Play
from ttc.language.common.span_extensions import (
    is_parenthesized,
    fills_line,
    adj_prev_sent,
    expand_line_start,
    expand_line_end,
    trim_non_word,
)
from ttc.language.common.token_extensions import as_span, expand_to_noun_chunk
from ttc.language.russian.constants import REFERRAL_PRON
from ttc.language.russian.dependency_patterns import (
    SPEAKING_VERB_TO_SPEAKER,
    SPEAKING_VERB_CONJUNCT_SPEAKER,
    VOICE_TO_AMOD,
)
from ttc.language.russian.token_extensions import ref_matches


def true(*args, **kwargs):
    return True


def ref_search_ctx(
    play: Play,
    *,
    last_replica: Optional[Span] = None,
    ref: Union[Token, Span],
):
    if isinstance(ref, Token):
        ref = expand_to_noun_chunk(ref)
    doc = ref.doc
    reps = list(play.replicas) + (
        [last_replica] if last_replica and last_replica not in play else []
    )
    all_reps = reps
    reps = reps[max(0, len(reps) - 5) :]
    search_ctx = []

    if reps[0]._.is_after_author_starting:
        # take this: ______________
        # [replica?] <author text>:
        # - <reps[0]> - <author text>
        # ...
        # - <reps[-1]> - <author... |-ref-| ...text>
        above_line = expand_line_start(reps[0][0].nbor(-2).sent)
        try:
            above_rep = all_reps[all_reps.index(reps[0]) - 1]
            filtered = doc[max(above_rep.end, above_line.start) : above_line.end]
            search_ctx.append(filtered)
        except IndexError:
            pass

    # add context pieces between some replicas
    # laying before the referral pronoun
    for l, r in zip(reps, reps[1:]):
        between_reps = trim_non_word(doc[l.end : r.start])
        if len(between_reps) > 1:
            search_ctx.append(between_reps)

    # add context piece between the last replica to the left
    # of the referral pronoun and the pronoun itself
    l_bound = max(reps[-1].end, ref.start) if reps else ref.start
    if len(nearest_r_ctx := trim_non_word(doc[reps[-1].end : l_bound])) > 1:
        search_ctx.append(nearest_r_ctx)

    return search_ctx[::-1]


def is_referential(noun: Union[Span, Token]):
    if isinstance(noun, Token):
        noun = as_span(noun)
    if any(t.lemma_ in REFERRAL_PRON for t in noun):
        return True
    dm = DependencyMatcher(noun.vocab)
    dm.add(0, [VOICE_TO_AMOD])
    return bool(dm(expand_to_noun_chunk(noun)))


def potential_speaker(word):
    return (
        not word.is_punct
        and word.dep in {obj}
        or "nsubj" in word.dep_
        # exact subject noun must be present for word to be a speaker
        # e.g.: [слушатели<--NEEDED] повернулись [к __] != снова посмотрела [на __]
        or (
            word.dep == obl
            and word.pos == PROPN
            and any(c.dep == nsubj for c in word.head.children)
        )
        or is_referential(word)
    )


def verb_child_nouns(region: Span, replica: Optional[Span]) -> List[Token]:
    replica = replica or region[0:0]

    def pick_verb(top_token):
        if top_token.pos == VERB:
            return top_token
        if top_token.pos == AUX and top_token.dep == cop:
            # e.g.: ее голос [cop=был]->тихим, как шепот
            return top_token.head
        return None

    def adequate(t: Token):
        return t not in replica and not t.is_punct

    candidates = DefaultDict[Token, List[Token]](list)
    for t in region:
        if not (verb := pick_verb(t)):
            continue
        if verb.dep == advcl and verb.head.pos in {VERB, AUX}:
            # TODO check skip adverbial clause modifiers
            continue
        if verb.dep in {amod} and adequate(verb.head):
            candidates[verb].append(verb.head)
        for child in verb.children:
            if potential_speaker(child) and adequate(child):
                candidates[verb].append(child)

    results: List[Token] = list(flatten(*candidates.values()))
    return results


# TODO(tool): [prop names] -> [run] -> [get bool expr for all reached values]
def search_for_speaker(
    s_region: Span,
    play: Play,
    dep_matcher: DependencyMatcher,
    *,
    replica: Span,
    noun_predicate: Optional[Callable[[Token], bool]] = None,
):
    for noun in verb_child_nouns(s_region, replica):
        if noun_predicate and not noun_predicate(noun):
            continue

        # try to find the exact noun subject for the given verb
        # => walk on previous sentences until the start of line.
        if noun.dep != nsubj and (
            (x := adj_prev_sent(s_region)).start != s_region.start
            and (
                s := search_for_speaker(
                    x[: -len(s_region)],
                    play,
                    dep_matcher,
                    replica=replica,
                    noun_predicate=(
                        lambda nn: (noun_predicate or true)(nn)
                        and ref_matches(nn, as_span(noun.head))
                    ),
                )
            )
        ):
            return expand_to_noun_chunk(s)

        if not is_referential(noun):
            return expand_to_noun_chunk(noun)

        preceding_play = play.slice(lambda r, _: r.start < replica.start)
        if not preceding_play.lines:
            return noun

        speakers = play.unique_speaker_lemmas()
        spans = ref_search_ctx(play, last_replica=replica, ref=noun)
        for span in spans:
            if s := search_for_speaker(
                span,
                preceding_play,
                dep_matcher,
                replica=preceding_play.last_replica or replica,
                noun_predicate=(
                    (lambda nn: ref_matches(noun, as_span(nn)))
                    if noun_predicate
                    else (
                        lambda nn: ref_matches(noun, as_span(nn))
                        # exclude the last speaker name as he won't need the reference
                        and (not speakers or nn.lemma_ != speakers[-1])
                    )
                ),
            ):
                return expand_to_noun_chunk(s)

    # try harder to find some noun (whatever)
    # => walk on previous sentences until the start of line.
    if (x := adj_prev_sent(s_region)).start != s_region.start and (
        s := search_for_speaker(
            x[: -len(s_region)],
            play,
            dep_matcher,
            replica=replica,
            noun_predicate=noun_predicate,
        )
    ):
        return expand_to_noun_chunk(s)

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
            search_start_region = expand_line_start(leading)

            # start with the nearest sentence before ":"
            if len(sents := list(search_start_region.sents)) > 1:
                search_start_region = sents[-2]
            # if search_start_region[0].pos == ADV and len(sents) > 2:
            #     search_start_region = sents[-3]

            play[replica] = search_for_speaker(
                search_start_region, play, dep_matcher, replica=replica
            )

        # Before author ending ([-] ... [\n])
        elif replica._.is_before_author_ending:
            trailing = doc[replica.end : max(replica.end, replica.sent.end)]
            if len(trailing) == 0 and trailing.end + 1 < len(doc):
                trailing = doc[trailing.start : trailing.end + 1]
            search_start_region = expand_line_end(trailing)

            play[replica] = search_for_speaker(
                search_start_region, play, dep_matcher, replica=replica
            )
            if not play[replica]:
                del play[replica]
                if alt := play.alternated(p_replica, replica):
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
