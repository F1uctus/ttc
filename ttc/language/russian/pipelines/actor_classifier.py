from collections import Counter
from itertools import chain
from typing import Optional, List, Callable, Union, Dict, Generator, Final

from spacy import Language
from spacy.matcher import DependencyMatcher
from spacy.symbols import (  # type: ignore
    VERB,
    AUX,
    PRON,
    PROPN,
    DET,
    NOUN,
    obj,
    iobj,
    obl,
    advcl,
    parataxis,
)
from spacy.tokens import Token, Span

from ttc.iterables import iter_by_triples, flatten
from ttc.language import Dialogue, Play
from ttc.language.common.span_extensions import (
    is_parenthesized,
    fills_line,
    sents_between,
    line_breaks_between,
    expand_line_start,
    expand_line_end,
    trim_non_word,
    contiguous,
)
from ttc.language.common.token_extensions import (
    has_dep,
    noun_chunk,
    morph_equals,
    morph_distance,
)
from ttc.language.russian.constants import REFERRAL_PRON, PRON_MORPHS
from ttc.language.russian.dependency_patterns import (
    VOICE_TO_AMOD,
)
from ttc.language.russian.token_extensions import is_copula
from ttc.language.types import Morph

Gender: Final[Morph] = "Gender"
Number: Final[Morph] = "Number"
Tense: Final[Morph] = "Tense"


def ref_matches(ref: Token, target: Token) -> bool:
    return morph_distance(target, ref, Gender, Number, Tense) < 2


def is_ref(noun: Union[Span, Token]):
    if isinstance(noun, Token):
        return noun.pos == PRON or noun.lemma_ in REFERRAL_PRON
    if any(t.pos == PRON for t in noun) or any(t.lemma_ in REFERRAL_PRON for t in noun):
        return True
    dm = DependencyMatcher(noun.vocab)
    dm.add(0, [VOICE_TO_AMOD])
    return bool(dm(noun_chunk(noun)))


def predicate_verbs(span: Span) -> List[Token]:
    verbs: List[Token] = []
    for t in span:
        if (
            t.pos == VERB
            and t.dep not in {advcl}
            and (
                has_dep(t, parataxis, "ROOT", "ccomp", "xcomp", "acl")
                or t.head in verbs  # FIXME: also pick children of root verbs?
                or t.head not in span
            )
        ):
            if t.text.endswith("вшись"):  # verbal adverb
                # e.g.: t=напившись --> становится[VERB]
                verbs.extend(tt for tt in t.children if tt.pos in {VERB, AUX})
            else:
                verbs.append(t)
        elif is_copula(t):
            verbs.append(t.head)
    return verbs


def potential_actors(verb: Token, span: Span) -> Generator[Token, None, None]:
    """actor, related verb"""

    if any(
        t in span and t.i < verb.i and any(has_dep(c, "nsubj") for c in t.children)
        for t in verb.conjuncts
    ):
        return

    for word in verb.children:
        if word not in span or word.is_punct:
            continue

        if verb.dep_ == "acl:relcl" and word.pos == PRON:
            continue

        if word.dep == obl:
            if det := next((c for c in word.children if c.pos == DET), None):
                yield det  # rel. verb == None
            elif (
                "Animacy=Inan" not in word.morph or word.ent_type_ == "PER"
            ) and word.pos in {NOUN, PRON, PROPN}:
                yield word  # rel. verb == verb

        # if verb.dep_ in {"ccomp", "xcomp"} and word.pos

        if (
            (word.dep == obj and ref_matches(verb, word))
            or has_dep(word, iobj, "nsubj")
            # exact subject noun must be present for word to be an actor
            # e.g.: [слушатели<--NEEDED] повернулись [к __] != снова посмотрела [на __]
            or (
                word.dep == obl
                and word.pos == PROPN
                and any("nsubj" in c.dep_ for c in verb.children)
            )
            or is_ref(word)
        ):
            yield word  # rel. verb == verb


def morph_aligns_with(target: Token) -> Callable[[Token], bool]:
    def aligned_gender(tk) -> Optional[str]:
        t = noun_chunk(tk)
        if len(t) == 1:
            return [*t.root.morph.get(Gender), None][0]  # type: ignore
        morphs: Dict[str, str] = next(
            (
                v
                for tk in reversed(t)
                for k, v in PRON_MORPHS.items()
                if tk.lemma_.startswith(k)
            ),
            {},
        )
        stats = Counter([*tk.morph.get(Gender), None][0] for tk in t)  # type: ignore
        return morphs.get(Gender, max(stats, key=stats.get))  # type: ignore

    target_gender = aligned_gender(target)
    return lambda ref: (
        aligned_gender(ref) == target_gender and morph_equals(ref, target, Number)
        if target_gender
        else ref_matches(ref, target)
        or (morph_equals(ref, target, Number) and ref_matches(ref.head, target))
    )


def head_not_in(t: Token, span: Union[Span, List[Token]]):
    h = t.head
    while True:
        if h in span:
            return h == t.head
        if h == h.head:
            return True
        h = h.head


def actor_search(
    span: Span,
    play: Play,
    replica: Span,
    *,
    ref_chain: Optional[List[Token]] = None,
    resolve_refs: bool = True,
) -> Optional[Span]:
    if ref_chain is None:
        ref_chain = []
    ref = ref_chain[-1] if ref_chain else None
    ref_matcher = morph_aligns_with(ref) if ref else lambda _: True

    # Pick all the matching actor candidates from the search [span]
    pred_verbs = predicate_verbs(span)
    candidates = list(flatten(potential_actors(v, span) for v in pred_verbs))

    if all("Gender=Neut" in v.morph for v in pred_verbs if head_not_in(v, pred_verbs)):
        # Neutral-gender root verb usually indicates
        # a description of situation, not actor denotation.
        matching = []
    else:
        matching = list(filter(ref_matcher, candidates))

    def is_proper_subject(n):
        return not is_ref(n) and (
            (has_dep(n, "nsubj") and ("Case=Nom" in n.morph or "Case=Acc" in n.morph))
            or n.ent_iob_ in "IB"
            or n.ent_type_ == "PER"
        )

    psubjs = {noun_chunk(m).root.lemma_: m for m in matching if is_proper_subject(m)}
    if len(psubjs) == 1:
        first_subj = psubjs[next(iter(psubjs))]
        actor = noun_chunk(first_subj)
        # TODO: Here we need to remove more actors that are already present
        #  in the play (need to check more contiguous replicas than just this & prev)
        if actor == play.last_actor and contiguous(play.last_replica, replica):
            # Discard repeated actor
            candidates.remove(first_subj)
            matching.remove(first_subj)
        else:
            # Single matching non-ref candidate is good enough
            return actor

    # Reference resolution
    if resolve_refs:
        m_refs = list(filter(is_ref, matching))
        if (
            not ref
            and len(m_refs) == 1
            and play.last_replica
            and contiguous(play.last_replica, replica)
            and (alt := play.penult())
        ):
            ref_chain.extend(m_refs)
            return alt

        m_verbs = list(filter(ref_matcher, pred_verbs))
        if m_refs:
            ref_roots = m_refs + m_verbs  # Verbs improve the ref resolution
        elif matching:
            # Pick only verbs which are distinct from the found actors
            # (this is empty most of the time).
            m_verbs = [
                v
                for v in m_verbs
                if set(v.children).isdisjoint(matching) and "Gender=Neut" not in v.morph
            ]
            ref_roots = m_verbs
        elif ref or not candidates:
            ref_roots = m_verbs
        else:
            ref_roots = []

        search_ctx = sents_between(
            # Spans between which the antecedent will be searched
            list(play.replicas)
            + ([] if ref else ([replica] if replica else []))
            + [span],
            reverse=True,
        )
        cached_ctx: List[Span] = []
        for word in ref_roots:
            if bound := play.reference(word):
                return bound
            ante = None
            depth = 1
            trailing_span: Optional[Span] = (
                span[: word.i - span.start]
                if span.start + 1 < word.i <= span.end
                else None
            )
            trailing_ctx = [trailing_span] if trailing_span else []
            for i, region in enumerate(chain(trailing_ctx, cached_ctx, search_ctx)):
                if depth == 5:
                    break
                if region != trailing_span:
                    if i == len(cached_ctx):
                        cached_ctx.append(region)
                    if word.i <= region.end or ref and ref.i <= region.end:
                        continue
                depth += 1
                ref_chain.append(word)
                ante = actor_search(region, play, replica, ref_chain=ref_chain)
                if ante and ante.start != word.i:
                    return noun_chunk(ante)
                else:
                    ref_chain.pop()
            else:
                if ref and ante and ref.lemma_ == ante.lemma_:
                    return noun_chunk(ante)

    # TODO: Add "ref_resolved" to the resolved and check it in is_ref()
    if first := next((m for m in matching if not is_ref(m)), None):
        if (actor := noun_chunk(first)) == play.last_actor and (
            line_breaks_between(first, ref_chain[0]) < 2
        ):
            pass
        else:
            return actor

    return noun_chunk(ref) if ref else noun_chunk(matching[0]) if matching else None


def classify_actors(
    language: Language,
    dialogue: Dialogue,
) -> Play:
    p = Play(language)

    if len(dialogue.replicas) == 0:
        return p

    doc = dialogue.doc

    for p_replica, replica, n_replica in iter_by_triples(dialogue.replicas):
        ref_chain: List[Token] = []
        # On the same line as prev replica
        if (
            p_replica
            and p_replica._.end_line_no == replica._.start_line_no
            and p_replica in p
        ):
            # Replica is on the same line - probably separated by author speech
            p[replica] = p[p_replica]  # <=> previous actor

        # Alternation
        # - ["xxx"] - A - ["yyy"] <- pick A <-|
        # - ["zzz"] - B - ["www"]             |- 2 line breaks between
        # - @replica                       <--|
        elif fills_line(replica) and (
            (penult := p.penult())
            and line_breaks_between(list(p.replicas_of(penult))[-1], replica) == 2
        ):
            p[replica] = penult

        # After author starting ( ... [:])
        elif replica._.is_after_author_starting:
            leading = doc[min(replica.start, replica.sent.start) : replica.start]
            if (
                len(leading) < 2 < leading.start
                and doc[leading.start - 1]._.has_newline
            ):
                # Check if colon was on a previous line
                # That indicates that actor definition may be on that line.
                leading = doc[leading.start - 2 : leading.end]
            search_span = expand_line_start(leading)

            # start with the nearest sentence before ":"
            if len(sents := list(search_span.sents)) > 1:
                search_span = sents[-2]

            p[replica] = (
                actor_search(search_span, p, replica, ref_chain=ref_chain),
                ref_chain,
            )

        # Before author ending ([-] ... [\n])
        elif replica._.is_before_author_ending:
            trailing = doc[replica.end : max(replica.end, replica.sent.end)]
            if len(trailing) == 0 and trailing.end + 1 < len(doc):
                trailing = doc[trailing.start : trailing.end + 1]

            if n_replica and line_breaks_between(replica, n_replica) == 0:
                # BUG: Sometimes ending is actually an insertion
                search_span = doc[trailing.start : n_replica.start]
            else:
                search_span = expand_line_end(trailing)
            search_span = trim_non_word(search_span)

            p[replica] = (
                actor_search(search_span, p, replica, ref_chain=ref_chain),
                ref_chain,
            )
            if not p[replica]:
                del p[replica]
                if penult := p.penult():
                    # Author speech is present, but it has
                    # no reference to the actor => actor alternation
                    p[replica] = penult

        # Author insertion
        elif replica._.is_before_author_insertion and n_replica:
            search_span = doc[replica.end : n_replica.start]
            if is_parenthesized(search_span):
                # Author is commenting on a situation, there should be
                # no reference to the actor.
                search_span = doc[replica.end : replica.end]

            search_span = trim_non_word(search_span)
            # Limit to the sentence nearest to the replica
            if len(sents := list(search_span.sents)) > 1:
                search_span = sents[0]

            p[replica] = (
                actor_search(search_span, p, replica, ref_chain=ref_chain),
                ref_chain,
            )
            if not p[replica]:
                del p[replica]
                if penult := p.penult():
                    # Author speech is present, but it has
                    # no reference to the actor => actor alternation
                    p[replica] = penult

        # Fallback, similar to ( ... [:]), but
        # constrained to a single line between replicas
        elif fills_line(replica) and (
            search_span := next(
                filter(
                    # Span contains words
                    lambda s: any(t.is_alpha for t in s)
                    and (
                        fills_line(s)
                        # Span does not immediately precede the replica
                        or any(t.is_alpha for t in doc[s.end : replica.start])
                    ),
                    # Checking every sentence between replicas from bottom to top
                    sents_between(
                        [replica] + list(reversed(p.replicas)) + [doc[0:0]],
                        reverse=True,
                    ),
                ),
                None,
            )
        ):
            # TODO: check for appeal in the replica text, then fallback on actor_search.
            # if len(list(search_span.sents)) == 1:
            search_span = list(trim_non_word(search_span).sents)[-1]
            p[replica] = actor_search(search_span, p, replica)
            # elif penult := p.penult():
            #     # Author speech has multiple sentences, so it more possibly has
            #     # no reference to the actor => actor alternation
            #     p[replica] = penult

        # Fallback - repeat actor from prev replica
        elif fills_line(replica) and p_replica and p_replica in p:
            p[replica] = p[p_replica]  # <=> previous actor

        else:
            p[replica] = None
            print("MISS", replica)  # TODO: handle

    return p
