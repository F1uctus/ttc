from typing import Optional, List, Callable, Union, Dict, Generator
from collections import Counter

from itertools import chain
from spacy import Language
from spacy.matcher import DependencyMatcher
from spacy.symbols import (  # type: ignore
    VERB,
    AUX,
    PROPN,
    DET,
    obj,
    obl,
    acl,
    advcl,
    parataxis,
)
from spacy.tokens import Token, Span

from ttc.iterables import iter_by_triples, flatten
from ttc.language import Dialogue, Play
from ttc.language.common.span_extensions import (
    is_parenthesized,
    fills_line,
    line_breaks_between,
    expand_line_start,
    expand_line_end,
    trim_non_word,
    contiguous,
    line_above,
)
from ttc.language.common.token_extensions import (
    as_span,
    noun_chunk,
    morph_equals,
    morph_distance,
)
from ttc.language.russian.token_extensions import is_copula
from ttc.language.russian.constants import REFERRAL_PRON, PRON_MORPHS
from ttc.language.russian.dependency_patterns import (
    ACTION_VERB_TO_ACTOR,
    ACTION_VERB_CONJUNCT_ACTOR,
    VOICE_TO_AMOD,
)


def ref_matches(ref: Token, target: Token) -> bool:
    return morph_distance(target, ref, "Gender", "Number", "Tense") < 2


def is_ref(noun: Union[Span, Token]):
    if isinstance(noun, Token):
        noun = as_span(noun)
    if any(t.lemma_ in REFERRAL_PRON for t in noun):
        return True
    dm = DependencyMatcher(noun.vocab)
    dm.add(0, [VOICE_TO_AMOD])
    return bool(dm(noun_chunk(noun)))


def top_verbs(span: Span, replica: Span) -> List[Token]:
    verbs: List[Token] = []
    for t in span:
        if (
            t.pos == VERB
            and t.dep not in {advcl}
            and (
                t.dep_ == "ROOT"
                or t.dep in {acl, parataxis}
                or t.head in list(replica) + verbs
            )
        ):
            if t.text.endswith("вшись"):
                verbs.extend(tt for tt in t.children if tt.pos in {VERB, AUX})
            else:
                verbs.append(t)
        elif is_copula(t):
            verbs.append(t.head)
    return verbs


def potential_actors(verb: Token, replica: Span) -> Generator[Token, None, None]:
    """actor, related verb"""

    def in_region(t):
        return t not in replica and not t.is_punct

    for word in verb.children:
        if not in_region(word):
            continue
        if word.dep == obl and (
            det := next((c for c in word.children if c.pos == DET), None)
        ):
            yield det  # rel. verb == None
        if (
            (word.dep in {obj} and ref_matches(verb, word))
            or "nsubj" in word.dep_
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
            return t.root.morph.get("Gender", [None])[0]  # type: ignore
        morphs: Dict[str, str] = next(
            (
                v
                for tk in reversed(t)
                for k, v in PRON_MORPHS.items()
                if tk.lemma_.startswith(k)
            ),
            {},
        )
        stats = Counter(tk.morph.get("Gender", [None])[0] for tk in t)  # type: ignore
        return morphs.get("Gender", max(stats, key=stats.get))  # type: ignore

    target_gender = aligned_gender(target)
    return lambda ref: (
        aligned_gender(ref) == target_gender and morph_equals(ref, target, "Number")
        if target_gender
        else ref_matches(ref, target)
        or (morph_equals(ref, target, "Number") and ref_matches(ref.head, target))
    )


def reference_resolution_context(bounds: List[Span]) -> Generator[Span, None, None]:
    if not bounds:
        return
    doc = bounds[0].doc
    bounds = sorted(bounds, key=lambda sp: sp.start, reverse=True) + [doc[0:0]]
    # Read context pieces between bounds
    for r_bound, l_bound in zip(bounds, bounds[1:]):
        if not (bet := trim_non_word(doc[l_bound.end : r_bound.start])):
            continue
        split_idxs = sorted(
            (i for i in doc._.nl_indices if bet.start_char <= i <= bet.end_char),
            reverse=True,
        )
        if not split_idxs:
            yield bet
            continue
        if trailing := doc.char_span(split_idxs[0] + 1, bet.end_char):
            yield trailing
        for ri, li in zip(split_idxs, split_idxs[1:]):
            if middle := doc.char_span(li + 1, ri):
                yield middle
        if leading := doc.char_span(bet.start_char, split_idxs[-1]):
            yield leading


def actor_search(
    span: Span,
    play: Play,
    dep_matcher: DependencyMatcher,
    replica: Span,
    *,
    ref_chain: Optional[List[Token]] = None,
    resolve_refs: bool = True,
) -> Optional[Span]:
    ref_chain = ref_chain or []
    ref = ref_chain[-1] if ref_chain else None
    ref_matcher = morph_aligns_with(ref) if ref else lambda _: True

    # Pick all the matching actor candidates from the search [span]
    root_verbs = top_verbs(span, replica)
    candidates = list(flatten(list(potential_actors(rv, replica)) for rv in root_verbs))

    if (roots := [r for r in span if r.dep_ == "ROOT"]) and (
        all("Neut" in r.morph.get("Gender", []) for r in roots)
    ):
        # Neutral-gender root verb usually indicates
        # a description of situation, not actor denotation.
        matching = []
    else:
        # Order of candidates is better to be preserved.
        matching = list(filter(ref_matcher, candidates))

    def exact_enough(n):
        return not is_ref(n) and ("nsubj" in n.dep_ or n.ent_iob_ in "IB")

    exacts = {noun_chunk(m).root.lemma_: m for m in matching if exact_enough(m)}
    if len(exacts) == 1:
        m = exacts[next(iter(exacts))]
        actor = noun_chunk(m)
        if actor == play.last_actor:
            if contiguous(play.last_replica, replica):
                # Discard repeated actor
                matching.remove(m)
        # Index check ensures that no other matches closer to the replica are missed
        elif candidates.index(m) == 0:
            # Single matching non-ref candidate is good enough
            return actor

    # Reference resolution
    if resolve_refs:
        m_refs = list(filter(is_ref, matching))
        m_verbs = list(filter(ref_matcher, root_verbs))
        if m_refs:
            ref_roots = m_refs + m_verbs  # Verbs improve the ref resolution
        else:
            # Pick only verbs which are distinct from the found actors
            # (this is empty most of the time).
            m_verbs = [
                v
                for v in m_verbs
                if set(v.children).isdisjoint(matching)
                and "Neut" not in v.morph.get("Gender", [])
            ]
            ref_roots = m_verbs

        # Spans between which the antecedent will be searched
        # (replicas & currently processed span)
        search_ctx = reference_resolution_context(
            list(play.replicas)
            + ([] if ref else ([replica] if replica else []))
            + [span]
        )
        cached_ctx: List[Span] = []
        for word in ref_roots:
            ante = None
            depth = 1
            for i, region in enumerate(chain(cached_ctx, search_ctx)):
                if depth == 5:
                    break
                if i == len(cached_ctx):
                    cached_ctx.append(region)
                if word.i <= region.end or ref and ref.i <= region.end:
                    continue
                depth += 1
                ante = actor_search(
                    region, play, dep_matcher, replica, ref_chain=ref_chain + [word]
                )
                if ante and ante.start != word.i:
                    return noun_chunk(ante)
            else:
                if ref and ante and ref.lemma_ == ante.lemma_:
                    return noun_chunk(ante)

    if len(matching) > 0:
        if not ref:
            return noun_chunk(matching[0])
        elif first := next((m for m in matching if not is_ref(m)), None):
            if (actor := noun_chunk(first)) == play.last_actor and (
                line_breaks_between(first, ref_chain[0]) < 2
            ):
                pass
            else:
                return actor

    return noun_chunk(ref) if ref else None


def classify_actors(
    language: Language,
    dialogue: Dialogue,
) -> Play:
    p = Play(language, {})

    if len(dialogue.replicas) == 0:
        return p

    doc = dialogue.doc

    dep_matcher = DependencyMatcher(language.vocab)
    dep_matcher.add("*", [ACTION_VERB_TO_ACTOR])
    dep_matcher.add("**", [ACTION_VERB_CONJUNCT_ACTOR])

    for p_replica, replica, n_replica in iter_by_triples(dialogue.replicas):
        # On the same line as prev replica
        if (
            p_replica
            and p_replica._.end_line_no == replica._.start_line_no
            and p_replica in p
        ):
            # Replica is on the same line - probably separated by author speech
            p[replica] = p[p_replica]  # <=> previous actor

        # Non-first replica fills line
        elif (
            fills_line(replica)
            and line_breaks_between(replica, p_replica) == 1
            and (alt := p.alternated())
        ):
            # Line has no author speech => actor alternation
            p[replica] = alt  # <=> penultimate actor

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

            p[replica] = actor_search(search_span, p, dep_matcher, replica)

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

            p[replica] = actor_search(search_span, p, dep_matcher, replica)
            if not p[replica]:
                del p[replica]
                if alt := p.alternated():
                    # Author speech is present, but it has
                    # no reference to the actor => actor alternation
                    p[replica] = alt  # <=> penultimate actor

        # Author insertion
        elif replica._.is_before_author_insertion and n_replica:
            search_span = doc[replica.end : n_replica.start]
            if is_parenthesized(search_span):
                # Author is commenting on a situation, there should be
                # no reference to the actor.
                search_span = doc[replica.end : replica.end]

            p[replica] = actor_search(search_span, p, dep_matcher, replica)
            if not p[replica]:
                del p[replica]
                if alt := p.alternated():
                    # Author speech is present, but it has
                    # no reference to the actor => actor alternation
                    p[replica] = alt  # <=> penultimate actor

        # Fallback, similar to ( ... [:]), but constrained
        # to a single line between replicas (purely heuristically)
        elif (
            fills_line(replica)
            and p_replica
            and line_breaks_between(replica, p_replica) == 2
            and (
                (search_span := line_above(replica))
                and sum(not t.is_stop and not t.is_punct for t in search_span) >= 5
            )
        ):
            # start with the nearest sentence before ":"
            if len(sents := list(search_span.sents)) > 1:
                search_span = sents[-1]

            s = actor_search(search_span, p, dep_matcher, replica, resolve_refs=False)
            if s and is_ref(s) and (alt := p.alternated()):
                # Author speech is present, but it has
                # no reference to the actor => actor alternation
                p[replica] = alt  # <=> penultimate actor
            else:
                p[replica] = s

        # Fallback - repeat actor from prev replica
        elif fills_line(replica) and p_replica and p_replica in p:
            p[replica] = p[p_replica]  # <=> previous actor

        else:
            p[replica] = None
            print("MISS", replica)  # TODO: handle

    return p
