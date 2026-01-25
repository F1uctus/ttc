from typing import Optional, List, Callable, Union, Dict, Generator, Final
from collections import Counter

from itertools import chain
from spacy import Language
from spacy.matcher import DependencyMatcher
from spacy.symbols import (  # type: ignore
    VERB,
    AUX,
    PRON,
    PROPN,
    NOUN,
    NUM,
    ADJ,
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
from ttc.language.types import Morph
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
    noun_chunk,
    morph_equals,
    morph_distance,
)
from ttc.language.russian.token_extensions import is_copula
from ttc.language.russian.constants import REFERRAL_PRON, PRON_MORPHS
from ttc.language.common.constants import HYPHENS as HYPHENS_STR
from ttc.language.russian.dependency_patterns import (
    VOICE_TO_AMOD,
)

Gender: Final[Morph] = "Gender"
Number: Final[Morph] = "Number"
Tense: Final[Morph] = "Tense"
HYPHENS: Final = set(HYPHENS_STR)
ANIMACY_ANIM: Final = "Animacy=Anim"
ANIMACY_INAN: Final = "Animacy=Inan"
NUMBER_PLUR: Final = "Number=Plur"

REFLEXIVE_PRONOUNS: Final = frozenset({"себя"})
ADJ_ENDINGS: Final = (
    "ый",
    "ий",
    "ой",
    "ая",
    "яя",
    "ое",
    "ее",
    "ые",
    "ие",
    "ого",
    "его",
    "ому",
    "ему",
    "ым",
    "им",
    "ом",
    "ем",
    "ую",
    "юю",
    "ыми",
    "ими",
)


def is_reflexive_pronoun(token: Token) -> bool:
    return token.lemma_ in REFLEXIVE_PRONOUNS


def looks_like_adj(token: Token) -> bool:
    if not token.is_alpha:
        return False
    return token.text.lower().endswith(ADJ_ENDINGS)


def has_noun_or_propn(span: Span) -> bool:
    return any(t.pos in {NOUN, PROPN} for t in span)


def is_pronoun_vague(token: Token) -> bool:
    if token.pos != PRON:
        return False
    if token.morph.get("Person"):
        return False
    pron_types = set(token.morph.get("PronType"))
    if "Prs" in pron_types:
        return False
    return True if pron_types or token.pos == PRON else False


def is_generic_person_noun(span: Optional[Span]) -> bool:
    if not span or span.root.pos != NOUN:
        return False
    lemma = span.root.lemma_.lower()
    return any(lemma.startswith(k) for k in PRON_MORPHS)


def is_inanimate(span: Span) -> bool:
    return span_has_animacy(span, ANIMACY_INAN) and not span_has_animacy(span, ANIMACY_ANIM)


def expand_hyphenated_span(span: Span) -> Span:
    if not span:
        return span
    doc = span.doc
    start, end = span.start, span.end
    contains_hyphen = any(t.text in HYPHENS for t in span)
    if doc[start].text in HYPHENS and start > 0:
        start -= 1
        contains_hyphen = True
    if contains_hyphen and start > 0:
        prev = doc[start - 1]
        if prev.is_alpha and looks_like_adj(prev):
            start -= 1
    return doc[start:end]


def span_has_animacy(span: Span, value: str) -> bool:
    return any(value in t.morph for t in span)


def span_has_animate_noun(span: Span) -> bool:
    return any(t.pos == NOUN and ANIMACY_ANIM in t.morph for t in span)


def span_is_collective(span: Span) -> bool:
    if not span:
        return False
    if span.root.lemma_ == "голос" and NUMBER_PLUR in span.root.morph:
        return True
    return any(NUMBER_PLUR in t.morph for t in span if t.is_alpha)


def is_nominative(token: Token) -> bool:
    return "Case=Nom" in token.morph


def is_oblique(token: Token) -> bool:
    cases = token.morph.get("Case")
    return bool(cases) and "Nom" not in cases


def has_first_person(span: Span) -> bool:
    return any("First" in t.morph.get("Person") for t in span if t.is_alpha)


def has_second_person(span: Span) -> bool:
    return any("Second" in t.morph.get("Person") for t in span if t.is_alpha)


def has_specific_role(span: Span) -> bool:
    return any(
        t.pos == NOUN
        and t.lemma_ not in REFERRAL_PRON
        and span_has_animacy(noun_chunk(t), ANIMACY_ANIM)
        for t in span
    )


def is_vague_head(span: Span) -> bool:
    root = span.root
    if root.pos == PRON and is_pronoun_vague(root):
        return not has_specific_role(span)
    if root.pos in {NUM, DET}:
        core_noun = any(
            t.pos in {NOUN, PROPN} and t.dep_ not in {"obl", "nmod", "case"}
            for t in span
        )
        if not core_noun:
            return True
    return False


def find_appositive_descriptor(span: Span) -> Optional[Token]:
    for t in span:
        if t.dep_ == "appos" and t.pos in {NOUN, PROPN, ADJ}:
            return t
    root = span.root
    if root.pos in {PRON, NUM, DET}:
        for t in span:
            if t.pos == ADJ and morph_equals(t, root, Gender, Number):
                return t
    return None


def normalize_span(span: Span) -> Span:
    if not span:
        return span
    span = expand_hyphenated_span(span)
    if span.start > 0:
        prev = span.doc[span.start - 1]
        if (
            prev.pos in {NUM, DET}
            and prev.head == span.root
            and prev.dep_ in {"nummod", "det"}
        ):
            span = span.doc[prev.i : span.end]
    if any(t.text in HYPHENS for t in span):
        right = next((t for t in reversed(span) if t.pos in {NOUN, PROPN}), None)
        left = next((t for t in span if t.pos in {NOUN, PROPN}), None)
        if right and left and right != left:
            if right.pos == PROPN or right.ent_type_ == "PER":
                return right.doc[right.i : right.i + 1]
            if right == span.root and left.pos == NOUN:
                modifier_deps = {"amod", "compound", "appos", "nmod", "flat"}
                if not any(c.dep_ in modifier_deps for c in left.children):
                    return right.doc[right.i : right.i + 1]
    if is_vague_head(span) and (appos := find_appositive_descriptor(span)):
        return appos.doc[appos.i : appos.i + 1]
    return span


def is_pronoun_span(span: Optional[Span]) -> bool:
    return bool(span) and span.root.pos == PRON


def is_vague_actor(span: Optional[Span]) -> bool:
    if not span:
        return False
    if span.root.pos == ADJ and not has_noun_or_propn(span):
        return bool(span.root.morph.get("PronType") or span.root.morph.get("NumType"))
    return is_vague_head(span)


def is_indefinite_actor(span: Optional[Span]) -> bool:
    if not span:
        return False
    if is_vague_actor(span):
        return True
    return span.root.pos == NUM and any(
        t.dep_ == "nmod" and t.head == span.root for t in span
    )


def is_brief_reply(span: Span) -> bool:
    return sum(t.is_alpha for t in span) <= 2


def actor_key(span: Optional[Span]) -> str:
    if not span:
        return ""
    if any(t.pos == PROPN or t.ent_type_ == "PER" for t in span):
        propn = " ".join(t.lemma_.lower() for t in span if t.pos == PROPN)
        return propn or span.lemma_.lower()
    if span.root.pos == PRON:
        return span.lemma_.lower()
    if any(t.text in HYPHENS for t in span) and span.root.pos in {NOUN, PROPN}:
        return span.root.lemma_.lower()
    return span.text.lower()


def is_human_like(span: Optional[Span]) -> bool:
    if not span:
        return False
    if any(t.ent_type_ == "PER" for t in span):
        return True
    if any(t.pos == NOUN and ANIMACY_ANIM in t.morph for t in span):
        return True
    if any(t.pos == PROPN and ANIMACY_ANIM in t.morph for t in span):
        return True
    return any(t.pos == PRON and t.morph.get("Person") for t in span)


def has_voice_intro(replica: Span) -> bool:
    intro = replica.doc[expand_line_start(replica).start : replica.start]
    if any(t.lemma_ == "голос" for t in intro):
        return True
    if above := line_above(replica):
        return any(t.lemma_ == "голос" for t in above)
    return False


def refined_noun_chunk(token: Union[Token, Span]) -> Span:
    return normalize_span(expand_hyphenated_span(noun_chunk(token)))


def best_candidate(candidates: List[Token]) -> Optional[Span]:
    if not candidates:
        return None

    prepared = [(c, refined_noun_chunk(c)) for c in candidates]
    any_human = any(is_human_like(span) for _, span in prepared)
    if any_human:
        prepared = [
            (c, span)
            for c, span in prepared
            if is_human_like(span) or not is_inanimate(span)
        ]

    scored: List[tuple] = []
    for c, span in prepared:
        alpha_len = sum(t.is_alpha for t in span)
        length_score = min(alpha_len, 3)
        score = length_score
        if "nsubj" in c.dep_:
            score += 2
        if c.dep == obj:
            score -= 2
        elif c.dep == obl:
            score -= 1
        if c.pos in {PROPN, NOUN} and not is_nominative(c):
            score -= 3
        if not is_ref(c):
            score += 3
        if c.pos == PROPN or c.ent_type_ == "PER" or any(
            t.ent_type_ == "PER" for t in span
        ):
            score += 3
        if c.dep_ == "appos":
            score += 1
        if span.root.pos == ADJ and c.dep_ == "appos":
            score += 2
        if c.pos in {NUM, PRON, DET} and span.root.pos == ADJ:
            score += 2
        if c.pos == PRON:
            score -= 2
        if is_reflexive_pronoun(c):
            score -= 5
        is_human = is_human_like(span)
        if is_human:
            score += 2
        if any_human and not is_human:
            score -= 2
        if any_human and is_inanimate(span):
            score -= 6
        scored.append((score, length_score, -c.i, span))

    best = max(scored, key=lambda item: (item[0], item[1], item[2]))
    return best[3]


def ref_matches(ref: Token, target: Token) -> bool:
    return morph_distance(target, ref, Gender, Number, Tense) < 2


def is_ref(noun: Union[Span, Token]):
    if isinstance(noun, Token):
        if noun.pos == PRON:
            return True
        if noun.lemma_ in REFERRAL_PRON:
            if noun.pos == NOUN and ANIMACY_ANIM in noun.morph:
                return False
            return True
        return False
    if any(t.pos == PRON for t in noun):
        return True
    if any(t.lemma_ in REFERRAL_PRON for t in noun):
        return not span_has_animate_noun(noun)
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

    for word in verb.children:
        if word in replica or word.is_punct:
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


def resolve_recent_actor(play: Play, ref: Token) -> Optional[Span]:
    matcher = morph_aligns_with(ref)
    fallback = None
    named_fallback = None
    named_keys: List[str] = []
    for actor in reversed(list(play.actors)):
        if not actor or is_ref(actor):
            continue
        if matcher(actor.root):
            if any(t.pos == PROPN or t.ent_type_ == "PER" for t in actor):
                return actor
            if fallback is None:
                fallback = actor
        if any(t.pos == PROPN or t.ent_type_ == "PER" for t in actor):
            key = actor_key(actor)
            if key not in named_keys:
                named_keys.append(key)
                if named_fallback is None:
                    named_fallback = actor
                if len(named_keys) >= 2:
                    break
    if named_fallback and len(named_keys) == 1:
        return named_fallback
    return fallback


def resolve_role_actor(play: Play, actor: Span, *, prefer_recent_actor: bool) -> Span:
    if not actor:
        return actor
    if actor.root.pos == PROPN or actor.root.ent_type_ == "PER":
        return actor
    if any(t.pos == PROPN or t.ent_type_ == "PER" for t in actor) and any(
        t.pos == NOUN for t in actor
    ):
        return actor
    named = next(
        (t for t in actor if t.pos == PROPN or t.ent_type_ == "PER"),
        None,
    )
    if named:
        return named.doc[named.i : named.i + 1]
    return actor


def resolve_named_case(play: Play, actor: Span, *, force: bool = False) -> Span:
    if not (actor.root.pos == PROPN or actor.root.ent_type_ == "PER"):
        return actor
    if (
        not force
        and actor.root.dep_ in {"nmod", "nmod:poss", "appos"}
        and actor.root.head.pos == NOUN
    ):
        return actor
    if (
        is_nominative(actor.root)
        and actor.root.text.lower() == actor.root.lemma_.lower()
        and not actor.root.text.lower().endswith(("и", "ы"))
    ):
        return actor
    root_text = actor.root.text.lower()
    stem = root_text[:-1] if len(root_text) > 2 else ""

    def find_by_lemma(tokens):
        for token in tokens:
            if (
                token.lemma_ == actor.root.lemma_
                and token.pos == PROPN
                and is_nominative(token)
            ):
                return actor.doc[token.i : token.i + 1]
        return None

    def find_by_stem(tokens):
        if not stem:
            return None
        for token in tokens:
            if (
                token.pos == PROPN
                and is_nominative(token)
                and token.text.lower().startswith(stem)
            ):
                return actor.doc[token.i : token.i + 1]
        return None

    if force:
        for finder in (find_by_lemma, find_by_stem):
            for tokens in (reversed(actor.doc[: actor.start]), actor.doc[actor.end :]):
                if found := finder(tokens):
                    return found

    for prev in reversed(list(play.actors)):
        if not prev or prev.root.lemma_ != actor.root.lemma_:
            continue
        if prev.root.pos == PROPN and is_nominative(prev.root):
            return prev

    for finder in (find_by_lemma, find_by_stem):
        for tokens in (reversed(actor.doc[: actor.start]), actor.doc[actor.end :]):
            if found := finder(tokens):
                return found
    return actor


def resolve_noun_case(play: Play, actor: Span) -> Span:
    if (
        not actor
        or actor.root.pos != NOUN
        or ANIMACY_ANIM not in actor.root.morph
        or not is_generic_person_noun(actor)
    ):
        return actor
    for prev in reversed(list(play.actors)):
        if (
            prev
            and prev.root.lemma_ == actor.root.lemma_
            and prev.root.pos == NOUN
            and is_oblique(prev.root)
        ):
            return prev
    for token in reversed(actor.doc[: actor.start]):
        if (
            token.lemma_ == actor.root.lemma_
            and token.pos == NOUN
            and is_oblique(token)
        ):
            return refined_noun_chunk(token)
    return actor


def recent_named_actor(play: Play) -> Optional[Span]:
    for actor in reversed(list(play.actors)):
        if actor and any(t.pos == PROPN or t.ent_type_ == "PER" for t in actor):
            return actor
    return None


def find_named_antecedent(span: Span, ref: Token) -> Optional[Span]:
    matcher = morph_aligns_with(ref)
    for token in span:
        if token.pos in {PROPN, NOUN, ADJ} and matcher(token):
            chunk = refined_noun_chunk(token)
            if any(t.pos == PROPN or t.ent_type_ == "PER" for t in chunk) or any(
                t.is_title for t in chunk
            ):
                return chunk
    return None


def is_known_actor(play: Play, actor: Optional[Span]) -> bool:
    if not actor:
        return False
    key = actor_key(actor)
    return any(actor_key(a) == key for a in play.actors if a)


def mentions_actor_with_speech_verb(span: Span, actor: Span) -> bool:
    if not actor:
        return False
    actor_lemmas = {t.lemma_ for t in actor if t.is_alpha}
    if not actor_lemmas:
        return False
    for t in span:
        if t.pos == VERB and t._.is_action_verb:
            for child in t.children:
                if "nsubj" in child.dep_ and child.lemma_ in actor_lemmas:
                    return True
    return False


def mentions_actor_with_copula(span: Span, actor: Span) -> bool:
    if not actor:
        return False
    actor_lemmas = {t.lemma_ for t in actor if t.is_alpha}
    if not actor_lemmas:
        return False
    for t in span:
        if "nsubj" in t.dep_ and t.lemma_ in actor_lemmas:
            head = t.head
            if head.pos in {ADJ, NOUN, PROPN} and any(
                ch._.is_copula for ch in head.children
            ):
                return True
    return False


def mentions_actor_as_subject(span: Span, actor: Span) -> bool:
    if not actor:
        return False
    actor_lemmas = {t.lemma_ for t in actor if t.is_alpha}
    if not actor_lemmas:
        return False
    for t in span:
        if "nsubj" in t.dep_ and t.lemma_ in actor_lemmas:
            return True
        if (
            t.dep_ == "appos"
            and t.lemma_ in actor_lemmas
            and "nsubj" in t.head.dep_
        ):
            return True
    return False


def mentions_actor_as_verbal_subject(span: Span, actor: Span) -> bool:
    if not actor:
        return False
    actor_lemmas = {t.lemma_ for t in actor if t.is_alpha}
    if not actor_lemmas:
        return False
    for t in span:
        if "nsubj" in t.dep_ and t.lemma_ in actor_lemmas:
            if t.head.pos == VERB and not t.head._.is_copula:
                return True
    return False


def reference_resolution_context(bounds: List[Span]) -> Generator[Span, None, None]:
    """Yields all the spans between `bounds`, from bottom to the top.
    Each span is split into sentences, if needed.
    """
    if not bounds:
        return
    doc = bounds[0].doc
    bounds = sorted(bounds, key=lambda sp: sp.start, reverse=True) + [doc[0:0]]
    # Read context pieces between bounds
    for r_bound, l_bound in zip(bounds, bounds[1:]):
        if not (bet := trim_non_word(doc[l_bound.end : r_bound.start])):
            continue
        split_idxs = sorted(
            (
                i
                for s in doc.sents
                if bet.start_char <= (i := s.end_char) <= bet.end_char
            ),
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
    replica: Span,
    *,
    ref_chain: Optional[List[Token]] = None,
    resolve_refs: bool = True,
    prefer_recent_actor: bool = False,
) -> Optional[Span]:
    if ref_chain is None:
        ref_chain = []
    ref = ref_chain[-1] if ref_chain else None
    ref_matcher = morph_aligns_with(ref) if ref else lambda _: True

    def finalize_actor(actor: Optional[Span]) -> Optional[Span]:
        if not actor:
            return None
        actor = resolve_role_actor(play, actor, prefer_recent_actor=prefer_recent_actor)
        actor = resolve_named_case(play, actor)
        if (
            prefer_recent_actor
            and actor.root.pos in {NOUN, PROPN}
            and is_oblique(actor.root)
            and not mentions_actor_as_subject(span, actor)
        ):
            return None
        if (
            prefer_recent_actor
            and actor.doc is span.doc
            and span.start <= actor.start
            and actor.end <= span.end
            and not mentions_actor_as_subject(span, actor)
        ):
            return None
        if prefer_recent_actor and not is_human_like(actor):
            allows_role = (
                actor.root.pos in {ADJ, NUM, DET}
                and mentions_actor_as_subject(span, actor)
            )
            allows_name_like = (
                actor.root.pos != PRON
                and actor.root.is_alpha
                and actor.root.is_title
                and len(actor.root.text) > 1
                and mentions_actor_as_subject(span, actor)
            )
            if not allows_role and not allows_name_like and not (
                mentions_actor_with_speech_verb(span, actor)
                or mentions_actor_with_copula(span, actor)
            ):
                return None
        if is_pronoun_span(actor):
            if prefer_recent_actor and (above := line_above(replica)):
                if named := find_named_antecedent(above, actor.root):
                    return resolve_named_case(play, named, force=True)
            if resolved := resolve_recent_actor(play, actor.root):
                return resolve_named_case(play, resolved, force=True)
            if not is_nominative(actor.root):
                return None
        base_actor = actor
        actor = resolve_noun_case(play, actor)
        if prefer_recent_actor and is_generic_person_noun(base_actor) and actor == base_actor:
            candidate = None
            if play.last_replica and line_breaks_between(play.last_replica, replica) == 1:
                candidate = play.penult()
            if not candidate:
                candidate = play.last_actor
                if candidate and not any(
                    t.pos == PROPN or t.ent_type_ == "PER" for t in candidate
                ):
                    candidate = recent_named_actor(play) or candidate
            if (
                candidate
                and (above := line_above(replica))
                and any(t.lemma_ == base_actor.root.lemma_ for t in above)
                and (penult := play.penult())
                and actor_key(penult) != actor_key(candidate)
            ):
                candidate = penult
            if (
                candidate
                and not is_ref(candidate)
                and not is_vague_actor(candidate)
                and not is_generic_person_noun(candidate)
            ):
                actor = candidate
        return actor

    # Pick all the matching actor candidates from the search [span]
    root_verbs = top_verbs(span, replica)
    candidates = list(flatten(potential_actors(rv, replica) for rv in root_verbs))

    if (roots := [r for r in span if r.dep_ == "ROOT"]) and (
        all("Gender=Neut" in r.morph for r in roots)
    ):
        # Neutral-gender root verb usually indicates
        # a description of situation, not actor denotation.
        matching = []
    else:
        # Order of candidates is better to be preserved.
        matching = list(filter(ref_matcher, candidates))
        if any(not is_reflexive_pronoun(m) for m in matching):
            matching = [m for m in matching if not is_reflexive_pronoun(m)]
    if prefer_recent_actor:
        matching = [
            m
            for m in matching
            if not (
                m.pos == PRON
                and "nsubj" in m.dep_
                and m.head.dep_ not in {"ROOT", "parataxis"}
            )
        ]
    has_strong_non_ref = any(not is_ref(m) and "nsubj" in m.dep_ for m in matching)
    pronoun_subjects = [
        m
        for m in matching
        if m.pos == PRON
        and "nsubj" in m.dep_
        and m.head.dep_ in {"ROOT", "parataxis"}
    ]
    if pronoun_subjects:
        non_ref_spans = [refined_noun_chunk(m) for m in matching if not is_ref(m)]
        if non_ref_spans and all(
            not span_has_animate_noun(s)
            and not any(t.pos == PROPN or t.ent_type_ == "PER" for t in s)
            for s in non_ref_spans
        ):
            return finalize_actor(refined_noun_chunk(pronoun_subjects[0]))
        if not non_ref_spans or all(not is_human_like(s) for s in non_ref_spans):
            if prefer_recent_actor and (above := line_above(replica)):
                if named := find_named_antecedent(above, pronoun_subjects[0]):
                    return resolve_named_case(play, named, force=True)
            resolved = resolve_recent_actor(play, pronoun_subjects[0])
            if resolved and (
                is_human_like(resolved)
                or any(t.pos == PROPN or t.ent_type_ == "PER" for t in resolved)
            ):
                return resolve_named_case(play, resolved, force=True)

    def exact_enough(n):
        return not is_ref(n) and (
            "nsubj" in n.dep_ or n.ent_iob_ in "IB" or n.ent_type_ == "PER"
        )

    exacts = {noun_chunk(m).root.lemma_: m for m in matching if exact_enough(m)}
    if len(exacts) == 1:
        m = exacts[next(iter(exacts))]
        actor = refined_noun_chunk(m)
        if actor == play.last_actor:
            if contiguous(play.last_replica, replica):
                # Discard repeated actor
                matching.remove(m)
        # Ensures that no other matches closer to the replica are missed
        elif all(is_ref(c) for c in candidates[: candidates.index(m)]):
            # Single matching non-ref candidate is good enough
            return finalize_actor(actor)

    # Reference resolution
    if resolve_refs:
        m_refs = [] if (has_strong_non_ref and not ref) else list(filter(is_ref, matching))
        m_verbs = list(filter(ref_matcher, root_verbs))
        person = m_refs[0].morph.get("Person") if m_refs else []
        if (
            not ref
            and len(m_refs) == 1
            and play.last_replica
            and line_breaks_between(play.last_replica, replica) == 1
            and (alt := play.penult())
            and ("First" in person or "Second" in person)
        ):
            ref_chain.extend(m_refs)
            return alt
        if m_refs:
            ref_roots = m_refs + m_verbs  # Verbs improve the ref resolution
        else:
            if any(not is_ref(m) for m in matching):
                ref_roots = []
            else:
                # Pick only verbs which are distinct from the found actors
                # (this is empty most of the time).
                m_verbs = [
                    v
                    for v in m_verbs
                    if set(v.children).isdisjoint(matching)
                    and "Gender=Neut" not in v.morph
                ]
                ref_roots = m_verbs

        search_ctx = reference_resolution_context(
            # Spans between which the antecedent will be searched
            list(play.replicas)
            + ([] if ref else ([replica] if replica else []))
            + [span]
        )
        cached_ctx: List[Span] = []
        for word in ref_roots:
            if bound := play.reference(word):
                return bound
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
                ref_chain.append(word)
                ante = actor_search(region, play, replica, ref_chain=ref_chain)
                if ante and ante.start != word.i:
                    return finalize_actor(refined_noun_chunk(ante))
                else:
                    ref_chain.pop()
            else:
                if ref and ante and ref.lemma_ == ante.lemma_:
                    return finalize_actor(refined_noun_chunk(ante))

    if len(matching) > 0:
        if not ref:
            if picked := best_candidate(matching):
                return finalize_actor(picked)
        elif (non_ref := [m for m in matching if not is_ref(m)]) and (
            (first := non_ref[0]) and (actor := best_candidate(non_ref))
        ):
            if actor == play.last_actor and (
                line_breaks_between(first, ref_chain[0]) < 2
            ):
                pass
            else:
                return finalize_actor(actor)

    return finalize_actor(refined_noun_chunk(ref)) if ref else None


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

        # Non-first replica fills line
        elif (
            p_replica
            and fills_line(replica)
            and line_breaks_between(replica, p_replica) == 1
        ):
            if p_replica._.is_after_author_starting and p_replica in p and has_voice_intro(
                p_replica
            ):
                p[replica] = p[p_replica]
                continue
            if (
                p_replica
                and p_replica in p
                and (above := line_above(replica))
                and not (
                    p_replica._.is_before_author_insertion
                    or p_replica._.is_after_author_starting
                )
                and not (
                    p_replica._.start_line_no == above._.start_line_no
                    and p_replica._.end_line_no == above._.end_line_no
                    and above.start == p_replica.start
                    and above.end == p_replica.end
                )
                and not (p_replica.text.strip().endswith("?") and is_brief_reply(replica))
                and p[p_replica]
                and any(t.pos == PROPN or t.ent_type_ == "PER" for t in p[p_replica])
                and any(t.pos == PROPN or t.ent_type_ == "PER" for t in above)
                and sum(t.is_alpha for t in above) > 50
                and mentions_actor_with_speech_verb(above, p[p_replica])
            ):
                p[replica] = p[p_replica]
                continue
            if penult := p.penult():
                # Line has no author speech => speakers alternation
                actor = penult
                if (
                    p_replica
                    and has_first_person(replica)
                    and has_second_person(p_replica)
                    and is_indefinite_actor(penult)
                    and (named := recent_named_actor(p))
                    and actor_key(named) != actor_key(p[p_replica])
                ):
                    actor = named
                p[replica] = actor
            elif p_replica.start > 1:
                # [replica] is second in play; [p_replica] & [replica] are contiguous
                # => search for the speaker above the [p_replica].
                if above := line_above(p_replica):
                    if not (
                        p_replica._.start_line_no == above._.start_line_no
                        and p_replica._.end_line_no == above._.end_line_no
                    ):
                        if actor := actor_search(
                            above,
                            p,
                            replica,
                            ref_chain=ref_chain,
                            prefer_recent_actor=True,
                        ):
                            p[replica] = actor
                            continue
                leading = doc[
                    min(p_replica.start, p_replica.sent.start) - 2 : p_replica.start
                ]
                search_span = trim_non_word(expand_line_start(leading))
                actor = None
                if sents := list(search_span.sents):
                    for sent in reversed(sents):
                        clipped = search_span.doc[
                            max(sent.start, search_span.start) : min(
                                sent.end, search_span.end
                            )
                        ]
                        if not clipped:
                            continue
                        actor = actor_search(
                            clipped,
                            p,
                            replica,
                            ref_chain=ref_chain,
                            prefer_recent_actor=True,
                        )
                        if actor:
                            break
                    if not actor and len(sents) > 1:
                        actor = actor_search(
                            search_span,
                            p,
                            replica,
                            ref_chain=ref_chain,
                            prefer_recent_actor=True,
                        )
                else:
                    actor = actor_search(
                        search_span,
                        p,
                        replica,
                        ref_chain=ref_chain,
                        prefer_recent_actor=True,
                    )
                p[replica] = (actor, ref_chain)

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

            actor = None
            fallback = None
            if sents := list(search_span.sents):
                for sent in reversed(sents):
                    clipped = search_span.doc[
                        max(sent.start, search_span.start) : min(
                            sent.end, search_span.end
                        )
                    ]
                    if not clipped:
                        continue
                    candidate = actor_search(
                        clipped, p, replica, ref_chain=ref_chain, prefer_recent_actor=True
                    )
                    if candidate and (
                        is_human_like(candidate)
                        or mentions_actor_with_speech_verb(clipped, candidate)
                        or mentions_actor_with_copula(clipped, candidate)
                        or is_known_actor(p, candidate)
                        or (
                            candidate.root.pos in {ADJ, NUM, DET}
                            and mentions_actor_as_subject(clipped, candidate)
                        )
                    ):
                        if any(
                            t.pos == PROPN or t.ent_type_ == "PER" for t in clipped
                        ):
                            actor = candidate
                            break
                        if fallback is None:
                            fallback = candidate
                if not actor and len(sents) > 1:
                    if any(
                        t.pos == PROPN or t.ent_type_ == "PER" for t in search_span
                    ):
                        actor = actor_search(search_span, p, replica, ref_chain=ref_chain)
                    else:
                        actor = fallback
            else:
                actor = actor_search(
                    search_span, p, replica, ref_chain=ref_chain, prefer_recent_actor=True
                )

            p[replica] = (actor, ref_chain)

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
            if len(sents := list(search_span.sents)) > 1:
                search_span = search_span.doc[
                    max(sents[0].start, search_span.start) : min(
                        sents[0].end, search_span.end
                    )
                ]

            prev_penult = p.penult()
            prev_actor = p[p_replica] if (p_replica and p_replica in p) else None
            p[replica] = (
                (actor := actor_search(
                    search_span,
                    p,
                    replica,
                    ref_chain=ref_chain,
                    prefer_recent_actor=True,
                )),
                ref_chain,
            )
            if (
                actor
                and prev_actor
                and has_first_person(replica)
                and has_second_person(p_replica)
                and prev_penult
                and actor_key(prev_penult) != actor_key(prev_actor)
                and actor.root.pos not in {ADJ, NUM, DET}
                and any(
                    t.pos == PROPN or t.ent_type_ == "PER" for t in prev_penult
                )
                and not any(t.pos == PROPN or t.ent_type_ == "PER" for t in actor)
            ):
                p[replica] = prev_penult
            if not p[replica]:
                if (above := line_above(replica)) and not (
                    p_replica
                    and p_replica._.start_line_no == above._.start_line_no
                    and p_replica._.end_line_no == above._.end_line_no
                ):
                    if candidate := actor_search(
                        above, p, replica, ref_chain=ref_chain, prefer_recent_actor=True
                    ):
                        if (
                            is_human_like(candidate)
                            or mentions_actor_with_speech_verb(above, candidate)
                            or mentions_actor_as_verbal_subject(above, candidate)
                        ):
                            p[replica] = candidate
                if not p[replica] and prev_penult:
                    # Author speech is present, but it has
                    # no reference to the actor => actor alternation
                    p[replica] = prev_penult

        # Author insertion
        elif replica._.is_before_author_insertion and n_replica:
            search_span = doc[replica.end : n_replica.start]
            if is_parenthesized(search_span):
                # Author is commenting on a situation, there should be
                # no reference to the actor.
                search_span = doc[replica.end : replica.end]

            search_span = trim_non_word(search_span)

            prev_penult = p.penult()
            prev_actor = p[p_replica] if (p_replica and p_replica in p) else None
            p[replica] = (
                (actor := actor_search(
                    search_span,
                    p,
                    replica,
                    ref_chain=ref_chain,
                    prefer_recent_actor=True,
                )),
                ref_chain,
            )
            if (
                actor
                and prev_actor
                and has_first_person(replica)
                and has_second_person(p_replica)
                and prev_penult
                and actor_key(prev_penult) != actor_key(prev_actor)
                and actor.root.pos not in {ADJ, NUM, DET}
                and any(
                    t.pos == PROPN or t.ent_type_ == "PER" for t in prev_penult
                )
                and not any(t.pos == PROPN or t.ent_type_ == "PER" for t in actor)
            ):
                p[replica] = prev_penult
            if not p[replica]:
                if (above := line_above(replica)) and not (
                    p_replica
                    and p_replica._.start_line_no == above._.start_line_no
                    and p_replica._.end_line_no == above._.end_line_no
                ):
                    if candidate := actor_search(
                        above, p, replica, ref_chain=ref_chain, prefer_recent_actor=True
                    ):
                        if (
                            is_human_like(candidate)
                            or mentions_actor_with_speech_verb(above, candidate)
                            or mentions_actor_as_verbal_subject(above, candidate)
                        ):
                            p[replica] = candidate
                if not p[replica] and prev_penult:
                    # Author speech is present, but it has
                    # no reference to the actor => actor alternation
                    p[replica] = prev_penult

        # Fallback, similar to ( ... [:]), but
        # constrained to a single line between replicas
        elif (
            fills_line(replica)
            and p_replica
            and line_breaks_between(replica, p_replica) == 2
            and (
                (search_span := line_above(replica))
                and sum(not t.is_stop and not t.is_punct for t in search_span) >= 5
            )
        ):
            # TODO: check for appeal in the replica text, then fallback on actor_search.

            # Start with the sentence nearest to the replica
            full_search_span = search_span
            if len(sents := list(search_span.sents)) > 1:
                search_span = sents[-1]

            actor = actor_search(
                search_span, p, replica, prefer_recent_actor=True
            )
            full_span_has_animate = any(
                t.pos == NOUN and ANIMACY_ANIM in t.morph for t in full_search_span
            )
            needs_rescan = actor and (
                (
                    not is_human_like(actor)
                    and not (
                        mentions_actor_with_speech_verb(full_search_span, actor)
                        or mentions_actor_with_copula(full_search_span, actor)
                    )
                )
                or (full_span_has_animate and not span_has_animate_noun(actor))
            )
            if needs_rescan and (sents := list(full_search_span.sents)):
                fallback = None
                for sent in reversed(sents):
                    clipped = full_search_span.doc[
                        max(sent.start, full_search_span.start) : min(
                            sent.end, full_search_span.end
                        )
                    ]
                    if not clipped:
                        continue
                    candidate = actor_search(
                        clipped, p, replica, prefer_recent_actor=True
                    )
                    if candidate and (
                        is_human_like(candidate)
                        or mentions_actor_with_speech_verb(clipped, candidate)
                        or mentions_actor_with_copula(clipped, candidate)
                        or is_known_actor(p, candidate)
                    ):
                        if full_span_has_animate and not span_has_animate_noun(candidate):
                            if fallback is None:
                                fallback = candidate
                            continue
                        actor = candidate
                        break
                if not actor and fallback:
                    actor = fallback
            if (
                actor
                and p_replica
                and p_replica in p
                and (prev_actor := p[p_replica])
                and actor_key(actor) != actor_key(prev_actor)
                and mentions_actor_with_speech_verb(search_span, prev_actor)
            ):
                actor = prev_actor
            if (
                actor
                and p_replica
                and p_replica in p
                and (prev_actor := p[p_replica])
                and actor_key(actor) == actor_key(prev_actor)
                and not mentions_actor_with_speech_verb(search_span, prev_actor)
                and (penult := p.penult())
            ):
                actor = penult
            if not actor and replica._.is_unannotated_alternation and (penult := p.penult()):
                actor = penult
            if actor and span_is_collective(actor) and (penult := p.penult()):
                actor = penult
            p[replica] = actor

        # Fallback - repeat actor from prev replica
        elif fills_line(replica) and p_replica and p_replica in p:
            p[replica] = p[p_replica]  # <=> previous actor

        else:
            p[replica] = None
            print("MISS", replica)  # TODO: handle

    return p
