from collections import deque
from typing import Literal, Callable, Optional, List, Dict, Set, Deque, Final, Any

from spacy import Language
from spacy.matcher import Matcher, DependencyMatcher
from spacy.symbols import NOUN, PRON, PROPN, VERB, AUX, parataxis  # type: ignore
from spacy.tokens import Token, Span, Doc

from ttc.language.common.span_extensions import (
    is_before_author_insertion,
    is_before_author_ending,
    is_after_author_starting,
    is_unannotated_alternation,
    trim_non_word,
)
from ttc.language.common.token_extensions import (
    is_open_quote,
    is_close_quote,
    has_newline,
)
from ttc.language.russian.dependency_patterns import (
    ACTION_VERB_TO_ACTOR,
    ACTION_VERB_CONJUNCT_ACTOR,
)
from ttc.language.russian.token_extensions import (
    is_hyphen,
)
from ttc.language.russian.token_patterns import TokenMatcherClass


def depends_on(match: Span, phrase: Set[Token]):
    """
    Checks if some word in a match is a semantic children of a phrase.
    That is a sign of a complex sentence construct containing a hyphen,
    so that match does not belong to an author speech.
    """
    return any(
        ancs <= phrase
        for t in match
        if t.is_alpha and (ancs := set(t.ancestors)) and t.dep != parataxis
    )


def extract_replicas(
    doc: Doc,
    language: Language,
    matchers: Dict[TokenMatcherClass, Matcher],
) -> List[Span]:
    """
    Extract replicas from a Doc object.
    Replicas are speech elements in a text that are often attributed to specific characters.
    The function uses a state machine to process the text and identify the replicas.
    """
    replicas: List[Span] = []
    tokens: Final[List[Token]] = []

    dep_matcher = DependencyMatcher(language.vocab)
    dep_matcher.add("*", [ACTION_VERB_TO_ACTOR])
    dep_matcher.add("**", [ACTION_VERB_CONJUNCT_ACTOR])

    SENTENCE_END_PUNCT = {".", "!", "?", "…", "..."}
    MAX_AUTHOR_TOKENS = 30

    def has_explicit_subject(match: Span) -> bool:
        sample = match
        if len(match) > 2 and is_hyphen(match[-2]) and match[-1].is_title:
            sample = match[:-2]
        sample = trim_non_word(sample)
        return any(
            "nsubj" in t.dep_ and t.pos in (NOUN, PROPN, PRON) for t in sample
        )

    def is_author_annotation(match: Span) -> bool:
        sample = match
        if len(match) > 2 and is_hyphen(match[-2]) and match[-1].is_title:
            sample = match[:-2]

        sample = trim_non_word(sample)
        if not sample:
            return False

        if any(dep_matcher(sample)):
            return True

        alpha_tokens = [t for t in sample if t.is_alpha]
        if not alpha_tokens:
            return False

        if any(
            p in ("First", "Second")
            for t in alpha_tokens
            for p in t.morph.get("Person")
        ):
            return False

        if len(alpha_tokens) > MAX_AUTHOR_TOKENS:
            return False

        if not any(t.pos in (VERB, AUX) for t in sample):
            return False

        if not any(t.pos in (NOUN, PROPN, PRON) for t in sample):
            return False

        return True

    def flush_replica(*tags: Callable[[Span], Any]):
        if tokens:
            replica_span = doc[tokens[0].i : tokens[-1].i + 1]
            for tag in tags:
                replica_span._.set(getattr(tag, "__name__", str(tag)), True)
            replicas.append(replica_span)
            tokens.clear()

    states: Deque[
        Literal[
            "author",
            "author_insertion",
            "replica_by_newline_and_hyphen",
            "replica_by_colon_and_quote",
            "replica_by_quote",  # Can be a mistreated author's speech
        ]
    ] = deque(maxlen=3)
    states.append("author")

    ti = -1  # token index
    doc_length = len(doc)
    while ti + 1 < doc_length:
        ti += 1

        pt: Optional[Token] = doc[ti - 1] if ti > 0 else None
        nt: Optional[Token] = doc[ti + 1] if ti + 1 < doc_length else None
        nnt: Optional[Token] = doc[ti + 2] if ti + 2 < doc_length else None
        t: Token = doc[ti]

        state = states[-1]

        if state == "replica_by_quote":
            if is_close_quote(t):
                if has_newline(t) or (nt and has_newline(nt)):
                    # "\n OR ".\n
                    flush_replica()
                elif nt and (is_hyphen(nt) or (nt.is_punct and nnt and is_hyphen(nnt))):
                    # " — OR ". —
                    flush_replica(is_before_author_ending)
                else:
                    # skip, just a quoted author speech
                    tokens.clear()
                states.append("author")
            elif pt and pt.is_punct and is_hyphen(t):
                flush_replica(is_before_author_insertion)
                states.append("author_insertion")
            else:
                tokens.append(t)

        elif state == "replica_by_colon_and_quote":
            if is_close_quote(t):
                flush_replica(is_after_author_starting)
                states.append("author")
            else:
                tokens.append(t)

        elif state == "replica_by_newline_and_hyphen":
            if has_newline(t):
                tokens.append(t)
                if doc[tokens[0].i - 2].text == ":":
                    flush_replica(is_after_author_starting)
                else:
                    flush_replica(is_unannotated_alternation)
                states.append("author")

            elif pt and pt.is_punct and is_hyphen(t):
                # (+) onomatopoeia
                # — Все тихо, и вдруг — бам-бам-бам! — заколотили в дверь...
                # — Из кустов — кря! — ...
                # (-) not an onomatopoeia
                # — То — дедам! — Сказал ...
                if (
                    pt.text in ("!",)
                    and nt
                    and not nt.is_title
                    and len(tokens) > 2
                    and (
                        # found a — [word] [punct] — construct
                        is_hyphen(tokens[-3])
                        # where [word] is not a noun
                        and tokens[-2].pos not in (NOUN, PRON, PROPN)
                        or
                        # found a repetitive sequence (x-x)
                        is_hyphen(tokens[-3])
                        and tokens[-2].lemma_ == tokens[-4].lemma_
                    )
                ):
                    tokens.append(t)
                    continue

                phrase = {t for t in tokens if t.is_alpha}

                # checking for author insertion
                line_end_i = doc_length
                for i, token in enumerate(doc[pt.i :], pt.i):
                    if has_newline(token) or token.i == doc_length - 1:
                        line_end_i = i
                        break
                results = matchers["AUTHOR_INSERTION"](
                    doc[pt.i : line_end_i + 1], as_spans=True
                )
                match: Optional[Span] = None
                for m in results:
                    if m.start != pt.i:
                        continue
                    if not is_author_annotation(m):
                        continue
                    if depends_on(m, phrase) and not (
                        (pt and pt.text in SENTENCE_END_PUNCT)
                        or has_explicit_subject(m)
                    ):
                        continue
                    match = m
                    break
                if match:
                    flush_replica(is_before_author_insertion)
                    # skip to the end of author insertion
                    ti = match[:-2].end
                    continue

                # checking for author ending
                results = matchers["AUTHOR_ENDING"](
                    doc[pt.i : line_end_i + 1], as_spans=True
                )
                for match in results:
                    match: Span = match
                    if match.start != pt.i:
                        continue
                    # may be an interrogative or exclamatory ending of a speech
                    # e.g. После всего этого, — что ты еще сказал?
                    if match[0] != pt and match[0].is_punct:
                        tokens.append(t)
                        break
                    if depends_on(match, phrase) and not (
                        (pt and pt.text in SENTENCE_END_PUNCT)
                        or has_explicit_subject(match)
                    ):
                        continue
                    if is_author_annotation(match):
                        flush_replica(is_before_author_ending)
                        states.append("author")
                        # skip to the end of author ending
                        ti = match.end - 1
                        break
                    # - 1 is a line break offset
                    if is_author_annotation(match) and (
                        match.end >= doc_length - 1
                        or has_newline(doc[match.end - 1])
                    ):
                        flush_replica(is_before_author_ending)
                        states.append("author")
                        # skip to the end of author ending
                        ti = match.end - 1
                        break
                else:
                    tokens.append(t)

            else:
                tokens.append(t)

        # author* -> replica* transitions
        elif (pt is None or pt.is_space or has_newline(pt)) and is_hyphen(t):
            # [Автор:]\n— Реплика
            states.append("replica_by_newline_and_hyphen")

        elif pt and ":" in pt.text and is_open_quote(t):
            # Автор: [«"]Реплика[»"]
            states.append("replica_by_colon_and_quote")

        elif (not pt or pt.is_punct) and is_open_quote(t):
            # "Реплика" — автор
            states.append("replica_by_quote")

        elif state == "author_insertion" and pt and pt.is_punct and is_hyphen(t):
            # — автор<punct> — [Рр]еплика
            #                 ^
            # return to the state preceding the author insertion
            states.append(states[-2])

    if "replica" in states[-1]:
        flush_replica()  # may have a trailing replica

    def copy_replica_tags(source: Span, target: Span) -> None:
        for tag in (
            is_before_author_insertion,
            is_before_author_ending,
            is_after_author_starting,
            is_unannotated_alternation,
        ):
            if getattr(source._, tag.__name__, False):
                target._.set(tag.__name__, True)

    def extend_interrogative_tail(replica: Span) -> Span:
        if not replica or replica.end >= len(doc):
            return replica
        if replica[-1].text != "," or not is_hyphen(doc[replica.end]):
            return replica
        idx = replica.end + 1
        steps = 0
        has_second_person = False
        has_question = False
        tail_end = None
        while idx < len(doc):
            token = doc[idx]
            steps += 1
            if steps > 15:
                break
            if has_newline(token):
                break
            if "Second" in token.morph.get("Person"):
                has_second_person = True
            if token.text in {"?", "!"}:
                has_question = True
            if is_hyphen(token) and has_question:
                tail_end = token.i
                break
            idx += 1
        if tail_end and has_second_person:
            extended = doc[replica.start : tail_end]
            copy_replica_tags(replica, extended)
            return extended
        return replica

    replicas = [extend_interrogative_tail(replica) for replica in replicas]

    return replicas
