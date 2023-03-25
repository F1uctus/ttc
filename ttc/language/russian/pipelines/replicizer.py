from collections import deque
from typing import Literal, Callable, Optional, List, Dict, Set, Deque, Final, Any

from spacy import Language
from spacy.matcher import Matcher, DependencyMatcher
from spacy.symbols import NOUN, PRON, PROPN, parataxis  # type: ignore
from spacy.tokens import Token, Span, Doc

from ttc.language.russian.dependency_patterns import (
    SPEAKING_VERB_TO_SPEAKER,
    SPEAKING_VERB_CONJUNCT_SPEAKER,
)
from ttc.language.russian.span_extensions import (
    is_before_author_insertion,
    is_before_author_ending,
    is_after_author_starting,
    is_unannotated_alternation,
)
from ttc.language.russian.token_extensions import (
    has_newline,
    is_open_quote,
    is_close_quote,
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
    replicas: Final[List[Span]] = []
    tokens: Final[List[Token]] = []

    dep_matcher = DependencyMatcher(language.vocab)
    dep_matcher.add("*", [SPEAKING_VERB_TO_SPEAKER])
    dep_matcher.add("**", [SPEAKING_VERB_CONJUNCT_SPEAKER])

    def flush_replica(*tags: Callable[[Span], Any]):
        if tokens:
            replica_span = doc[tokens[0].i : tokens[-1].i + 1]
            for tag in tags:
                replica_span._.set(tag.__name__, True)
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
                match: Span = next(iter(results), None)  # type: ignore
                if match:
                    if any(dep_matcher(match)):
                        flush_replica(is_before_author_insertion)
                        # skip to the end of author insertion
                        ti = match[:-2].end
                        continue
                    if depends_on(match, phrase):
                        continue
                    flush_replica(is_before_author_insertion)
                    # skip to the end of author insertion
                    ti = match[:-2].end
                    continue

                # checking for author ending
                results = matchers["AUTHOR_ENDING"](
                    doc[pt.i : line_end_i + 1], as_spans=True
                )
                for match in results:
                    match: Span = match  # type: ignore
                    # may be an interrogative or exclamatory ending of a speech
                    # e.g. После всего этого, — что ты еще сказал?
                    if match[0] != pt and match[0].is_punct:
                        tokens.append(t)
                        break
                    if any(dep_matcher(match)):
                        flush_replica(is_before_author_ending)
                        states.append("author")
                        # skip to the end of author ending
                        ti = match.end - 1
                        break
                    if depends_on(match, phrase):
                        continue
                    # - 1 is a line break offset
                    if match.end >= doc_length - 1 or has_newline(doc[match.end - 1]):
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

        elif is_open_quote(t):
            # "Реплика" — автор
            states.append("replica_by_quote")

        elif state == "author_insertion" and pt and pt.is_punct and is_hyphen(t):
            # — автор<punct> — [Рр]еплика
            #                 ^
            # return to the state preceding the author insertion
            states.append(states[-2])

    if "replica" in states[-1]:
        flush_replica()  # may have a trailing replica

    return replicas
