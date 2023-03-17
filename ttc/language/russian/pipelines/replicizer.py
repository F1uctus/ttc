from collections import deque
from typing import Literal, Callable, Optional, Union, List, Dict, Tuple, Deque, Final

from spacy.matcher import Matcher
from spacy.symbols import NOUN, PRON, PROPN  # type: ignore
from spacy.tokens import Token, Span, Doc

from ttc.language.russian.token_extensions import (
    has_newline,
    is_open_quote,
    is_close_quote,
    is_hyphen,
)
from ttc.language.russian.token_patterns import TokenMatcherClass


def next_matching(
    doc_like: Union[Span, Doc],
    predicate: Callable[[Token], bool],
    *,
    start: int = 0,
) -> Tuple[Optional[Token], int]:
    for i, token in enumerate(doc_like[start:], start):
        if predicate(token):
            return token, i
    return None, len(doc_like)


def extract_replicas(
    doc: Doc,
    matchers: Dict[TokenMatcherClass, Matcher],
) -> List[Span]:
    """
    Extract replicas from a Doc object.
    Replicas are speech elements in a text that are often attributed to specific characters.
    The function uses a state machine to process the text and identify the replicas.
    """
    replicas: Final[List[Span]] = []
    tokens: Final[List[Token]] = []

    def flush_replica():
        if tokens:
            replicas.append(doc[tokens[0].i : tokens[-1].i + 1])
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

        if t.is_space and not has_newline(t):
            if state.startswith("replica"):
                tokens.append(t)
            else:
                continue

        if state == "replica_by_quote":
            if is_close_quote(t):
                if has_newline(t) or (
                    nt
                    and (
                        (is_hyphen(nt) or has_newline(nt))
                        or (nt.is_punct and nnt and is_hyphen(nnt))
                    )
                ):
                    # matches all below cases:
                    # "\n
                    # " —
                    # ".\n
                    # ". —
                    flush_replica()
                else:
                    # skip, just a quoted author speech
                    tokens.clear()
                states.append("author")
            elif pt and pt.is_punct and is_hyphen(t):
                flush_replica()
                states.append("author_insertion")
            else:
                tokens.append(t)

        elif state == "replica_by_colon_and_quote":
            if is_close_quote(t):
                flush_replica()
                states.append("author")
            else:
                tokens.append(t)

        elif state == "replica_by_newline_and_hyphen":
            if has_newline(t):
                tokens.append(t)
                flush_replica()
                states.append("author")
            elif pt and pt.is_punct and is_hyphen(t):
                # checking for author insertion
                par_end_idx = next_matching(  # paragraph ending index
                    doc,
                    lambda x: has_newline(x) or x.i == doc_length - 1,
                    start=pt.i,
                )[1]
                results = matchers["AUTHOR_INSERTION"](
                    doc[pt.i : par_end_idx + 1], as_spans=True
                )
                if len(results) > 0:
                    flush_replica()
                    # skip to the end of author insertion
                    ti = results[0][:-2].end
                else:
                    # (+) Onomatopoeia
                    # — Все тихо, и вдруг — бам-бам-бам! — заколотили в дверь...
                    # — Из кустов — кря! — ...
                    # (-) Not an onomatopoeia
                    # — То — дедам! — сказал ...
                    onomatopoeia = (
                        pt.text in ("!",)
                        and nt
                        and not nt.is_title
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
                    )
                    if pt.text not in (",", ";") and not onomatopoeia:
                        # certainly not a complex sentence construct
                        # and not an onomatopoeia
                        flush_replica()
                        states.append("author_insertion")
                    elif onomatopoeia:
                        tokens.append(t)
                    else:
                        # checking for author ending
                        results = matchers["AUTHOR_ENDING"](
                            doc[pt.i : par_end_idx + 1], as_spans=True
                        )
                        for match in results:
                            # - 1 is a line break offset
                            if match.end >= doc_length - 1 or has_newline(
                                doc[match.end - 1]
                            ):
                                flush_replica()
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
