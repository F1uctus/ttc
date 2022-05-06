from collections import deque
from typing import Literal, Callable, Optional, Union, List, Dict, Tuple, Deque

from spacy.matcher import Matcher
from spacy.tokens import Token, Span, Doc

from ttc.language.russian.token_patterns import TokenMatcherClass


def next_matching(
    doc_like: Union[Span, Doc],
    predicate: Callable[[Token], bool],
    *,
    start: int = 0,
    step: int = 1,
) -> Tuple[Optional[Token], int]:
    delta = 0
    sub_piece = None
    while 0 <= start + delta < len(doc_like):
        checked = doc_like[start + delta]
        if predicate(checked):
            sub_piece = checked
            break
        delta += step
    return sub_piece, start + delta


def extract_replicas(
    doc: Doc,
    matchers: Dict[TokenMatcherClass, Matcher],
) -> List[Span]:
    replicas: List[Span] = []
    tokens: List[Token] = []

    def flush_replica():
        nonlocal tokens
        if len(tokens) > 0:
            replicas.append(doc[tokens[0].i : tokens[-1].i + 1])
            tokens = []

    states: Deque[
        Literal[
            "author",
            "author_insertion",
            "replica_hyphen_after_newline",
            "replica_quote_after_colon",
            "replica_quote_before_hyphen",
        ]
    ] = deque(maxlen=3)
    states.append("author")

    ti = -1  # token index
    doc_length = len(doc)
    while ti + 1 < doc_length:
        ti += 1

        pt: Optional[Token] = doc[ti - 1] if ti - 1 >= 0 else None
        nt: Optional[Token] = doc[ti + 1] if ti + 1 < len(doc) else None
        t: Token = doc[ti]

        state = states[-1]

        if state == "replica_quote_before_hyphen":
            if t._.is_close_quote:
                if t._.has_newline or (nt and nt._.is_hyphen):
                    flush_replica()
                else:
                    # skip, just a quoted author speech
                    tokens = []
                states.append("author")
            elif pt and pt.is_punct and t._.is_hyphen:
                flush_replica()
                states.append("author_insertion")
            else:
                tokens.append(t)

        elif state == "replica_quote_after_colon":
            if t._.is_close_quote:
                flush_replica()
                states.append("author")
            else:
                tokens.append(t)

        elif state == "replica_hyphen_after_newline":
            if t._.has_newline:
                flush_replica()
                states.append("author")
            elif pt and pt.is_punct and t._.is_hyphen:
                # checking for author insertion
                par_end_idx = next_matching(  # paragraph ending index
                    doc,
                    lambda x: x._.has_newline or x.i == len(doc) - 1,
                    start=pt.i,
                )[1]
                results = [
                    match
                    for match in matchers["AUTHOR_INSERTION"](
                        doc[pt.i : par_end_idx + 1], as_spans=True
                    )
                ]
                if len(results) > 0:
                    flush_replica()
                    # skip to the end of author insertion
                    ti = results[0][:-2].end
                elif pt.text not in (",", ";"):
                    # certainly not a complex sentence construct
                    flush_replica()
                    states.append("author_insertion")
                else:
                    # checking for author ending
                    results = [
                        match
                        for match in matchers["AUTHOR_ENDING"](
                            doc[pt.i : par_end_idx + 1], as_spans=True
                        )
                    ]
                    for match in results:
                        # - 1 is a line break offset
                        if (
                            match.end >= len(doc) - 1
                            or doc[match.end - 1]._.has_newline
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
        elif (pt is None or pt._.has_newline) and t._.is_hyphen:
            # [Автор:]\n— Реплика
            states.append("replica_hyphen_after_newline")

        elif pt and ":" in pt.text and t._.is_open_quote:
            # Автор: "Реплика" | Автор: «Реплика»
            states.append("replica_quote_after_colon")

        elif t._.is_open_quote:
            # "Реплика" — автор
            states.append("replica_quote_before_hyphen")

        elif state == "author_insertion" and pt and pt.is_punct and t._.is_hyphen:
            # — автор<punct> — [Рр]еплика
            #                 ^
            # return to the state preceding the author insertion
            states.append(states[-2])

    if "replica" in states[-1]:
        flush_replica()  # may have a trailing replica

    return replicas
