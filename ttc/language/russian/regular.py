from collections import deque
from typing import Literal, Callable, cast

from spacy import Language
from spacy.matcher import Matcher, DependencyMatcher
from spacy.tokens import Token, Span, Doc

from ttc.iterables import (
    iter_by_triples,
    canonical_int_enumeration,
)
from ttc.language import Speaker, Dialogue
from ttc.language.russian.constants import REFERRAL_PRON
from ttc.language.russian.dependency_patterns import (
    SPEAKER_TO_SPECIAL_VERB,
    SPEAKER_TO_VERB_CONJ_SPECIAL_VERB,
)
from ttc.language.russian.token_checks import morph_equals
from ttc.language.russian.token_patterns import TokenMatcherClass


def next_matching(
    doc_like: Span | Doc,
    predicate: Callable[[Token], bool],
    *,
    start: int = 0,
    step: int = 1
) -> tuple[Token | None, int]:
    delta = 0
    sub_piece = None
    while 0 <= start + delta < len(doc_like):
        checked = doc_like[start + delta]
        if predicate(checked):
            sub_piece = checked
            break
        delta += step
    return sub_piece, start + delta


# noinspection PyProtectedMember
# (spaCy "._." extensions)
def extract_replicas(
    doc: Doc,
    matchers: dict[TokenMatcherClass, Matcher],
) -> list[Span]:
    replicas: list[Span] = []
    tokens: list[Token] = []

    # noinspection PyProtectedMember
    # (spaCy "._." extensions)
    def flush_replica():
        nonlocal tokens
        if len(tokens) > 0:
            replicas.append(doc[tokens[0].i : tokens[-1].i + 1])
            tokens = []

    states: deque[
        Literal[
            "author",
            "author_insertion",
            "replica_dash_after_newline",
            "replica_quote_after_colon",
            "replica_quote_before_dash",
        ]
    ] = deque(maxlen=3)
    states.append("author")

    ti = -1  # token index
    doc_length = len(doc)
    while ti + 1 < doc_length:
        ti += 1

        pt: Token | None = doc[ti - 1] if ti - 1 >= 0 else None
        nt: Token | None = doc[ti + 1] if ti + 1 < len(doc) else None
        t: Token = doc[ti]

        state = states[-1]

        if state == "replica_quote_before_dash":
            if t._.is_close_quote:
                if nt and (nt._.is_dash or nt._.is_newline):
                    flush_replica()
                else:
                    # skip, just a quoted author speech
                    tokens = []
                states.append("author")
            elif pt and pt.is_punct and t._.is_dash:
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

        elif state == "replica_dash_after_newline":
            if t._.is_newline:
                flush_replica()
                states.append("author")
            elif pt and pt.is_punct and t._.is_dash:
                # checking for author insertion
                par_end_idx = next_matching(  # paragraph ending index
                    doc,
                    lambda x: x._.is_newline or x.i == len(doc) - 1,
                    start=pt.i,
                )[1]
                results = [
                    cast(Span, match)
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
                        cast(Span, match)
                        for match in matchers["AUTHOR_ENDING"](
                            doc[pt.i : par_end_idx + 1], as_spans=True
                        )
                    ]
                    for match in results:
                        if match.end >= len(doc) - 1 or doc[match.end]._.is_newline:
                            flush_replica()
                            states.append("author")
                            # skip to the end of author ending
                            ti = match.end
                            break
                    else:
                        tokens.append(t)
            else:
                tokens.append(t)

        # author* -> replica* transitions
        elif (pt is None or pt._.is_newline) and t._.is_dash:
            # [Автор:]\n— Реплика
            states.append("replica_dash_after_newline")

        elif pt and ":" in pt.text and t._.is_open_quote:
            # Автор: "Реплика" | Автор: «Реплика»
            states.append("replica_quote_after_colon")

        elif t._.is_open_quote:
            # "Реплика" — автор
            states.append("replica_quote_before_dash")

        elif state == "author_insertion" and pt and pt.is_punct and t._.is_dash:
            # — автор<punct> — [Рр]еплика
            #                 ^
            # return to the state preceding the author insertion
            states.append(states[-2])

    if "replica" in states[-1]:
        flush_replica()  # may have a trailing replica

    return replicas


def find_by_reference(span: Span, reference: Token) -> Span | None:
    """
    Inside the given span, find a noun chunk that might be
    a definition of given reference token.

    Parameters:
        span: sentence or other doc-like span to search definition in.
        reference: reference to some speaker (e.g. он, она, оно, ...)
    """
    for nc in span.noun_chunks:
        if any(morph_equals(t, reference, "Gender", "Number") for t in nc):
            return nc
    return None


# noinspection PyProtectedMember
# (spaCy "._." extensions)
def classify_speakers(
    language: Language,
    dialogue: Dialogue,
) -> dict[Span, Speaker | None]:
    relations: dict[Span, Speaker | None] = {}

    sents = list(dialogue.doc.sents)

    # - Go through each extracted replica;
    # - Look into the next sentence after the replica -
    #   (most commonly there will be a speaker specification);
    # - Look into the previous sentence;
    # - Then go with next-previous alternation (same as above) until we hit one of:
    #       - Start of text;
    #       - Previous replica;
    #       - Next replica;
    #       - End of text.
    # - If speaker was still not found, try to get the forward/backward reference of
    #   second interlocutor from the nearest (prev/next) replicas.

    for prev_replica, replica, next_replica in iter_by_triples(dialogue.replicas):
        # "p_" - previous; "n_" - next; "r_" - replica
        r_start_sent_i = sents.index(replica[0].sent)
        r_start_sent = sents[r_start_sent_i]
        r_end_sent_i = sents.index(replica[-1].sent)
        p_r_sent_i = sents.index(prev_replica[-1].sent) if prev_replica else None
        n_r_sent_i = sents.index(next_replica[0].sent) if next_replica else None

        back_first = any(
            abs(replica[0].i - t.i) < 4 and t.lemma_ == ":" for t in r_start_sent
        )  # replica is after colon - means back-first speaker search

        sent_backward_done = sent_forward_done = False

        for offset in canonical_int_enumeration():
            # Iterate over sentences near the current replica.
            # They have "up-down" indices
            # (e.g i + 0, i + 1, i - 1, i + 2, i - 2, ... if it's forward-first);
            # (and i + 0, i - 1, i + 1, i - 2, i + 2, ... if it's backward-first).
            sent_i = r_start_sent_i - offset if back_first else r_end_sent_i + offset

            sent_backward_done = sent_backward_done or sent_i == p_r_sent_i
            sent_forward_done = sent_forward_done or sent_i == n_r_sent_i

            if sent_backward_done and sent_forward_done:
                # We've got up to the sentences containing
                # the previous AND next replicas -> cannot find
                # the speaker this way, use alternative method.
                break

            sent = sents[sent_i]
            prev_sent = sents[sent_i - 1] if sent_i > 0 else None
            next_sent = sents[sent_i + 1] if sent_i < len(sents) - 1 else None

            # First, look for obvious references, such as
            # ... verb ... noun ... || ... noun ... verb ...
            # where verb and noun are syntactically related.
            dep_matcher = DependencyMatcher(language.vocab)
            dep_matcher.add(
                "SPEAKER_TO_SPECIAL_VERB",
                [SPEAKER_TO_SPECIAL_VERB],
            )
            dep_matcher.add(
                "SPEAKER_TO_VERB_CONJ_SPECIAL_VERB",
                [SPEAKER_TO_VERB_CONJ_SPECIAL_VERB],
            )

            for match_id, token_ids in dep_matcher(sent):
                # for this matcher speaker is the top token in pattern
                token: Token = sent[token_ids[0]]
                if prev_sent and token.lemma_ in REFERRAL_PRON:
                    speaker = find_by_reference(prev_sent, token)
                else:
                    # increase speaker "breadth" using noun chunks
                    for nc in sent.noun_chunks:
                        if token in nc:
                            speaker = nc
                            break
                    else:
                        speaker = dialogue.doc[token.i : token.i]  # TODO check
                relations[replica] = Speaker(list(speaker) if speaker else [token])
                if replica in relations:
                    break

            if replica in relations:
                break

    return relations
