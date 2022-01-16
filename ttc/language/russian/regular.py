from collections import deque
from typing import (
    Literal,
    Callable,
    cast,
    Collection,
    Optional,
    Union,
    List,
    Dict,
    Tuple,
    Deque,
)

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
    SPEAKER_TO_SPEAKING_VERB,
    SPEAKER_CONJUNCT_SPEAKING_VERB,
)
from ttc.language.russian.token_checks import morph_equals
from ttc.language.russian.token_patterns import TokenMatcherClass


def next_matching(
    doc_like: Union[Span, Doc],
    predicate: Callable[[Token], bool],
    *,
    start: int = 0,
    step: int = 1
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


# noinspection PyProtectedMember
# (spaCy "._." extensions)
def extract_replicas(
    doc: Doc,
    matchers: Dict[TokenMatcherClass, Matcher],
) -> List[Span]:
    replicas: List[Span] = []
    tokens: List[Token] = []

    # noinspection PyProtectedMember
    # (spaCy "._." extensions)
    def flush_replica():
        nonlocal tokens
        if len(tokens) > 0:
            replicas.append(doc[tokens[0].i : tokens[-1].i + 1])
            tokens = []

    states: Deque[
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

        pt: Optional[Token] = doc[ti - 1] if ti - 1 >= 0 else None
        nt: Optional[Token] = doc[ti + 1] if ti + 1 < len(doc) else None
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


def find_by_reference(spans: list[Span], reference: Token, misses=0) -> Span | None:
    """
    Inside the given span, find a noun chunk that might be
    a definition of given reference token.

    Parameters:
        spans: list of sentences or other doc-like spans to search definition in.
        reference: reference to some speaker (e.g. он, она, оно, ...)
    """
    for nc in spans[-1].noun_chunks:
        for t in nc:
            if morph_equals(t, reference, "Gender", "Number"):
                if nc.lemma_ in REFERRAL_PRON:
                    return find_by_reference(spans[:-1], t, misses)
                return nc
    else:
        if len(spans) > 1 and misses < 4:
            return find_by_reference(spans[:-1], reference, misses + 1)
    return None


# noinspection PyProtectedMember
# (spaCy "._." extensions)
def replica_fills_line(replica: Span) -> bool:
    doc = replica.doc
    return (
        replica.start - 3 >= 0
        and replica.end + 3 < len(doc)  # TODO: Check end-of-doc case
        and any(t._.is_newline for t in doc[replica.start - 3 : replica.start])
        # colon means that replica is still annotated by author, just on previous line
        and not any(t.text == ":" for t in doc[replica.start - 3 : replica.start])
        and any(t._.is_newline for t in doc[replica.end : replica.end + 3])
    )


# noinspection PyProtectedMember
# (spaCy "._." extensions)
def classify_speakers(
    language: Language,
    dialogue: Dialogue,
) -> Dict[Span, Optional[Speaker]]:
    doc = dialogue.doc
    relations: Dict[Span, Optional[Speaker]] = {}
    sents = list(doc.sents)

    dep_matcher = DependencyMatcher(language.vocab)
    dep_matcher.add("*", [SPEAKER_TO_SPEAKING_VERB])
    dep_matcher.add("**", [SPEAKER_CONJUNCT_SPEAKING_VERB])

    # Handles speakers alteration case
    sp_queue: Dict[str, Speaker] = {}

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

            # got up to the previous replica - stop going back
            sent_backward_done = sent_backward_done or sent_i == p_r_sent_i

            # got up to the next replica - stop going forward
            sent_forward_done = sent_forward_done or sent_i == n_r_sent_i

            if sent_backward_done and sent_forward_done:
                # We've got up to the sentences containing
                # the previous AND next replicas -> cannot find
                # the speaker this way, use alternative method.
                break
            if sent_backward_done and offset < 0:
                continue
            if sent_forward_done and offset > 0:
                continue

            sent = sents[sent_i]
            prev_sent = sents[sent_i - 1] if sent_i > 0 else None
            next_sent = sents[sent_i + 1] if sent_i < len(sents) - 1 else None

            if prev_replica and prev_replica._.end_line_no == replica._.start_line_no:
                # Replica is on the same line - probably separated by author speech
                # Assign to the previous speaker
                relations[replica] = list(sp_queue.values())[-1]
                break

            if prev_replica and replica_fills_line(replica):
                if 1 < abs(replica._.start_line_no - prev_replica._.end_line_no) < 4:
                    # If there was 1-2 lines filled with author speech, it is more
                    # probably a continuation, and not alteration, because the reader
                    # will likely lose context if this replica is not annotated by
                    # speaker name and follows some lines of author text
                    relations[replica] = list(sp_queue.values())[-1]
                    break
                if len(sp_queue) > 1:
                    # Line has no author speech -> speakers alteration
                    # Assign replica to the penultimate speaker
                    relations[replica] = penult = list(sp_queue.values())[-2]
                    # reinsert to the end of queue
                    sp_queue[penult.lemma] = sp_queue.pop(penult.lemma)
                    break

            # Look for obvious references, such as
            # ... verb ... noun ... || ... noun ... verb ...
            # where verb and noun are syntactically related.
            for match_id, token_ids in dep_matcher(sent):
                # For this matcher speaker is the top token in pattern
                token: Token = sent[token_ids[0]]
                if prev_sent and token.lemma_ in REFERRAL_PRON:
                    speaker_span = find_by_reference(sents[:sent_i], token)
                else:
                    # increase speaker "breadth" by matching noun chunk
                    for nc in sent.noun_chunks:
                        if token in nc:
                            speaker_span = nc
                            break
                    else:
                        # cannot expand noun, use token as-is
                        speaker_span = doc[token.i : token.i + 1]

                sp = Speaker(list(speaker_span) if speaker_span else [token])
                if sp.lemma in sp_queue:
                    # Reinsert speaker to the end of queue
                    sp_queue[sp.lemma] = relations[replica] = sp_queue.pop(sp.lemma)
                else:
                    sp_queue[sp.lemma] = relations[replica] = sp
                break  # dependency matching when speaker was found

            if replica in relations:
                break  # sentences walking if speaker was found

    return relations
