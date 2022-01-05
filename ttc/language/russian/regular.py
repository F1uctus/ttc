from collections import deque
from enum import Enum
from typing import Optional, Generator, Literal, Callable, cast

from spacy import Language
from spacy.matcher import Matcher
from spacy.tokens import Token, Span, Doc

from ttc.language import Deps, Morphs, Replica
from ttc.language import Speaker
from ttc.language.dialogue import Dialogue
from ttc.language.russian.constants import (
    DASHES,
    REFERRAL_PRON,
    PERSONAL_PRON,
)
from ttc.language.russian.token_checks import (
    is_name,
    maybe_name,
    is_special_verb,
    is_names_verb,
    morph_equals,
)
from ttc.language.russian.token_patterns import TokenPatterns, MatcherClass


def sentence_index_by_char_index(sentences: list[Span], index: int) -> int:
    for i, sentence in enumerate(sentences):
        if sentence.start_char <= index <= sentence.end_char:
            return i
    return -1


def main_speaker_by_reference(
    sentences: list[Span], ref: Speaker, idx: Optional[int] = None
) -> Generator[Speaker, None, None]:
    idx = idx or len(sentences) - 1
    for t in sentences[idx]:
        if t.lemma_ in REFERRAL_PRON:
            if idx - 1 > 0:
                yield from main_speaker_by_reference(sentences, ref, idx - 1)
        elif maybe_name(t):
            yield Speaker([t])


def next_matching(
    doc_like: Span | Doc,
    predicate: Callable[[Token], bool],
    *,
    start: int = 0,
    step: int = 1
) -> tuple[Optional[Token], int]:
    delta = 0
    sub_piece = None
    while 0 <= start + delta < len(doc_like):
        checked = doc_like[start + delta]
        if predicate(checked):
            sub_piece = checked
            break
        delta += step
    return sub_piece, start + delta


def next_non_empty_s(
    doc_like: list[Span], start: int = 0, *, step: int = 1
) -> tuple[Optional[Span], int]:
    delta = 0
    sub_piece = None
    while 0 <= start + delta < len(doc_like):
        checked = doc_like[start + delta]
        if len(checked.text.strip()) > 0:
            sub_piece = checked
            break
        delta += step
    return sub_piece, start + delta


# noinspection PyProtectedMember
# (spaCy extensions)
def extract_replicas(doc: Doc, matchers: dict[MatcherClass, Matcher]) -> list[Replica]:
    replicas: list[Replica] = []
    tokens: list[Token] = []

    def flush_replica():
        nonlocal tokens
        if len(tokens) > 0:
            replicas.append(Replica(tokens))
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
                    cast(Span, m)
                    for m in matchers["AUTHOR_INSERTION"](
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
                        cast(Span, m)
                        for m in matchers["AUTHOR_ENDING"](
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


def get_speaker(span: list[Token]) -> Optional[Speaker]:
    speaker_tokens = []
    linked_token = None

    class State(Enum):
        START, AFTER_VERB, AFTER_SPECIAL_VERB, AFTER_NAME = range(4)

    state = State.START
    for t in span:
        if t.text in DASHES:
            if state and len(t.text) == len(t.text_with_ws):
                speaker_tokens.append(t)
            else:
                speaker_tokens.clear()
                state = State.START
        elif state == State.AFTER_SPECIAL_VERB:
            if t.dep_ in [Deps.PUNCT, Deps.CC]:
                break
            if len(speaker_tokens) == 0 and t.dep_ == Deps.ADVMOD:
                continue
            speaker_tokens.append(t)
            if is_name(t):
                break
        elif state == State.AFTER_VERB:
            speaker_tokens.clear()
            if is_name(t, linked_verb=linked_token):
                speaker_tokens.append(t)
        elif state == State.AFTER_NAME:
            if is_names_verb(t, linked_name=linked_token):
                break
            elif maybe_name(t) or t.dep_ == Deps.CASE:
                speaker_tokens.append(t)
        else:
            speaker_tokens.clear()
            if maybe_name(t) and t.morph.get(Morphs.CASE)[0] != "Dat":
                linked_token = t
                speaker_tokens.append(t)
                state = State.AFTER_NAME
            elif is_special_verb(t):
                linked_token = t
                state = State.AFTER_SPECIAL_VERB
            elif is_names_verb(t):
                linked_token = t
                state = State.AFTER_VERB

    if len(speaker_tokens) == 0:
        return None

    return Speaker(speaker_tokens)


def classify_speakers(
    language: Language, dialogue: Dialogue
) -> dict[Replica, Optional[Speaker]]:
    matcher = Matcher(language.vocab)

    for name, value in TokenPatterns.entries():
        if name not in matcher:
            matcher.add(name, [value])

    relations: dict[Replica, Optional[Speaker]] = {}

    sents = [s for s in dialogue.doc.sents]
    speakers_queue: deque[Speaker] = deque()
    replica_tokens = {t for r in dialogue.replicas for t in r.tokens}

    for i_r, replica in enumerate(dialogue.replicas):
        sent_idx = sentence_index_by_char_index(sents, replica.tokens[0].idx)
        sent: Span = sents[sent_idx]

        sent_prev, sent_prev_idx = next_non_empty_s(sents, sent_idx - 1, step=-1)
        sent_next, _ = next_non_empty_s(sents, sent_idx + 1)

        newline_after = sent.text.endswith("\n") or (
            sent_next and "\n" in sent_next[0].text
        )

        rotate_speakers = True

        sent_non_replica = [t for t in sent if t not in replica_tokens]
        speaker = get_speaker(sent_non_replica)
        if sent_next and not speaker and not newline_after:
            # First check for [DASH VERB NOUN+] pattern
            results: list[tuple[int, int, int]] = matcher(sent_next)
            for match_id, start, end in results:
                string_id = language.vocab.strings[match_id]
                if string_id == TokenPatterns.DASH_VERB_NOUN.name:
                    span = sent_next[start:end]
                    if span[2] not in replica_tokens:
                        speaker = Speaker(span[2:])

            if not speaker:
                sent_next_non_replica = [
                    t for t in sent_next if t not in replica_tokens
                ]
                speaker = get_speaker(sent_next_non_replica)

        if not speaker and sent_prev:
            pt = list(sent_prev)
            if sum("\n" in t.text for t in sent) > 1:
                pt += list(sent)
            # if prev sent was not replica, it is probably not an alteration
            prev_is_replica = any(t in pt for t in replica_tokens)
            if not prev_is_replica:
                rotate_speakers = False

        if sent_prev and speaker and speaker.lemma in REFERRAL_PRON:
            # extract possible source names from previous sentence
            possible_speakers = list(
                main_speaker_by_reference(
                    sents,
                    speaker,
                    sent_prev_idx,
                )
            )

            # If last 2 speakers have same gender & there is a reference to other with
            # gender pronoun, then the speaker is meant to be the 1st of them.
            if len(speakers_queue) == 2:
                gender1 = speakers_queue[0].gender
                gender2 = speakers_queue[1].gender
                if gender1 == gender2 and speaker.lemma in PERSONAL_PRON.get(gender1):
                    speakers = speakers_queue[::-1]
                else:
                    speakers = deque()
            else:
                speakers = speakers_queue

            for s in speakers:
                if morph_equals(Morphs.GENDER, s, speaker) and morph_equals(
                    Morphs.NUMBER, s, speaker
                ):
                    possible_speakers.append(s)

            cap_names = []
            for i, ps in enumerate(possible_speakers):
                if ps.text[0].isupper():
                    cap_names.append(i)
            if len(cap_names) == 1:
                speaker = possible_speakers[cap_names[0]]
            elif len(possible_speakers) > 0:
                speaker = possible_speakers[-1]

        if speaker and speaker.text not in {s.text for s in speakers_queue}:
            speakers_queue.append(speaker)
            rotate_speakers = False

        if len(speakers_queue) > 2:
            while len(speakers_queue) > 2:
                speakers_queue.popleft()
        elif rotate_speakers:
            speakers_queue.rotate()

        if not speaker and len(speakers_queue) > 0:
            speaker = speakers_queue[-1]

        # print(speakers_queue)

        relations[replica] = speaker

    return relations
