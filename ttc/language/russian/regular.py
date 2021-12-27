from collections import deque
from enum import Enum
from typing import Union, Optional, overload, Generator, Literal

from spacy import Language
from spacy.matcher import Matcher
from spacy.tokens import Token, Span, Doc

from ttc.language import Deps, Morphs, Replica
from ttc.language import Speaker
from ttc.language.dialogue import Dialogue
from ttc.language.russian import Patterns
from ttc.language.russian.constants import DASHES, REFERRAL_PRON, PERSONAL_PRON
from ttc.language.russian.token_checks import (
    is_name,
    maybe_name,
    is_special_verb,
    is_names_verb,
    morph_equals,
)


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


@overload
def next_non_empty(
    doc_like: list[Span], idx: int = 0, *, step: int = 1
) -> tuple[Optional[Span], int]:
    ...


@overload
def next_non_empty(
    doc_like: Span, idx: int = 0, *, step: int = 1
) -> tuple[Optional[Token], int]:
    ...


def next_non_empty(
    doc_like: Union[Span, list[Span]], idx: int = 0, *, step: int = 1
) -> tuple[Union[Span, Token, None], int]:
    delta = 0
    sub_piece = None
    while 0 <= idx + delta < len(doc_like):
        sub_piece = doc_like[idx + delta]
        if len(sub_piece.text.strip()) > 0:
            break
        delta += step
    return sub_piece, idx + delta


# noinspection PyProtectedMember
# (spaCy extensions)
def extract_replicas(doc: Doc) -> list[Replica]:
    doc_length = len(doc)

    replicas: list[Replica] = []
    tokens: list[Token] = []

    states: deque[
        Literal[
            "author",
            "author_insertion",
            "replica_newline_dash",
            "replica_colon_quote",
            "replica_quote",
        ]
    ] = deque(maxlen=3)
    states.append("author")

    ti = -1  # token index
    while ti + 1 < doc_length:
        ti += 1

        pt: Optional[Token] = doc[ti - 1] if ti - 1 >= 0 else None
        nt: Optional[Token] = doc[ti + 1] if ti + 1 < len(doc) else None
        t: Token = doc[ti]

        state = states[-1]
        if state == "replica_quote":
            if t._.is_close_quote:
                if len(tokens) > 0:
                    replicas.append(Replica(tokens))
                    tokens = []
                    states.append("author")
                    continue
            elif t.is_punct and nt and nt._.is_dash:
                tokens.append(t)
                replicas.append(Replica(tokens))
                tokens = []
                states.append("author_insertion")
                ti += 1
                continue
            tokens.append(t)
        elif state == "replica_colon_quote":
            if t._.is_close_quote and len(tokens) > 0:
                replicas.append(Replica(tokens))
                tokens = []
                states.append("author")
                continue
            tokens.append(t)
        elif state == "replica_newline_dash":
            if t._.is_newline:
                if len(tokens) > 0:
                    replicas.append(Replica(tokens))
                    tokens = []
                states.append("author")
                continue
            elif (
                # sentence starts with dash
                (t._.is_sent_end or next_non_empty(t.sent)[0]._.is_dash)
                and nt
                and nt._.is_dash
            ):
                tokens.append(t)
                replicas.append(Replica(tokens))
                tokens = []
                states.append("author_insertion")
                ti += 1
                continue
            tokens.append(t)
        elif t._.is_close_quote:
            if len(tokens) > 0:
                replicas.append(Replica(tokens))
                tokens = []
                states.append("author")
                continue
            tokens.append(t)
        else:  # author -> ?
            if (pt is None or pt._.is_newline) and t._.is_dash:
                states.append("replica_newline_dash")  # [Автор:]\n— Реплика
            elif pt and ":" in pt.text and t._.is_open_quote:
                # Автор: "Реплика" | Автор: «Реплика»
                states.append("replica_colon_quote")
            elif t._.is_open_quote:
                states.append("replica_quote")  # "Реплика" — автор
            if len(states) > 1 and states[-1] == "author_insertion":
                if states[-1] == states[-2] and t.is_sent_end:
                    # not a replica continuation, sentence ended
                    states.append("author")
                if t.is_punct and nt and nt._.is_dash:
                    # — Реплика, — автор. — Реплика
                    #                       ^
                    states.append(states[-2])
                    ti += 1

    if len(tokens) > 0 and "replica" in states[-1]:
        replicas.append(Replica(tokens))

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

    relations: dict[Replica, Optional[Speaker]] = {}

    sents = [s for s in dialogue.doc.sents]
    speakers_queue: deque[Speaker] = deque()
    replica_tokens = {t for r in dialogue.replicas for t in r.tokens}

    for name, value in Patterns.entries():
        if name not in matcher:
            matcher.add(name, [value])

    for i_r, replica in enumerate(dialogue.replicas):
        sent_idx = sentence_index_by_char_index(sents, replica.tokens[0].idx)
        sent: Span = sents[sent_idx]

        sent_prev, sent_prev_idx = next_non_empty(sents, sent_idx - 1, step=-1)
        sent_next, _ = next_non_empty(sents, sent_idx + 1)

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
                if string_id == Patterns.DASH_VERB_NOUN.name:
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
