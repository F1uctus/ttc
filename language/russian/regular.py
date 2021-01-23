# encoding: utf-8
from collections import deque
from typing import Union, Optional, overload

import spacy
from spacy.tokens import Doc, Token, Span, MorphAnalysis

from language.russian.common.constants import (
    NAME_EXCEPTIONS,
    SPECIAL_VERBS,
    DASHES,
    GENERIC_NAMES
)
from language.replica import Replica


def process_doc(text: str):
    nlp = spacy.load('ru_core_news_md')
    return nlp(text)


def sentence_index_by_span(sents: list[Span], index: int):
    for i, sentence in enumerate(sents):
        if sentence.start_char <= index <= sentence.end_char:
            return i
    return -1


@overload
def next_non_empty(
    doc_piece: list[Span], idx: int, *, step: int = 1
) -> tuple[Optional[Span], int]: ...


@overload
def next_non_empty(
    doc_piece: Span, idx: int, *, step: int = 1
) -> tuple[Optional[Token], int]: ...


def next_non_empty(
    doc_piece: Union[Span, list[Span]], idx: int, *, step: int = 1
) -> tuple[Union[Span, Token, None], int]:
    delta = 0
    sub_piece = None
    while 0 <= idx + delta < len(doc_piece):
        sub_piece = doc_piece[idx + delta]
        if len(sub_piece.text.strip()) > 0:
            break
        delta += step
    return sub_piece, delta


def extract_replicas(doc: Doc) -> list[Replica]:
    doc_length = len(doc)
    replicas: list[Replica] = []
    character_speech: list[Token] = []

    def prev_token() -> Optional[Token]:
        return doc[i_t - 1] if i_t - 1 >= 0 else None

    def token() -> Token:
        return doc[i_t]

    def speaker_text():
        if i_t >= doc_length: return
        nonlocal character_speech
        character_speech.append(token())

    def author_text():
        nonlocal character_speech
        # flush speaker tokens
        if len(character_speech) > 0:
            replicas.append(Replica(None, character_speech))
        character_speech = []

    transitions = {
        'speaker-text': speaker_text,
        'newline': author_text
    }

    state = 'author-text'
    i_t = 0
    while i_t < doc_length:
        if '\n' in token().text:
            state = 'newline'
        elif token().text in DASHES:
            if state == 'newline':
                # "\n -" begins a replica
                state = 'speaker-text'
                i_t += 1
            elif prev_token().dep_ == 'punct':
                if state == 'speaker-text':
                    # replica break
                    state = 'author-text'
                elif state == 'author-text' and len(character_speech) > 0:
                    # character replica after author speech
                    state = 'speaker-text'
                    i_t += 1
        elif state == 'newline':
            state = 'author-text'

        transitions.get(state, lambda: ...)()

        i_t += 1

    return replicas


def is_name(t: Token, linked_verb_morph: MorphAnalysis = None):
    return (t.pos_ in ['NOUN', 'PROPN', 'ADJ']
            and t.lemma not in NAME_EXCEPTIONS
            and t.text[0].isupper()
            and (linked_verb_morph is None
                 or linked_verb_morph.get('Number') == t.morph.get('Number')))


def is_names_verb(t: Token, linked_name_morph: MorphAnalysis = None):
    return (t.pos_ == 'VERB'
            and (linked_name_morph is None
                 or linked_name_morph.get('Number') == t.morph.get('Number')))


def get_speaker(span: list[Token], speaker: Token = None) -> Union[Token, str]:
    possible_new_speaker = None
    long_speaker_len = 0
    morph = None
    state = ''
    for token in span:
        if token.text in DASHES:
            long_speaker_len = 0
            state = ''
        elif state == 'after_special_verb':
            if token.dep_ == 'punct':
                break
            speaker = token
            if is_name(token):
                break
            long_speaker_len += 1
            if long_speaker_len > 1:
                # multi-word names are most probably indicate secondary character.
                return '!generic'
        elif state == 'after_verb':
            long_speaker_len = 0
            if is_name(token, linked_verb_morph=morph):
                speaker = token
        elif state == 'after_name':
            long_speaker_len = 0
            if is_names_verb(token, linked_name_morph=morph):
                speaker = possible_new_speaker
        else:
            long_speaker_len = 0
            if is_name(token):
                morph = token.morph
                possible_new_speaker = token
                state = 'after_name'
            elif token.lemma_ in SPECIAL_VERBS:
                morph = token.morph
                state = 'after_special_verb'
            elif is_names_verb(token):
                morph = token.morph
                state = 'after_verb'

    if speaker and speaker.lemma_ in GENERIC_NAMES:
        return '!generic'

    return speaker


def classify_speakers(doc: Doc, replicas: list[Replica]):
    sents = [s for s in doc.sents]
    speakers_queue: deque[Token] = deque()

    for i_e, replica in enumerate(replicas):
        sent_idx = sentence_index_by_span(sents, replica.tokens[0].idx)

        sent: Span = sents[sent_idx]
        sent_prev, _ = next_non_empty(sents, sent_idx - 1, step=-1)
        sent_next, _ = next_non_empty(sents, sent_idx + 1)

        newline_after = (sent.text[-1] == '\n'
                         or (sent_next and '\n' in sent_next[0].text))

        speaker = get_speaker([t for t in sent if t not in replica.tokens])
        if speaker is None and not newline_after:
            speaker = get_speaker([t for t in sent_next if t not in replica.tokens])

        if isinstance(speaker, Token):
            if sent_prev and speaker.lemma_ in {'он', 'она'}:
                # extract possible source names from previous sentence
                src_speakers = [t for t in sent_prev
                                if t not in replica.tokens and is_name(t)]
                if len(src_speakers) > 0:
                    speaker = src_speakers[-1]

        # if isinstance(speaker, Token):
        #     if speaker.lemma_ not in speakers_queue:
        #         speakers_queue.append(speaker.lemma_)
        # else:
        #     speaker = speakers_queue[-1]
        #
        # if len(speakers_queue) > 2:
        #     while len(speakers_queue) > 1:
        #         speakers_queue.popleft()
        # else:
        #     speakers_queue.rotate()
        #
        # print(speakers_queue)

        replica.speaker = speaker

    return replicas
