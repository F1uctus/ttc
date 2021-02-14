# encoding: utf-8
from collections import deque
from typing import Union, Optional, overload

import spacy
from spacy import Language
from spacy.matcher import Matcher
from spacy.tokens import Doc, Token, Span, MorphAnalysis

from language.replica import Replica
from language.russian.common.constants import SPECIAL_VERBS, DASHES


class Processor:
    def __init__(self, doc: Doc, nlp: Language, matcher: Matcher):
        self.doc = doc
        self.nlp = nlp
        self.matcher = matcher

    def __iter__(self):
        return iter((self.doc, self.nlp, self.matcher))


def process_doc(text: str) -> Processor:
    nlp = spacy.load("ru_core_news_md")
    matcher = Matcher(nlp.vocab)
    return Processor(nlp(text), nlp, matcher)


def sentence_index_by_span(sents: list[Span], index: int):
    for i, sentence in enumerate(sents):
        if sentence.start_char <= index <= sentence.end_char:
            return i
    return -1


@overload
def next_non_empty(
    doc_like: list[Span], idx: int, *, step: int = 1
) -> tuple[Optional[Span], int]:
    ...


@overload
def next_non_empty(
    doc_like: Span, idx: int, *, step: int = 1
) -> tuple[Optional[Token], int]:
    ...


def next_non_empty(
    doc_like: Union[Span, list[Span]], idx: int, *, step: int = 1
) -> tuple[Union[Span, Token, None], int]:
    delta = 0
    sub_piece = None
    while 0 <= idx + delta < len(doc_like):
        sub_piece = doc_like[idx + delta]
        if len(sub_piece.text.strip()) > 0:
            break
        delta += step
    return sub_piece, delta


def extract_replicas(processor: Processor) -> list[Replica]:
    doc = processor.doc
    doc_length = len(doc)
    replicas: list[Replica] = []
    character_speech: list[Token] = []

    def prev_token() -> Optional[Token]:
        return doc[i_t - 1] if i_t - 1 >= 0 else None

    def token() -> Token:
        return doc[i_t]

    def speaker_text():
        if i_t >= doc_length:
            return
        nonlocal character_speech
        character_speech.append(token())

    def author_text():
        nonlocal character_speech
        # flush speaker tokens
        if len(character_speech) > 0:
            replicas.append(Replica(None, character_speech))
        character_speech = []

    transitions = {"speaker-text": speaker_text, "newline": author_text}

    state = "author-text"
    i_t = 0
    while i_t < doc_length:
        if "\n" in token().text:
            state = "newline"
        elif token().text in DASHES:
            if state == "newline":
                # "\n -" begins a replica
                state = "speaker-text"
                i_t += 1
            elif prev_token().dep_ == "punct":
                if state == "speaker-text":
                    # replica break
                    state = "author-text"
                elif state == "author-text" and len(character_speech) > 0:
                    # character replica after author speech
                    state = "speaker-text"
                    i_t += 1
        elif state == "newline":
            state = "author-text"

        transitions.get(state, lambda: ...)()

        i_t += 1

    return replicas


def is_name_weak(t: Token):
    return (
        t.pos_ in ["NOUN", "PROPN", "ADJ", "PRON"]
        and t.dep_ not in ["obl"]
        and "mod" not in t.dep_
    )


def is_name(t: Token, linked_verb_morph: MorphAnalysis = None):
    return (
        t.pos_ in ["NOUN", "PROPN", "ADJ", "PRON"]
        and t.text[0].isupper()
        and (
            linked_verb_morph is None
            or linked_verb_morph.get("Number") == t.morph.get("Number")
        )
    )


def is_special_verb(t: Token):
    return t.lemma_ in SPECIAL_VERBS and t.morph.get("VerbForm")[0] not in [
        "Conv",
        "Part",
    ]


def is_names_verb(t: Token, linked_name_morph: MorphAnalysis = None):
    return (
        t.pos_ == "VERB"
        and t.morph.get("VerbForm")[0] not in ["Conv", "Part"]
        and (
            linked_name_morph is None
            or linked_name_morph.get("Number") == t.morph.get("Number")
        )
    )


def get_speaker(span: list[Token]) -> Union[Token, str, None]:
    speaker_tokens = []
    morph = None
    state = ""
    for t in span:
        if t.text in DASHES:
            if state in ["after_name", "after_special_verb"] and len(t.text) == len(
                t.text_with_ws
            ):
                speaker_tokens.append(t)
            else:
                speaker_tokens.clear()
                state = ""
        elif state == "after_special_verb":
            if t.dep_ in ["punct", "cc"]:
                break
            if len(speaker_tokens) == 0 and t.dep_ == "advmod":
                continue
            speaker_tokens.append(t)
            if is_name(t):
                break
        elif state == "after_verb":
            speaker_tokens.clear()
            if is_name(t, linked_verb_morph=morph):
                speaker_tokens.append(t)
        elif state == "after_name":
            if is_name_weak(t):
                speaker_tokens.append(t)
            elif is_names_verb(t, linked_name_morph=morph):
                break
        else:
            speaker_tokens.clear()
            if is_name_weak(t):
                morph = t.morph
                speaker_tokens.append(t)
                state = "after_name"
            elif is_special_verb(t):
                morph = t.morph
                state = "after_special_verb"
            elif is_names_verb(t):
                morph = t.morph
                state = "after_verb"

    if len(speaker_tokens) == 0:
        return None

    if len(speaker_tokens) == 1:
        return speaker_tokens[0]

    # multi-word names are most probably indicate secondary character.
    return f'!generic[{" ".join(t.text for t in speaker_tokens)}]'


def classify_speakers(processor: Processor, replicas: list[Replica]):
    doc: Doc
    nlp: Language
    matcher: Matcher
    (doc, nlp, matcher) = processor

    sents = [s for s in doc.sents]
    speakers_queue: deque[Token] = deque()
    replica_tokens = {t for r in replicas for t in r.tokens}

    if "Replica1" not in matcher:
        pattern = [
            {"TEXT": {"IN": list(DASHES)}},
            {"POS": "VERB"},
            {"POS": {"IN": ["NOUN", "PROPN", "PRON", "ADJ"]}, "OP": "+"},
        ]
        matcher.add("Replica1", [pattern])

    for i_r, replica in enumerate(replicas):
        sent_idx = sentence_index_by_span(sents, replica.tokens[0].idx)
        sent: Span = sents[sent_idx]

        sent_prev, _ = next_non_empty(sents, sent_idx - 1, step=-1)
        # piece_prev = doc[sent_prev[0].i : replica.tokens[0].i - 1]

        sent_next, _ = next_non_empty(sents, sent_idx + 1)
        # piece_next = doc[replica.tokens[-1].i + 1 : sent_next[-1].i + 1]

        newline_after = sent.text[-1] == "\n" or (
            sent_next and "\n" in sent_next[0].text
        )

        do_rotate = True

        sent_non_replica = [t for t in sent if t not in replica_tokens]
        speaker = get_speaker(sent_non_replica)
        if speaker is None and not newline_after:
            # First check for [DASH VERB NOUN] pattern
            matches = matcher(sent_next)
            for match_id, start, end in matches:
                string_id = nlp.vocab.strings[match_id]
                if string_id == "Replica1":
                    span = sent_next[start:end]
                    if span[2] not in replica_tokens:
                        speaker = f"[{' '.join(t.lemma_ for t in span[2:])}]"

            if speaker is None:
                sent_next_non_replica = [
                    t for t in sent_next if t not in replica_tokens
                ]
                speaker = get_speaker(sent_next_non_replica)
        if speaker is None and sent_prev:
            pt = list(sent_prev)
            if sum("\n" in t.text for t in sent) > 1:
                pt += list(sent)
            # if prev sent was not replica, it is probably not an alteration
            prev_is_replica = any(t in pt for t in replica_tokens)
            if not prev_is_replica:
                do_rotate = False

        if isinstance(speaker, Token):
            if sent_prev and speaker.lemma_ in {"он", "она", "мужчина", "женщина"}:
                # extract possible source names from previous sentence
                speakers = [
                    t
                    for t in [t for t in sent_prev] + list(speakers_queue)[::-1]
                    if isinstance(t, Token)
                    and t not in replica.tokens
                    and is_name_weak(t)
                    and t.morph.get("Case") == speaker.morph.get("Case")
                    and t.morph.get("Number") == speaker.morph.get("Number")
                ]
                if len(speakers) > 0:
                    speaker = speakers[-1]

        if isinstance(speaker, Token):
            if hasattr(speaker, "lemma_") and speaker.lemma_ not in [
                s.lemma_ if isinstance(s, Token) else s for s in speakers_queue
            ]:
                speakers_queue.append(speaker)
                do_rotate = False

        if len(speakers_queue) > 2:
            while len(speakers_queue) > 2:
                speakers_queue.popleft()
        elif do_rotate:
            speakers_queue.rotate()

        if not speaker and len(speakers_queue) > 0:
            speaker = speakers_queue[-1]

        print(speakers_queue)

        replica.speaker = speaker

    return replicas
