# encoding: utf-8
from pprint import pprint
from typing import Union

from natasha import (
    Segmenter,
    MorphVocab,

    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,

    Doc
)
from natasha.doc import DocToken
from razdel.segmenters.punct import DASHES

from language.russian.common.constants import NAME_EXCEPTIONS, SPECIAL_VERBS
from language.russian.common.natasha_span import NatashaSpan
from language.span import Span


def process_doc(text: str):
    doc = Doc(text)
    emb = NewsEmbedding()

    segmenter = Segmenter()
    doc.segment(segmenter)

    morph_tagger = NewsMorphTagger(emb)
    doc.tag_morph(morph_tagger)

    morph_vocab = MorphVocab()
    for token in doc.tokens:
        token.lemmatize(morph_vocab)

    syntax_parser = NewsSyntaxParser(emb)
    doc.parse_syntax(syntax_parser)
    return doc


def sentence_by_span(doc: Doc, index: int):
    for sentence in doc.sents:
        if sentence.start <= index <= sentence.stop:
            return sentence
    return None


def sentence_index_by_span(doc: Doc, index: int):
    for i, sentence in enumerate(doc.sents):
        if sentence.start <= index <= sentence.stop:
            return i
    return None


def token_by_span(doc: Doc, index: int):
    for token in doc.tokens:
        if token.start <= index <= token.stop:
            return token
    return None


def extract_replicas(doc: Doc) -> list[Span]:
    replicas = []
    take_token = False
    # split text into replicas
    for i_s, sent in enumerate(doc.sents):
        prev_sent = None if i_s < 1 else doc.sents[i_s - 1]

        # newline usually begins a new speaker's replica
        if prev_sent and doc.text[prev_sent.stop] == '\n':
            take_token = False

        offset = 0
        start_dash = sent.tokens[0].text in DASHES
        if start_dash:
            offset = 1
        else:
            for i_t, token in enumerate(sent.tokens):
                # exceptional case 1:
                # "author text"... - ...character speech
                if token.text in DASHES \
                    and i_t + 1 < len(sent.tokens) \
                    and sent.tokens[i_t + 1].text == '…':
                    offset = i_t + 1
                    take_token = True
                    break
                # exceptional case 2:
                # author text: - Character speech
                if token.text == ':' \
                    and i_t + 1 < len(sent.tokens) \
                    and sent.tokens[i_t + 1].text in DASHES:
                    offset = i_t + 2
                    take_token = True
                    break

        if take_token or start_dash:
            take_token = take_token != start_dash
            person_speech_tokens = []
            for i_t, token in enumerate(sent.tokens[offset:]):
                if start_dash \
                    and token.text in DASHES \
                    and (i_t - 1 < 0
                         or sent.tokens[i_t + offset - 1].rel == 'punct'):
                    # author's speech alternates with person's through dash
                    take_token = not take_token
                    continue
                if take_token:
                    person_speech_tokens.append(token)

            if len(person_speech_tokens) == 0:
                continue
            # here we have a clean speech text
            replicas.append(NatashaSpan(None, person_speech_tokens))
    return replicas


def is_name(t: DocToken, linked_verb_feats: dict = None):
    return (t.pos == 'PROPN'
            and t.lemma not in NAME_EXCEPTIONS
            and t.text[0].isupper()
            and (linked_verb_feats is None
                 or linked_verb_feats.get('Number', None)
                 == t.feats.get('Number', None)
                 ))


def is_names_verb(t: DocToken, linked_name_feats: dict = None):
    return (t.pos == 'VERB'
            and (linked_name_feats is None
                 or linked_name_feats.get('Number', None)
                 == t.feats.get('Number', None)
                 ))


def get_speaker(
    tokens: list[DocToken],
    speaker: DocToken = None
) -> Union[DocToken, str]:
    possible_new_speaker = None
    state = ''
    i = 0
    long_speaker_len = 0
    feats = None
    while i < len(tokens):
        prev_token = tokens[i - 1] if i - 1 > 0 else None
        token = tokens[i]
        if token.text in DASHES:
            long_speaker_len = 0
            state = ''
        elif state == 'after_special_verb':
            if token.rel == 'punct':
                break
            speaker = token
            if is_name(token):
                break
            long_speaker_len += 1
            if long_speaker_len > 1:
                return '!generic'
        elif state == 'after_verb':
            long_speaker_len = 0
            if prev_token and prev_token.lemma in SPECIAL_VERBS:
                state = 'after_special_verb'
                continue
            elif is_name(token, linked_verb_feats=feats):
                speaker = token
        elif state == 'after_name':
            long_speaker_len = 0
            if is_names_verb(token, linked_name_feats=feats):
                speaker = possible_new_speaker
        else:
            long_speaker_len = 0
            if is_name(token):
                feats = token.feats
                possible_new_speaker = token
                state = 'after_name'
            elif is_names_verb(token):
                feats = token.feats
                state = 'after_verb'

        i += 1

    return speaker


Speaker = Union[DocToken, str, None]


def classify_speakers(doc: Doc, replicas: list[Span]) -> list[Span]:
    last_speaker = None
    for replica in replicas:
        sentence_idx = sentence_index_by_span(doc, replica.tokens[0].start)
        sentence = doc.sents[sentence_idx]
        pprint(sentence.tokens)
        sentence.syntax.print()

        sentence_next = (doc.sents[sentence_idx + 1]
                         if sentence_idx + 1 < len(doc.sents)
                         else None)
        sentence_prev = (doc.sents[sentence_idx - 1]
                         if sentence_idx - 1 > 0
                         else None)
        newline_before = (sentence_prev is not None
                          and doc.text[sentence_prev.stop] == '\n')
        newline_after = (doc.text[sentence.stop] == '\n')

        speaker = None

        if not newline_before and sentence_prev is not None:
            new_speaker = get_speaker(sentence_prev.tokens, last_speaker)
            if new_speaker is not None:
                speaker = new_speaker

        if speaker is None:
            speaker = get_speaker(sentence.tokens)

        if speaker is None and not newline_after and sentence_next is not None:
            new_speaker = get_speaker(sentence_next.tokens)
            if new_speaker is not None:
                speaker = new_speaker

        last_speaker = speaker
        replica.speaker = last_speaker
        print(replica)

    return replicas
