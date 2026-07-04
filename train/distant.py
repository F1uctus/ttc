"""Distant supervision: the rule pipeline labels raw RU fiction."""

from collections.abc import Iterable, Iterator

from ttc.corpora.schema import Character, CorpusDoc, Mention, Replica
from ttc.corpus import normalize_name


def label_fiction(
    texts: Iterable[str], cc, doc_id_prefix: str = "distant"
) -> Iterator[CorpusDoc]:
    for n, text in enumerate(texts):
        dialogue = cc.extract_dialogue(text)
        play = cc.connect_play(dialogue)
        chars: dict[str, Character] = {}
        mentions: list[Mention] = []
        replicas: list[Replica] = []
        for replica, actor in play.lines:
            speaker = None
            if actor is not None and len(actor):
                key = normalize_name(actor.lemma_)
                if key not in chars:
                    chars[key] = Character(id=f"c{len(chars)}", name=actor.text)
                speaker = chars[key].id
                mentions.append(
                    Mention(actor.start_char, actor.end_char, chars[key].id)
                )
            replicas.append(Replica(replica.start_char, replica.end_char, speaker))
        yield CorpusDoc(
            doc_id=f"{doc_id_prefix}/{n}",
            lang="ru",
            domain="prose",
            source="distant",
            license="unknown",
            text=text,
            replicas=replicas,
            characters=list(chars.values()),
            mentions=mentions,
        )
