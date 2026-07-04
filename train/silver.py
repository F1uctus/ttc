"""LLM silver labeling: rule-segmented replicas, LLM-assigned speakers."""

from collections.abc import Callable, Iterable, Iterator

from ttc.corpora.schema import Character, CorpusDoc, Replica
from ttc.corpus import normalize_name


def silver_label(
    texts: Iterable[str],
    llm: Callable[[str], list[dict]],
    cc,
    doc_id_prefix: str = "silver",
) -> Iterator[CorpusDoc]:
    for n, text in enumerate(texts):
        dialogue = cc.extract_dialogue(text)
        replicas_spans = list(dialogue.replicas)
        labels = {d["replica_index"]: d["speaker"] for d in llm(text)}
        chars: dict[str, Character] = {}
        replicas: list[Replica] = []
        for i, rspan in enumerate(replicas_spans):
            speaker = None
            name = labels.get(i)
            if name:
                key = normalize_name(name)
                if key not in chars:
                    chars[key] = Character(id=f"c{len(chars)}", name=name)
                speaker = chars[key].id
            replicas.append(Replica(rspan.start_char, rspan.end_char, speaker))
        yield CorpusDoc(
            doc_id=f"{doc_id_prefix}/{n}",
            lang="ru",
            domain="prose",
            source="silver_llm",
            license="unknown",
            text=text,
            replicas=replicas,
            characters=list(chars.values()),
            mentions=[],
        )
