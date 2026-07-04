"""LLM silver labeling: rule-segmented replicas, LLM-assigned speakers."""

import json
import re
import subprocess
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


# Backends supply respond(prompt) -> text; make_llm does the rest.

Command = Callable[[str], list[str]]
Respond = Callable[[str], str]


def build_prompt(text: str, replicas: list) -> str:
    """The attribution prompt: enumerate replicas, ask for a JSON array."""
    enumerated = "\n".join(f"{i}: {r!s}" for i, r in enumerate(replicas))
    return (
        "Attribute each Russian direct-speech replica to the character who "
        "speaks it, using the surrounding narration.\n\n"
        f"Text:\n{text}\n\n"
        f"Replicas (index: text):\n{enumerated}\n\n"
        'Return ONLY a JSON array like [{"replica_index":0,"speaker":"Имя"}] '
        "with one object per replica, the speaker name in nominative case. "
        "If a replica's speaker is unknown, use null. No prose, no code fences."
    )


def _extract_label_array(response: str) -> list[dict]:
    """Parse the JSON array of {replica_index, speaker} from model output."""
    match = re.search(r"\[.*\]", response, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return [d for d in data if isinstance(d, dict) and d.get("speaker")]


def make_llm(
    cc,
    respond: Respond,
    prompt_builder: Callable[[str, list], str] = build_prompt,
) -> Callable[[str], list[dict]]:
    """Build a silver llm(text) on top of a respond(prompt) backend."""

    def llm(text: str) -> list[dict]:
        replicas = list(cc.extract_dialogue(text).replicas)
        if not replicas:
            return []
        return _extract_label_array(respond(prompt_builder(text, replicas)))

    return llm


def cli_llm(cc, command: Command, timeout: int = 300, **kwargs):
    """Generic subprocess-CLI backend: ``command(prompt) -> argv``."""

    def respond(prompt: str) -> str:
        return subprocess.run(
            command(prompt),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        ).stdout

    return make_llm(cc, respond, **kwargs)


def agent_cli_llm(cc, model: str = "composer-2.5", **kwargs):
    """Cursor `agent` CLI backend (see `cli_llm`)."""
    return cli_llm(
        cc,
        lambda p: ["agent", "--print", "--output-format", "text", "--model", model, p],
        **kwargs,
    )


def openrouter_llm(
    cc,
    model: str = "meta-llama/llama-3.3-70b-instruct:free",
    api_key: str | None = None,
    timeout: int = 300,
    **kwargs,
):
    """OpenRouter backend; model may be a list of free model ids to rotate."""
    import os
    import urllib.request

    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OpenRouter API key required: set OPENROUTER_API_KEY or pass api_key="
        )

    def respond(prompt: str) -> str:
        body = json.dumps(
            {"model": model, "messages": [{"role": "user", "content": prompt}]}
        ).encode("utf-8")
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"]

    return make_llm(cc, respond, **kwargs)
