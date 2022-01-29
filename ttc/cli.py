import itertools
import random
from typing import TextIO, Tuple, Dict, Optional

import click
from colorama import init as colorama_init, Fore, Style
from spacy.tokens import Span

import ttc

colorama_init()

COLORS = [
    value
    for name, value in Fore.__dict__.items()
    if name[0] != "_"
    and not any(c in name for c in ["BLACK", "WHITE", "RESET", "LIGHT"])
]


@click.command(name="print")
@click.argument("file", type=click.File("r", encoding="utf-8"), nargs=1)
@click.argument("language", type=str, nargs=1)
def print_play(file: TextIO, language):
    cc = ttc.load(language)

    if cc is None:
        print("Specified language is not supported")
        exit(1)

    text = file.read().split("-" * 20)[0]

    print("Extracting replicas...")
    dialogue = cc.extract_dialogue(text)

    print("Connecting dialogue into the play...")
    play = cc.connect_play(dialogue)

    rs_indexed: Dict[int, Tuple[Span, Optional[Span]]] = {
        k.start_char: (k, v) for k, v in play.content.items()
    }
    r_starts = list(rs_indexed.keys())
    start_i = 0

    colors = [c for c in COLORS]
    random.shuffle(colors)
    color_stream = itertools.cycle(colors)
    speaker_colors: Dict[Optional[Span], str] = {
        s: c for s, c in zip(play.content.values(), color_stream)
    }

    print("Speakers found:")
    print(
        ", ".join(
            c + s.text + Style.RESET_ALL
            for s, c in speaker_colors.items()
            if s is not None
        )
    )
    print("Marked play:")
    for i, c in enumerate(text):
        replica: Optional[Span]
        speaker: Optional[Span]
        replica, speaker = (
            rs_indexed[r_starts[start_i]] if start_i < len(r_starts) else (None, None)
        )

        if replica and i >= replica.start_char:
            if i == replica.start_char:
                if speaker is not None:
                    print(speaker_colors[speaker], end=" ")

            if i >= replica.end_char:
                start_i += 1
                print(Style.RESET_ALL, end="")

        print(c, end="")

    print()
