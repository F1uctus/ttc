import itertools
import random
from typing import TextIO, Tuple, Dict, Optional

import click
from click import echo
from colorama import init as colorama_init, Fore, Back, Style  # type: ignore
from spacy.tokens import Span

import ttc

colorama_init()

COLORS = [
    value
    for name, value in Fore.__dict__.items()
    if name[0] != "_"
    and not any(c in name for c in ["BLACK", "WHITE", "RESET", "LIGHT"])
]


@click.group
def cli():
    pass


@cli.command("print-play")  # type: ignore
@click.argument("file", type=click.File("r", encoding="utf-8"), nargs=1)
@click.argument("language", type=str, nargs=1)
@click.option("--with-text", is_flag=True)
def print_play(file: TextIO, language, with_text: bool):
    cc = ttc.load(language)

    if cc is None:
        echo("Specified language is not supported")
        exit(1)

    text = file.read().split("-" * 20)[0]
    file.close()

    echo("Extracting replicas...")
    dialogue = cc.extract_dialogue(text)

    echo("Connecting replicas into the play...")
    play = cc.connect_play(dialogue)

    colors = list(COLORS)
    random.shuffle(colors)
    speaker_colors: Dict[str, Tuple[Span, str]] = {
        s.lemma_: (s, c) for s, c in zip(play.speakers, itertools.cycle(colors)) if s
    }

    echo("Speakers found:")
    echo(
        ", ".join(c + s.text + Style.RESET_ALL for _, (s, c) in speaker_colors.items())
    )

    first_col_w = max(len(str(s)) for s in play.speakers)
    for r, s in play.lines:
        if s:
            echo(speaker_colors[s.lemma_][1] + " ", nl=False)
        echo(f"{{:<{first_col_w}}}  ".format(str(s)), nl=False)
        echo(str(r))

    echo(Style.RESET_ALL, nl=False)

    if with_text:
        echo("\nMarked play:")
        rs_indexed: Dict[int, Tuple[Span, Optional[Span]]] = {
            r.start_char: (r, s) for r, s in play.lines
        }
        r_starts = list(rs_indexed.keys())
        r_start_i = 0
        for i, c in enumerate(text):
            replica: Optional[Span]
            speaker: Optional[Span]
            replica, speaker = (
                rs_indexed[r_starts[r_start_i]]
                if r_start_i < len(r_starts)
                else (None, None)
            )

            if speaker:
                if i == speaker.start_char:
                    echo(Back.GREEN, nl=False)
                elif i == speaker.end_char:
                    echo(Style.RESET_ALL, nl=False)

            if replica and i >= replica.start_char:
                if i == replica.start_char:
                    if speaker:
                        echo(speaker_colors[speaker.lemma_][1], nl=False)

                if i == replica.end_char:
                    r_start_i += 1
                    echo(Style.RESET_ALL, nl=False)

            echo(c, nl=False)

    echo()


if __name__ == "__main__":
    cli()
