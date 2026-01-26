import itertools
import random
from typing import TextIO, Tuple, Dict, Optional

import click
from click import echo, style
from spacy.tokens import Span

import ttc

COLORS = ["red", "green", "yellow", "blue", "magenta", "cyan"]


@click.group
def cli():
    pass


@cli.command("print-play")
@click.argument("file", type=click.File("r", encoding="utf-8"), nargs=1)
@click.argument("language", type=str, nargs=1)
@click.option("--with-text", is_flag=True)
def print_play(file: TextIO, language, with_text: bool):
    cc = ttc.load(language)

    if cc is None:
        echo("Specified language is not supported")
        exit(1)

    assert cc is not None

    text = file.read().split("-" * 20)[0]
    file.close()

    echo("Extracting replicas...")
    dialogue = cc.extract_dialogue(text)

    echo("Connecting replicas into the play...")
    play = cc.connect_play(dialogue)

    colors = list(COLORS)
    random.shuffle(colors)
    actor_colors: Dict[str, Tuple[Span, str]] = {
        s.lemma_: (s, c) for s, c in zip(play.actors, itertools.cycle(colors)) if s
    }

    echo("Actors found:")
    echo(
        ", ".join(
            style(s.text, fg=c) for _, (s, c) in actor_colors.items()
        )
    )

    first_col_w = max(len(str(s)) for s in play.actors)
    for r, s in play.lines:
        if s:
            echo(style(" ", fg=actor_colors[s.lemma_][1]), nl=False)
        echo(f"{str(s):<{first_col_w}}  ", nl=False)
        echo(str(r))

    if with_text:
        echo("\nMarked play:")
        rs_indexed: Dict[int, Tuple[Span, Optional[Span]]] = {
            r.start_char: (r, s) for r, s in play.lines
        }
        r_starts = list(rs_indexed.keys())
        r_start_i = 0
        for i, c in enumerate(text):
            replica: Optional[Span]
            actor: Optional[Span]
            replica, actor = (
                rs_indexed[r_starts[r_start_i]]
                if r_start_i < len(r_starts)
                else (None, None)
            )

            if actor:
                if i == actor.start_char:
                    echo(style("", bg="green", reset=False), nl=False)
                elif i == actor.end_char:
                    echo(style("", reset=True), nl=False)

            if replica and i >= replica.start_char:
                if i == replica.start_char:
                    if actor:
                        echo(
                            style(
                                "", fg=actor_colors[actor.lemma_][1], reset=False
                            ),
                            nl=False,
                        )

                if i == replica.end_char:
                    r_start_i += 1
                    echo(style("", reset=True), nl=False)

            echo(c, nl=False)

    echo()


if __name__ == "__main__":
    cli()
