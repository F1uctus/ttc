import itertools
import json as jsonlib
import random
from dataclasses import asdict
from pathlib import Path
from typing import TextIO, Tuple, Dict, Optional

import click
from click import echo, style
from spacy.tokens import Span

import ttc

COLORS = ["red", "green", "yellow", "blue", "magenta", "cyan"]

MODEL_SIZES = click.Choice(["sm", "md", "lg"])


@click.group
def cli():
    pass


@cli.command("eval")
@click.argument(
    "paths", type=click.Path(exists=True, path_type=Path), nargs=-1, required=False
)
@click.option("--model", type=MODEL_SIZES, default=None, help="spaCy model size.")
@click.option("--by-file", is_flag=True, help="Report per-file metrics.")
@click.option("--errors", "show_errors", is_flag=True, help="List mismatches.")
@click.option(
    "--unblind-heldout",
    is_flag=True,
    help="Allow --errors on held-out texts (breaks tuning blindness!).",
)
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output.")
def eval_corpus(paths, model, by_file, show_errors, unblind_heldout, as_json):
    """Measure extraction/attribution accuracy on annotated corpus PATHS.

    PATHS are corpus .txt files or directories of them; defaults to
    tests/russian/texts/{tune,heldout} relative to the current directory.
    """
    from ttc.eval import aggregate, evaluate_paths, format_report

    if not paths:
        texts = Path("tests/russian/texts")
        paths = tuple(d for d in (texts / "tune", texts / "heldout") if d.is_dir())
        if not paths:
            echo("No corpus paths given and no default corpus found.")
            exit(1)

    if show_errors and not unblind_heldout:
        blocked = [p for p in paths if "heldout" in (part.lower() for part in p.parts)]
        if blocked:
            echo(
                "Refusing to list per-replica errors for held-out texts:"
                f" {', '.join(map(str, blocked))}\n"
                "Held-out data is for aggregate numbers only while tuning;"
                " pass --unblind-heldout if you really need this."
            )
            exit(2)

    cc = ttc.load("ru", model_size=model)
    assert cc is not None

    exit_code = 0
    for path in paths:
        reports = evaluate_paths(cc, [path])
        if not reports:
            echo(f"{path}: no corpus files found")
            exit_code = 1
            continue
        if as_json:
            total = aggregate(reports)
            echo(
                jsonlib.dumps(
                    {
                        "path": str(path),
                        "files": [
                            {**asdict(r), "path": str(r.path)} for r in reports
                        ],
                        "end_to_end_accuracy": total.end_to_end_accuracy,
                        "attribution_accuracy": total.attribution_accuracy,
                        "extraction_precision": total.extraction_precision,
                        "extraction_recall": total.extraction_recall,
                    },
                    ensure_ascii=False,
                )
            )
        else:
            echo(f"== {path}")
            echo(format_report(reports, by_file=by_file, show_errors=show_errors))
    exit(exit_code)


@cli.command("annotate")
@click.argument("text_file", type=click.Path(exists=True, path_type=Path), nargs=1)
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    required=True,
    help="Corpus file to write, e.g. tests/russian/texts/tune/author-work-1.txt",
)
@click.option("--model", type=MODEL_SIZES, default=None, help="spaCy model size.")
@click.option("--port", type=int, default=8765, show_default=True)
def annotate(text_file: Path, out: Path, model, port: int):
    """Annotate TEXT_FILE speakers in the browser, prefilled by the pipeline.

    TEXT_FILE is raw text, or an existing corpus file to re-annotate
    (its gold pairs are used as the prefill instead of predictions).
    """
    from ttc.annotate import run

    cc = ttc.load("ru", model_size=model)
    assert cc is not None
    run(cc, text_file, out, port)


@cli.command("print-play")
@click.argument("file", type=click.File("r", encoding="utf-8"), nargs=1)
@click.argument("language", type=str, nargs=1)
@click.option("--with-text", is_flag=True)
@click.option("--model", type=MODEL_SIZES, default=None, help="spaCy model size.")
def print_play(file: TextIO, language, with_text: bool, model):
    cc = ttc.load(language, model_size=model)

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
