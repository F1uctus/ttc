import itertools
import json as jsonlib
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TextIO

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
@click.option(
    "--jsonl",
    "jsonl_paths",
    type=click.Path(exists=True, path_type=Path),
    multiple=True,
    help="Interchange JSONL corpora (multi-corpus/multi-language).",
)
def eval_corpus(
    paths, model, by_file, show_errors, unblind_heldout, as_json, jsonl_paths
):
    """Measure extraction/attribution accuracy on annotated corpus PATHS.

    PATHS are corpus .txt files or directories of them; defaults to
    tests/russian/texts/{tune,heldout} relative to the current directory.
    Pass --jsonl to evaluate interchange corpora (with a qtype breakdown).
    """
    from ttc.eval import aggregate, evaluate_paths, format_report

    if not paths and not jsonl_paths:
        texts = Path("tests/russian/texts")
        paths = tuple(d for d in (texts / "tune", texts / "heldout") if d.is_dir())
        if not paths:
            echo("No corpus paths given and no default corpus found.")
            sys.exit(1)

    if show_errors and not unblind_heldout:
        blocked = [p for p in paths if "heldout" in (part.lower() for part in p.parts)]
        if blocked:
            echo(
                "Refusing to list per-replica errors for held-out texts:"
                f" {', '.join(str(p) for p in blocked)}\n"
                "Held-out data is for aggregate numbers only while tuning;"
                " pass --unblind-heldout if you really need this."
            )
            sys.exit(2)

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
                        "files": [{**asdict(r), "path": str(r.path)} for r in reports],
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

    for jp in jsonl_paths:
        from ttc.corpora.schema import read_jsonl
        from ttc.eval import evaluate_interchange_doc

        reports = []
        for doc in read_jsonl(jp):
            doc_cc = cc if doc.lang == "ru" else ttc.load(doc.lang)
            if doc_cc is None:
                echo(f"{doc.doc_id}: no classifier for lang {doc.lang!r}, skipped")
                continue
            reports.append(evaluate_interchange_doc(doc_cc, doc))
        if reports:
            echo(f"== {jp}")
            echo(format_report(reports, by_file=by_file, show_errors=show_errors))

    sys.exit(exit_code)


@cli.group("corpus")
def corpus_group():
    """Corpus conversion, statistics and auditing."""


@corpus_group.command("convert")
@click.argument("source", type=str)
@click.argument("in_path", type=click.Path(exists=True, path_type=Path))
@click.option("--out", type=click.Path(path_type=Path), required=True)
@click.option(
    "--split",
    type=click.Choice(["tune", "heldout", "all"]),
    default="all",
    show_default=True,
    help="Keep only docs of this split (native docs are never filtered).",
)
def corpus_convert(source: str, in_path: Path, out: Path, split: str):
    """Convert corpus SOURCE at IN_PATH into interchange JSONL."""
    from ttc.corpora import get_adapter
    from ttc.corpora.schema import validate, write_jsonl
    from ttc.corpora.splits import split_of

    try:
        adapter = get_adapter(source)
    except KeyError as e:
        raise click.ClickException(str(e.args[0]))

    docs = []
    n_issues = 0
    for doc in adapter(in_path):
        if source != "native" and split != "all" and split_of(doc.doc_id) != split:
            continue
        n_issues += len(issues := validate(doc))
        for issue in issues:
            echo(style(issue, fg="yellow"))
        docs.append(doc)
    n = write_jsonl(docs, out)
    echo(
        f"{n} doc(s) -> {out}"
        + (f" ({n_issues} validation issues)" if n_issues else "")
    )


@corpus_group.command("stats")
@click.argument(
    "jsonl", type=click.Path(exists=True, path_type=Path), nargs=-1, required=True
)
def corpus_stats(jsonl):
    """Per-source/language document, replica and character counts."""
    from collections import Counter

    from ttc.corpora.schema import read_jsonl

    docs = Counter()
    replicas = Counter()
    chars = Counter()
    for path in jsonl:
        for doc in read_jsonl(path):
            key = (doc.source, doc.lang, doc.domain)
            docs[key] += 1
            replicas[key] += len(doc.replicas)
            chars[key] += len(doc.characters)
    echo(
        f"{'source':<14}{'lang':<6}{'domain':<8}"
        f"{'docs':>7}{'replicas':>10}{'chars':>8}"
    )
    for key in sorted(docs):
        s, l, d = key
        echo(f"{s:<14}{l:<6}{d:<8}{docs[key]:>7}{replicas[key]:>10}{chars[key]:>8}")


@corpus_group.command("audit")
@click.argument(
    "paths", type=click.Path(exists=True, path_type=Path), nargs=-1, required=True
)
@click.option("--report", "report_path", type=click.Path(path_type=Path), default=None)
@click.option("--skip-disagreements", is_flag=True, help="Mechanical checks only.")
@click.option("--model", type=MODEL_SIZES, default=None, help="spaCy model size.")
def corpus_audit(paths, report_path, skip_disagreements, model):
    """Audit native RU gold before it is used as training seed."""
    from ttc.corpora.audit import audit_native, format_report

    cc = None
    if not skip_disagreements:
        cc = ttc.load("ru", model_size=model)
    report = audit_native(list(paths), cc=cc)
    text = format_report(report)
    if report_path:
        report_path.write_text(text, encoding="utf-8")
        echo(f"report -> {report_path}")
    else:
        echo(text)
    sys.exit(0 if report.clean else 1)


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
        sys.exit(1)

    assert cc is not None

    text = file.read().split("-" * 20)[0]
    file.close()

    echo("Extracting replicas...")
    dialogue = cc.extract_dialogue(text)

    echo("Connecting replicas into the play...")
    play = cc.connect_play(dialogue)

    colors = list(COLORS)
    random.shuffle(colors)
    actor_colors: dict[str, tuple[Span, str]] = {
        s.lemma_: (s, c) for s, c in zip(play.actors, itertools.cycle(colors)) if s
    }

    echo("Actors found:")
    echo(", ".join(style(s.text, fg=c) for _, (s, c) in actor_colors.items()))

    first_col_w = max(len(str(s)) for s in play.actors)
    for r, s in play.lines:
        if s:
            echo(style(" ", fg=actor_colors[s.lemma_][1]), nl=False)
        echo(f"{s!s:<{first_col_w}}  ", nl=False)
        echo(str(r))

    if with_text:
        echo("\nMarked play:")
        rs_indexed: dict[int, tuple[Span, Span | None]] = {
            r.start_char: (r, s) for r, s in play.lines
        }
        r_starts = list(rs_indexed.keys())
        r_start_i = 0
        for i, c in enumerate(text):
            replica: Span | None
            actor: Span | None
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
                if i == replica.start_char and actor:
                    echo(
                        style("", fg=actor_colors[actor.lemma_][1], reset=False),
                        nl=False,
                    )

                if i == replica.end_char:
                    r_start_i += 1
                    echo(style("", reset=True), nl=False)

            echo(c, nl=False)

    echo()


if __name__ == "__main__":
    cli()
