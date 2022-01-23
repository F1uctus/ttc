import click

import ttc


@click.command(name="extract-dialogue")
@click.argument("input", type=click.File("rb"), nargs=-1)
@click.argument("language", type=str, nargs=-1)
def extract_dialogue(input, language):
    cc = ttc.load(language)
    dialogue = cc.extract_dialogue(input.read())
    print(dialogue)
