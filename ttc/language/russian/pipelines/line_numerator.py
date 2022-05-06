from spacy import Language
from spacy.tokens import Doc, Token, Span

NAME = "line_numerator"

if not Token.has_extension("line_no"):
    Token.set_extension("line_no", default=1)

if not Span.has_extension("start_line_no"):
    Span.set_extension("start_line_no", getter=lambda s: s[0]._.line_no)

if not Span.has_extension("end_line_no"):
    Span.set_extension("end_line_no", getter=lambda s: s[-1]._.line_no)


@Language.component(NAME)
def _mark_line_numbers(doc: Doc):
    line_no = 1

    for token in doc:
        token._.line_no = line_no
        if token._.is_newline:
            line_no += 1

    return doc
