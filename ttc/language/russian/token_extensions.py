from spacy.tokens import Token, Span


def morph_equals(
    a: Token,
    b: Token,
    *morphs: str,  # Morph
) -> bool:
    if a.morph is None or b.morph is None:
        return False
    return all(a.morph.get(k) == b.morph.get(k) for k in morphs)


def token_as_span(t: Token) -> Span:
    return t.doc[t.i : t.i + 1]
