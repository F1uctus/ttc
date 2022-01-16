from typing import Union

from spacy.tokens import Token

from ttc.language import Speaker


def morph_equals(
    a: Union[Token, Speaker],
    b: Union[Token, Speaker],
    *morphs: str,  # Morph
) -> bool:
    if a.morph is None or b.morph is None:
        return False
    return all(a.morph.get(k) == b.morph.get(k) for k in morphs)
