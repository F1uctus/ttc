from spacy.symbols import AUX, cop  # type: ignore
from spacy.tokens import Token

from ttc.language.common.constants import HYPHENS
from ttc.language.russian.constants import ACTION_VERBS


def is_hyphen(self: Token):
    return self.text in HYPHENS


def is_action_verb(self: Token):
    """Used in dependency matching"""
    return any(v in self.lemma_ for v in ACTION_VERBS)


def is_copula(self: Token):
    """e.g.: ее голос [cop=был]->тихим, как шепот"""
    return self.dep == cop and self.pos == AUX


def is_not_second_person(self: Token):
    return "Person=Second" not in self.morph


TOKEN_EXTENSIONS = {
    name: {"getter": f}
    for name, f in locals().items()
    if callable(f) and f.__module__ == __name__
}
