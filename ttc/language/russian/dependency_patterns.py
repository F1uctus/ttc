from ttc.language.russian.constants import REFERRAL_PRON

ACTION_VERB_TO_ACTOR = [
    {  # (anchor) actor
        "RIGHT_ID": "actor",
        "RIGHT_ATTRS": {
            "DEP": {"IN": ["nsubj", "obj", "acl", "acl:relcl"]},
            "POS": {"NOT_IN": ["SPACE", "PUNCT"]},
        },
    },
    {  # actor <--- action verb
        "LEFT_ID": "actor",
        "REL_OP": "<",
        "RIGHT_ID": "action_verb",
        "RIGHT_ATTRS": {
            "POS": "VERB",
            "_": {"is_action_verb": True},
        },
    },
]

ACTION_VERB_CONJUNCT_ACTOR = [
    {  # (anchor) actor
        "RIGHT_ID": "actor",
        "RIGHT_ATTRS": {
            "DEP": "nsubj",
            "POS": {"NOT_IN": ["SPACE", "PUNCT"]},
        },
    },
    {  # actor <--- conjunct
        "LEFT_ID": "actor",
        "REL_OP": "<",
        "RIGHT_ID": "conjunct",
        "RIGHT_ATTRS": {},
    },
    {  # conjunct ---> action verb
        "LEFT_ID": "conjunct",
        "REL_OP": ">>",  # TODO: Replace by search in Token.conjuncts?
        "RIGHT_ID": "action_verb",
        "RIGHT_ATTRS": {
            "POS": "VERB",
            "_": {"is_action_verb": True},
        },
    },
]

VOICE_TO_AMOD = [
    {
        "RIGHT_ID": "voice_gender_specifier",
        "RIGHT_ATTRS": {
            "DEP": "amod",
            "LEMMA": {"IN": ["женский", "мужской"] + list(REFERRAL_PRON)},
        },
    },
    {
        "LEFT_ID": "voice_gender_specifier",
        "REL_OP": "<",
        "RIGHT_ID": "voice_noun",
        "RIGHT_ATTRS": {
            "LEMMA": "голос",
        },
    },
]
