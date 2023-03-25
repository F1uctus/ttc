SPEAKING_VERB_TO_SPEAKER = [
    {  # (anchor) speaker
        "RIGHT_ID": "speaker",
        "RIGHT_ATTRS": {
            "DEP": {"IN": ["nsubj", "obj"]},
            "POS": {"NOT_IN": ["SPACE", "PUNCT"]},
        },
    },
    {  # speaker <--- speaking verb
        "LEFT_ID": "speaker",
        "REL_OP": "<",
        "RIGHT_ID": "speaking_verb",
        "RIGHT_ATTRS": {
            "POS": "VERB",
            "_": {"is_speaking_verb": True},
        },
    },
]

SPEAKING_VERB_CONJUNCT_SPEAKER = [
    {  # (anchor) speaker
        "RIGHT_ID": "speaker",
        "RIGHT_ATTRS": {
            "DEP": "nsubj",
            "POS": {"NOT_IN": ["SPACE", "PUNCT"]},
        },
    },
    {  # speaker <--- conjunct
        "LEFT_ID": "speaker",
        "REL_OP": "<",
        "RIGHT_ID": "conjunct",
        "RIGHT_ATTRS": {},
    },
    {  # conjunct ---> speaking verb
        "LEFT_ID": "conjunct",
        "REL_OP": ">>",  # TODO: Replace by search in Token.conjuncts?
        "RIGHT_ID": "speaking_verb",
        "RIGHT_ATTRS": {
            "POS": "VERB",
            "_": {"is_speaking_verb": True},
        },
    },
]
