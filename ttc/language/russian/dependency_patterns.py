SPEAKER_TO_SPECIAL_VERB = [
    {  # (anchor) speaker
        "RIGHT_ID": "speaker",
        "RIGHT_ATTRS": {
            "DEP": "nsubj",
        },
    },
    {  # speaker <--- special verb
        "LEFT_ID": "speaker",
        "REL_OP": "<",
        "RIGHT_ID": "special_verb",
        "RIGHT_ATTRS": {
            "POS": "VERB",
            "_": {"is_author_verb": True},
        },
    },
]

SPEAKER_TO_VERB_CONJ_SPECIAL_VERB = [
    {  # (anchor) speaker
        "RIGHT_ID": "speaker",
        "RIGHT_ATTRS": {
            "DEP": "nsubj",
        },
    },
    {  # speaker <--- verb
        "LEFT_ID": "speaker",
        "REL_OP": "<",
        "RIGHT_ID": "verb",
        "RIGHT_ATTRS": {
            "POS": "VERB",
        },
    },
    {  # verb --conj-> special verb
        "LEFT_ID": "verb",
        "REL_OP": ">",
        "RIGHT_ID": "special_verb",
        "RIGHT_ATTRS": {
            "POS": "VERB",
            "_": {"is_author_verb": True},
        },
    },
]
