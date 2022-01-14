SPEAKER_TO_SPEAKING_VERB = [
    {  # (anchor) speaker
        "RIGHT_ID": "speaker",
        "RIGHT_ATTRS": {
            "DEP": "nsubj",
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

SPEAKER_CONJUNCT_SPEAKING_VERB = [
    {  # (anchor) speaker
        "RIGHT_ID": "speaker",
        "RIGHT_ATTRS": {
            "DEP": "nsubj",
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
        "REL_OP": ">",
        "RIGHT_ID": "speaking_verb",
        "RIGHT_ATTRS": {
            "POS": "VERB",
            "_": {"is_speaking_verb": True},
        },
    },
]
