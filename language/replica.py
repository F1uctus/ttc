class Replica:
    def __init__(self, speaker, tokens):
        self.speaker = speaker
        self.tokens = tokens

    def to_string(self):
        return f'{self.speaker}: {" ".join(t.text for t in self.tokens)}'

    def __repr__(self):
        return self.to_string()
