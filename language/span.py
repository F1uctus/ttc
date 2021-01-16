from abc import ABCMeta, abstractmethod


class Span(metaclass=ABCMeta):
    def __init__(self, speaker, tokens):
        self.speaker = speaker
        self.tokens = tokens

    @abstractmethod
    def to_string(self): pass

    def __repr__(self):
        return self.to_string()