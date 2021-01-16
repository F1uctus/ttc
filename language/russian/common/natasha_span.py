from natasha.doc import DocToken

from language.span import Span


class NatashaSpan(Span):
    def to_string(self):
        spkr = (self.speaker.text
                if isinstance(self.speaker, DocToken)
                else self.speaker)
        return f'{spkr}: {" ".join([t.text for t in self.tokens])}'