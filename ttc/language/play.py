from dataclasses import dataclass
from typing import Optional, Dict

from spacy import Language
from spacy.tokens import Span


@dataclass
class Play:
    language: Language
    content: Dict[Span, Optional[Span]]

    def __repr__(self):
        s = ""
        first_col_w = max(len(str(s)) for s in self.content.values())
        for replica, speaker in self.content.items():
            s += f"{{:<{first_col_w}}}  {{:<200}}\n".format(str(speaker), str(replica))
        return s
