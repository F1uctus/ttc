from dataclasses import dataclass
from typing import Optional, Callable, List, Dict, Tuple

from spacy import Language
from spacy.tokens import Span

from ttc.iterables import deduplicate


@dataclass
class Play:
    language: Language
    _rels: Dict[Span, Optional[Span]]
    """Replica -> Speaker"""

    @property
    def lines(self):
        """Replica -> Speaker"""
        return self._rels.items()

    @property
    def replicas(self):
        return self._rels.keys()

    @property
    def last_replica(self):
        return next(reversed(self.replicas), None)

    @property
    def speakers(self):
        return self._rels.values()

    @property
    def last_speaker(self):
        return self[lr] if (lr := self.last_replica) else None

    def unique_speaker_lemmas(self) -> List[str]:
        return deduplicate(
            [s.lemma_ if s else "" for s in self._rels.values()][::-1],
        )[::-1]

    def alternated(
        self,
        p_replica: Optional[Span],
        replica: Span,
    ) -> Optional[Span]:
        if not p_replica or replica._.start_line_no - p_replica._.end_line_no != 1:
            return None
        uniq_speakers = self.unique_speaker_lemmas()
        return (
            next(
                (s for s in self._rels.values() if s and s.lemma_ == uniq_speakers[-2]),
                None,
            )
            if len(uniq_speakers) >= 2
            else None
        )

    def slice(self, predicate: Callable[[Span, Optional[Span]], bool]):
        return Play(
            self.language,
            {r: s for r, s in self._rels.items() if predicate(r, s)},
        )

    def __len__(self):
        return len(self._rels)

    def __contains__(self, item):
        return item in self._rels

    def __getitem__(self, item):
        return self._rels[item]

    def __setitem__(self, replica, speaker):
        self._rels[replica] = speaker

    def __delitem__(self, key):
        del self._rels[key]

    def __repr__(self):
        s = ""
        first_col_w = max(len(str(s)) for s in self._rels.values())
        for replica, speaker in self._rels.items():
            s += f"{{:<{first_col_w}}}  {{:<200}}\n".format(str(speaker), str(replica))
        return s
