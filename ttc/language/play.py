from dataclasses import dataclass
from typing import Optional, Callable, List, Dict

from spacy import Language
from spacy.tokens import Span

from ttc.iterables import deduplicate


@dataclass
class Play:
    language: Language
    _rels: Dict[Span, Optional[Span]]
    """Replica -> Actor"""

    @property
    def lines(self):
        """Replica -> Actor"""
        return self._rels.items()

    @property
    def replicas(self):
        return self._rels.keys()

    @property
    def last_replica(self):
        return next(reversed(self.replicas), None)

    @property
    def actors(self):
        return self._rels.values()

    @property
    def last_actor(self):
        return self[lr] if (lr := self.last_replica) else None

    def unique_actor_lemmas(self) -> List[str]:
        return deduplicate(
            [s.lemma_ if s else "" for s in self._rels.values()][::-1],
        )[::-1]

    def alternated(self) -> Optional[Span]:
        uniq = self.unique_actor_lemmas()
        return (
            next(
                (s for s in self._rels.values() if s and s.lemma_ == uniq[-2]),
                None,
            )
            if len(uniq) >= 2
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

    def __setitem__(self, replica, actor):
        self._rels[replica] = actor

    def __delitem__(self, key):
        del self._rels[key]

    def __repr__(self):
        s = ""
        first_col_w = max(len(str(s)) for s in self._rels.values())
        for replica, actor in self._rels.items():
            s += f"{{:<{first_col_w}}} | {{:<200}}\n".format(str(actor), str(replica))
        return s
