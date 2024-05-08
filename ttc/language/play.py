from dataclasses import dataclass, field
from typing import Optional, Callable, List, Tuple, Dict, Union

from spacy import Language
from spacy.tokens import Span


@dataclass
class Play:
    language: Language

    _rels: Dict[Span, Optional[Span]] = field(default_factory=dict)
    """Replica -> Actor"""

    _refs: Dict[Span, Optional[Span]] = field(default_factory=dict)
    """Reference -> Actor"""

    @property
    def lines(self):
        """Replica -> Actor"""
        return self._rels.items()

    @property
    def replicas(self):
        return self._rels.keys()

    def replicas_of(self, actor: Span):
        actor = self._rels.get(actor, actor)  # type: ignore
        for r, a in self._rels.items():
            if a and (a.start == actor.start or a.lemma_ == actor.lemma_):
                yield r

    @property
    def last_replica(self):
        return next(reversed(self.replicas), None)

    @property
    def actors(self):
        return self._rels.values()

    @property
    def last_actor(self):
        return self[lr] if (lr := self.last_replica) else None

    def penult(self) -> Optional[Span]:
        if len(self.actors) < 2:
            return None
        last_lemma = self.last_actor.lemma_
        for a in reversed(self._rels.values()):
            if a and a.lemma_ != last_lemma:
                return a
        return None

    def reference(self, word) -> Optional[Span]:
        return self._refs.get(word, None)

    def slice(self, predicate: Callable[[Span, Optional[Span]], bool]):
        return Play(
            self.language,
            {r: s for r, s in self._rels.items() if predicate(r, s)},
        )

    def __len__(self):
        return len(self._rels)

    def __contains__(self, item):
        return item in self._rels

    def __getitem__(self, item: Union[Span, int]):
        if isinstance(item, int):
            if item < 0:
                item += len(self._rels)
            for i, key in enumerate(self._rels.keys()):
                if i == item:
                    return self._rels[key]
            return list(self._rels)[item]
        return self._rels[item]

    def __setitem__(self, replica, val):
        if isinstance(val, Tuple):
            if (isinstance(actor := val[0], Span) or actor is None) and isinstance(
                ref_chain := val[1], List
            ):
                for ref in ref_chain:
                    self._refs[ref] = actor
                self._rels[replica] = actor
            else:
                raise ValueError
        else:
            self._rels[replica] = val

    def __delitem__(self, key):
        del self._rels[key]

    def __repr__(self):
        s = ""
        first_col_w = max(len(str(s)) for s in self._rels.values())
        for replica, actor in self._rels.items():
            s += f"{{:<{first_col_w}}} | {{:<200}}\n".format(str(actor), str(replica))
        return s
