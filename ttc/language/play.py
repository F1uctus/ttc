from dataclasses import dataclass, field

from spacy import Language
from spacy.tokens import Span


@dataclass
class Play:
    language: Language

    _rels: dict[Span, Span | None] = field(default_factory=dict)
    """Replica -> Actor"""

    _refs: dict[Span, Span | None] = field(default_factory=dict)
    """Reference -> Actor"""

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

    def _actor_key(self, span: Span | None) -> str:
        if not span:
            return ""
        if any(t.pos_ == "PROPN" or t.ent_type_ == "PER" for t in span):
            propn = " ".join(t.lemma_.lower() for t in span if t.pos_ == "PROPN")
            return propn or span.lemma_.lower()
        if span.root.pos_ == "PRON":
            return span.lemma_.lower()
        return span.text.lower()

    def penult(self) -> Span | None:
        if not (last := self.last_actor):
            return None
        last_key = self._actor_key(last)
        for actor in reversed(list(self._rels.values())):
            if actor and self._actor_key(actor) != last_key:
                return actor
        return None

    def reference(self, word) -> Span | None:
        return self._refs.get(word, None)

    def __len__(self):
        return len(self._rels)

    def __contains__(self, item):
        return item in self._rels

    def __getitem__(self, item):
        return self._rels[item]

    def __setitem__(self, replica, val):
        if isinstance(val, tuple):
            if (isinstance(actor := val[0], Span) or actor is None) and isinstance(
                ref_chain := val[1], list
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
