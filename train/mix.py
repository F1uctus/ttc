"""Temperature-sampled corpus mixing."""

import random


def temperature_weights(counts: dict[str, int], t: float) -> dict[str, float]:
    scaled = {k: v**t for k, v in counts.items() if v > 0}
    total = sum(scaled.values())
    return {k: v / total for k, v in scaled.items()}


class MixtureSampler:
    def __init__(
        self, examples_by_source: dict[str, list[dict]], t: float, seed: int
    ) -> None:
        self.data = {k: v for k, v in examples_by_source.items() if v}
        self.weights = temperature_weights({k: len(v) for k, v in self.data.items()}, t)
        self.rng = random.Random(seed)
        self.sources = list(self.weights)
        self.cum = []
        acc = 0.0
        for s in self.sources:
            acc += self.weights[s]
            self.cum.append(acc)

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        u = self.rng.random()
        source = next(s for s, c in zip(self.sources, self.cum) if u <= c)
        return self.rng.choice(self.data[source])
