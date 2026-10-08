import pytest

pytestmark = pytest.mark.train


def test_temperature_weights_flatten():
    from train.mix import temperature_weights

    counts = {"pdnc": 36000, "jy": 8000, "ru": 500}
    flat = temperature_weights(counts, 0.5)
    raw = temperature_weights(counts, 1.0)
    assert abs(sum(flat.values()) - 1) < 1e-9
    assert flat["ru"] > raw["ru"]
    assert flat["pdnc"] < raw["pdnc"]


def test_mixture_sampler_deterministic():
    from train.mix import MixtureSampler

    data = {"a": [{"i": 1}, {"i": 2}], "b": [{"i": 3}]}
    s1 = MixtureSampler(data, t=0.5, seed=1)
    s2 = MixtureSampler(data, t=0.5, seed=1)
    assert [next(s1)["i"] for _ in range(10)] == [next(s2)["i"] for _ in range(10)]
