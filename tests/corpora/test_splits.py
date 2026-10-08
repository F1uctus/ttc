from ttc.corpora.splits import split_of


def test_split_deterministic_and_roughly_proportional():
    ids = [f"pdnc/novel/{i}" for i in range(1000)]
    first = [split_of(i) for i in ids]
    assert first == [split_of(i) for i in ids]  # deterministic
    heldout = first.count("heldout")
    assert 120 <= heldout <= 280  # ~20% of 1000, generous tolerance


def test_split_values():
    assert {split_of(f"x/{i}") for i in range(50)} <= {"tune", "heldout"}
