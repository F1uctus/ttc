from ttc.eval import b_cubed


def test_b_cubed_perfect():
    clusters = [{1, 2}, {3}]
    assert b_cubed(clusters, clusters) == (1.0, 1.0, 1.0)


def test_b_cubed_merge_hurts_precision():
    p, r, f1 = b_cubed([{1, 2}, {3, 4}], [{1, 2, 3, 4}])
    assert r == 1.0 and p == 0.5 and 0.6 < f1 < 0.7


def test_span_f1():
    from ttc.eval import span_f1

    p, r, f1 = span_f1([(0, 5), (10, 14)], [(0, 5), (20, 24)])
    assert p == 0.5 and r == 0.5 and f1 == 0.5
    assert span_f1([], []) == (0.0, 0.0, 0.0)
