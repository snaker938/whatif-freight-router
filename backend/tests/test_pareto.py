from app.pareto import dominates, pareto_filter


def test_dominates_basic():
    assert dominates((1, 2, 3), (2, 2, 3))
    assert not dominates((2, 2, 3), (1, 2, 3))
    assert not dominates((1, 2, 3), (1, 2, 3))


def test_pareto_filter():
    items = [
        (10, 10, 10),
        (9, 11, 10),
        (8, 9, 9),
        (12, 12, 12),
    ]
    kept = pareto_filter(items, key=lambda x: x)
    assert (12, 12, 12) not in kept
    assert (8, 9, 9) in kept
