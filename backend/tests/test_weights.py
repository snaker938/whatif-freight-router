from app.objectives_selection import normalise_weights


def test_normalise_weights_zero_sum():
    wt, wm, we = normalise_weights(0, 0, 0)
    assert abs(wt - 1 / 3) < 1e-9
    assert abs(wm - 1 / 3) < 1e-9
    assert abs(we - 1 / 3) < 1e-9


def test_normalise_weights():
    wt, wm, we = normalise_weights(2, 1, 1)
    assert abs(wt - 0.5) < 1e-9
    assert abs(wm - 0.25) < 1e-9
    assert abs(we - 0.25) < 1e-9
