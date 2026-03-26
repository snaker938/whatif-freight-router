from __future__ import annotations

import math

from app.risk_model import cvar, quantile, robust_objective


def test_quantile_uses_hyndman_fan_type7_interpolation() -> None:
    values = [1.0, 2.0, 3.0, 4.0]

    assert quantile(values, 0.25) == 1.75
    assert quantile(values, 0.50) == 2.5
    assert quantile(values, 0.75) == 3.25


def test_cvar_matches_hand_computed_interpolated_tail_average() -> None:
    values = [1.0, 2.0, 3.0, 4.0]

    assert math.isclose(cvar(values, alpha=0.75), 3.625, rel_tol=0.0, abs_tol=1e-9)
    assert cvar(values, alpha=1.0) == 4.0


def test_robust_objective_surrogates_remain_monotone_in_tail_excess() -> None:
    mean_value = 10.0
    lower_tail = 11.0
    higher_tail = 13.0

    for family in ("cvar_excess", "entropic", "downside_semivariance"):
        low = robust_objective(
            mean_value=mean_value,
            cvar_value=lower_tail,
            risk_aversion=1.0,
            risk_family=family,
            risk_theta=0.75,
        )
        high = robust_objective(
            mean_value=mean_value,
            cvar_value=higher_tail,
            risk_aversion=1.0,
            risk_family=family,
            risk_theta=0.75,
        )
        assert high >= low >= mean_value
