from app.models import GeoJSONLineString, RouteMetrics, RouteOption
from app.objectives_selection import normalise_weights, pick_best_by_weighted_sum


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


def _option(route_id: str) -> RouteOption:
    return RouteOption(
        id=route_id,
        geometry=GeoJSONLineString(type="LineString", coordinates=[(-1.0, 52.0), (-0.5, 51.5)]),
        metrics=RouteMetrics(
            distance_km=10.0,
            duration_s=100.0,
            monetary_cost=50.0,
            emissions_kg=20.0,
            avg_speed_kmh=60.0,
        ),
    )


def test_weighted_sum_uses_route_id_tie_break_for_equal_scores() -> None:
    later = _option("z_route")
    earlier = _option("a_route")

    selected = pick_best_by_weighted_sum(
        [later, earlier],
        w_time=1.0,
        w_money=1.0,
        w_co2=1.0,
    )

    assert selected.id == "a_route"
