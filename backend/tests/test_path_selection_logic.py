from __future__ import annotations

from app.main import _finalize_pareto_options, _select_ranked_candidate_routes
from app.models import GeoJSONLineString, RouteMetrics, RouteOption


def _option(route_id: str, *, duration: float, money: float, co2: float) -> RouteOption:
    return RouteOption(
        id=route_id,
        geometry=GeoJSONLineString(
            type="LineString",
            coordinates=[(-1.0, 52.0), (-0.5, 51.7)],
        ),
        metrics=RouteMetrics(
            distance_km=10.0,
            duration_s=duration,
            monetary_cost=money,
            emissions_kg=co2,
            avg_speed_kmh=40.0,
        ),
    )


def _raw_route(duration_s: float, lon0: float) -> dict[str, object]:
    return {
        "duration": duration_s,
        "geometry": {
            "type": "LineString",
            "coordinates": [[lon0, 52.0], [lon0 + 0.3, 51.7]],
        },
        "legs": [{"annotation": {"distance": [1000.0], "duration": [duration_s]}}],
    }


def test_finalize_pareto_prefers_true_nondominated_set() -> None:
    options = [
        _option("r0", duration=100.0, money=10.0, co2=10.0),
        _option("r1", duration=95.0, money=15.0, co2=15.0),
        _option("r2", duration=120.0, money=20.0, co2=20.0),
    ]

    out = _finalize_pareto_options(options, max_alternatives=5)
    ids = {o.id for o in out}
    assert ids == {"r0", "r1"}


def test_finalize_pareto_fallback_returns_multiple_choices() -> None:
    options = [
        _option("best", duration=100.0, money=10.0, co2=10.0),
        _option("dominated_a", duration=110.0, money=12.0, co2=12.0),
        _option("dominated_b", duration=130.0, money=14.0, co2=13.0),
    ]

    out = _finalize_pareto_options(options, max_alternatives=5)
    assert len(out) >= 2
    assert any(o.id == "best" for o in out)


def test_select_ranked_candidate_routes_sorts_by_duration() -> None:
    routes = [
        _raw_route(300.0, -1.5),
        _raw_route(120.0, -1.8),
        _raw_route(180.0, -1.2),
    ]
    selected = _select_ranked_candidate_routes(routes, max_routes=2)
    durations = [float(route["duration"]) for route in selected]
    assert durations == [120.0, 180.0]
