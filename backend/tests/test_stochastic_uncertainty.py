from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from app.main import build_option
from app.models import CostToggles, StochasticConfig
from app.scenario import ScenarioMode


def _route(*, distance_m: float, duration_s: float) -> dict[str, Any]:
    coords = [[-1.8904, 52.4862], [-1.2, 52.0], [-0.1276, 51.5072]]
    return {
        "distance": distance_m,
        "duration": duration_s,
        "geometry": {"type": "LineString", "coordinates": coords},
        "legs": [
            {
                "annotation": {
                    "distance": [distance_m / 2.0, distance_m / 2.0],
                    "duration": [duration_s / 2.0, duration_s / 2.0],
                }
            }
        ],
    }


def test_stochastic_uncertainty_is_deterministic_for_same_seed() -> None:
    route = _route(distance_m=50_000.0, duration_s=3_000.0)
    stochastic = StochasticConfig(enabled=True, seed=123, sigma=0.1, samples=40)

    first = build_option(
        route,
        option_id="route_seed",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.NO_SHARING,
        cost_toggles=CostToggles(),
        stochastic=stochastic,
        departure_time_utc=datetime(2026, 2, 12, 9, 0, tzinfo=UTC),
    )
    second = build_option(
        route,
        option_id="route_seed",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.NO_SHARING,
        cost_toggles=CostToggles(),
        stochastic=stochastic,
        departure_time_utc=datetime(2026, 2, 12, 9, 0, tzinfo=UTC),
    )

    assert first.uncertainty is not None
    assert second.uncertainty is not None
    assert first.uncertainty == second.uncertainty


def test_stochastic_uncertainty_changes_for_different_seed() -> None:
    route = _route(distance_m=50_000.0, duration_s=3_000.0)
    base_kwargs = {
        "route": route,
        "option_id": "route_seed",
        "vehicle_type": "rigid_hgv",
        "scenario_mode": ScenarioMode.NO_SHARING,
        "cost_toggles": CostToggles(),
        "departure_time_utc": datetime(2026, 2, 12, 9, 0, tzinfo=UTC),
    }

    first = build_option(**base_kwargs, stochastic=StochasticConfig(enabled=True, seed=1))
    second = build_option(**base_kwargs, stochastic=StochasticConfig(enabled=True, seed=2))

    assert first.uncertainty is not None
    assert second.uncertainty is not None
    assert first.uncertainty["p95_duration_s"] != second.uncertainty["p95_duration_s"]
