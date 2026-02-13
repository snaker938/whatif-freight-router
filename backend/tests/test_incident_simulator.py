from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from app.incident_simulator import simulate_incident_events
from app.main import build_option
from app.models import CostToggles, IncidentSimulatorConfig
from app.scenario import ScenarioMode


def _route(*, distance_m: float, duration_s: float) -> dict[str, Any]:
    return {
        "distance": distance_m,
        "duration": duration_s,
        "geometry": {
            "type": "LineString",
            "coordinates": [[-1.8904, 52.4862], [-1.3, 52.1], [-0.1276, 51.5072]],
        },
        "legs": [
            {
                "annotation": {
                    "distance": [distance_m / 3.0, distance_m / 3.0, distance_m / 3.0],
                    "duration": [duration_s / 3.0, duration_s / 3.0, duration_s / 3.0],
                }
            }
        ],
    }


def _config(seed: int) -> IncidentSimulatorConfig:
    return IncidentSimulatorConfig(
        enabled=True,
        seed=seed,
        dwell_rate_per_100km=120.0,
        accident_rate_per_100km=80.0,
        closure_rate_per_100km=40.0,
        max_events_per_route=20,
    )


def test_incident_simulator_is_deterministic_for_same_seed() -> None:
    distances = [12_000.0, 10_000.0, 8_000.0]
    durations = [700.0, 600.0, 500.0]
    cfg = _config(seed=42)
    first = simulate_incident_events(
        config=cfg,
        segment_distances_m=distances,
        segment_durations_s=durations,
        weather_incident_multiplier=1.4,
        route_key="route-abc",
    )
    second = simulate_incident_events(
        config=cfg,
        segment_distances_m=distances,
        segment_durations_s=durations,
        weather_incident_multiplier=1.4,
        route_key="route-abc",
    )
    assert [item.model_dump(mode="json") for item in first] == [
        item.model_dump(mode="json") for item in second
    ]


def test_incident_simulator_changes_with_seed_and_respects_bounds() -> None:
    distances = [12_000.0, 10_000.0, 8_000.0]
    durations = [700.0, 600.0, 500.0]
    first = simulate_incident_events(
        config=_config(seed=7),
        segment_distances_m=distances,
        segment_durations_s=durations,
        weather_incident_multiplier=1.2,
        route_key="route-abc",
    )
    second = simulate_incident_events(
        config=_config(seed=8),
        segment_distances_m=distances,
        segment_durations_s=durations,
        weather_incident_multiplier=1.2,
        route_key="route-abc",
    )

    assert [item.event_id for item in first] != [item.event_id for item in second]
    assert all(item.delay_s >= 0.0 for item in first)
    assert all(0 <= item.segment_index < len(distances) for item in first)


def test_build_option_includes_incident_stage_and_metrics() -> None:
    option = build_option(
        _route(distance_m=42_000.0, duration_s=2_700.0),
        option_id="incident",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.NO_SHARING,
        cost_toggles=CostToggles(),
        incident_simulation=_config(seed=123),
        departure_time_utc=datetime(2026, 2, 12, 8, 30, tzinfo=UTC),
    )
    stages = [str(item["stage"]) for item in option.eta_timeline]
    assert "incidents" in stages
    assert option.metrics.incident_delay_s >= 0.0
    assert option.incident_events
    assert all(event.source == "synthetic" for event in option.incident_events)

