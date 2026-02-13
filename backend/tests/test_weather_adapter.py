from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from app.main import build_option
from app.models import CostToggles, WeatherImpactConfig
from app.scenario import ScenarioMode
from app.weather_adapter import weather_incident_multiplier, weather_speed_multiplier, weather_summary


def _route(*, distance_m: float, duration_s: float) -> dict[str, Any]:
    return {
        "distance": distance_m,
        "duration": duration_s,
        "geometry": {
            "type": "LineString",
            "coordinates": [[-1.8904, 52.4862], [-1.2, 52.0], [-0.1276, 51.5072]],
        },
        "legs": [
            {
                "annotation": {
                    "distance": [distance_m / 2.0, distance_m / 2.0],
                    "duration": [duration_s / 2.0, duration_s / 2.0],
                }
            }
        ],
    }


def test_weather_multiplier_clamps_and_is_deterministic() -> None:
    cfg = WeatherImpactConfig(enabled=True, profile="storm", intensity=2.0)
    first = weather_speed_multiplier(cfg)
    second = weather_speed_multiplier(cfg)
    assert first == second
    assert first > 1.0

    summary = weather_summary(cfg)
    assert summary["enabled"] is True
    assert float(summary["intensity"]) == 2.0


def test_weather_incident_multiplier_honors_uplift_toggle() -> None:
    cfg_off = WeatherImpactConfig(
        enabled=True,
        profile="snow",
        intensity=1.0,
        apply_incident_uplift=False,
    )
    cfg_on = WeatherImpactConfig(
        enabled=True,
        profile="snow",
        intensity=1.0,
        apply_incident_uplift=True,
    )
    assert weather_incident_multiplier(cfg_off) == 1.0
    assert weather_incident_multiplier(cfg_on) > 1.0


def test_build_option_weather_profile_adds_eta_stage_and_delay() -> None:
    route = _route(distance_m=45_000.0, duration_s=2_700.0)
    clear = build_option(
        route,
        option_id="clear",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.FULL_SHARING,
        cost_toggles=CostToggles(),
        weather=WeatherImpactConfig(enabled=True, profile="clear", intensity=1.0),
        departure_time_utc=datetime(2026, 2, 12, 8, 30, tzinfo=UTC),
    )
    rain = build_option(
        route,
        option_id="rain",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.FULL_SHARING,
        cost_toggles=CostToggles(),
        weather=WeatherImpactConfig(enabled=True, profile="rain", intensity=1.0),
        departure_time_utc=datetime(2026, 2, 12, 8, 30, tzinfo=UTC),
    )

    assert rain.metrics.duration_s > clear.metrics.duration_s
    assert rain.metrics.weather_delay_s > 0.0
    assert rain.weather_summary is not None
    assert str(rain.weather_summary["profile"]) == "rain"
    stages = [str(item["stage"]) for item in rain.eta_timeline]
    assert stages == ["baseline", "time_of_day", "scenario", "weather", "gradient"]
    assert any("Weather profile 'rain'" in msg for msg in rain.eta_explanations)
