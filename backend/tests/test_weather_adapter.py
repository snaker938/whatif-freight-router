from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

import app.calibration_loader as calibration_loader
import app.carbon_model as carbon_model
import app.fuel_energy_model as fuel_energy_model
import app.main as main_module
import app.scenario as scenario_module
from app.departure_profile import DepartureMultiplier
from app.main import build_option
from app.models import CostToggles, WeatherImpactConfig
from app.scenario import ScenarioMode, ScenarioPolicy
from app.toll_engine import TollComputation
from app.weather_adapter import (
    weather_incident_multiplier,
    weather_speed_multiplier,
    weather_summary,
)


@pytest.fixture(autouse=True)
def _stub_scenario_profile(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("STRICT_RUNTIME_TEST_BYPASS", "1")

    def _policy(*_args: Any, **_kwargs: Any) -> ScenarioPolicy:
        return ScenarioPolicy(
            duration_multiplier=1.0,
            incident_rate_multiplier=1.0,
            incident_delay_multiplier=1.0,
            fuel_consumption_multiplier=1.0,
            emissions_multiplier=1.0,
            stochastic_sigma_multiplier=1.0,
            source="pytest",
            version="pytest",
        )

    def _tod(
        departure_time_utc: datetime | None,
        *,
        route_points: list[tuple[float, float]] | None = None,
        road_class_counts: dict[str, int] | None = None,
    ) -> DepartureMultiplier:
        _ = (route_points, road_class_counts)
        hour = (
            int(departure_time_utc.astimezone(UTC).hour)
            if departure_time_utc is not None
            else 12
        )
        multiplier = 1.20 if 7 <= hour <= 10 else 0.90 if 0 <= hour <= 5 else 1.00
        return DepartureMultiplier(
            multiplier=multiplier,
            profile_source="pytest",
            local_time_iso=departure_time_utc.isoformat() if departure_time_utc is not None else None,
            profile_day="weekday",
            profile_key="uk_default.mixed.weekday",
            profile_version="pytest",
            profile_as_of_utc=datetime.now(UTC).isoformat(),
            profile_refreshed_at_utc=datetime.now(UTC).isoformat(),
        )

    monkeypatch.setattr(main_module, "resolve_scenario_profile", _policy)
    monkeypatch.setattr(scenario_module, "resolve_scenario_profile", _policy)
    monkeypatch.setattr(main_module, "time_of_day_multiplier_uk", _tod)
    monkeypatch.setattr(
        main_module,
        "compute_toll_cost",
        lambda **_kwargs: TollComputation(
            contains_toll=False,
            toll_distance_km=0.0,
            toll_cost_gbp=0.0,
            confidence=1.0,
            source="pytest",
            details={"segments_matched": 0, "classified_steps": 0},
        ),
    )
    monkeypatch.setattr(
        main_module,
        "resolve_carbon_price",
        lambda **_kwargs: carbon_model.CarbonPricingContext(
            price_per_kg=0.10,
            source="pytest",
            schedule_year=2026,
            scope_mode="ttw",
            uncertainty_low=0.08,
            uncertainty_high=0.12,
        ),
    )
    monkeypatch.setattr(
        fuel_energy_model,
        "load_fuel_price_snapshot",
        lambda as_of_utc=None: calibration_loader.FuelPriceSnapshot(
            prices_gbp_per_l={"diesel": 1.52, "petrol": 1.58, "lng": 1.05},
            grid_price_gbp_per_kwh=0.28,
            regional_multipliers={"uk_default": 1.0},
            as_of=datetime.now(UTC).isoformat(),
            source="pytest",
            signature="pytest",
            live_diagnostics={},
        ),
    )
    monkeypatch.setattr(
        main_module,
        "_route_stochastic_uncertainty",
        lambda *args, **kwargs: (
            {"duration_p50_s": 0.0, "monetary_p50_gbp": 0.0, "emissions_p50_kg": 0.0},
            {"sample_count": 1},
        ),
    )
    monkeypatch.setattr(
        main_module,
        "load_risk_normalization_reference",
        lambda **_kwargs: calibration_loader.RiskNormalizationReference(
            duration_s_per_km=90.0,
            monetary_gbp_per_km=1.0,
            emissions_kg_per_km=0.5,
            source="pytest",
            version="pytest",
            as_of_utc=datetime.now(UTC).isoformat(),
            corridor_bucket="uk_default",
            day_kind="weekday",
            local_time_slot="h12",
        ),
    )
    monkeypatch.setattr(
        main_module,
        "load_fuel_consumption_calibration",
        lambda: SimpleNamespace(
            source="pytest",
            version="pytest",
            as_of_utc=datetime.now(UTC).isoformat(),
        ),
    )
    calibration_loader.load_scenario_profiles.cache_clear()
    yield
    calibration_loader.load_scenario_profiles.cache_clear()


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
