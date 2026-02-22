from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

import app.calibration_loader as calibration_loader
import app.main as main_module
import app.scenario as scenario_module
from app.main import build_option
from app.models import CostToggles
from app.scenario import ScenarioMode, ScenarioPolicy
from app.settings import settings


@pytest.fixture(autouse=True)
def _scenario_require_url_relaxed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STRICT_RUNTIME_TEST_BYPASS", "1")
    monkeypatch.setattr(settings, "strict_live_data_required", False)
    monkeypatch.setattr(settings, "live_runtime_data_enabled", False)
    monkeypatch.setattr(settings, "live_scenario_require_url_in_strict", False)
    monkeypatch.setattr(settings, "live_scenario_allow_signed_fallback", True)
    monkeypatch.setattr(settings, "live_fuel_require_url_in_strict", False)
    monkeypatch.setattr(settings, "live_fuel_allow_signed_fallback", True)
    monkeypatch.setattr(settings, "live_toll_tariffs_require_url_in_strict", False)
    monkeypatch.setattr(settings, "live_toll_tariffs_allow_signed_fallback", True)
    monkeypatch.setattr(settings, "live_toll_topology_require_url_in_strict", False)
    monkeypatch.setattr(settings, "live_toll_topology_allow_signed_fallback", True)
    monkeypatch.setattr(settings, "live_carbon_require_url_in_strict", False)
    monkeypatch.setattr(settings, "live_carbon_allow_signed_fallback", True)
    monkeypatch.setattr(
        main_module,
        "resolve_scenario_profile",
        lambda *_args, **_kwargs: ScenarioPolicy(
            duration_multiplier=1.0,
            incident_rate_multiplier=1.0,
            incident_delay_multiplier=1.0,
            fuel_consumption_multiplier=1.0,
            emissions_multiplier=1.0,
            stochastic_sigma_multiplier=1.0,
            source="test",
            version="test",
        ),
    )
    monkeypatch.setattr(
        scenario_module,
        "resolve_scenario_profile",
        lambda *_args, **_kwargs: ScenarioPolicy(
            duration_multiplier=1.0,
            incident_rate_multiplier=1.0,
            incident_delay_multiplier=1.0,
            fuel_consumption_multiplier=1.0,
            emissions_multiplier=1.0,
            stochastic_sigma_multiplier=1.0,
            source="test",
            version="test",
        ),
    )
    calibration_loader.load_scenario_profiles.cache_clear()
    yield
    calibration_loader.load_scenario_profiles.cache_clear()


def _route() -> dict[str, Any]:
    coords = [[-3.2, 57.0], [-2.4, 56.2], [-1.6, 54.9], [-0.1, 51.5]]
    return {
        "distance": 120_000.0,
        "duration": 7_400.0,
        "geometry": {"type": "LineString", "coordinates": coords},
        "legs": [
            {
                "annotation": {
                    "distance": [30_000.0, 30_000.0, 30_000.0, 30_000.0],
                    "duration": [1_850.0, 1_850.0, 1_850.0, 1_850.0],
                }
            }
        ],
    }


def test_hillier_profile_increases_duration_and_emissions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "terrain_allow_synthetic_grid", True)
    flat = build_option(
        _route(),
        option_id="terrain_uplift_flat",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.NO_SHARING,
        cost_toggles=CostToggles(use_tolls=False),
        terrain_profile="flat",
        departure_time_utc=datetime(2026, 2, 18, 3, 30, tzinfo=UTC),
    )
    hilly = build_option(
        _route(),
        option_id="terrain_uplift_hilly",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.NO_SHARING,
        cost_toggles=CostToggles(use_tolls=False),
        terrain_profile="hilly",
        departure_time_utc=datetime(2026, 2, 18, 3, 30, tzinfo=UTC),
    )
    assert hilly.metrics.duration_s >= flat.metrics.duration_s
    assert hilly.metrics.emissions_kg >= flat.metrics.emissions_kg
    assert hilly.terrain_summary is not None
    assert hilly.terrain_summary.source == "dem_real"
