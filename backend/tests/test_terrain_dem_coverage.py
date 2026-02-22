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


def _uk_route() -> dict[str, Any]:
    coords = [[-1.8904, 52.4862], [-1.2, 52.0], [-0.1276, 51.5072]]
    return {
        "distance": 42_000.0,
        "duration": 2_600.0,
        "geometry": {"type": "LineString", "coordinates": coords},
        "legs": [
            {
                "annotation": {
                    "distance": [14_000.0, 14_000.0, 14_000.0],
                    "duration": [860.0, 860.0, 880.0],
                }
            }
        ],
    }


def test_terrain_summary_reports_dem_coverage_for_uk_route(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "terrain_allow_synthetic_grid", True)
    option = build_option(
        _uk_route(),
        option_id="terrain_cov",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.NO_SHARING,
        cost_toggles=CostToggles(),
        terrain_profile="hilly",
        departure_time_utc=datetime(2026, 2, 18, 8, 30, tzinfo=UTC),
    )
    assert option.terrain_summary is not None
    assert option.terrain_summary.source == "dem_real"
    assert option.terrain_summary.coverage_ratio >= settings.terrain_dem_coverage_min_uk
    assert option.terrain_summary.confidence > 0.0
    assert option.terrain_summary.version
    assert option.weather_summary is not None
    assert "terrain_live_source" in option.weather_summary
    assert "terrain_live_cache_hit_rate" in option.weather_summary
