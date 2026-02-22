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
from app.terrain_dem import TerrainCoverageError


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


def _non_uk_route() -> dict[str, Any]:
    coords = [[-122.4194, 37.7749], [-121.8863, 37.3382], [-118.2437, 34.0522]]
    return {
        "distance": 610_000.0,
        "duration": 22_000.0,
        "geometry": {"type": "LineString", "coordinates": coords},
        "legs": [
            {
                "annotation": {
                    "distance": [305_000.0, 305_000.0],
                    "duration": [11_000.0, 11_000.0],
                }
            }
        ],
    }


def test_non_uk_route_is_rejected_as_unsupported_region() -> None:
    with pytest.raises(TerrainCoverageError) as exc_info:
        build_option(
            _non_uk_route(),
            option_id="terrain_non_uk",
            vehicle_type="rigid_hgv",
            scenario_mode=ScenarioMode.NO_SHARING,
            cost_toggles=CostToggles(),
            terrain_profile="hilly",
            departure_time_utc=datetime(2026, 2, 18, 8, 30, tzinfo=UTC),
        )
    assert exc_info.value.reason_code == "terrain_region_unsupported"
