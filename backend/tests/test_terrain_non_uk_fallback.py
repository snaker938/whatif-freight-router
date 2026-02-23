from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

import app.calibration_loader as calibration_loader
import app.carbon_model as carbon_model
import app.fuel_energy_model as fuel_energy_model
import app.main as main_module
import app.scenario as scenario_module
from app.calibration_loader import FuelPriceSnapshot
from app.departure_profile import DepartureMultiplier
from app.main import build_option
from app.models import CostToggles
from app.scenario import ScenarioMode, ScenarioPolicy
from app.settings import settings
from app.terrain_dem import TerrainCoverageError
from app.toll_engine import TollComputation


@pytest.fixture(autouse=True)
def _scenario_require_url_relaxed(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
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
        "resolve_carbon_price",
        lambda **_kwargs: carbon_model.CarbonPricingContext(
            price_per_kg=0.10,
            source="pytest_live",
            schedule_year=2026,
            scope_mode="ttw",
            uncertainty_low=0.08,
            uncertainty_high=0.12,
        ),
    )
    monkeypatch.setattr(
        main_module,
        "compute_toll_cost",
        lambda **_kwargs: TollComputation(
            contains_toll=False,
            toll_distance_km=0.0,
            toll_cost_gbp=0.0,
            confidence=1.0,
            source="pytest_live",
            details={"segments_matched": 0, "classified_steps": 0},
        ),
    )
    monkeypatch.setattr(
        fuel_energy_model,
        "load_fuel_price_snapshot",
        lambda as_of_utc=None: FuelPriceSnapshot(
            prices_gbp_per_l={"diesel": 1.52, "petrol": 1.58},
            grid_price_gbp_per_kwh=0.28,
            regional_multipliers={"uk_default": 1.0},
            as_of="2026-01-15T00:00:00Z",
            source="pytest_live",
            signature="pytest",
            live_diagnostics={"cache_hit": True, "source_url": "pytest://fuel"},
        ),
    )
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
            source="pytest_live",
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
            source="pytest_live",
            version="pytest",
            as_of_utc=datetime.now(UTC).isoformat(),
        ),
    )
    monkeypatch.setattr(
        main_module,
        "time_of_day_multiplier_uk",
        lambda departure_time_utc, **_kwargs: DepartureMultiplier(
            multiplier=1.0,
            profile_source="pytest_live",
            local_time_iso=departure_time_utc.isoformat() if departure_time_utc is not None else None,
            profile_day="weekday",
            profile_key="uk_default.mixed.weekday",
            profile_version="pytest",
            profile_as_of_utc=datetime.now(UTC).isoformat(),
            profile_refreshed_at_utc=datetime.now(UTC).isoformat(),
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
