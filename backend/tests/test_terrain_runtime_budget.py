from __future__ import annotations

import time
from collections.abc import Iterator
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

import app.calibration_loader as calibration_loader
import app.carbon_model as carbon_model
import app.departure_profile as departure_profile_module
import app.fuel_energy_model as fuel_energy_model
import app.main as main_module
import app.scenario as scenario_module
from app.calibration_loader import FuelPriceSnapshot
from app.departure_profile import DepartureMultiplier
from app.main import build_option
from app.models import CostToggles
from app.scenario import ScenarioMode, ScenarioPolicy
from app.settings import settings
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
        departure_profile_module,
        "time_of_day_multiplier_uk",
        lambda *_args, **_kwargs: DepartureMultiplier(
            multiplier=1.0,
            profile_source="pytest",
            local_time_iso=None,
            profile_day="weekday",
            profile_key="pytest.departure",
        ),
    )
    monkeypatch.setattr(
        main_module,
        "time_of_day_multiplier_uk",
        lambda *_args, **_kwargs: DepartureMultiplier(
            multiplier=1.0,
            profile_source="pytest",
            local_time_iso=None,
            profile_day="weekday",
            profile_key="pytest.departure",
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
            source="pytest",
            details={},
        ),
    )
    monkeypatch.setattr(
        main_module,
        "_route_stochastic_uncertainty",
        lambda *_args, **_kwargs: (
            {"q95_duration_s": 1.0, "q95_monetary_cost": 1.0, "q95_emissions_kg": 1.0},
            {"sample_count": 0, "seed": "pytest", "sigma": 0.0},
        ),
    )
    monkeypatch.setattr(
        main_module,
        "load_risk_normalization_reference",
        lambda **_kwargs: SimpleNamespace(
            source="pytest",
            version="pytest",
            as_of_utc="2026-01-15T00:00:00Z",
            corridor_bucket="uk_default",
            day_kind="weekday",
            local_time_slot="h08",
        ),
    )
    calibration_loader.load_scenario_profiles.cache_clear()
    yield
    calibration_loader.load_scenario_profiles.cache_clear()


def _route() -> dict[str, Any]:
    coords = [[-4.2, 57.5], [-3.0, 56.3], [-1.9, 54.5], [-0.1, 51.5]]
    return {
        "distance": 180_000.0,
        "duration": 11_000.0,
        "geometry": {"type": "LineString", "coordinates": coords},
        "legs": [
            {
                "annotation": {
                    "distance": [60_000.0, 60_000.0, 60_000.0],
                    "duration": [3_666.0, 3_667.0, 3_667.0],
                }
            }
        ],
    }


def test_terrain_path_runtime_p95_under_two_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "terrain_allow_synthetic_grid", True)
    timings_ms: list[float] = []
    route = _route()
    for idx in range(24):
        t0 = time.perf_counter()
        build_option(
            route,
            option_id=f"terrain_runtime_{idx}",
            vehicle_type="rigid_hgv",
            scenario_mode=ScenarioMode.NO_SHARING,
            cost_toggles=CostToggles(),
            terrain_profile="hilly",
            departure_time_utc=datetime(2026, 2, 18, 8, 30, tzinfo=UTC),
        )
        timings_ms.append((time.perf_counter() - t0) * 1000.0)

    sorted_timings = sorted(timings_ms)
    p95_idx = max(0, min(len(sorted_timings) - 1, int(0.95 * len(sorted_timings)) - 1))
    assert sorted_timings[p95_idx] < 2000.0
