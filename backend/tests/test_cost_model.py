from __future__ import annotations

import math
import os
from datetime import UTC, datetime
from typing import Any

import pytest

import app.calibration_loader as calibration_loader
import app.carbon_model as carbon_model
from app.main import build_option
from app.model_data_errors import ModelDataError
from app.models import CostToggles
from app.scenario import ScenarioMode
from app.settings import settings


def _toll_tariffs_ready() -> bool:
    try:
        calibration_loader.load_toll_tariffs.cache_clear()
        table = calibration_loader.load_toll_tariffs()
        return bool(table.rules)
    except Exception:
        return False


TOLL_TARIFFS_READY = _toll_tariffs_ready()


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _scenario_profiles_payload(now_iso: str) -> dict[str, Any]:
    transform = calibration_loader._default_scenario_transform_params()
    transform["fit_strategy"] = "empirical_temporal_forward"
    transform["scenario_edge_scaling_version"] = "v4_live_empirical"
    transform["context_similarity"]["max_distance"] = 10.0

    def _q(v: float) -> dict[str, float]:
        return {"p10": round(v * 0.97, 6), "p50": float(v), "p90": round(v * 1.03, 6)}

    base_profiles = {
        "no_sharing": {
            "duration_multiplier": _q(1.10),
            "incident_rate_multiplier": _q(1.08),
            "incident_delay_multiplier": _q(1.12),
            "fuel_consumption_multiplier": _q(1.06),
            "emissions_multiplier": _q(1.05),
            "stochastic_sigma_multiplier": _q(1.10),
        },
        "partial_sharing": {
            "duration_multiplier": _q(1.02),
            "incident_rate_multiplier": _q(1.01),
            "incident_delay_multiplier": _q(1.03),
            "fuel_consumption_multiplier": _q(1.01),
            "emissions_multiplier": _q(1.01),
            "stochastic_sigma_multiplier": _q(1.02),
        },
        "full_sharing": {
            "duration_multiplier": _q(0.95),
            "incident_rate_multiplier": _q(0.94),
            "incident_delay_multiplier": _q(0.95),
            "fuel_consumption_multiplier": _q(0.96),
            "emissions_multiplier": _q(0.96),
            "stochastic_sigma_multiplier": _q(0.95),
        },
    }
    contexts: list[dict[str, Any]] = []
    for corridor_idx in range(8):
        corridor = f"c{corridor_idx:02d}"
        for slot in (0, 4, 8, 12, 16, 20):
            contexts.append(
                {
                    "context_key": f"{corridor}|h{slot:02d}|weekday|mixed|rigid_hgv|clear",
                    "corridor_bucket": corridor,
                    "corridor_geohash5": corridor,
                    "road_mix_bucket": "mixed",
                    "vehicle_class": "rigid_hgv",
                    "day_kind": "weekday",
                    "weather_bucket": "clear",
                    "weather_regime": "clear",
                    "hour_slot_local": slot,
                    "road_mix_vector": {"mixed": 1.0},
                    "mode_observation_source": "observed_live",
                    "mode_projection_ratio": 0.0,
                    "profiles": base_profiles,
                }
            )

    return {
        "version": "scenario_live_test_v1",
        "calibration_basis": "empirical",
        "as_of_utc": now_iso,
        "generated_at_utc": now_iso,
        "split_strategy": "temporal_forward_plus_corridor_block",
        "holdout_metrics": {
            "mode_separation_mean": 0.12,
            "duration_mape": 0.04,
            "monetary_mape": 0.04,
            "emissions_mape": 0.04,
            "coverage": 0.97,
            "hour_slot_coverage": 12.0,
            "corridor_coverage": 10.0,
            "full_identity_share": 0.12,
            "projection_dominant_context_share": 0.0,
            "observed_mode_row_share": 1.0,
        },
        "profiles": base_profiles,
        "contexts": contexts,
        "transform_params": transform,
    }


def _scenario_context_payload(now_iso: str) -> dict[str, Any]:
    fetch = {
        "source_url": "https://live.example/source",
        "fetch_error": None,
        "cache_hit": False,
        "stale_cache_used": False,
        "status_code": 200,
        "as_of_utc": now_iso,
    }
    return {
        "as_of_utc": now_iso,
        "source_set": {
            "webtris": "https://live.example/webtris",
            "traffic_england": "https://live.example/traffic",
            "dft_counts": "https://live.example/dft",
            "open_meteo": "https://live.example/meteo",
        },
        "coverage": {
            "webtris": 1.0,
            "traffic_england": 1.0,
            "dft": 1.0,
            "open_meteo": 1.0,
            "overall": 1.0,
        },
        "traffic_features": {"flow_index": 120.0, "speed_index": 62.0},
        "incident_features": {"delay_pressure": 2.5, "severity_index": 0.7},
        "weather_features": {"weather_severity_index": 0.3, "weather_bucket": "clear"},
        "source_diagnostics": {
            "webtris": {"fetch": dict(fetch)},
            "traffic_england": {"fetch": dict(fetch)},
            "dft_counts": {"fetch": dict(fetch)},
            "open_meteo": {"fetch": dict(fetch)},
        },
    }


def _departure_profile_payload(now_iso: str) -> dict[str, Any]:
    return {
        "version": "departure_live_test_v1",
        "calibration_basis": "empirical",
        "as_of_utc": now_iso,
        "weekday": [1.0] * 1440,
        "weekend": [0.95] * 1440,
        "holiday": [0.90] * 1440,
    }


def _bank_holidays_payload(now_iso: str) -> dict[str, Any]:
    return {
        "as_of_utc": now_iso,
        "england-and-wales": {"events": [{"date": "2026-12-25"}]},
        "scotland": {"events": [{"date": "2026-01-01"}]},
    }


def _fuel_payload(now_iso: str) -> dict[str, Any]:
    return {
        "as_of_utc": now_iso,
        "source": "live_runtime:fuel_prices",
        "calibration_basis": "empirical",
        "prices_gbp_per_l": {"diesel": 1.55, "petrol": 1.62, "lng": 1.05},
        "grid_price_gbp_per_kwh": 0.27,
        "regional_multipliers": {"uk_default": 1.0},
    }


def _toll_tariffs_payload(now_iso: str) -> dict[str, Any]:
    return {
        "as_of_utc": now_iso,
        "defaults": {"crossing_fee_gbp": 0.0, "distance_fee_gbp_per_km": 0.0},
        "rules": [
            {
                "id": "rule_dartford_hgv",
                "operator": "nh",
                "crossing_id": "dartford",
                "road_class": "motorway",
                "direction": "both",
                "start_minute": 0,
                "end_minute": 1439,
                "crossing_fee_gbp": 2.5,
                "distance_fee_gbp_per_km": 0.15,
                "vehicle_classes": ["rigid_hgv", "artic_hgv", "van"],
                "axle_classes": ["3to4", "5plus", "2"],
                "payment_classes": ["cash", "electronic"],
                "exemptions": [],
            }
        ],
    }


def _toll_topology_payload(now_iso: str) -> dict[str, Any]:
    return {
        "as_of_utc": now_iso,
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "id": "dartford_seg",
                    "name": "Dartford Crossing",
                    "operator": "nh",
                    "road_class": "motorway",
                    "crossing_id": "dartford",
                    "direction": "both",
                    "crossing_fee_gbp": 2.5,
                    "distance_fee_gbp_per_km": 0.15,
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[-1.7, 52.3], [-0.9, 51.9]],
                },
            }
        ],
    }


def _stochastic_payload(now_iso: str) -> dict[str, Any]:
    context_probs: dict[str, dict[str, float]] = {}
    corridors = [f"c{i:02d}" for i in range(8)]
    slots = ["h00", "h04", "h08", "h12", "h16", "h20"]
    for corridor in corridors:
        for slot in slots:
            key = f"{corridor}|weekday|{slot}|mixed|clear|default"
            context_probs[key] = {"weekday_offpeak": 1.0}
    context_probs["*|weekday|h12|mixed|clear|default"] = {"weekday_offpeak": 1.0}
    context_probs["*|weekday|*|mixed|clear|default"] = {"weekday_offpeak": 1.0}
    return {
        "as_of_utc": now_iso,
        "calibration_basis": "empirical",
        "calibration_version": "stochastic_live_test_v1",
        "copula_id": "gaussian_5x5_v2",
        "split_strategy": "temporal_forward_plus_corridor_block",
        "holdout_window": {"start_utc": "2025-01-01T00:00:00Z", "end_utc": "2025-12-31T23:59:59Z"},
        "holdout_metrics": {"pit_mean": 0.5, "coverage": 0.95, "crps_mean": 0.2, "duration_mape": 0.10},
        "coverage_metrics": {"hour_slot_coverage": 12.0, "corridor_coverage": 10.0},
        "posterior_model": {"context_to_regime_probs": context_probs},
        "regimes": {
            "weekday_offpeak": {
                "sigma_scale": 1.0,
                "traffic_scale": 1.0,
                "incident_scale": 1.0,
                "weather_scale": 1.0,
                "price_scale": 1.0,
                "eco_scale": 1.0,
                "corr": [
                    [1.0, 0.20, 0.15, 0.10, 0.08],
                    [0.20, 1.0, 0.20, 0.12, 0.10],
                    [0.15, 0.20, 1.0, 0.14, 0.12],
                    [0.10, 0.12, 0.14, 1.0, 0.20],
                    [0.08, 0.10, 0.12, 0.20, 1.0],
                ],
                "spread_floor": 0.05,
                "spread_cap": 1.25,
                "factor_low": 0.55,
                "factor_high": 2.2,
                "duration_mix": [1.0, 1.0, 1.0],
                "monetary_mix": [0.62, 0.38],
                "emissions_mix": [0.72, 0.28],
                "transform_family": "quantile_mapping_v1",
                "shock_quantile_mapping": {
                    "traffic": [[-2.0, 0.75], [0.0, 1.0], [2.0, 1.35]],
                    "incident": [[-2.0, 0.72], [0.0, 1.0], [2.0, 1.40]],
                    "weather": [[-2.0, 0.78], [0.0, 1.0], [2.0, 1.30]],
                    "price": [[-2.0, 0.82], [0.0, 1.0], [2.0, 1.25]],
                    "eco": [[-2.0, 0.80], [0.0, 1.0], [2.0, 1.22]],
                },
            }
        },
    }


def _carbon_schedule_payload(now_iso: str) -> dict[str, Any]:
    return {
        "as_of_utc": now_iso,
        "source": "desnz_live_schedule",
        "calibration_basis": "empirical",
        "prices_gbp_per_kg": {
            "central": {"2025": 0.10, "2026": 0.11, "2027": 0.12},
        },
        "uncertainty_distribution_by_year": {
            "2025": {"p10": 0.08, "p50": 0.10, "p90": 0.12},
            "2026": {"p10": 0.09, "p50": 0.11, "p90": 0.13},
            "2027": {"p10": 0.10, "p50": 0.12, "p90": 0.14},
        },
        "ev_grid_intensity_kg_per_kwh_by_region": {"uk_default": [0.20] * 24},
        "non_ev_scope_factors": {"wtw": {"2026": 1.20}, "lca": {"2026": 1.32}},
    }


@pytest.fixture(autouse=True)
def _strict_runtime_test_bypass(monkeypatch: pytest.MonkeyPatch):
    if os.environ.get("STRICT_RUNTIME_TEST_BYPASS") is None:
        monkeypatch.setenv("STRICT_RUNTIME_TEST_BYPASS", "1")
    now_iso = _now_iso()
    monkeypatch.setattr(settings, "scenario_require_signature", False)
    monkeypatch.setattr(settings, "live_fuel_require_signature", False)
    monkeypatch.setattr(settings, "live_scenario_coefficient_url", "https://live.example/scenario")
    monkeypatch.setattr(settings, "live_departure_profile_url", "https://live.example/departure")
    monkeypatch.setattr(settings, "live_fuel_price_url", "https://live.example/fuel")
    monkeypatch.setattr(settings, "live_toll_tariffs_url", "https://live.example/tariffs")
    monkeypatch.setattr(settings, "live_toll_topology_url", "https://live.example/topology")
    monkeypatch.setattr(settings, "live_stochastic_regimes_url", "https://live.example/stochastic")
    monkeypatch.setattr(settings, "live_carbon_schedule_url", "https://live.example/carbon")
    monkeypatch.setattr(calibration_loader, "live_scenario_profiles", lambda: _scenario_profiles_payload(now_iso))
    monkeypatch.setattr(calibration_loader, "live_scenario_context", lambda _ctx: _scenario_context_payload(now_iso))
    monkeypatch.setattr(calibration_loader, "live_departure_profiles", lambda: _departure_profile_payload(now_iso))
    monkeypatch.setattr(calibration_loader, "live_bank_holidays", lambda: _bank_holidays_payload(now_iso))
    monkeypatch.setattr(calibration_loader, "live_fuel_prices", lambda _as_of: _fuel_payload(now_iso))
    monkeypatch.setattr(calibration_loader, "live_toll_tariffs", lambda: _toll_tariffs_payload(now_iso))
    monkeypatch.setattr(calibration_loader, "live_toll_topology", lambda: _toll_topology_payload(now_iso))
    monkeypatch.setattr(calibration_loader, "live_stochastic_regimes", lambda: _stochastic_payload(now_iso))
    monkeypatch.setattr(carbon_model, "live_carbon_schedule", lambda: _carbon_schedule_payload(now_iso))
    calibration_loader.load_scenario_profiles.cache_clear()
    calibration_loader.load_live_scenario_context.cache_clear()
    calibration_loader.load_departure_profile.cache_clear()
    calibration_loader.load_uk_bank_holidays.cache_clear()
    calibration_loader.load_toll_tariffs.cache_clear()
    calibration_loader.load_toll_segments_seed.cache_clear()
    calibration_loader.load_fuel_price_snapshot.cache_clear()
    calibration_loader.load_stochastic_regimes.cache_clear()
    yield
    calibration_loader.load_scenario_profiles.cache_clear()
    calibration_loader.load_live_scenario_context.cache_clear()
    calibration_loader.load_departure_profile.cache_clear()
    calibration_loader.load_uk_bank_holidays.cache_clear()
    calibration_loader.load_toll_tariffs.cache_clear()
    calibration_loader.load_toll_segments_seed.cache_clear()
    calibration_loader.load_fuel_price_snapshot.cache_clear()
    calibration_loader.load_stochastic_regimes.cache_clear()


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


def test_scenario_delay_increases_duration_cost_and_emissions() -> None:
    base_route = _route(distance_m=50_000.0, duration_s=3_600.0)

    full = build_option(
        base_route,
        option_id="full",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.FULL_SHARING,
        cost_toggles=CostToggles(),
    )
    partial = build_option(
        base_route,
        option_id="partial",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.PARTIAL_SHARING,
        cost_toggles=CostToggles(),
    )
    no_share = build_option(
        base_route,
        option_id="no_sharing",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.NO_SHARING,
        cost_toggles=CostToggles(),
    )

    assert no_share.metrics.duration_s >= partial.metrics.duration_s >= full.metrics.duration_s
    assert no_share.metrics.monetary_cost >= partial.metrics.monetary_cost >= full.metrics.monetary_cost
    assert no_share.metrics.emissions_kg >= partial.metrics.emissions_kg >= full.metrics.emissions_kg
    assert no_share.metrics.duration_s > full.metrics.duration_s
    assert no_share.metrics.monetary_cost > full.metrics.monetary_cost
    assert no_share.metrics.emissions_kg > full.metrics.emissions_kg
    assert no_share.scenario_summary is not None
    assert no_share.scenario_summary.mode.value == "no_sharing"
    assert partial.scenario_summary is not None
    assert partial.scenario_summary.mode.value == "partial_sharing"
    assert full.scenario_summary is not None
    assert full.scenario_summary.mode.value == "full_sharing"


def test_build_option_uses_segment_totals_when_top_level_missing() -> None:
    route = _route(distance_m=10_000.0, duration_s=1_000.0)
    route["distance"] = 0.0
    route["duration"] = 0.0

    option = build_option(
        route,
        option_id="seg_fallback",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.FULL_SHARING,
        cost_toggles=CostToggles(),
    )

    assert math.isclose(option.metrics.distance_km, 10.0, rel_tol=0.0, abs_tol=0.001)
    assert math.isclose(option.metrics.duration_s, 1_000.0, rel_tol=0.0, abs_tol=0.01)


def test_carbon_price_toggle_increases_monetary_cost() -> None:
    route = _route(distance_m=35_000.0, duration_s=2_400.0)

    base = build_option(
        route,
        option_id="base",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.FULL_SHARING,
        cost_toggles=CostToggles(),
    )
    priced = build_option(
        route,
        option_id="priced",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.FULL_SHARING,
        cost_toggles=CostToggles(carbon_price_per_kg=0.50),
    )

    assert priced.metrics.monetary_cost > base.metrics.monetary_cost


@pytest.mark.skipif(not TOLL_TARIFFS_READY, reason="Toll tariff corpus unavailable in local strict runtime.")
def test_toll_toggle_only_applies_to_toll_flagged_routes() -> None:
    no_toll_route = _route(distance_m=10_000.0, duration_s=800.0)
    toll_route = _route(distance_m=10_000.0, duration_s=800.0)
    toll_route["contains_toll"] = True

    toggles = CostToggles(use_tolls=True, toll_cost_per_km=0.35)
    no_toll = build_option(
        no_toll_route,
        option_id="no_toll",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.FULL_SHARING,
        cost_toggles=toggles,
    )
    with pytest.raises(ValueError, match="tariff resolution"):
        build_option(
            toll_route,
            option_id="with_toll",
            vehicle_type="rigid_hgv",
            scenario_mode=ScenarioMode.FULL_SHARING,
            cost_toggles=toggles,
        )

    assert no_toll.metrics.monetary_cost > 0.0


@pytest.mark.skipif(not TOLL_TARIFFS_READY, reason="Toll tariff corpus unavailable in local strict runtime.")
def test_toll_toggle_detects_toll_classes_from_osrm_steps() -> None:
    no_toll_route = _route(distance_m=20_000.0, duration_s=1_400.0)
    toll_class_route = _route(distance_m=20_000.0, duration_s=1_400.0)
    toll_class_route["legs"] = [
        {
            "annotation": {
                "distance": [10_000.0, 10_000.0],
                "duration": [700.0, 700.0],
            },
            "steps": [
                {"distance": 8_000.0, "classes": ["motorway"]},
                {"distance": 12_000.0, "classes": ["toll"]},
            ],
        }
    ]

    toggles = CostToggles(use_tolls=True, toll_cost_per_km=0.50)
    no_toll = build_option(
        no_toll_route,
        option_id="no_toll_step_classes",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.FULL_SHARING,
        cost_toggles=toggles,
    )
    tolled = build_option(
        toll_class_route,
        option_id="with_toll_step_classes",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.FULL_SHARING,
        cost_toggles=toggles,
    )

    assert no_toll.metrics.monetary_cost > 0.0
    assert sum(float(row.get("toll_cost", 0.0)) for row in tolled.segment_breakdown) > 0.0
    assert bool((tolled.toll_metadata or {}).get("tariff_rule_ids"))
    assert tolled.metrics.monetary_cost > no_toll.metrics.monetary_cost


@pytest.mark.skipif(not TOLL_TARIFFS_READY, reason="Toll tariff corpus unavailable in local strict runtime.")
def test_toll_toggle_detects_toll_classes_from_intersections() -> None:
    no_toll_route = _route(distance_m=20_000.0, duration_s=1_400.0)
    toll_intersection_route = _route(distance_m=20_000.0, duration_s=1_400.0)
    toll_intersection_route["legs"] = [
        {
            "annotation": {
                "distance": [10_000.0, 10_000.0],
                "duration": [700.0, 700.0],
            },
            "steps": [
                {
                    "distance": 20_000.0,
                    "intersections": [{"classes": ["toll"]}],
                }
            ],
        }
    ]

    toggles = CostToggles(use_tolls=True, toll_cost_per_km=0.50)
    no_toll = build_option(
        no_toll_route,
        option_id="no_toll_intersections",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.FULL_SHARING,
        cost_toggles=toggles,
    )
    tolled = build_option(
        toll_intersection_route,
        option_id="with_toll_intersections",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.FULL_SHARING,
        cost_toggles=toggles,
    )

    assert no_toll.metrics.monetary_cost > 0.0
    assert sum(float(row.get("toll_cost", 0.0)) for row in tolled.segment_breakdown) > 0.0
    assert bool((tolled.toll_metadata or {}).get("tariff_rule_ids"))
    assert tolled.metrics.monetary_cost > no_toll.metrics.monetary_cost


@pytest.mark.skipif(not TOLL_TARIFFS_READY, reason="Toll tariff corpus unavailable in local strict runtime.")
def test_toll_free_candidate_meta_overrides_legacy_toll_flag() -> None:
    route = _route(distance_m=16_000.0, duration_s=1_200.0)
    route["contains_toll"] = True

    toll_toggles = CostToggles(use_tolls=True, toll_cost_per_km=0.60)
    with pytest.raises(ValueError, match="tariff resolution"):
        build_option(
            route,
            option_id="legacy_tolled",
            vehicle_type="rigid_hgv",
            scenario_mode=ScenarioMode.FULL_SHARING,
            cost_toggles=toll_toggles,
        )

    toll_free_route = _route(distance_m=16_000.0, duration_s=1_200.0)
    toll_free_route["contains_toll"] = True
    toll_free_route["_candidate_meta"] = {
        "source_labels": ["exclude:toll"],
        "seen_by_exclude_toll": True,
        "seen_by_non_exclude_toll": False,
        "toll_exclusion_available": True,
    }
    toll_free = build_option(
        toll_free_route,
        option_id="meta_toll_free",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.FULL_SHARING,
        cost_toggles=toll_toggles,
    )

    assert toll_free.metrics.monetary_cost > 0.0


def test_fuel_multiplier_changes_monetary_ranking() -> None:
    route_fast = _route(distance_m=18_000.0, duration_s=900.0)
    route_slow = _route(distance_m=12_000.0, duration_s=1_500.0)

    base_fast = build_option(
        route_fast,
        option_id="fast",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.FULL_SHARING,
        cost_toggles=CostToggles(),
    )
    base_slow = build_option(
        route_slow,
        option_id="slow",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.FULL_SHARING,
        cost_toggles=CostToggles(),
    )
    boosted_fast = build_option(
        route_fast,
        option_id="fast_boost",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.FULL_SHARING,
        cost_toggles=CostToggles(fuel_price_multiplier=3.0),
    )
    boosted_slow = build_option(
        route_slow,
        option_id="slow_boost",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.FULL_SHARING,
        cost_toggles=CostToggles(fuel_price_multiplier=3.0),
    )

    def _fuel_total(option: Any) -> float:
        return sum(float(row.get("fuel_cost", 0.0)) for row in option.segment_breakdown)

    fast_ratio = _fuel_total(boosted_fast) / max(1e-6, _fuel_total(base_fast))
    slow_ratio = _fuel_total(boosted_slow) / max(1e-6, _fuel_total(base_slow))
    assert fast_ratio > 2.5
    assert slow_ratio > 2.5


def test_segment_breakdown_includes_cost_decomposition_keys() -> None:
    route = _route(distance_m=25_000.0, duration_s=1_600.0)
    route["legs"] = [
        {
            "annotation": {
                "distance": [12_000.0, 13_000.0],
                "duration": [760.0, 840.0],
            },
            "steps": [
                {"distance": 12_000.0, "classes": ["primary"]},
                {"distance": 13_000.0, "classes": ["motorway"]},
            ],
        }
    ]

    option = build_option(
        route,
        option_id="decomp",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.FULL_SHARING,
        cost_toggles=CostToggles(use_tolls=True, toll_cost_per_km=0.55, carbon_price_per_kg=0.12),
    )
    assert option.segment_breakdown
    sample = option.segment_breakdown[0]
    assert "toll_cost" in sample
    assert "fuel_cost" in sample
    assert "carbon_cost" in sample
    assert "energy_kwh" in sample
    assert "fuel_liters_p10" in sample
    assert "fuel_liters_p50" in sample
    assert "fuel_liters_p90" in sample
    assert "fuel_cost_p10_gbp" in sample
    assert "fuel_cost_p50_gbp" in sample
    assert "fuel_cost_p90_gbp" in sample
    assert float(sample["fuel_liters_p10"]) <= float(sample["fuel_liters_p50"]) <= float(sample["fuel_liters_p90"])
    assert float(sample["fuel_cost_p10_gbp"]) <= float(sample["fuel_cost_p50_gbp"]) <= float(sample["fuel_cost_p90_gbp"])
    ws = option.weather_summary or {}
    assert "fuel_price_source" in ws
    assert "fuel_price_as_of" in ws
    assert "consumption_model_version" in ws
    assert "vehicle_profile_id" in ws
    assert "vehicle_profile_version" in ws
    assert "vehicle_profile_source" in ws


def test_strict_live_url_missing_with_fallback_disabled_returns_source_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "live_fuel_price_url", "")
    monkeypatch.setattr(settings, "live_fuel_require_url_in_strict", True)
    monkeypatch.setattr(settings, "live_fuel_allow_signed_fallback", False)
    calibration_loader.load_fuel_price_snapshot.cache_clear()
    try:
        with pytest.raises(ModelDataError) as exc:
            calibration_loader.load_fuel_price_snapshot()
        assert exc.value.reason_code == "fuel_price_source_unavailable"
    finally:
        calibration_loader.load_fuel_price_snapshot.cache_clear()


def test_strict_live_auth_failure_maps_to_auth_reason_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "live_fuel_price_url", "https://live.example/fuel")
    monkeypatch.setattr(settings, "live_fuel_require_url_in_strict", True)
    monkeypatch.setattr(settings, "live_fuel_allow_signed_fallback", False)
    monkeypatch.setattr(
        calibration_loader,
        "live_fuel_prices",
        lambda _as_of: {
            "_live_error": {
                "reason_code": "fuel_price_auth_unavailable",
                "message": "auth failed",
                "diagnostics": {"status_code": 401},
            }
        },
    )
    calibration_loader.load_fuel_price_snapshot.cache_clear()
    try:
        with pytest.raises(ModelDataError) as exc:
            calibration_loader.load_fuel_price_snapshot()
        assert exc.value.reason_code == "fuel_price_auth_unavailable"
    finally:
        calibration_loader.load_fuel_price_snapshot.cache_clear()


def test_strict_live_stale_without_fallback_returns_source_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "live_fuel_price_url", "https://live.example/fuel")
    monkeypatch.setattr(settings, "live_fuel_require_url_in_strict", True)
    monkeypatch.setattr(settings, "live_fuel_allow_signed_fallback", False)
    # This test targets freshness handling, not signature checks.
    monkeypatch.setattr(settings, "live_fuel_require_signature", False)
    monkeypatch.setattr(
        calibration_loader,
        "live_fuel_prices",
        lambda _as_of: {
            "as_of_utc": "2020-01-01T00:00:00Z",
            "source": "live-test",
            "prices_gbp_per_l": {"diesel": 1.5, "petrol": 1.45, "lng": 0.95},
            "grid_price_gbp_per_kwh": 0.25,
            "regional_multipliers": {"uk_default": 1.0},
        },
    )
    calibration_loader.load_fuel_price_snapshot.cache_clear()
    try:
        with pytest.raises(ModelDataError) as exc:
            calibration_loader.load_fuel_price_snapshot()
        assert exc.value.reason_code == "fuel_price_source_unavailable"
    finally:
        calibration_loader.load_fuel_price_snapshot.cache_clear()
