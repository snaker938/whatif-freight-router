from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import time
from types import SimpleNamespace
from typing import Any

import app.live_data_sources as live_data_sources
import app.main as main_module
import pytest
from app.model_data_errors import ModelDataError
from app.models import (
    CostToggles,
    EvidenceProvenance,
    EvidenceSourceRecord,
    EmissionsContext,
    GeoJSONLineString,
    IncidentSimulatorConfig,
    LatLng,
    RouteRequest,
    RouteMetrics,
    RouteOption,
    ScenarioSummary,
    StochasticConfig,
    Waypoint,
    WeatherImpactConfig,
)
from app.settings import settings


class _NoopOSRM:
    async def fetch_routes(self, **_: Any) -> list[dict[str, Any]]:
        return []


def _collect_kwargs() -> dict[str, Any]:
    return {
        "osrm": _NoopOSRM(),
        "origin": LatLng(lat=53.94633, lon=-1.02722),
        "destination": LatLng(lat=51.48539, lon=-0.13184),
        "waypoints": None,
        "max_alternatives": 3,
        "vehicle_type": "rigid_hgv",
        "scenario_mode": main_module.ScenarioMode.NO_SHARING,
        "cost_toggles": CostToggles(),
        "terrain_profile": "flat",
        "stochastic": StochasticConfig(),
        "emissions_context": EmissionsContext(),
        "weather": WeatherImpactConfig(),
        "incident_simulation": IncidentSimulatorConfig(),
        "departure_time_utc": None,
        "pareto_method": "dominance",
        "epsilon": None,
        "optimization_mode": "expected_value",
        "risk_aversion": 1.0,
        "option_prefix": "route",
        "progress_cb": None,
    }


def _make_leg_route(origin: LatLng, destination: LatLng) -> dict[str, Any]:
    return {
        "distance": 50_000.0,
        "duration": 2400.0,
        "geometry": {
            "type": "LineString",
            "coordinates": [
                [origin.lon, origin.lat],
                [destination.lon, destination.lat],
            ],
        },
        "legs": [
            {
                "annotation": {
                    "distance": [50_000.0],
                    "duration": [2400.0],
                }
            }
        ],
    }


def _make_leg_option(option_id: str, scenario_mode: main_module.ScenarioMode) -> RouteOption:
    return RouteOption(
        id=option_id,
        geometry=GeoJSONLineString(
            type="LineString",
            coordinates=[[-1.0, 52.0], [-1.1, 52.1]],
        ),
        metrics=RouteMetrics(
            distance_km=50.0,
            duration_s=2400.0,
            monetary_cost=80.0,
            emissions_kg=25.0,
            avg_speed_kmh=75.0,
        ),
        scenario_summary=ScenarioSummary(
            mode=scenario_mode,
            duration_multiplier=1.0,
            incident_rate_multiplier=1.0,
            incident_delay_multiplier=1.0,
            fuel_consumption_multiplier=1.0,
            emissions_multiplier=1.0,
            stochastic_sigma_multiplier=1.0,
            source="pytest",
            version="pytest",
        ),
    )


def _make_hydrated_option(option_id: str) -> RouteOption:
    option = _make_leg_option(option_id, main_module.ScenarioMode.NO_SHARING)
    option.eta_explanations = ["ok"]
    option.segment_breakdown = [{"segment_index": 0, "distance_km": 50.0}]
    option.counterfactuals = [{"id": "fuel_plus_10pct", "delta": 1.0, "improves": False}]
    option.uncertainty = {"q95_duration_s": 2500.0}
    option.uncertainty_samples_meta = {"sample_count": 25}
    option.evidence_provenance = EvidenceProvenance(
        active_families=["scenario"],
        families=[EvidenceSourceRecord(family="scenario", source="pytest", active=True, confidence=1.0)],
    )
    return option


def _make_summary_option(option_id: str) -> RouteOption:
    option = _make_leg_option(option_id, main_module.ScenarioMode.NO_SHARING)
    option.evidence_provenance = EvidenceProvenance(
        active_families=["scenario"],
        families=[EvidenceSourceRecord(family="scenario", source="pytest", active=True, confidence=1.0)],
    )
    return option


def test_route_option_lightweight_copy_keeps_compact_segment_summary() -> None:
    option = _make_hydrated_option("route_0").model_copy(
        update={
            "segment_breakdown": [
                {
                    "segment_index": 0,
                    "distance_km": 40.0,
                    "duration_s": 1800.0,
                    "toll_cost": 1.25,
                    "fuel_cost": 4.5,
                    "carbon_cost": 0.75,
                },
                {
                    "segment_index": 1,
                    "distance_km": 10.0,
                    "duration_s": 600.0,
                    "toll_cost": 0.5,
                    "fuel_cost": 1.0,
                    "carbon_cost": 0.2,
                },
            ],
            "weather_summary": {
                "toll_cost_total_gbp": 1.75,
                "fuel_cost_total_gbp": 5.5,
                "carbon_cost_total_gbp": 0.95,
            },
        },
        deep=True,
    )

    lightweight = main_module._route_option_lightweight_copy(option)

    assert len(lightweight.segment_breakdown) == 1
    summary = lightweight.segment_breakdown[0]
    assert summary["segment_index"] == 0
    assert summary["segment_count"] == 2
    assert summary["distance_km"] == pytest.approx(50.0, rel=0.0, abs=1e-6)
    assert summary["duration_s"] == pytest.approx(2400.0, rel=0.0, abs=1e-6)
    assert summary["toll_cost"] == pytest.approx(1.75, rel=0.0, abs=1e-6)
    assert summary["fuel_cost"] == pytest.approx(5.5, rel=0.0, abs=1e-6)
    assert summary["carbon_cost"] == pytest.approx(0.95, rel=0.0, abs=1e-6)
    assert lightweight.eta_explanations == []
    assert lightweight.eta_timeline == []
    assert lightweight.counterfactuals == []


def test_hydrate_priority_route_options_skips_already_hydrated(monkeypatch) -> None:
    routes = [{"_built_option_id": "route_0", "geometry": {"coordinates": [[-1.0, 52.0], [-1.1, 52.1]]}}]
    option = _make_hydrated_option("route_0")
    calls = {"count": 0}

    def _unexpected_build_option(*args: Any, **kwargs: Any) -> RouteOption:
        _ = (args, kwargs)
        calls["count"] += 1
        return _make_hydrated_option("route_0")

    monkeypatch.setattr(main_module, "build_option", _unexpected_build_option)

    rebuilt_options, rebuilt_frontier, rebuilt_display, rebuilt_selected, hydration_ms = (
        main_module._hydrate_priority_route_options(
            routes=routes,
            options=[option],
            strict_frontier=[option],
            display_candidates=[option],
            selected=option,
            vehicle_type="rigid_hgv",
            scenario_mode=main_module.ScenarioMode.NO_SHARING,
            cost_toggles=CostToggles(),
            departure_time_utc=None,
        )
    )

    assert calls["count"] == 0
    assert rebuilt_options[0].id == "route_0"
    assert rebuilt_frontier[0].uncertainty is not None
    assert rebuilt_display[0].segment_breakdown
    assert rebuilt_selected.counterfactuals
    assert hydration_ms == 0.0


def test_hydrate_priority_route_options_rebuilds_summary_option(monkeypatch) -> None:
    routes = [{"_built_option_id": "route_0", "geometry": {"coordinates": [[-1.0, 52.0], [-1.1, 52.1]]}}]
    summary_option = _make_summary_option("route_0")
    calls = {"count": 0}

    def _build_full_option(*args: Any, **kwargs: Any) -> RouteOption:
        _ = (args, kwargs)
        calls["count"] += 1
        return _make_hydrated_option("route_0")

    monkeypatch.setattr(main_module, "build_option", _build_full_option)

    rebuilt_options, rebuilt_frontier, rebuilt_display, rebuilt_selected, hydration_ms = (
        main_module._hydrate_priority_route_options(
            routes=routes,
            options=[summary_option],
            strict_frontier=[summary_option],
            display_candidates=[summary_option],
            selected=summary_option,
            vehicle_type="rigid_hgv",
            scenario_mode=main_module.ScenarioMode.NO_SHARING,
            cost_toggles=CostToggles(),
            departure_time_utc=None,
        )
    )

    assert calls["count"] == 1
    assert rebuilt_options[0].uncertainty is not None
    assert rebuilt_frontier[0].segment_breakdown
    assert rebuilt_display[0].counterfactuals
    assert rebuilt_selected.eta_explanations
    assert hydration_ms >= 0.0


def test_hydrate_priority_route_options_uses_cached_full_option(monkeypatch) -> None:
    routes = [
        {
            "_built_option_id": "route_0",
            "distance": 50_000.0,
            "duration": 2400.0,
            "geometry": {"type": "LineString", "coordinates": [[-1.0, 52.0], [-1.1, 52.1]]},
            "legs": [{"annotation": {"distance": [50_000.0], "duration": [2400.0]}}],
        }
    ]
    summary_option = _make_summary_option("route_0")
    cached_option = _make_hydrated_option("route_0")
    build_calls = {"count": 0}

    def _unexpected_build_option(*args: Any, **kwargs: Any) -> RouteOption:
        _ = (args, kwargs)
        build_calls["count"] += 1
        return _make_hydrated_option("route_0")

    monkeypatch.setattr(main_module, "build_option", _unexpected_build_option)
    monkeypatch.setattr(
        main_module,
        "get_cached_route_option_build",
        lambda key: main_module.CachedRouteOptionBuild(option=cached_option, estimated_build_ms=12.0),
    )

    rebuilt_options, rebuilt_frontier, rebuilt_display, rebuilt_selected, hydration_ms = (
        main_module._hydrate_priority_route_options(
            routes=routes,
            options=[summary_option],
            strict_frontier=[summary_option],
            display_candidates=[summary_option],
            selected=summary_option,
            vehicle_type="rigid_hgv",
            scenario_mode=main_module.ScenarioMode.NO_SHARING,
            cost_toggles=CostToggles(),
            departure_time_utc=None,
        )
    )

    assert build_calls["count"] == 0
    assert rebuilt_options[0].uncertainty is not None
    assert rebuilt_frontier[0].segment_breakdown
    assert rebuilt_display[0].counterfactuals
    assert rebuilt_selected.eta_explanations
    assert hydration_ms >= 0.0


def test_should_hydrate_priority_route_options_respects_evaluation_lean_mode() -> None:
    lean_req = RouteRequest(
        origin=LatLng(lat=52.0, lon=-1.5),
        destination=LatLng(lat=51.5, lon=-1.2),
        evaluation_lean_mode=True,
    )
    full_req = RouteRequest(
        origin=LatLng(lat=52.0, lon=-1.5),
        destination=LatLng(lat=51.5, lon=-1.2),
    )

    assert main_module._should_hydrate_priority_route_options(lean_req) is False
    assert main_module._should_hydrate_priority_route_options(full_req) is True


def test_route_option_detail_level_respects_pipeline_and_lean_mode() -> None:
    lean_req = RouteRequest(
        origin=LatLng(lat=52.0, lon=-1.5),
        destination=LatLng(lat=51.5, lon=-1.2),
        evaluation_lean_mode=True,
    )
    full_req = RouteRequest(
        origin=LatLng(lat=52.0, lon=-1.5),
        destination=LatLng(lat=51.5, lon=-1.2),
    )

    assert main_module._route_option_detail_level(req=lean_req, pipeline_mode="voi") == "summary"
    assert main_module._route_option_detail_level(req=full_req, pipeline_mode="voi") == "full"
    assert main_module._route_option_detail_level(req=full_req, pipeline_mode="dccs") == "summary"


def test_route_state_cache_profile_separates_hydrated_and_lean_dccs_requests() -> None:
    lean_req = RouteRequest(
        origin=LatLng(lat=52.0, lon=-1.5),
        destination=LatLng(lat=51.5, lon=-1.2),
        evaluation_lean_mode=True,
    )
    full_req = RouteRequest(
        origin=LatLng(lat=52.0, lon=-1.5),
        destination=LatLng(lat=51.5, lon=-1.2),
    )

    lean_profile = main_module._route_state_cache_profile(req=lean_req, pipeline_mode="dccs")
    full_profile = main_module._route_state_cache_profile(req=full_req, pipeline_mode="dccs")

    assert lean_profile["route_option_detail_level"] == "summary"
    assert lean_profile["priority_hydration_enabled"] is False
    assert full_profile["route_option_detail_level"] == "summary"
    assert full_profile["priority_hydration_enabled"] is True
    assert lean_profile != full_profile


def test_build_options_lightweight_keeps_uncertainty_for_robust_mode(monkeypatch) -> None:
    routes = [_make_leg_route(_collect_kwargs()["origin"], _collect_kwargs()["destination"])]
    observed_force_uncertainty: list[bool] = []

    def _build_option_stub(route: dict[str, Any], *, option_id: str, force_uncertainty: bool = False, **kwargs: Any) -> RouteOption:
        _ = (route, kwargs)
        observed_force_uncertainty.append(bool(force_uncertainty))
        return _make_summary_option(option_id)

    monkeypatch.setattr(main_module, "build_option", _build_option_stub)
    monkeypatch.setattr(main_module, "terrain_begin_route_run", lambda: None)
    monkeypatch.setattr(main_module, "resolve_vehicle_profile", lambda *_: SimpleNamespace(vehicle_class="rigid_hgv"))

    options, warnings, terrain_diag = main_module._build_options(
        routes,
        vehicle_type="rigid_hgv",
        scenario_mode=main_module.ScenarioMode.NO_SHARING,
        cost_toggles=CostToggles(),
        terrain_profile="flat",
        stochastic=StochasticConfig(enabled=True, samples=24),
        emissions_context=EmissionsContext(),
        weather=WeatherImpactConfig(),
        incident_simulation=IncidentSimulatorConfig(),
        departure_time_utc=None,
        utility_weights=(1.0, 1.0, 1.0),
        risk_aversion=1.0,
        option_prefix="route",
        optimization_mode="robust",
        route_option_cache_runtime={},
        scenario_policy_cache={},
        lightweight=True,
    )

    assert len(options) == 1
    assert warnings == []
    assert terrain_diag.fail_closed_count == 0
    assert observed_force_uncertainty == [True]


def test_build_options_threads_optimization_fields_into_cache_key(monkeypatch) -> None:
    routes = [_make_leg_route(_collect_kwargs()["origin"], _collect_kwargs()["destination"])]
    captured: dict[str, Any] = {}

    def _cache_key_stub(route: dict[str, Any], **kwargs: Any) -> str | None:
        _ = route
        captured.update(kwargs)
        return None

    def _build_option_stub(route: dict[str, Any], *, option_id: str, **kwargs: Any) -> RouteOption:
        _ = (route, kwargs)
        return _make_summary_option(option_id)

    monkeypatch.setattr(main_module, "build_route_option_cache_key", _cache_key_stub)
    monkeypatch.setattr(main_module, "build_option", _build_option_stub)
    monkeypatch.setattr(main_module, "terrain_begin_route_run", lambda: None)
    monkeypatch.setattr(main_module, "resolve_vehicle_profile", lambda *_: SimpleNamespace(vehicle_class="rigid_hgv"))

    options, warnings, terrain_diag = main_module._build_options(
        routes,
        vehicle_type="rigid_hgv",
        scenario_mode=main_module.ScenarioMode.NO_SHARING,
        cost_toggles=CostToggles(),
        terrain_profile="flat",
        stochastic=StochasticConfig(enabled=True, samples=24),
        emissions_context=EmissionsContext(),
        weather=WeatherImpactConfig(),
        incident_simulation=IncidentSimulatorConfig(),
        departure_time_utc=None,
        utility_weights=(1.0, 1.0, 1.0),
        risk_aversion=1.0,
        option_prefix="route",
        optimization_mode="robust",
        pareto_method="epsilon_constraint",
        epsilon=SimpleNamespace(duration_s=3600.0),
        max_alternatives=7,
        route_option_cache_runtime={},
        scenario_policy_cache={},
        lightweight=True,
    )

    assert len(options) == 1
    assert warnings == []
    assert terrain_diag.fail_closed_count == 0
    assert captured["optimization_mode"] == "robust"
    assert captured["pareto_method"] == "epsilon_constraint"
    assert captured["max_alternatives"] == 7
    assert captured["detail_level"] == "summary"
    assert captured["epsilon"] is not None


def test_build_option_uses_core_cache_for_lightweight_copy(monkeypatch) -> None:
    route = _make_leg_route(_collect_kwargs()["origin"], _collect_kwargs()["destination"])
    cached_option = _make_hydrated_option("cached-route")

    monkeypatch.setattr(main_module, "terrain_begin_route_run", lambda: None)
    monkeypatch.setattr(main_module, "get_cached_route_option_core", lambda key: main_module.CachedRouteOptionCore(option=cached_option, estimated_build_ms=12.0))

    option = main_module.build_option(
        route,
        option_id="route_0",
        vehicle_type="rigid_hgv",
        scenario_mode=main_module.ScenarioMode.NO_SHARING,
        cost_toggles=CostToggles(),
        optimization_mode="expected_value",
        pareto_method="dominance",
        epsilon=None,
        max_alternatives=4,
        lightweight=True,
    )

    assert option.id == "route_0"
    assert len(option.segment_breakdown) == 1
    summary = option.segment_breakdown[0]
    assert summary["segment_index"] == 0
    assert summary["segment_count"] == 1
    assert summary["distance_km"] == pytest.approx(50.0, rel=0.0, abs=1e-6)
    assert summary["duration_s"] == pytest.approx(2400.0, rel=0.0, abs=1e-6)
    assert summary["toll_cost"] == pytest.approx(0.0, rel=0.0, abs=1e-6)
    assert summary["fuel_cost"] == pytest.approx(0.0, rel=0.0, abs=1e-6)
    assert summary["carbon_cost"] == pytest.approx(0.0, rel=0.0, abs=1e-6)
    assert option.eta_explanations == []
    assert option.counterfactuals == []
    assert option.evidence_provenance is not None


def test_build_option_lightweight_coarsens_segment_economics_and_keeps_summary_fields(monkeypatch) -> None:
    origin = _collect_kwargs()["origin"]
    destination = _collect_kwargs()["destination"]
    route = {
        "distance": 120_000.0,
        "duration": 7_200.0,
        "geometry": {
            "type": "LineString",
            "coordinates": [
                [origin.lon, origin.lat],
                [destination.lon, destination.lat],
            ],
        },
        "legs": [
            {
                "annotation": {
                    "distance": [1_000.0] * 120,
                    "duration": [60.0] * 120,
                }
            }
        ],
    }
    energy_calls = {"count": 0}
    segment_grade_probe_flags: list[bool] = []

    monkeypatch.setattr(main_module, "terrain_begin_route_run", lambda: None)
    monkeypatch.setattr(
        main_module,
        "_validate_osrm_geometry",
        lambda _route: _route["geometry"]["coordinates"],
    )
    monkeypatch.setattr(
        main_module,
        "extract_segment_annotations",
        lambda _route: ([1_000.0] * 120, [60.0] * 120),
    )
    def _segment_grade_profile(**kwargs: Any) -> list[float]:
        segment_grade_probe_flags.append(bool(kwargs.get("probe_segment_boundaries", True)))
        return [0.0] * len(kwargs["segment_distances_m"])

    monkeypatch.setattr(main_module, "segment_grade_profile", _segment_grade_profile)
    monkeypatch.setattr(
        main_module,
        "build_scenario_route_context",
        lambda **kwargs: SimpleNamespace(context_key="test-context"),
    )
    monkeypatch.setattr(
        main_module,
        "resolve_scenario_profile",
        lambda *_args, **_kwargs: SimpleNamespace(
            context_key="test-context",
            duration_multiplier=1.0,
            incident_rate_multiplier=1.0,
            incident_delay_multiplier=1.0,
            fuel_consumption_multiplier=1.0,
            emissions_multiplier=1.0,
            stochastic_sigma_multiplier=1.0,
            source="pytest",
            version="1",
            calibration_basis="pytest",
            as_of_utc=None,
            live_as_of_utc=None,
            live_source_set={},
            live_coverage={"overall": 1.0},
            live_traffic_pressure=0.0,
            live_incident_pressure=0.0,
            live_weather_pressure=0.0,
            scenario_edge_scaling_version="1",
            mode_observation_source="pytest",
            mode_projection_ratio=1.0,
        ),
    )
    monkeypatch.setattr(
        main_module,
        "time_of_day_multiplier_uk",
        lambda *_args, **_kwargs: SimpleNamespace(
            multiplier=1.0,
            local_time_iso="2026-03-25T08:00:00+00:00",
            profile_day="weekday",
            profile_source="pytest",
            profile_key="pytest",
            profile_version="1",
            profile_as_of_utc=None,
            profile_refreshed_at_utc=None,
            confidence_low=None,
            confidence_high=None,
        ),
    )
    monkeypatch.setattr(
        main_module,
        "estimate_terrain_summary",
        lambda **_kwargs: SimpleNamespace(
            confidence=1.0,
            duration_multiplier=1.0,
            emissions_multiplier=1.0,
            source="missing",
            version="1",
            ascent_m=0.0,
            descent_m=0.0,
            coverage_ratio=1.0,
            as_route_option_payload=lambda: {
                "source": "missing",
                "version": "1",
                "confidence": 1.0,
                "coverage_ratio": 1.0,
                "ascent_m": 0.0,
                "descent_m": 0.0,
            },
        ),
    )
    monkeypatch.setattr(main_module, "params_for_vehicle", lambda *_args, **_kwargs: SimpleNamespace())
    monkeypatch.setattr(main_module, "segment_duration_multiplier", lambda **_kwargs: 1.0)
    monkeypatch.setattr(main_module, "simulate_incident_events", lambda **_kwargs: [])
    monkeypatch.setattr(
        main_module,
        "compute_toll_cost",
        lambda **_kwargs: SimpleNamespace(
            contains_toll=False,
            toll_distance_km=0.0,
            toll_cost_gbp=0.0,
            source="pytest",
            confidence=1.0,
            details={
                "fallback_policy_used": False,
                "fallback_reason": "",
                "pricing_unresolved": False,
                "as_of_utc": None,
                "tariff_rule_ids": "",
                "matched_asset_ids": "",
            },
        ),
    )
    monkeypatch.setattr(
        main_module,
        "resolve_carbon_price",
        lambda **_kwargs: SimpleNamespace(
            source="pytest",
            schedule_year=2026,
            price_per_kg=0.0,
            uncertainty_low=0.0,
            uncertainty_high=0.0,
            scope_mode="included",
        ),
    )
    monkeypatch.setattr(main_module, "apply_scope_emissions_adjustment", lambda **kwargs: kwargs["emissions_kg"])
    monkeypatch.setattr(
        main_module,
        "resolve_vehicle_profile",
            lambda *_args, **_kwargs: SimpleNamespace(
                id="rigid_hgv",
                schema_version=1,
                profile_source="pytest",
                vehicle_class="rigid_hgv",
                risk_bucket="rigid",
                powertrain="diesel",
                cost_per_hour=60.0,
                fuel_type="diesel",
                euro_class="euro6",
                ambient_temp_c=15.0,
            ev_kwh_per_km=None,
            grid_co2_kg_per_kwh=None,
        ),
    )

    def _segment_energy_and_emissions(**kwargs: Any) -> SimpleNamespace:
        _ = kwargs
        energy_calls["count"] += 1
        return SimpleNamespace(
            fuel_cost_gbp=1.0,
            fuel_liters_p10=0.1,
            fuel_liters_p50=0.2,
            fuel_liters_p90=0.3,
            fuel_cost_p10_gbp=1.1,
            fuel_cost_p50_gbp=1.2,
            fuel_cost_p90_gbp=1.3,
            fuel_cost_uncertainty_low_gbp=0.9,
            fuel_cost_uncertainty_high_gbp=1.4,
            energy_kwh=2.0,
            emissions_kg=0.5,
            emissions_uncertainty_low_kg=0.4,
            emissions_uncertainty_high_kg=0.6,
            fuel_liters=0.25,
            price_source="pytest",
            price_as_of="2026-03-25T00:00:00Z",
        )

    monkeypatch.setattr(main_module, "segment_energy_and_emissions", _segment_energy_and_emissions)
    monkeypatch.setattr(settings, "route_option_cache_enabled", False)

    option = main_module.build_option(
        route,
        option_id="route_0",
        vehicle_type="rigid_hgv",
        scenario_mode=main_module.ScenarioMode.NO_SHARING,
        cost_toggles=CostToggles(),
        terrain_profile="flat",
        stochastic=StochasticConfig(),
        emissions_context=EmissionsContext(),
        weather=WeatherImpactConfig(),
        incident_simulation=IncidentSimulatorConfig(),
        departure_time_utc=None,
        utility_weights=(1.0, 1.0, 1.0),
        risk_aversion=1.0,
        scenario_policy_cache={},
        lightweight=True,
        force_uncertainty=False,
    )

    assert energy_calls["count"] <= 8
    assert segment_grade_probe_flags == [False]
    assert len(option.segment_breakdown) == 1
    summary = option.segment_breakdown[0]
    assert summary["segment_index"] == 0
    assert summary["segment_count"] == 15
    assert summary["distance_km"] == pytest.approx(120.0, rel=0.0, abs=1e-6)
    assert summary["duration_s"] == pytest.approx(7200.0, rel=0.0, abs=1e-6)
    assert summary["toll_cost"] == pytest.approx(0.0, rel=0.0, abs=1e-6)
    assert summary["fuel_cost"] == pytest.approx(120.0, rel=0.0, abs=1e-6)
    assert summary["carbon_cost"] == pytest.approx(0.0, rel=0.0, abs=1e-6)
    assert option.eta_explanations == []
    assert option.eta_timeline == []
    assert option.counterfactuals == []
    assert option.uncertainty is None
    assert option.metrics.distance_km == pytest.approx(120.0, rel=0.0, abs=1e-6)
    assert option.metrics.duration_s > 0.0
    assert option.metrics.monetary_cost > 0.0
    assert option.metrics.emissions_kg > 0.0
    assert option.scenario_summary is not None
    assert option.terrain_summary is not None


def test_build_option_lightweight_keeps_higher_segment_floor_for_long_routes(monkeypatch) -> None:
    origin = _collect_kwargs()["origin"]
    destination = _collect_kwargs()["destination"]
    route = {
        "distance": 240_000.0,
        "duration": 14_400.0,
        "geometry": {
            "type": "LineString",
            "coordinates": [
                [origin.lon, origin.lat],
                [destination.lon, destination.lat],
            ],
        },
        "legs": [
            {
                "annotation": {
                    "distance": [1_000.0] * 240,
                    "duration": [60.0] * 240,
                }
            }
        ],
    }
    captured_max_segments = {"value": None}

    monkeypatch.setattr(main_module, "terrain_begin_route_run", lambda: None)
    monkeypatch.setattr(
        main_module,
        "_validate_osrm_geometry",
        lambda _route: _route["geometry"]["coordinates"],
    )
    monkeypatch.setattr(
        main_module,
        "extract_segment_annotations",
        lambda _route: ([1_000.0] * 240, [60.0] * 240),
    )
    monkeypatch.setattr(
        main_module,
        "segment_grade_profile",
        lambda **kwargs: [0.0] * len(kwargs["segment_distances_m"]),
    )
    monkeypatch.setattr(
        main_module,
        "_aggregate_route_segments",
        lambda *, segment_distances_m, segment_durations_s, segment_grades, max_segments: (
            captured_max_segments.__setitem__("value", int(max_segments)) or (
                segment_distances_m,
                segment_durations_s,
                segment_grades,
            )
        ),
    )
    monkeypatch.setattr(
        main_module,
        "build_scenario_route_context",
        lambda **kwargs: SimpleNamespace(context_key="test-context"),
    )
    monkeypatch.setattr(
        main_module,
        "resolve_scenario_profile",
        lambda *_args, **_kwargs: SimpleNamespace(
            context_key="test-context",
            duration_multiplier=1.0,
            incident_rate_multiplier=1.0,
            incident_delay_multiplier=1.0,
            fuel_consumption_multiplier=1.0,
            emissions_multiplier=1.0,
            stochastic_sigma_multiplier=1.0,
            source="pytest",
            version="1",
            calibration_basis="pytest",
            as_of_utc=None,
            live_as_of_utc=None,
            live_source_set={},
            live_coverage={"overall": 1.0},
            live_traffic_pressure=0.0,
            live_incident_pressure=0.0,
            live_weather_pressure=0.0,
            scenario_edge_scaling_version="1",
            mode_observation_source="pytest",
            mode_projection_ratio=1.0,
        ),
    )
    monkeypatch.setattr(
        main_module,
        "time_of_day_multiplier_uk",
        lambda *_args, **_kwargs: SimpleNamespace(
            multiplier=1.0,
            local_time_iso="2026-03-25T08:00:00+00:00",
            profile_day="weekday",
            profile_source="pytest",
            profile_key="pytest",
            profile_version="1",
            profile_as_of_utc=None,
            profile_refreshed_at_utc=None,
            confidence_low=None,
            confidence_high=None,
        ),
    )
    monkeypatch.setattr(
        main_module,
        "estimate_terrain_summary",
        lambda **_kwargs: SimpleNamespace(
            confidence=1.0,
            duration_multiplier=1.0,
            emissions_multiplier=1.0,
            source="missing",
            version="1",
            ascent_m=0.0,
            descent_m=0.0,
            coverage_ratio=1.0,
            as_route_option_payload=lambda: {
                "source": "missing",
                "version": "1",
                "confidence": 1.0,
                "coverage_ratio": 1.0,
                "ascent_m": 0.0,
                "descent_m": 0.0,
            },
        ),
    )
    monkeypatch.setattr(main_module, "params_for_vehicle", lambda *_args, **_kwargs: SimpleNamespace())
    monkeypatch.setattr(main_module, "segment_duration_multiplier", lambda **_kwargs: 1.0)
    monkeypatch.setattr(main_module, "simulate_incident_events", lambda **_kwargs: [])
    monkeypatch.setattr(
        main_module,
        "compute_toll_cost",
        lambda **_kwargs: SimpleNamespace(
            contains_toll=False,
            toll_distance_km=0.0,
            toll_cost_gbp=0.0,
            source="pytest",
            confidence=1.0,
            details={
                "fallback_policy_used": False,
                "fallback_reason": "",
                "pricing_unresolved": False,
                "as_of_utc": None,
                "tariff_rule_ids": "",
                "matched_asset_ids": "",
            },
        ),
    )
    monkeypatch.setattr(
        main_module,
        "resolve_carbon_price",
        lambda **_kwargs: SimpleNamespace(
            source="pytest",
            schedule_year=2026,
            price_per_kg=0.0,
            uncertainty_low=0.0,
            uncertainty_high=0.0,
            scope_mode="included",
        ),
    )
    monkeypatch.setattr(main_module, "apply_scope_emissions_adjustment", lambda **kwargs: kwargs["emissions_kg"])
    monkeypatch.setattr(
        main_module,
        "resolve_vehicle_profile",
        lambda *_args, **_kwargs: SimpleNamespace(
            id="rigid_hgv",
            schema_version=1,
            profile_source="pytest",
            vehicle_class="rigid_hgv",
            risk_bucket="rigid",
            powertrain="diesel",
            cost_per_hour=60.0,
            fuel_type="diesel",
            euro_class="euro6",
            ambient_temp_c=15.0,
            ev_kwh_per_km=None,
            grid_co2_kg_per_kwh=None,
        ),
    )
    monkeypatch.setattr(
        main_module,
        "segment_energy_and_emissions",
        lambda **_kwargs: SimpleNamespace(
            fuel_cost_gbp=1.0,
            fuel_liters_p10=0.1,
            fuel_liters_p50=0.2,
            fuel_liters_p90=0.3,
            fuel_cost_p10_gbp=1.1,
            fuel_cost_p50_gbp=1.2,
            fuel_cost_p90_gbp=1.3,
            fuel_cost_uncertainty_low_gbp=0.9,
            fuel_cost_uncertainty_high_gbp=1.4,
            energy_kwh=2.0,
            emissions_kg=0.5,
            emissions_uncertainty_low_kg=0.4,
            emissions_uncertainty_high_kg=0.6,
            fuel_liters=0.25,
            price_source="pytest",
            price_as_of="2026-03-25T00:00:00Z",
        ),
    )
    monkeypatch.setattr(settings, "route_option_cache_enabled", False)

    option = main_module.build_option(
        route,
        option_id="route_0",
        vehicle_type="rigid_hgv",
        scenario_mode=main_module.ScenarioMode.NO_SHARING,
        cost_toggles=CostToggles(),
        terrain_profile="flat",
        stochastic=StochasticConfig(),
        emissions_context=EmissionsContext(),
        weather=WeatherImpactConfig(),
        incident_simulation=IncidentSimulatorConfig(),
        departure_time_utc=None,
        utility_weights=(1.0, 1.0, 1.0),
        risk_aversion=1.0,
        scenario_policy_cache={},
        lightweight=True,
        force_uncertainty=False,
    )

    assert captured_max_segments["value"] is not None
    assert captured_max_segments["value"] >= 20
    assert option.metrics.distance_km == pytest.approx(240.0, rel=0.0, abs=1e-6)
    assert option.metrics.duration_s > 0.0
    assert option.metrics.monetary_cost > 0.0
    assert option.metrics.emissions_kg > 0.0


def test_build_option_lightweight_preserves_uncertainty_when_forced(monkeypatch) -> None:
    origin = _collect_kwargs()["origin"]
    destination = _collect_kwargs()["destination"]
    route = {
        "distance": 60_000.0,
        "duration": 3_600.0,
        "geometry": {
            "type": "LineString",
            "coordinates": [
                [origin.lon, origin.lat],
                [destination.lon, destination.lat],
            ],
        },
        "legs": [
            {
                "annotation": {
                    "distance": [1_000.0] * 60,
                    "duration": [60.0] * 60,
                }
            }
        ],
    }
    energy_calls = {"count": 0}
    uncertainty_calls = {"count": 0}
    observed_uncertainty_samples: list[int] = []

    monkeypatch.setattr(main_module, "terrain_begin_route_run", lambda: None)
    monkeypatch.setattr(
        main_module,
        "_validate_osrm_geometry",
        lambda _route: _route["geometry"]["coordinates"],
    )
    monkeypatch.setattr(
        main_module,
        "extract_segment_annotations",
        lambda _route: ([1_000.0] * 60, [60.0] * 60),
    )
    monkeypatch.setattr(
        main_module,
        "segment_grade_profile",
        lambda **kwargs: [0.0] * len(kwargs["segment_distances_m"]),
    )
    monkeypatch.setattr(
        main_module,
        "build_scenario_route_context",
        lambda **kwargs: SimpleNamespace(context_key="test-context"),
    )
    monkeypatch.setattr(
        main_module,
        "resolve_scenario_profile",
        lambda *_args, **_kwargs: SimpleNamespace(
            context_key="test-context",
            duration_multiplier=1.0,
            incident_rate_multiplier=1.0,
            incident_delay_multiplier=1.0,
            fuel_consumption_multiplier=1.0,
            emissions_multiplier=1.0,
            stochastic_sigma_multiplier=1.0,
            source="pytest",
            version="1",
            calibration_basis="pytest",
            as_of_utc=None,
            live_as_of_utc=None,
            live_source_set={},
            live_coverage={"overall": 1.0},
            live_traffic_pressure=0.0,
            live_incident_pressure=0.0,
            live_weather_pressure=0.0,
            scenario_edge_scaling_version="1",
            mode_observation_source="pytest",
            mode_projection_ratio=1.0,
        ),
    )
    monkeypatch.setattr(
        main_module,
        "time_of_day_multiplier_uk",
        lambda *_args, **_kwargs: SimpleNamespace(
            multiplier=1.0,
            local_time_iso="2026-03-25T08:00:00+00:00",
            profile_day="weekday",
            profile_source="pytest",
            profile_key="pytest",
            profile_version="1",
            profile_as_of_utc=None,
            profile_refreshed_at_utc=None,
            confidence_low=None,
            confidence_high=None,
        ),
    )
    monkeypatch.setattr(
        main_module,
        "estimate_terrain_summary",
        lambda **_kwargs: SimpleNamespace(
            confidence=1.0,
            duration_multiplier=1.0,
            emissions_multiplier=1.0,
            source="missing",
            version="1",
            ascent_m=0.0,
            descent_m=0.0,
            coverage_ratio=1.0,
            as_route_option_payload=lambda: {
                "source": "missing",
                "version": "1",
                "confidence": 1.0,
                "coverage_ratio": 1.0,
                "ascent_m": 0.0,
                "descent_m": 0.0,
            },
        ),
    )
    monkeypatch.setattr(main_module, "params_for_vehicle", lambda *_args, **_kwargs: SimpleNamespace())
    monkeypatch.setattr(main_module, "segment_duration_multiplier", lambda **_kwargs: 1.0)
    monkeypatch.setattr(main_module, "simulate_incident_events", lambda **_kwargs: [])
    monkeypatch.setattr(
        main_module,
        "compute_toll_cost",
        lambda **_kwargs: SimpleNamespace(
            contains_toll=False,
            toll_distance_km=0.0,
            toll_cost_gbp=0.0,
            source="pytest",
            confidence=1.0,
            details={
                "fallback_policy_used": False,
                "fallback_reason": "",
                "pricing_unresolved": False,
                "as_of_utc": None,
                "tariff_rule_ids": "",
                "matched_asset_ids": "",
            },
        ),
    )
    monkeypatch.setattr(
        main_module,
        "resolve_carbon_price",
        lambda **_kwargs: SimpleNamespace(
            source="pytest",
            schedule_year=2026,
            price_per_kg=0.0,
            uncertainty_low=0.0,
            uncertainty_high=0.0,
            scope_mode="included",
        ),
    )
    monkeypatch.setattr(main_module, "apply_scope_emissions_adjustment", lambda **kwargs: kwargs["emissions_kg"])
    monkeypatch.setattr(
        main_module,
        "resolve_vehicle_profile",
            lambda *_args, **_kwargs: SimpleNamespace(
                id="rigid_hgv",
                schema_version=1,
                profile_source="pytest",
                vehicle_class="rigid_hgv",
                risk_bucket="rigid",
                powertrain="diesel",
                cost_per_hour=60.0,
                fuel_type="diesel",
                euro_class="euro6",
                ambient_temp_c=15.0,
            ev_kwh_per_km=None,
            grid_co2_kg_per_kwh=None,
        ),
    )

    def _segment_energy_and_emissions(**kwargs: Any) -> SimpleNamespace:
        _ = kwargs
        energy_calls["count"] += 1
        return SimpleNamespace(
            fuel_cost_gbp=1.0,
            fuel_liters_p10=0.1,
            fuel_liters_p50=0.2,
            fuel_liters_p90=0.3,
            fuel_cost_p10_gbp=1.1,
            fuel_cost_p50_gbp=1.2,
            fuel_cost_p90_gbp=1.3,
            fuel_cost_uncertainty_low_gbp=0.9,
            fuel_cost_uncertainty_high_gbp=1.4,
            energy_kwh=2.0,
            emissions_kg=0.5,
            emissions_uncertainty_low_kg=0.4,
            emissions_uncertainty_high_kg=0.6,
            fuel_liters=0.25,
            price_source="pytest",
            price_as_of="2026-03-25T00:00:00Z",
        )

    def _route_stochastic_uncertainty(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        _ = (args, kwargs)
        uncertainty_calls["count"] += 1
        observed_uncertainty_samples.append(int(kwargs["stochastic"].samples))
        return (
            {
                "mean_duration_s": 3600.0,
                "q95_duration_s": 3720.0,
                "mean_monetary_cost": 42.0,
                "q95_monetary_cost": 45.0,
                "mean_emissions_kg": 11.0,
                "q95_emissions_kg": 12.0,
            },
            {"sample_count": 24, "regime": "pytest"},
        )

    monkeypatch.setattr(main_module, "segment_energy_and_emissions", _segment_energy_and_emissions)
    monkeypatch.setattr(main_module, "_route_stochastic_uncertainty", _route_stochastic_uncertainty)
    monkeypatch.setattr(
        main_module,
        "load_risk_normalization_reference",
        lambda **_kwargs: SimpleNamespace(
            source="pytest",
            version="1",
            as_of_utc="2026-03-25T00:00:00Z",
            corridor_bucket="pytest",
            day_kind="weekday",
            local_time_slot="h12",
        ),
    )
    monkeypatch.setattr(settings, "route_option_cache_enabled", False)

    option = main_module.build_option(
        route,
        option_id="route_0",
        vehicle_type="rigid_hgv",
        scenario_mode=main_module.ScenarioMode.NO_SHARING,
        cost_toggles=CostToggles(),
        terrain_profile="flat",
        stochastic=StochasticConfig(enabled=True, samples=72),
        emissions_context=EmissionsContext(),
        weather=WeatherImpactConfig(),
        incident_simulation=IncidentSimulatorConfig(),
        departure_time_utc=None,
        utility_weights=(1.0, 1.0, 1.0),
        risk_aversion=1.0,
        scenario_policy_cache={},
        lightweight=True,
        force_uncertainty=True,
    )

    assert energy_calls["count"] <= int(settings.route_option_segment_cap_long)
    assert uncertainty_calls["count"] == 1
    assert observed_uncertainty_samples == [24]
    assert option.uncertainty is not None
    assert option.uncertainty_samples_meta is not None
    assert option.uncertainty_samples_meta["lightweight_uncertainty_sample_cap_applied"] is True
    assert option.uncertainty_samples_meta["lightweight_uncertainty_requested_samples"] == 72
    assert len(option.segment_breakdown) == 1
    summary = option.segment_breakdown[0]
    assert summary["segment_index"] == 0
    assert summary["segment_count"] >= 1
    assert summary["distance_km"] == pytest.approx(60.0, rel=0.0, abs=1e-6)
    assert summary["duration_s"] == pytest.approx(3600.0, rel=0.0, abs=1e-6)
    assert option.metrics.duration_s > 0.0
    assert option.metrics.monetary_cost > 0.0
    assert option.metrics.emissions_kg > 0.0


def test_collect_route_options_runs_single_precheck(monkeypatch) -> None:
    precheck_calls = {"count": 0}

    async def _ok_precheck(*, origin: LatLng, destination: LatLng) -> dict[str, Any]:
        _ = (origin, destination)
        precheck_calls["count"] += 1
        return {"ok": True, "reason_code": "ok", "message": "ok"}

    async def _scenario_context(**_: Any) -> Any:
        return None

    async def _scenario_modifiers(**_: Any) -> dict[str, Any]:
        return {}

    async def _candidate_routes(**_: Any):
        return [], [], 0, main_module.CandidateDiagnostics()

    def _build_options(*args: Any, **kwargs: Any):
        _ = (args, kwargs)
        return [], [], main_module.TerrainDiagnostics()

    monkeypatch.setattr(main_module, "_route_graph_od_feasibility_async", _ok_precheck)
    monkeypatch.setattr(main_module, "refresh_live_runtime_route_caches", lambda **_: None)
    monkeypatch.setattr(main_module, "resolve_vehicle_profile", lambda *_: SimpleNamespace(vehicle_class="rigid_hgv"))
    monkeypatch.setattr(main_module, "_scenario_context_from_od", _scenario_context)
    monkeypatch.setattr(main_module, "_scenario_candidate_modifiers_async", _scenario_modifiers)
    monkeypatch.setattr(main_module, "_collect_candidate_routes", _candidate_routes)
    monkeypatch.setattr(main_module, "_build_options", _build_options)
    monkeypatch.setattr(settings, "strict_live_data_required", False)
    monkeypatch.setattr(settings, "live_route_compute_refresh_mode", "route_compute")
    monkeypatch.setattr(settings, "live_route_compute_require_all_expected", False)

    options, warnings, candidate_fetches, _terrain_diag, candidate_diag = asyncio.run(
        main_module._collect_route_options(**_collect_kwargs())
    )
    assert options == []
    assert warnings == []
    assert candidate_fetches == 0
    assert precheck_calls["count"] == 1
    assert candidate_diag.prefetch_total_sources == 0


def test_collect_route_options_initializes_route_option_cache_runtime(monkeypatch) -> None:
    captured_runtime: dict[str, Any] = {}
    origin = _collect_kwargs()["origin"]
    destination = _collect_kwargs()["destination"]

    async def _ok_precheck(*, origin: LatLng, destination: LatLng) -> dict[str, Any]:
        _ = (origin, destination)
        return {
            "ok": True,
            "reason_code": "ok",
            "message": "ok",
            "origin_node_id": "origin-node",
            "destination_node_id": "destination-node",
        }

    async def _scenario_context(**_: Any) -> Any:
        return None

    async def _scenario_modifiers(**_: Any) -> dict[str, Any]:
        return {}

    async def _candidate_routes(**_: Any):
        return [_make_leg_route(origin, destination)], [], 1, main_module.CandidateDiagnostics(raw_count=1, deduped_count=1)

    def _build_options(*args: Any, **kwargs: Any):
        _ = args
        captured_runtime.update(dict(kwargs.get("route_option_cache_runtime") or {}))
        return [], [], main_module.TerrainDiagnostics()

    monkeypatch.setattr(main_module, "_route_graph_od_feasibility_async", _ok_precheck)
    monkeypatch.setattr(main_module, "refresh_live_runtime_route_caches", lambda **_: None)
    monkeypatch.setattr(main_module, "resolve_vehicle_profile", lambda *_: SimpleNamespace(vehicle_class="rigid_hgv"))
    monkeypatch.setattr(main_module, "_scenario_context_from_od", _scenario_context)
    monkeypatch.setattr(main_module, "_scenario_candidate_modifiers_async", _scenario_modifiers)
    monkeypatch.setattr(main_module, "_collect_candidate_routes", _candidate_routes)
    monkeypatch.setattr(main_module, "_build_options", _build_options)
    monkeypatch.setattr(settings, "strict_live_data_required", False)
    monkeypatch.setattr(settings, "live_route_compute_refresh_mode", "route_compute")
    monkeypatch.setattr(settings, "live_route_compute_require_all_expected", False)

    asyncio.run(main_module._collect_route_options(**_collect_kwargs()))

    assert captured_runtime == {
        "cache_hits": 0,
        "cache_hits_local": 0,
        "cache_hits_global": 0,
        "cache_misses": 0,
        "cache_key_missing": 0,
        "cache_disabled": 0,
        "cache_set_failures": 0,
        "saved_ms_estimate": 0.0,
        "reuse_rate": 0.0,
        "last_cache_key": None,
    }


def test_legacy_final_route_trace_defaults_route_option_cache_runtime() -> None:
    option = _make_leg_option("route_0", main_module.ScenarioMode.NO_SHARING)
    trace = main_module._legacy_final_route_trace(
        selected=option,
        frontier_options=[option],
        candidate_fetches=1,
        candidate_diag=main_module.CandidateDiagnostics(selected_candidate_ids_json="[]"),
        run_seed=20260325,
        stage_timings_ms={"collecting_candidates": 1.0, "total": 2.0},
    )

    assert trace["route_option_cache_runtime"] == {
        "cache_hits": 0,
        "cache_hits_local": 0,
        "cache_hits_global": 0,
        "cache_misses": 0,
        "cache_key_missing": 0,
        "cache_disabled": 0,
        "cache_set_failures": 0,
        "saved_ms_estimate": 0.0,
        "reuse_rate": 0.0,
        "last_cache_key": None,
    }


def test_legacy_final_route_trace_no_runtime_dict_calls_do_not_leak_counters() -> None:
    option = _make_leg_option("route_0", main_module.ScenarioMode.NO_SHARING)
    first = main_module._legacy_final_route_trace(
        selected=option,
        frontier_options=[option],
        candidate_fetches=1,
        candidate_diag=main_module.CandidateDiagnostics(selected_candidate_ids_json="[]"),
        run_seed=20260325,
        stage_timings_ms={"collecting_candidates": 1.0, "total": 2.0},
    )
    first["route_option_cache_runtime"]["cache_hits"] = 17

    second = main_module._legacy_final_route_trace(
        selected=option,
        frontier_options=[option],
        candidate_fetches=1,
        candidate_diag=main_module.CandidateDiagnostics(selected_candidate_ids_json="[]"),
        run_seed=20260325,
        stage_timings_ms={"collecting_candidates": 1.0, "total": 2.0},
    )

    assert second["route_option_cache_runtime"] == {
        "cache_hits": 0,
        "cache_hits_local": 0,
        "cache_hits_global": 0,
        "cache_misses": 0,
        "cache_key_missing": 0,
        "cache_disabled": 0,
        "cache_set_failures": 0,
        "saved_ms_estimate": 0.0,
        "reuse_rate": 0.0,
        "last_cache_key": None,
    }


def test_collect_route_options_legacy_candidate_cache_key_changes_with_support_regime(monkeypatch) -> None:
    observed_cache_keys: list[str | None] = []
    short_haul_origin = LatLng(lat=51.4816, lon=-3.1791)
    short_haul_destination = LatLng(lat=51.4545, lon=-2.5879)

    async def _ok_precheck(*, origin: LatLng, destination: LatLng) -> dict[str, Any]:
        _ = (origin, destination)
        return {"ok": True, "reason_code": "ok", "message": "ok"}

    async def _scenario_context(**_: Any) -> Any:
        return None

    async def _scenario_modifiers(**_: Any) -> dict[str, Any]:
        return {}

    async def _candidate_routes(**kwargs: Any):
        observed_cache_keys.append(kwargs.get("cache_key"))
        return [_make_leg_route(short_haul_origin, short_haul_destination)], [], 1, main_module.CandidateDiagnostics(raw_count=1, deduped_count=1)

    def _build_options(*args: Any, **kwargs: Any):
        _ = (args, kwargs)
        return [], [], main_module.TerrainDiagnostics()

    monkeypatch.setattr(main_module, "_route_graph_od_feasibility_async", _ok_precheck)
    monkeypatch.setattr(main_module, "refresh_live_runtime_route_caches", lambda **_: None)
    monkeypatch.setattr(main_module, "resolve_vehicle_profile", lambda *_: SimpleNamespace(vehicle_class="rigid_hgv"))
    monkeypatch.setattr(main_module, "_scenario_context_from_od", _scenario_context)
    monkeypatch.setattr(main_module, "_scenario_candidate_modifiers_async", _scenario_modifiers)
    monkeypatch.setattr(main_module, "_collect_candidate_routes", _candidate_routes)
    monkeypatch.setattr(main_module, "_build_options", _build_options)
    monkeypatch.setattr(settings, "strict_live_data_required", False)
    monkeypatch.setattr(settings, "live_route_compute_refresh_mode", "route_compute")
    monkeypatch.setattr(settings, "live_route_compute_require_all_expected", False)

    short_kwargs = _collect_kwargs()
    short_kwargs["origin"] = short_haul_origin
    short_kwargs["destination"] = short_haul_destination

    asyncio.run(
        main_module._collect_route_options(
            **short_kwargs,
            refinement_policy="corridor_uniform",
            search_budget=2,
            od_ambiguity_index=0.77,
            od_engine_disagreement_prior=0.91,
            od_hard_case_prior=0.33,
            od_ambiguity_support_ratio=0.82,
            od_ambiguity_source_entropy=0.71,
            od_candidate_path_count=3,
            od_corridor_family_count=2,
            allow_supported_ambiguity_fast_fallback=True,
        )
    )
    asyncio.run(
        main_module._collect_route_options(
            **short_kwargs,
            refinement_policy="corridor_uniform",
            search_budget=2,
            od_ambiguity_index=0.22,
            od_engine_disagreement_prior=0.18,
            od_hard_case_prior=0.14,
            od_ambiguity_support_ratio=0.41,
            od_ambiguity_source_entropy=0.15,
            od_candidate_path_count=9,
            od_corridor_family_count=4,
            allow_supported_ambiguity_fast_fallback=True,
        )
    )

    assert len(observed_cache_keys) == 2
    assert observed_cache_keys[0] != observed_cache_keys[1]


def test_collect_route_options_legacy_candidate_cache_key_stays_stable_for_equivalent_support_regime(
    monkeypatch,
) -> None:
    observed_cache_keys: list[str | None] = []
    short_haul_origin = LatLng(lat=51.4816, lon=-3.1791)
    short_haul_destination = LatLng(lat=51.4545, lon=-2.5879)

    async def _ok_precheck(*, origin: LatLng, destination: LatLng) -> dict[str, Any]:
        _ = (origin, destination)
        return {"ok": True, "reason_code": "ok", "message": "ok"}

    async def _scenario_context(**_: Any) -> Any:
        return None

    async def _scenario_modifiers(**_: Any) -> dict[str, Any]:
        return {}

    async def _candidate_routes(**kwargs: Any):
        observed_cache_keys.append(kwargs.get("cache_key"))
        return [_make_leg_route(short_haul_origin, short_haul_destination)], [], 1, main_module.CandidateDiagnostics(raw_count=1, deduped_count=1)

    def _build_options(*args: Any, **kwargs: Any):
        _ = (args, kwargs)
        return [], [], main_module.TerrainDiagnostics()

    monkeypatch.setattr(main_module, "_route_graph_od_feasibility_async", _ok_precheck)
    monkeypatch.setattr(main_module, "refresh_live_runtime_route_caches", lambda **_: None)
    monkeypatch.setattr(main_module, "resolve_vehicle_profile", lambda *_: SimpleNamespace(vehicle_class="rigid_hgv"))
    monkeypatch.setattr(main_module, "_scenario_context_from_od", _scenario_context)
    monkeypatch.setattr(main_module, "_scenario_candidate_modifiers_async", _scenario_modifiers)
    monkeypatch.setattr(main_module, "_collect_candidate_routes", _candidate_routes)
    monkeypatch.setattr(main_module, "_build_options", _build_options)
    monkeypatch.setattr(settings, "strict_live_data_required", False)
    monkeypatch.setattr(settings, "live_route_compute_refresh_mode", "route_compute")
    monkeypatch.setattr(settings, "live_route_compute_require_all_expected", False)

    short_kwargs = _collect_kwargs()
    short_kwargs["origin"] = short_haul_origin
    short_kwargs["destination"] = short_haul_destination

    asyncio.run(
        main_module._collect_route_options(
            **short_kwargs,
            refinement_policy="corridor_uniform",
            search_budget=2,
            od_ambiguity_index=0.77,
            od_engine_disagreement_prior=0.91,
            od_hard_case_prior=0.33,
            od_ambiguity_support_ratio=0.82,
            od_ambiguity_source_entropy=0.71,
            od_candidate_path_count=3,
            od_corridor_family_count=2,
            allow_supported_ambiguity_fast_fallback=True,
        )
    )
    asyncio.run(
        main_module._collect_route_options(
            **short_kwargs,
            refinement_policy="corridor_uniform",
            search_budget=2,
            od_ambiguity_index=0.93,
            od_engine_disagreement_prior=0.64,
            od_hard_case_prior=0.41,
            od_ambiguity_support_ratio=0.85,
            od_ambiguity_source_entropy=0.74,
            od_candidate_path_count=3,
            od_corridor_family_count=2,
            allow_supported_ambiguity_fast_fallback=True,
        )
    )

    assert len(observed_cache_keys) == 2
    assert observed_cache_keys[0] == observed_cache_keys[1]


def test_collect_route_options_legacy_candidate_cache_key_changes_with_retry_suppression_regime(
    monkeypatch,
) -> None:
    observed_cache_keys: list[str | None] = []
    short_haul_origin = LatLng(lat=51.4816, lon=-3.1791)
    short_haul_destination = LatLng(lat=51.4545, lon=-2.5879)

    async def _ok_precheck(*, origin: LatLng, destination: LatLng) -> dict[str, Any]:
        _ = (origin, destination)
        return {"ok": True, "reason_code": "ok", "message": "ok"}

    async def _scenario_context(**_: Any) -> Any:
        return None

    async def _scenario_modifiers(**_: Any) -> dict[str, Any]:
        return {}

    async def _candidate_routes(**kwargs: Any):
        observed_cache_keys.append(kwargs.get("cache_key"))
        return [_make_leg_route(short_haul_origin, short_haul_destination)], [], 1, main_module.CandidateDiagnostics(raw_count=1, deduped_count=1)

    def _build_options(*args: Any, **kwargs: Any):
        _ = (args, kwargs)
        return [], [], main_module.TerrainDiagnostics()

    monkeypatch.setattr(main_module, "_route_graph_od_feasibility_async", _ok_precheck)
    monkeypatch.setattr(main_module, "refresh_live_runtime_route_caches", lambda **_: None)
    monkeypatch.setattr(main_module, "resolve_vehicle_profile", lambda *_: SimpleNamespace(vehicle_class="rigid_hgv"))
    monkeypatch.setattr(main_module, "_scenario_context_from_od", _scenario_context)
    monkeypatch.setattr(main_module, "_scenario_candidate_modifiers_async", _scenario_modifiers)
    monkeypatch.setattr(main_module, "_collect_candidate_routes", _candidate_routes)
    monkeypatch.setattr(main_module, "_build_options", _build_options)
    monkeypatch.setattr(settings, "strict_live_data_required", False)
    monkeypatch.setattr(settings, "live_route_compute_refresh_mode", "route_compute")
    monkeypatch.setattr(settings, "live_route_compute_require_all_expected", False)

    short_kwargs = _collect_kwargs()
    short_kwargs["origin"] = short_haul_origin
    short_kwargs["destination"] = short_haul_destination

    asyncio.run(
        main_module._collect_route_options(
            **short_kwargs,
            refinement_policy="corridor_uniform",
            search_budget=2,
            od_ambiguity_index=0.77,
            od_engine_disagreement_prior=0.91,
            od_hard_case_prior=0.33,
            od_ambiguity_support_ratio=0.82,
            od_ambiguity_source_entropy=0.71,
            od_candidate_path_count=9,
            od_corridor_family_count=4,
            allow_supported_ambiguity_fast_fallback=True,
        )
    )
    asyncio.run(
        main_module._collect_route_options(
            **short_kwargs,
            refinement_policy="corridor_uniform",
            search_budget=2,
            od_ambiguity_index=0.12,
            od_engine_disagreement_prior=0.08,
            od_hard_case_prior=0.05,
            od_ambiguity_support_ratio=0.22,
            od_ambiguity_source_entropy=0.10,
            od_candidate_path_count=9,
            od_corridor_family_count=4,
            allow_supported_ambiguity_fast_fallback=True,
        )
    )

    assert len(observed_cache_keys) == 2
    assert observed_cache_keys[0] != observed_cache_keys[1]


def test_collect_route_options_legacy_candidate_cache_key_stays_stable_for_equivalent_retry_suppression_regime(
    monkeypatch,
) -> None:
    observed_cache_keys: list[str | None] = []
    short_haul_origin = LatLng(lat=51.4816, lon=-3.1791)
    short_haul_destination = LatLng(lat=51.4545, lon=-2.5879)

    async def _ok_precheck(*, origin: LatLng, destination: LatLng) -> dict[str, Any]:
        _ = (origin, destination)
        return {"ok": True, "reason_code": "ok", "message": "ok"}

    async def _scenario_context(**_: Any) -> Any:
        return None

    async def _scenario_modifiers(**_: Any) -> dict[str, Any]:
        return {}

    async def _candidate_routes(**kwargs: Any):
        observed_cache_keys.append(kwargs.get("cache_key"))
        return [_make_leg_route(short_haul_origin, short_haul_destination)], [], 1, main_module.CandidateDiagnostics(raw_count=1, deduped_count=1)

    def _build_options(*args: Any, **kwargs: Any):
        _ = (args, kwargs)
        return [], [], main_module.TerrainDiagnostics()

    monkeypatch.setattr(main_module, "_route_graph_od_feasibility_async", _ok_precheck)
    monkeypatch.setattr(main_module, "refresh_live_runtime_route_caches", lambda **_: None)
    monkeypatch.setattr(main_module, "resolve_vehicle_profile", lambda *_: SimpleNamespace(vehicle_class="rigid_hgv"))
    monkeypatch.setattr(main_module, "_scenario_context_from_od", _scenario_context)
    monkeypatch.setattr(main_module, "_scenario_candidate_modifiers_async", _scenario_modifiers)
    monkeypatch.setattr(main_module, "_collect_candidate_routes", _candidate_routes)
    monkeypatch.setattr(main_module, "_build_options", _build_options)
    monkeypatch.setattr(settings, "strict_live_data_required", False)
    monkeypatch.setattr(settings, "live_route_compute_refresh_mode", "route_compute")
    monkeypatch.setattr(settings, "live_route_compute_require_all_expected", False)

    short_kwargs = _collect_kwargs()
    short_kwargs["origin"] = short_haul_origin
    short_kwargs["destination"] = short_haul_destination

    asyncio.run(
        main_module._collect_route_options(
            **short_kwargs,
            refinement_policy="corridor_uniform",
            search_budget=2,
            od_ambiguity_index=0.77,
            od_engine_disagreement_prior=0.91,
            od_hard_case_prior=0.33,
            od_ambiguity_support_ratio=0.82,
            od_ambiguity_source_entropy=0.71,
            od_candidate_path_count=9,
            od_corridor_family_count=4,
            allow_supported_ambiguity_fast_fallback=True,
        )
    )
    asyncio.run(
        main_module._collect_route_options(
            **short_kwargs,
            refinement_policy="corridor_uniform",
            search_budget=2,
            od_ambiguity_index=0.63,
            od_engine_disagreement_prior=0.74,
            od_hard_case_prior=0.29,
            od_ambiguity_support_ratio=0.84,
            od_ambiguity_source_entropy=0.76,
            od_candidate_path_count=9,
            od_corridor_family_count=4,
            allow_supported_ambiguity_fast_fallback=True,
        )
    )

    assert len(observed_cache_keys) == 2
    assert observed_cache_keys[0] == observed_cache_keys[1]


def test_collect_route_options_legacy_candidate_cache_does_not_share_across_support_regimes(monkeypatch) -> None:
    short_haul_origin = LatLng(lat=51.4816, lon=-3.1791)
    short_haul_destination = LatLng(lat=51.4545, lon=-2.5879)
    cache_store: dict[
        str | None,
        tuple[list[dict[str, Any]], list[str], int, main_module.CandidateDiagnostics],
    ] = {}
    observed_cache_keys: list[str | None] = []
    observed_regimes: list[str] = []

    async def _ok_precheck(*, origin: LatLng, destination: LatLng) -> dict[str, Any]:
        _ = (origin, destination)
        return {"ok": True, "reason_code": "ok", "message": "ok"}

    async def _scenario_context(**_: Any) -> Any:
        return None

    async def _scenario_modifiers(**_: Any) -> dict[str, Any]:
        return {}

    async def _candidate_routes(**kwargs: Any):
        cache_key = kwargs.get("cache_key")
        observed_cache_keys.append(cache_key)
        cached_result = cache_store.get(cache_key)
        if cached_result is not None:
            return cached_result
        regime = (
            "support_rich"
            if int(kwargs.get("od_candidate_path_count") or 0) <= 4
            and int(kwargs.get("od_corridor_family_count") or 0) <= 2
            else "graph_search"
        )
        routes = [_make_leg_route(short_haul_origin, short_haul_destination)]
        routes[0]["_support_regime"] = regime
        result = (routes, [], 1, main_module.CandidateDiagnostics(raw_count=1, deduped_count=1))
        cache_store[cache_key] = result
        return result

    def _build_options(*args: Any, **kwargs: Any):
        routes = list(args[0])
        observed_regimes.append(str(routes[0].get("_support_regime", "missing")))
        _ = kwargs
        return [], [], main_module.TerrainDiagnostics()

    monkeypatch.setattr(main_module, "_route_graph_od_feasibility_async", _ok_precheck)
    monkeypatch.setattr(main_module, "refresh_live_runtime_route_caches", lambda **_: None)
    monkeypatch.setattr(main_module, "resolve_vehicle_profile", lambda *_: SimpleNamespace(vehicle_class="rigid_hgv"))
    monkeypatch.setattr(main_module, "_scenario_context_from_od", _scenario_context)
    monkeypatch.setattr(main_module, "_scenario_candidate_modifiers_async", _scenario_modifiers)
    monkeypatch.setattr(main_module, "_collect_candidate_routes", _candidate_routes)
    monkeypatch.setattr(main_module, "_build_options", _build_options)
    monkeypatch.setattr(settings, "strict_live_data_required", False)
    monkeypatch.setattr(settings, "live_route_compute_refresh_mode", "route_compute")
    monkeypatch.setattr(settings, "live_route_compute_require_all_expected", False)

    short_kwargs = _collect_kwargs()
    short_kwargs["origin"] = short_haul_origin
    short_kwargs["destination"] = short_haul_destination

    asyncio.run(
        main_module._collect_route_options(
            **short_kwargs,
            refinement_policy="corridor_uniform",
            search_budget=2,
            od_ambiguity_index=0.77,
            od_engine_disagreement_prior=0.91,
            od_hard_case_prior=0.33,
            od_ambiguity_support_ratio=0.82,
            od_ambiguity_source_entropy=0.71,
            od_candidate_path_count=3,
            od_corridor_family_count=2,
            allow_supported_ambiguity_fast_fallback=True,
        )
    )
    asyncio.run(
        main_module._collect_route_options(
            **short_kwargs,
            refinement_policy="corridor_uniform",
            search_budget=2,
            od_ambiguity_index=0.22,
            od_engine_disagreement_prior=0.18,
            od_hard_case_prior=0.14,
            od_ambiguity_support_ratio=0.41,
            od_ambiguity_source_entropy=0.15,
            od_candidate_path_count=9,
            od_corridor_family_count=4,
            allow_supported_ambiguity_fast_fallback=True,
        )
    )

    assert len(observed_cache_keys) == 2
    assert observed_cache_keys[0] != observed_cache_keys[1]
    assert len(cache_store) == 2
    assert observed_regimes == ["support_rich", "graph_search"]


def test_lightweight_build_rehydrates_only_priority_subset(monkeypatch) -> None:
    build_calls: list[tuple[str, bool]] = []
    origin = _collect_kwargs()["origin"]
    destination = _collect_kwargs()["destination"]
    routes = [
        _make_leg_route(origin, destination),
        _make_leg_route(origin, destination),
        _make_leg_route(origin, destination),
    ]

    def _build_option(route: dict[str, Any], *, option_id: str, lightweight: bool = False, **kwargs: Any) -> RouteOption:
        _ = (route, kwargs)
        build_calls.append((option_id, bool(lightweight)))
        update = {
            "eta_explanations": [] if lightweight else ["full"],
            "eta_timeline": [] if lightweight else [{"stage": "baseline", "duration_s": 2400.0, "delta_s": 0.0}],
            "segment_breakdown": [] if lightweight else [{"segment_index": 0, "distance_km": 50.0}],
            "counterfactuals": [] if lightweight else [{"id": "cf"}],
            "weather_summary": {"mode": "light" if lightweight else "full"},
            "uncertainty": None if lightweight else {"time": 1.0, "money": 1.0, "co2": 1.0},
            "uncertainty_samples_meta": None if lightweight else {"objective_samples_json": "{\"route_0\":[1.0]}"},
        }
        return _make_leg_option(option_id, main_module.ScenarioMode.NO_SHARING).model_copy(
            update=update,
            deep=True,
        )

    monkeypatch.setattr(main_module, "build_option", _build_option)
    monkeypatch.setattr(main_module, "terrain_begin_route_run", lambda: None)
    monkeypatch.setattr(main_module, "resolve_vehicle_profile", lambda *_: SimpleNamespace(vehicle_class="rigid_hgv"))
    monkeypatch.setattr(settings, "route_option_cache_enabled", False)
    monkeypatch.setattr(settings, "route_option_reuse_scenario_policy", False)

    options, warnings, terrain_diag = main_module._build_options(
        routes,
        vehicle_type="rigid_hgv",
        scenario_mode=main_module.ScenarioMode.NO_SHARING,
        cost_toggles=CostToggles(),
        terrain_profile="flat",
        stochastic=StochasticConfig(),
        emissions_context=EmissionsContext(),
        weather=WeatherImpactConfig(),
        incident_simulation=IncidentSimulatorConfig(),
        departure_time_utc=None,
        option_prefix="route",
        route_option_cache_runtime={},
        scenario_policy_cache={},
        lightweight=True,
    )
    assert warnings == []
    assert terrain_diag.fail_closed_count == 0
    assert build_calls == [("route_0", True), ("route_1", True), ("route_2", True)]
    assert all(not option.eta_explanations for option in options)
    assert all(not option.segment_breakdown for option in options)
    assert all(not option.counterfactuals for option in options)
    assert all(option.uncertainty is None for option in options)
    assert all(option.uncertainty_samples_meta is None for option in options)

    hydrated_options, hydrated_frontier, hydrated_display, hydrated_selected, hydrate_ms = (
        main_module._hydrate_priority_route_options(
            routes=routes,
            options=options,
            strict_frontier=[options[0]],
            display_candidates=[options[0], options[1]],
            selected=options[0],
            vehicle_type="rigid_hgv",
            scenario_mode=main_module.ScenarioMode.NO_SHARING,
            cost_toggles=CostToggles(),
            terrain_profile="flat",
            stochastic=StochasticConfig(),
            emissions_context=EmissionsContext(),
            weather=WeatherImpactConfig(),
            incident_simulation=IncidentSimulatorConfig(),
            departure_time_utc=None,
            scenario_policy_cache={},
        )
    )

    assert hydrate_ms >= 0.0
    assert ("route_0", False) in build_calls
    assert ("route_1", False) in build_calls
    assert ("route_2", False) not in build_calls
    assert sum(1 for _, lightweight in build_calls if lightweight) == 3
    assert sum(1 for _, lightweight in build_calls if not lightweight) == 2
    assert hydrated_selected.eta_explanations == ["full"]
    assert hydrated_selected.uncertainty == {"time": 1.0, "money": 1.0, "co2": 1.0}
    assert hydrated_frontier[0].eta_explanations == ["full"]
    assert hydrated_display[0].eta_explanations == ["full"]
    assert hydrated_display[1].eta_explanations == ["full"]
    assert hydrated_options[0].eta_explanations == ["full"]
    assert hydrated_options[1].eta_explanations == ["full"]
    assert hydrated_options[2].eta_explanations == []
    assert hydrated_options[0].uncertainty == {"time": 1.0, "money": 1.0, "co2": 1.0}
    assert hydrated_options[1].uncertainty == {"time": 1.0, "money": 1.0, "co2": 1.0}
    assert hydrated_options[2].uncertainty is None


def test_collect_route_options_multileg_legacy_candidate_cache_ignores_top_level_support_regime(monkeypatch) -> None:
    observed_cache_keys: list[str | None] = []

    async def _ok_precheck(*, origin: LatLng, destination: LatLng) -> dict[str, Any]:
        _ = (origin, destination)
        return {
            "ok": True,
            "reason_code": "ok",
            "message": "ok",
            "origin_node_id": "origin-node",
            "destination_node_id": "destination-node",
        }

    async def _scenario_context(**_: Any) -> Any:
        return None

    async def _scenario_modifiers(**_: Any) -> dict[str, Any]:
        return {}

    async def _candidate_routes(**kwargs: Any):
        observed_cache_keys.append(kwargs.get("cache_key"))
        leg_origin = kwargs["origin"]
        leg_destination = kwargs["destination"]
        return [_make_leg_route(leg_origin, leg_destination)], [], 1, main_module.CandidateDiagnostics(raw_count=1, deduped_count=1)

    def _build_options(*args: Any, **kwargs: Any):
        _ = (args, kwargs)
        return [], [], main_module.TerrainDiagnostics()

    monkeypatch.setattr(main_module, "_route_graph_od_feasibility_async", _ok_precheck)
    monkeypatch.setattr(main_module, "refresh_live_runtime_route_caches", lambda **_: None)
    monkeypatch.setattr(main_module, "resolve_vehicle_profile", lambda *_: SimpleNamespace(vehicle_class="rigid_hgv"))
    monkeypatch.setattr(main_module, "_scenario_context_from_od", _scenario_context)
    monkeypatch.setattr(main_module, "_scenario_candidate_modifiers_async", _scenario_modifiers)
    monkeypatch.setattr(main_module, "_collect_candidate_routes", _candidate_routes)
    monkeypatch.setattr(main_module, "_build_options", _build_options)
    monkeypatch.setattr(settings, "strict_live_data_required", False)
    monkeypatch.setattr(settings, "live_route_compute_refresh_mode", "route_compute")
    monkeypatch.setattr(settings, "live_route_compute_require_all_expected", False)

    base_kwargs = _collect_kwargs()
    base_kwargs["waypoints"] = [Waypoint(lat=52.8, lon=-1.2, label="mid")]

    asyncio.run(
        main_module._collect_route_options(
            **base_kwargs,
            refinement_policy="corridor_uniform",
            search_budget=2,
            od_ambiguity_index=0.77,
            od_engine_disagreement_prior=0.91,
            od_hard_case_prior=0.33,
            od_ambiguity_support_ratio=0.82,
            od_ambiguity_source_entropy=0.71,
            od_candidate_path_count=3,
            od_corridor_family_count=2,
            allow_supported_ambiguity_fast_fallback=True,
        )
    )
    asyncio.run(
        main_module._collect_route_options(
            **base_kwargs,
            refinement_policy="corridor_uniform",
            search_budget=2,
            od_ambiguity_index=0.12,
            od_engine_disagreement_prior=0.08,
            od_hard_case_prior=0.05,
            od_ambiguity_support_ratio=0.22,
            od_ambiguity_source_entropy=0.10,
            od_candidate_path_count=11,
            od_corridor_family_count=5,
            allow_supported_ambiguity_fast_fallback=True,
        )
    )

    assert len(observed_cache_keys) in {2, 4}
    if len(observed_cache_keys) == 4:
        assert observed_cache_keys[:2] == observed_cache_keys[2:]


def test_legacy_final_route_trace_preserves_route_option_cache_runtime() -> None:
    option = _make_leg_option("route_0", main_module.ScenarioMode.NO_SHARING)
    trace = main_module._legacy_final_route_trace(
        selected=option,
        frontier_options=[option],
        candidate_fetches=1,
        candidate_diag=main_module.CandidateDiagnostics(selected_candidate_ids_json="[]"),
        run_seed=20260325,
        stage_timings_ms={"collecting_candidates": 1.0, "total": 2.0},
        route_option_cache_runtime={
            "cache_hits": 3,
            "cache_hits_local": 2,
            "cache_hits_global": 1,
            "cache_misses": 4,
            "cache_key_missing": 0,
            "cache_disabled": 0,
            "cache_set_failures": 0,
            "saved_ms_estimate": 12.5,
            "reuse_rate": 0.428571,
            "last_cache_key": "legacy-route-key",
        },
    )

    assert trace["route_option_cache_runtime"] == {
        "cache_hits": 3,
        "cache_hits_local": 2,
        "cache_hits_global": 1,
        "cache_misses": 4,
        "cache_key_missing": 0,
        "cache_disabled": 0,
        "cache_set_failures": 0,
        "saved_ms_estimate": 12.5,
        "reuse_rate": 0.428571,
        "last_cache_key": "legacy-route-key",
    }


def test_collect_route_options_populates_external_route_option_cache_runtime(monkeypatch) -> None:
    runtime_out: dict[str, Any] = {}
    origin = _collect_kwargs()["origin"]
    destination = _collect_kwargs()["destination"]

    async def _ok_precheck(*, origin: LatLng, destination: LatLng) -> dict[str, Any]:
        _ = (origin, destination)
        return {
            "ok": True,
            "reason_code": "ok",
            "message": "ok",
            "origin_node_id": "origin-node",
            "destination_node_id": "destination-node",
        }

    async def _scenario_context(**_: Any) -> Any:
        return None

    async def _scenario_modifiers(**_: Any) -> dict[str, Any]:
        return {}

    async def _candidate_routes(**_: Any):
        return [_make_leg_route(origin, destination)], [], 1, main_module.CandidateDiagnostics(raw_count=1, deduped_count=1)

    def _build_options(*args: Any, **kwargs: Any):
        runtime = dict(kwargs.get("route_option_cache_runtime") or {})
        runtime["cache_hits"] = 5
        kwargs["route_option_cache_runtime"].update(runtime)
        return [], [], main_module.TerrainDiagnostics()

    monkeypatch.setattr(main_module, "_route_graph_od_feasibility_async", _ok_precheck)
    monkeypatch.setattr(main_module, "refresh_live_runtime_route_caches", lambda **_: None)
    monkeypatch.setattr(main_module, "resolve_vehicle_profile", lambda *_: SimpleNamespace(vehicle_class="rigid_hgv"))
    monkeypatch.setattr(main_module, "_scenario_context_from_od", _scenario_context)
    monkeypatch.setattr(main_module, "_scenario_candidate_modifiers_async", _scenario_modifiers)
    monkeypatch.setattr(main_module, "_collect_candidate_routes", _candidate_routes)
    monkeypatch.setattr(main_module, "_build_options", _build_options)
    monkeypatch.setattr(settings, "strict_live_data_required", False)
    monkeypatch.setattr(settings, "live_route_compute_refresh_mode", "route_compute")
    monkeypatch.setattr(settings, "live_route_compute_require_all_expected", False)

    asyncio.run(
        main_module._collect_route_options(
            **_collect_kwargs(),
            route_option_cache_runtime_out=runtime_out,
        )
    )

    assert runtime_out["cache_hits"] == 5


def test_collect_route_options_honors_route_compute_refresh_mode_in_strict_runtime(monkeypatch) -> None:
    refresh_modes: list[str] = []

    async def _ok_precheck(*, origin: LatLng, destination: LatLng) -> dict[str, Any]:
        _ = (origin, destination)
        return {"ok": True, "reason_code": "ok", "message": "ok"}

    async def _prefetch_ok(**_: Any) -> dict[str, Any]:
        return {
            "source_total": 14,
            "source_success": 14,
            "source_failed": 0,
            "failed_source_keys": [],
            "rows": [],
        }

    async def _scenario_context(**_: Any) -> Any:
        return None

    async def _scenario_modifiers(**_: Any) -> dict[str, Any]:
        return {}

    async def _candidate_routes(**_: Any):
        return [], [], 0, main_module.CandidateDiagnostics(raw_count=0, deduped_count=0)

    def _build_options(*args: Any, **kwargs: Any):
        _ = (args, kwargs)
        return [], [], main_module.TerrainDiagnostics()

    monkeypatch.setattr(main_module, "_route_graph_od_feasibility_async", _ok_precheck)
    monkeypatch.setattr(
        main_module,
        "refresh_live_runtime_route_caches",
        lambda **kwargs: refresh_modes.append(str(kwargs.get("mode", ""))),
    )
    monkeypatch.setattr(main_module, "resolve_vehicle_profile", lambda *_: SimpleNamespace(vehicle_class="rigid_hgv"))
    monkeypatch.setattr(main_module, "_prefetch_expected_live_sources", _prefetch_ok)
    monkeypatch.setattr(main_module, "_scenario_context_from_od", _scenario_context)
    monkeypatch.setattr(main_module, "_scenario_candidate_modifiers_async", _scenario_modifiers)
    monkeypatch.setattr(main_module, "_collect_candidate_routes", _candidate_routes)
    monkeypatch.setattr(main_module, "_build_options", _build_options)
    monkeypatch.setattr(settings, "strict_live_data_required", True)
    monkeypatch.setattr(settings, "live_route_compute_refresh_mode", "route_compute")
    monkeypatch.setattr(settings, "live_route_compute_require_all_expected", True)

    asyncio.run(main_module._collect_route_options(**_collect_kwargs()))

    assert refresh_modes
    assert refresh_modes[0] == "route_compute"
    assert "all_sources" not in refresh_modes


def test_collect_route_options_skips_external_prefetch_under_repo_local_policy(monkeypatch) -> None:
    prefetch_called = {"value": False}

    async def _ok_precheck(*, origin: LatLng, destination: LatLng) -> dict[str, Any]:
        return {
            "ok": True,
            "reason_code": "ok",
            "message": "ok",
            "origin_node_id": "origin-node",
            "destination_node_id": "destination-node",
        }

    async def _scenario_context(**_: Any) -> Any:
        return None

    async def _scenario_modifiers(**_: Any) -> dict[str, Any]:
        return {}

    async def _candidate_routes(**_: Any):
        return [], [], 0, main_module.CandidateDiagnostics(raw_count=0, deduped_count=0)

    def _build_options(*args: Any, **kwargs: Any):
        _ = (args, kwargs)
        return [], [], main_module.TerrainDiagnostics()

    async def _prefetch_fail(**_: Any) -> dict[str, Any]:
        prefetch_called["value"] = True
        raise AssertionError("repo_local_fresh should not external-prefetch route-compute sources")

    monkeypatch.setattr(main_module, "_route_graph_od_feasibility_async", _ok_precheck)
    monkeypatch.setattr(main_module, "refresh_live_runtime_route_caches", lambda **_: None)
    monkeypatch.setattr(main_module, "resolve_vehicle_profile", lambda *_: SimpleNamespace(vehicle_class="rigid_hgv"))
    monkeypatch.setattr(main_module, "_prefetch_expected_live_sources", _prefetch_fail)
    monkeypatch.setattr(main_module, "_scenario_context_from_od", _scenario_context)
    monkeypatch.setattr(main_module, "_scenario_candidate_modifiers_async", _scenario_modifiers)
    monkeypatch.setattr(main_module, "_collect_candidate_routes", _candidate_routes)
    monkeypatch.setattr(main_module, "_build_options", _build_options)
    monkeypatch.setattr(settings, "strict_live_data_required", True)
    monkeypatch.setattr(settings, "live_route_compute_require_all_expected", True)
    monkeypatch.setattr(settings, "live_route_compute_refresh_mode", "route_compute")
    monkeypatch.setattr(settings, "live_source_policy", "repo_local_fresh")

    options, warnings, candidate_fetches, _terrain_diag, candidate_diag = asyncio.run(
        main_module._collect_route_options(**_collect_kwargs())
    )

    assert options == []
    assert warnings == []
    assert candidate_fetches == 0
    assert prefetch_called["value"] is False
    assert candidate_diag.prefetch_total_sources == 0


def test_collect_route_options_multileg_reuses_context_cache_and_avoids_per_leg_refresh(monkeypatch) -> None:
    refresh_modes: list[str] = []
    scenario_context_calls: list[tuple[float, float, float, float]] = []

    async def _ok_precheck(*, origin: LatLng, destination: LatLng) -> dict[str, Any]:
        return {
            "ok": True,
            "reason_code": "ok",
            "message": "ok",
            "origin_node_id": f"node_{origin.lat:.4f}_{origin.lon:.4f}",
            "destination_node_id": f"node_{destination.lat:.4f}_{destination.lon:.4f}",
        }

    async def _scenario_context(
        *,
        origin: LatLng,
        destination: LatLng,
        **_: Any,
    ) -> Any:
        scenario_context_calls.append((origin.lat, origin.lon, destination.lat, destination.lon))
        return {"key": f"{origin.lat:.6f},{origin.lon:.6f}->{destination.lat:.6f},{destination.lon:.6f}"}

    async def _scenario_modifiers(**_: Any) -> dict[str, Any]:
        return {}

    async def _candidate_routes(
        *,
        origin: LatLng,
        destination: LatLng,
        **_: Any,
    ):
        route = _make_leg_route(origin, destination)
        return [route], [], 1, main_module.CandidateDiagnostics(raw_count=1, deduped_count=1)

    def _build_options(*_: Any, option_prefix: str, **kwargs: Any):
        mode = kwargs.get("scenario_mode", main_module.ScenarioMode.NO_SHARING)
        return [_make_leg_option(f"{option_prefix}_0", mode)], [], main_module.TerrainDiagnostics()

    monkeypatch.setattr(main_module, "_route_graph_od_feasibility_async", _ok_precheck)
    monkeypatch.setattr(
        main_module,
        "refresh_live_runtime_route_caches",
        lambda **kwargs: refresh_modes.append(str(kwargs.get("mode", ""))),
    )
    monkeypatch.setattr(main_module, "resolve_vehicle_profile", lambda *_: SimpleNamespace(vehicle_class="rigid_hgv"))
    monkeypatch.setattr(main_module, "_scenario_context_from_od", _scenario_context)
    monkeypatch.setattr(main_module, "_scenario_candidate_modifiers_async", _scenario_modifiers)
    monkeypatch.setattr(main_module, "_collect_candidate_routes", _candidate_routes)
    monkeypatch.setattr(main_module, "_build_options", _build_options)
    monkeypatch.setattr(settings, "strict_live_data_required", False)
    monkeypatch.setattr(settings, "live_route_compute_refresh_mode", "route_compute")
    monkeypatch.setattr(settings, "live_route_compute_require_all_expected", False)

    kwargs = _collect_kwargs()
    kwargs["waypoints"] = [
        Waypoint(lat=53.2000, lon=-1.5000, label="A"),
        Waypoint(lat=kwargs["origin"].lat, lon=kwargs["origin"].lon, label="B"),
        Waypoint(lat=53.2000, lon=-1.5000, label="C"),
    ]

    options, warnings, candidate_fetches, _terrain_diag, candidate_diag = asyncio.run(
        main_module._collect_route_options(**kwargs)
    )

    assert options
    assert warnings == []
    assert candidate_fetches >= 1
    assert refresh_modes == ["route_compute"]
    # 1 base OD context + 3 unique leg contexts (origin->A repeats once and is cache-hit on 2nd occurrence).
    assert len(scenario_context_calls) == 4
    assert candidate_diag.scenario_context_ms >= 0.0


def test_collect_route_options_prefetch_gate_failure_maps_reason(monkeypatch) -> None:
    precheck_calls = {"count": 0}

    async def _ok_precheck(*, origin: LatLng, destination: LatLng) -> dict[str, Any]:
        _ = (origin, destination)
        precheck_calls["count"] += 1
        return {"ok": True, "reason_code": "ok", "message": "ok"}

    async def _prefetch_fail(**_: Any) -> dict[str, Any]:
        raise ModelDataError(
            reason_code="live_source_refresh_failed",
            message="Live-source prefetch failed strict freshness gate.",
            details={
                "source_total": 14,
                "source_success": 11,
                "source_failed": 3,
                "failed_source_keys": ["departure_profiles", "fuel_prices", "terrain_live_tile"],
            },
        )

    monkeypatch.setattr(main_module, "_route_graph_od_feasibility_async", _ok_precheck)
    monkeypatch.setattr(main_module, "refresh_live_runtime_route_caches", lambda **_: None)
    monkeypatch.setattr(main_module, "resolve_vehicle_profile", lambda *_: SimpleNamespace(vehicle_class="rigid_hgv"))
    monkeypatch.setattr(main_module, "_prefetch_expected_live_sources", _prefetch_fail)
    monkeypatch.setattr(settings, "live_route_compute_refresh_mode", "all_sources")
    monkeypatch.setattr(settings, "live_route_compute_require_all_expected", True)

    options, warnings, candidate_fetches, _terrain_diag, candidate_diag = asyncio.run(
        main_module._collect_route_options(**_collect_kwargs())
    )
    assert options == []
    assert candidate_fetches == 0
    # Prefetch now executes before graph feasibility precheck.
    assert precheck_calls["count"] == 0
    assert warnings
    assert "route_live_prefetch: live_source_refresh_failed" in warnings[0]
    assert candidate_diag.prefetch_total_sources == 14
    assert candidate_diag.prefetch_success_sources == 11
    assert candidate_diag.prefetch_failed_sources == 3
    assert "departure_profiles" in candidate_diag.prefetch_failed_keys


def test_collect_route_options_prefetch_summary_propagates_to_candidate_diag(monkeypatch) -> None:
    async def _ok_precheck(*, origin: LatLng, destination: LatLng) -> dict[str, Any]:
        _ = (origin, destination)
        return {"ok": True, "reason_code": "ok", "message": "ok"}

    async def _prefetch_ok(**_: Any) -> dict[str, Any]:
        return {
            "source_total": 14,
            "source_success": 14,
            "source_failed": 0,
            "failed_source_keys": [],
            "rows": [],
        }

    async def _scenario_context(**_: Any) -> Any:
        return None

    async def _scenario_modifiers(**_: Any) -> dict[str, Any]:
        return {}

    async def _candidate_routes(**_: Any):
        return [], [], 0, main_module.CandidateDiagnostics(raw_count=0, deduped_count=0)

    def _build_options(*args: Any, **kwargs: Any):
        _ = (args, kwargs)
        return [], [], main_module.TerrainDiagnostics()

    monkeypatch.setattr(main_module, "_route_graph_od_feasibility_async", _ok_precheck)
    monkeypatch.setattr(main_module, "refresh_live_runtime_route_caches", lambda **_: None)
    monkeypatch.setattr(main_module, "resolve_vehicle_profile", lambda *_: SimpleNamespace(vehicle_class="rigid_hgv"))
    monkeypatch.setattr(main_module, "_prefetch_expected_live_sources", _prefetch_ok)
    monkeypatch.setattr(main_module, "_scenario_context_from_od", _scenario_context)
    monkeypatch.setattr(main_module, "_scenario_candidate_modifiers_async", _scenario_modifiers)
    monkeypatch.setattr(main_module, "_collect_candidate_routes", _candidate_routes)
    monkeypatch.setattr(main_module, "_build_options", _build_options)
    monkeypatch.setattr(settings, "live_route_compute_refresh_mode", "all_sources")
    monkeypatch.setattr(settings, "live_route_compute_require_all_expected", True)

    _options, _warnings, _candidate_fetches, _terrain_diag, candidate_diag = asyncio.run(
        main_module._collect_route_options(**_collect_kwargs())
    )
    assert candidate_diag.prefetch_total_sources == 14
    assert candidate_diag.prefetch_success_sources == 14
    assert candidate_diag.prefetch_failed_sources == 0
    assert candidate_diag.prefetch_failed_keys == ""


def test_collect_route_options_prefetch_runs_before_disconnected_precheck(monkeypatch) -> None:
    call_order: list[str] = []

    async def _prefetch_ok(**_: Any) -> dict[str, Any]:
        call_order.append("prefetch")
        return {
            "source_total": 14,
            "source_success": 14,
            "source_failed": 0,
            "failed_source_keys": [],
            "rows": [],
        }

    async def _disconnected_precheck(*, origin: LatLng, destination: LatLng) -> dict[str, Any]:
        _ = (origin, destination)
        call_order.append("precheck")
        return {
            "ok": False,
            "reason_code": "routing_graph_disconnected_od",
            "message": "Origin and destination are disconnected in the loaded graph component map.",
            "origin_component": 1,
            "destination_component": 91,
            "origin_nearest_distance_m": 24.0,
            "destination_nearest_distance_m": 27.0,
            "origin_candidate_count": 16,
            "destination_candidate_count": 16,
        }

    async def _scenario_context_should_not_run(**_: Any) -> Any:
        raise AssertionError("scenario context should not run on disconnected precheck")

    monkeypatch.setattr(main_module, "_route_graph_od_feasibility_async", _disconnected_precheck)
    monkeypatch.setattr(main_module, "refresh_live_runtime_route_caches", lambda **_: None)
    monkeypatch.setattr(main_module, "resolve_vehicle_profile", lambda *_: SimpleNamespace(vehicle_class="rigid_hgv"))
    monkeypatch.setattr(main_module, "_prefetch_expected_live_sources", _prefetch_ok)
    monkeypatch.setattr(main_module, "_scenario_context_from_od", _scenario_context_should_not_run)
    monkeypatch.setattr(settings, "live_route_compute_refresh_mode", "all_sources")
    monkeypatch.setattr(settings, "live_route_compute_require_all_expected", True)

    options, warnings, candidate_fetches, _terrain_diag, candidate_diag = asyncio.run(
        main_module._collect_route_options(**_collect_kwargs())
    )
    assert options == []
    assert candidate_fetches == 0
    assert call_order == ["prefetch", "precheck"]
    assert warnings
    assert "routing_graph_disconnected_od" in warnings[0]
    assert candidate_diag.prefetch_total_sources == 14
    assert candidate_diag.prefetch_success_sources == 14
    assert candidate_diag.prefetch_failed_sources == 0
    assert candidate_diag.precheck_reason_code == "routing_graph_disconnected_od"
    assert candidate_diag.precheck_origin_candidate_count == 16
    assert candidate_diag.precheck_destination_candidate_count == 16


def test_prefetch_expected_live_sources_passes_correct_carbon_kwargs(monkeypatch) -> None:
    observed: dict[str, Any] = {}

    def _scenario_profiles() -> Any:
        return SimpleNamespace(transform_params={"mode_effect_scale": {"road": 1.0}})

    def _resolve_carbon_price(**kwargs: Any) -> Any:
        observed.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(settings, "strict_live_data_required", False)
    monkeypatch.setattr(settings, "live_route_compute_require_all_expected", False)
    monkeypatch.setattr(settings, "live_route_compute_probe_terrain", False)
    monkeypatch.setattr(main_module, "load_scenario_profiles", _scenario_profiles)
    monkeypatch.setattr(main_module, "load_live_scenario_context", lambda **_: {"ok": True})
    monkeypatch.setattr(main_module, "load_departure_profile", lambda: {"ok": True})
    monkeypatch.setattr(main_module, "load_stochastic_regimes", lambda: {"ok": True})
    monkeypatch.setattr(main_module, "load_toll_segments_seed", lambda: ())
    monkeypatch.setattr(main_module, "load_toll_tariffs", lambda: {"ok": True})
    monkeypatch.setattr(main_module, "load_fuel_price_snapshot", lambda **_: {"ok": True})
    monkeypatch.setattr(main_module, "resolve_carbon_price", _resolve_carbon_price)
    monkeypatch.setattr(main_module, "load_uk_bank_holidays", lambda: {"ok": True})

    departure = datetime(2026, 2, 26, 12, 0, tzinfo=UTC)
    summary = asyncio.run(
        main_module._prefetch_expected_live_sources(
            origin=LatLng(lat=53.94633, lon=-1.02722),
            destination=LatLng(lat=51.48539, lon=-0.13184),
            vehicle_class="rigid_hgv",
            departure_time_utc=departure,
            weather_bucket="clear",
            cost_toggles=CostToggles(carbon_price_per_kg=0.123),
        )
    )

    assert summary.get("source_failed") == 0
    assert observed.get("request_price_per_kg") == 0.123
    assert observed.get("departure_time_utc") == departure
    assert "as_of_utc" not in observed
    assert "price_override_per_kg" not in observed


def test_prefetch_expected_live_sources_parallel_calls_prevent_budget_starvation(monkeypatch) -> None:
    call_log: list[str] = []

    def _scenario_profiles() -> Any:
        call_log.append("scenario_coefficients")
        return SimpleNamespace(transform_params={"mode_effect_scale": {"road": 1.0}})

    def _slow_live_scenario_context(**_: Any) -> dict[str, Any]:
        call_log.append("scenario_live_context")
        time.sleep(1.20)
        return {"ok": True}

    def _mark(label: str):
        def _inner(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            call_log.append(label)
            return {"ok": True}

        return _inner

    monkeypatch.setattr(settings, "strict_live_data_required", False)
    monkeypatch.setattr(settings, "live_route_compute_require_all_expected", False)
    monkeypatch.setattr(settings, "live_route_compute_probe_terrain", False)
    monkeypatch.setattr(settings, "live_route_compute_prefetch_timeout_ms", 1_000)
    monkeypatch.setattr(main_module, "_should_prefetch_terrain_source", lambda: False)
    monkeypatch.setattr(main_module, "load_scenario_profiles", _scenario_profiles)
    monkeypatch.setattr(main_module, "load_live_scenario_context", _slow_live_scenario_context)
    monkeypatch.setattr(main_module, "load_departure_profile", _mark("departure_profiles"))
    monkeypatch.setattr(main_module, "load_stochastic_regimes", _mark("stochastic_regimes"))
    monkeypatch.setattr(main_module, "load_toll_segments_seed", _mark("toll_topology"))
    monkeypatch.setattr(main_module, "load_toll_tariffs", _mark("toll_tariffs"))
    monkeypatch.setattr(main_module, "load_fuel_price_snapshot", _mark("fuel_prices"))
    monkeypatch.setattr(main_module, "resolve_carbon_price", _mark("carbon_schedule"))
    monkeypatch.setattr(main_module, "load_uk_bank_holidays", _mark("bank_holidays"))

    summary = asyncio.run(
        main_module._prefetch_expected_live_sources(
            origin=LatLng(lat=53.94633, lon=-1.02722),
            destination=LatLng(lat=51.48539, lon=-0.13184),
            vehicle_class="rigid_hgv",
            departure_time_utc=None,
            weather_bucket="clear",
            cost_toggles=CostToggles(),
        )
    )

    rows = {
        str(row.get("source_key")): row
        for row in summary.get("rows", [])
        if isinstance(row, dict) and str(row.get("source_key"))
    }
    assert str(rows["scenario_live_context"].get("reason_code")) == "route_compute_timeout"
    assert bool(rows["departure_profiles"].get("ok")) is True
    assert bool(rows["stochastic_regimes"].get("ok")) is True
    assert "departure_profiles" in call_log
    assert "stochastic_regimes" in call_log


def test_prefetch_expected_live_sources_surfaces_missing_transform_params(monkeypatch) -> None:
    def _scenario_profiles() -> Any:
        return SimpleNamespace(transform_params=None)

    monkeypatch.setattr(settings, "strict_live_data_required", True)
    monkeypatch.setattr(settings, "live_route_compute_require_all_expected", True)
    monkeypatch.setattr(settings, "live_route_compute_probe_terrain", False)
    monkeypatch.setattr(settings, "live_route_compute_force_uncached", True)
    monkeypatch.setattr(settings, "live_route_compute_force_uncached", True)
    monkeypatch.setattr(main_module, "_should_prefetch_terrain_source", lambda: False)
    monkeypatch.setattr(main_module, "load_scenario_profiles", _scenario_profiles)
    monkeypatch.setattr(main_module, "load_live_scenario_context", lambda **_: {"ok": True})
    monkeypatch.setattr(main_module, "load_departure_profile", lambda: {"ok": True})
    monkeypatch.setattr(main_module, "load_stochastic_regimes", lambda: {"ok": True})
    monkeypatch.setattr(main_module, "load_toll_segments_seed", lambda: ())
    monkeypatch.setattr(main_module, "load_toll_tariffs", lambda: {"ok": True})
    monkeypatch.setattr(main_module, "load_fuel_price_snapshot", lambda **_: {"ok": True})
    monkeypatch.setattr(main_module, "resolve_carbon_price", lambda **_: {"ok": True})
    monkeypatch.setattr(main_module, "load_uk_bank_holidays", lambda: {"ok": True})

    with pytest.raises(ModelDataError) as excinfo:
        asyncio.run(
            main_module._prefetch_expected_live_sources(
                origin=LatLng(lat=53.94633, lon=-1.02722),
                destination=LatLng(lat=51.48539, lon=-0.13184),
                vehicle_class="rigid_hgv",
                departure_time_utc=None,
                weather_bucket="clear",
                cost_toggles=CostToggles(),
            )
        )

    assert excinfo.value.reason_code == "live_source_refresh_failed"
    detail = excinfo.value.details if isinstance(excinfo.value.details, dict) else {}
    failed_details = detail.get("failed_source_details", [])
    assert isinstance(failed_details, list)
    assert any(
        isinstance(row, dict)
        and str(row.get("source_key")) == "scenario_live_context"
        and str(row.get("reason_code")) == "scenario_profile_invalid"
        for row in failed_details
    )


def test_prefetch_expected_live_sources_fails_when_expected_family_not_observed(monkeypatch) -> None:
    def _scenario_profiles() -> Any:
        return SimpleNamespace(transform_params={"policy_adjustment": {"x": 1}})

    monkeypatch.setattr(settings, "strict_live_data_required", True)
    monkeypatch.setattr(settings, "live_route_compute_require_all_expected", True)
    monkeypatch.setattr(settings, "live_route_compute_probe_terrain", False)
    monkeypatch.setattr(settings, "live_route_compute_force_uncached", True)
    monkeypatch.setattr(main_module, "_should_prefetch_terrain_source", lambda: False)
    monkeypatch.setattr(main_module, "load_scenario_profiles", _scenario_profiles)
    monkeypatch.setattr(main_module, "load_live_scenario_context", lambda **_: {"ok": True})
    monkeypatch.setattr(main_module, "load_departure_profile", lambda: {"ok": True})
    monkeypatch.setattr(main_module, "load_stochastic_regimes", lambda: {"ok": True})
    monkeypatch.setattr(main_module, "load_toll_segments_seed", lambda: ())
    monkeypatch.setattr(main_module, "load_toll_tariffs", lambda: {"ok": True})
    monkeypatch.setattr(main_module, "load_fuel_price_snapshot", lambda **_: {"ok": True})
    monkeypatch.setattr(main_module, "resolve_carbon_price", lambda **_: {"ok": True})
    monkeypatch.setattr(main_module, "load_uk_bank_holidays", lambda: {"ok": True})
    monkeypatch.setattr(main_module, "current_live_trace_request_id", lambda: "prefetch-rid")
    monkeypatch.setattr(
        main_module,
        "get_live_call_trace",
        lambda _request_id: {
            "expected_rollup": [
                {"source_key": "fuel_prices", "required": True, "observed_calls": 0},
                {"source_key": "carbon_schedule", "required": True, "observed_calls": 1},
            ]
        },
    )

    with pytest.raises(ModelDataError) as excinfo:
        asyncio.run(
            main_module._prefetch_expected_live_sources(
                origin=LatLng(lat=53.94633, lon=-1.02722),
                destination=LatLng(lat=51.48539, lon=-0.13184),
                vehicle_class="rigid_hgv",
                departure_time_utc=None,
                weather_bucket="clear",
                cost_toggles=CostToggles(),
            )
        )

    assert excinfo.value.reason_code == "live_source_refresh_failed"
    detail = excinfo.value.details if isinstance(excinfo.value.details, dict) else {}
    assert "fuel_prices" in list(detail.get("missing_expected_sources", []))


def test_prefetch_expected_live_sources_uses_uncached_wrapped_loaders_for_strict_gate(monkeypatch) -> None:
    call_log: list[str] = []

    def _wrap_uncached(label: str, result: Any):
        def _inner(*_args: Any, **_kwargs: Any) -> Any:
            call_log.append(f"inner:{label}")
            return result

        def _outer(*_args: Any, **_kwargs: Any) -> Any:
            call_log.append(f"outer:{label}")
            return result

        _outer.__wrapped__ = _inner  # type: ignore[attr-defined]
        return _outer

    monkeypatch.setattr(settings, "strict_live_data_required", True)
    monkeypatch.setattr(settings, "live_route_compute_require_all_expected", True)
    monkeypatch.setattr(settings, "live_route_compute_probe_terrain", False)
    monkeypatch.setattr(settings, "live_route_compute_force_uncached", True)
    monkeypatch.setattr(main_module, "_should_prefetch_terrain_source", lambda: False)
    monkeypatch.setattr(
        main_module,
        "load_scenario_profiles",
        lambda: SimpleNamespace(transform_params={"policy_adjustment": {"x": 1}}),
    )
    monkeypatch.setattr(main_module, "load_live_scenario_context", lambda **_: {"ok": True})
    monkeypatch.setattr(main_module, "load_departure_profile", _wrap_uncached("departure_profiles", {"ok": True}))
    monkeypatch.setattr(main_module, "load_stochastic_regimes", _wrap_uncached("stochastic_regimes", {"ok": True}))
    monkeypatch.setattr(main_module, "load_toll_segments_seed", _wrap_uncached("toll_topology", ()))
    monkeypatch.setattr(main_module, "load_toll_tariffs", _wrap_uncached("toll_tariffs", {"ok": True}))
    monkeypatch.setattr(main_module, "load_fuel_price_snapshot", _wrap_uncached("fuel_prices", {"ok": True}))
    monkeypatch.setattr(main_module, "resolve_carbon_price", lambda **_: {"ok": True})
    monkeypatch.setattr(main_module, "load_uk_bank_holidays", _wrap_uncached("bank_holidays", {"ok": True}))
    monkeypatch.setattr(main_module, "current_live_trace_request_id", lambda: "prefetch-rid")
    monkeypatch.setattr(
        main_module,
        "get_live_call_trace",
        lambda _request_id: {
            "expected_rollup": [
                {"source_key": "scenario_coefficients", "required": True, "observed_calls": 1},
                {"source_key": "scenario_webtris_sites", "required": True, "observed_calls": 1},
                {"source_key": "scenario_webtris_daily", "required": True, "observed_calls": 1},
                {"source_key": "scenario_traffic_england", "required": True, "observed_calls": 1},
                {"source_key": "scenario_dft_counts", "required": True, "observed_calls": 1},
                {"source_key": "scenario_open_meteo", "required": True, "observed_calls": 1},
                {"source_key": "departure_profiles", "required": True, "observed_calls": 1},
                {"source_key": "stochastic_regimes", "required": True, "observed_calls": 1},
                {"source_key": "toll_topology", "required": True, "observed_calls": 1},
                {"source_key": "toll_tariffs", "required": True, "observed_calls": 1},
                {"source_key": "fuel_prices", "required": True, "observed_calls": 1},
                {"source_key": "carbon_schedule", "required": True, "observed_calls": 1},
                {"source_key": "bank_holidays", "required": True, "observed_calls": 1},
            ]
        },
    )

    summary = asyncio.run(
        main_module._prefetch_expected_live_sources(
            origin=LatLng(lat=53.94633, lon=-1.02722),
            destination=LatLng(lat=51.48539, lon=-0.13184),
            vehicle_class="rigid_hgv",
            departure_time_utc=None,
            weather_bucket="clear",
            cost_toggles=CostToggles(),
        )
    )

    assert summary.get("source_failed") == 0
    for label in (
        "departure_profiles",
        "stochastic_regimes",
        "toll_topology",
        "toll_tariffs",
        "fuel_prices",
        "bank_holidays",
    ):
        assert f"inner:{label}" in call_log
        assert f"outer:{label}" not in call_log


def test_prefetch_expected_live_sources_scenario_calls_use_uncached_wrappers(monkeypatch) -> None:
    call_log: list[str] = []

    def _inner_profiles() -> Any:
        call_log.append("inner:scenario_profiles")
        return SimpleNamespace(transform_params={"policy_adjustment": {"x": 1}})

    def _outer_profiles() -> Any:
        call_log.append("outer:scenario_profiles")
        return SimpleNamespace(transform_params={"policy_adjustment": {"x": 1}})

    _outer_profiles.__wrapped__ = _inner_profiles  # type: ignore[attr-defined]

    def _inner_context(**_kwargs: Any) -> Any:
        call_log.append("inner:scenario_live_context")
        return {"ok": True}

    def _outer_context(**_kwargs: Any) -> Any:
        call_log.append("outer:scenario_live_context")
        return {"ok": True}

    _outer_context.__wrapped__ = _inner_context  # type: ignore[attr-defined]

    monkeypatch.setattr(settings, "strict_live_data_required", True)
    monkeypatch.setattr(settings, "live_route_compute_require_all_expected", True)
    monkeypatch.setattr(settings, "live_route_compute_probe_terrain", False)
    monkeypatch.setattr(settings, "live_route_compute_force_uncached", True)
    monkeypatch.setattr(main_module, "_should_prefetch_terrain_source", lambda: False)
    monkeypatch.setattr(main_module, "load_scenario_profiles", _outer_profiles)
    monkeypatch.setattr(main_module, "load_live_scenario_context", _outer_context)
    monkeypatch.setattr(main_module, "load_departure_profile", lambda: {"ok": True})
    monkeypatch.setattr(main_module, "load_stochastic_regimes", lambda: {"ok": True})
    monkeypatch.setattr(main_module, "load_toll_segments_seed", lambda: ())
    monkeypatch.setattr(main_module, "load_toll_tariffs", lambda: {"ok": True})
    monkeypatch.setattr(main_module, "load_fuel_price_snapshot", lambda **_: {"ok": True})
    monkeypatch.setattr(main_module, "resolve_carbon_price", lambda **_: {"ok": True})
    monkeypatch.setattr(main_module, "load_uk_bank_holidays", lambda: {"ok": True})
    monkeypatch.setattr(main_module, "current_live_trace_request_id", lambda: "prefetch-rid")
    monkeypatch.setattr(
        main_module,
        "get_live_call_trace",
        lambda _request_id: {
            "expected_rollup": [
                {"source_key": "scenario_coefficients", "required": True, "observed_calls": 1},
                {"source_key": "scenario_webtris_sites", "required": True, "observed_calls": 1},
                {"source_key": "scenario_webtris_daily", "required": True, "observed_calls": 1},
                {"source_key": "scenario_traffic_england", "required": True, "observed_calls": 1},
                {"source_key": "scenario_dft_counts", "required": True, "observed_calls": 1},
                {"source_key": "scenario_open_meteo", "required": True, "observed_calls": 1},
                {"source_key": "departure_profiles", "required": True, "observed_calls": 1},
                {"source_key": "stochastic_regimes", "required": True, "observed_calls": 1},
                {"source_key": "toll_topology", "required": True, "observed_calls": 1},
                {"source_key": "toll_tariffs", "required": True, "observed_calls": 1},
                {"source_key": "fuel_prices", "required": True, "observed_calls": 1},
                {"source_key": "carbon_schedule", "required": True, "observed_calls": 1},
                {"source_key": "bank_holidays", "required": True, "observed_calls": 1},
            ]
        },
    )

    summary = asyncio.run(
        main_module._prefetch_expected_live_sources(
            origin=LatLng(lat=53.94633, lon=-1.02722),
            destination=LatLng(lat=51.48539, lon=-0.13184),
            vehicle_class="rigid_hgv",
            departure_time_utc=None,
            weather_bucket="clear",
            cost_toggles=CostToggles(),
        )
    )

    assert summary.get("source_failed") == 0
    assert "inner:scenario_profiles" in call_log
    assert "inner:scenario_live_context" in call_log
    assert "outer:scenario_profiles" not in call_log
    assert "outer:scenario_live_context" not in call_log


def test_route_expected_live_calls_terrain_parity(monkeypatch) -> None:
    monkeypatch.setattr(settings, "live_terrain_dem_url_template", "https://example.test/{z}/{x}/{y}.tif")
    monkeypatch.setattr(settings, "strict_live_data_required", False)
    monkeypatch.setattr(settings, "live_route_compute_require_all_expected", False)
    monkeypatch.setattr(settings, "live_route_compute_probe_terrain", False)
    without_terrain = main_module._route_compute_expected_live_calls()
    assert all(str(row.get("source_key")) != "terrain_live_tile" for row in without_terrain)

    monkeypatch.setattr(settings, "strict_live_data_required", True)
    monkeypatch.setattr(settings, "live_route_compute_require_all_expected", True)
    monkeypatch.setattr(settings, "live_route_compute_probe_terrain", False)
    with_terrain = main_module._route_compute_expected_live_calls()
    assert any(str(row.get("source_key")) == "terrain_live_tile" for row in with_terrain)


def test_prefetch_expected_live_sources_terrain_probe_succeeds_with_partial_coverage(monkeypatch) -> None:
    observed_probe_points: list[tuple[float, float]] = []

    def _scenario_profiles() -> Any:
        return SimpleNamespace(transform_params={"policy_adjustment": {"x": 1}})

    def _sample_elevation(lat: float, lon: float) -> tuple[float, bool, str]:
        observed_probe_points.append((float(lat), float(lon)))
        if len(observed_probe_points) == 2:
            return 123.4, True, "live_dem_z8"
        return float("nan"), False, "live_dem_z8"

    monkeypatch.setattr(settings, "strict_live_data_required", True)
    monkeypatch.setattr(settings, "live_route_compute_require_all_expected", True)
    monkeypatch.setattr(settings, "live_route_compute_probe_terrain", True)
    monkeypatch.setattr(settings, "live_terrain_prefetch_probe_fractions", "0.5,0.35,0.65")
    monkeypatch.setattr(settings, "live_terrain_prefetch_min_covered_points", 1)
    monkeypatch.setattr(main_module, "load_scenario_profiles", _scenario_profiles)
    monkeypatch.setattr(main_module, "load_live_scenario_context", lambda **_: {"ok": True})
    monkeypatch.setattr(main_module, "load_departure_profile", lambda: {"ok": True})
    monkeypatch.setattr(main_module, "load_stochastic_regimes", lambda: {"ok": True})
    monkeypatch.setattr(main_module, "load_toll_segments_seed", lambda: ())
    monkeypatch.setattr(main_module, "load_toll_tariffs", lambda: {"ok": True})
    monkeypatch.setattr(main_module, "load_fuel_price_snapshot", lambda **_: {"ok": True})
    monkeypatch.setattr(main_module, "resolve_carbon_price", lambda **_: {"ok": True})
    monkeypatch.setattr(main_module, "load_uk_bank_holidays", lambda: {"ok": True})
    monkeypatch.setattr(main_module, "terrain_begin_route_run", lambda: None)
    monkeypatch.setattr(main_module, "sample_elevation_m", _sample_elevation)

    summary = asyncio.run(
        main_module._prefetch_expected_live_sources(
            origin=LatLng(lat=54.94056, lon=-1.77979),
            destination=LatLng(lat=51.47633, lon=-0.17578),
            vehicle_class="rigid_hgv",
            departure_time_utc=None,
            weather_bucket="clear",
            cost_toggles=CostToggles(),
        )
    )

    assert summary.get("source_failed") == 0
    rows = summary.get("rows", [])
    terrain_row = next(
        (row for row in rows if isinstance(row, dict) and str(row.get("source_key")) == "terrain_live_tile"),
        None,
    )
    assert isinstance(terrain_row, dict)
    assert bool(terrain_row.get("ok")) is True
    assert len(observed_probe_points) == 3


def test_prefetch_expected_live_sources_terrain_probe_failure_contains_probe_details(monkeypatch) -> None:
    def _scenario_profiles() -> Any:
        return SimpleNamespace(transform_params={"policy_adjustment": {"x": 1}})

    monkeypatch.setattr(settings, "strict_live_data_required", True)
    monkeypatch.setattr(settings, "live_route_compute_require_all_expected", True)
    monkeypatch.setattr(settings, "live_route_compute_probe_terrain", True)
    monkeypatch.setattr(settings, "live_terrain_prefetch_probe_fractions", "0.5,0.35,0.65")
    monkeypatch.setattr(settings, "live_terrain_prefetch_min_covered_points", 1)
    monkeypatch.setattr(main_module, "load_scenario_profiles", _scenario_profiles)
    monkeypatch.setattr(main_module, "load_live_scenario_context", lambda **_: {"ok": True})
    monkeypatch.setattr(main_module, "load_departure_profile", lambda: {"ok": True})
    monkeypatch.setattr(main_module, "load_stochastic_regimes", lambda: {"ok": True})
    monkeypatch.setattr(main_module, "load_toll_segments_seed", lambda: ())
    monkeypatch.setattr(main_module, "load_toll_tariffs", lambda: {"ok": True})
    monkeypatch.setattr(main_module, "load_fuel_price_snapshot", lambda **_: {"ok": True})
    monkeypatch.setattr(main_module, "resolve_carbon_price", lambda **_: {"ok": True})
    monkeypatch.setattr(main_module, "load_uk_bank_holidays", lambda: {"ok": True})
    monkeypatch.setattr(main_module, "terrain_begin_route_run", lambda: None)
    monkeypatch.setattr(main_module, "sample_elevation_m", lambda _lat, _lon: (float("nan"), False, "live_dem_z8"))

    with pytest.raises(ModelDataError) as excinfo:
        asyncio.run(
            main_module._prefetch_expected_live_sources(
                origin=LatLng(lat=54.94056, lon=-1.77979),
                destination=LatLng(lat=51.47633, lon=-0.17578),
                vehicle_class="rigid_hgv",
                departure_time_utc=None,
                weather_bucket="clear",
                cost_toggles=CostToggles(),
            )
        )

    assert excinfo.value.reason_code == "live_source_refresh_failed"
    detail = excinfo.value.details if isinstance(excinfo.value.details, dict) else {}
    failed_details = detail.get("failed_source_details", [])
    assert isinstance(failed_details, list)
    terrain_row = next(
        (
            row
            for row in failed_details
            if isinstance(row, dict) and str(row.get("source_key")) == "terrain_live_tile"
        ),
        None,
    )
    assert isinstance(terrain_row, dict)
    detail_data = terrain_row.get("detail_data")
    assert isinstance(detail_data, dict)
    assert int(detail_data.get("probe_count", 0)) == 3
    assert int(detail_data.get("covered_count", 0)) == 0


def _live_diag_payload(url: str, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "_live_diagnostics": {
            "source_url": str(url),
            "fetch_error": None,
            "cache_hit": False,
            "stale_cache_used": False,
            "status_code": 200,
            "as_of_utc": "2026-02-26T12:00:00Z",
            "retry_attempts": 1,
            "retry_count": 0,
            "retry_total_backoff_ms": 0,
            "retry_last_error": None,
            "retry_last_status_code": None,
            "retry_deadline_exceeded": False,
        }
    }
    if isinstance(extra, dict):
        payload.update(extra)
    return payload


def _base_route_context() -> dict[str, Any]:
    return {
        "corridor_bucket": "gcpvj",
        "road_mix_bucket": "mixed",
        "vehicle_class": "rigid_hgv",
        "day_kind": "weekday",
        "hour_slot_local": 12,
        "weather_bucket": "clear",
        "centroid_lat": 52.38316,
        "centroid_lon": -2.47021,
        "road_hint": "",
    }


def test_live_scenario_context_sparse_observations_apply_partial_waiver(monkeypatch) -> None:
    monkeypatch.setattr(settings, "live_runtime_data_enabled", True)
    monkeypatch.setattr(live_data_sources, "_utc_now", lambda: datetime(2026, 2, 26, 12, 0, tzinfo=UTC))
    monkeypatch.setattr(live_data_sources, "_cache_get", lambda _key: None)
    monkeypatch.setattr(live_data_sources, "_cache_put", lambda _key, _payload: None)
    monkeypatch.setattr(live_data_sources, "_scenario_url_allowed", lambda _url: True)
    monkeypatch.setattr(
        live_data_sources,
        "_nearest_webtris_sites",
        lambda *_args, **_kwargs: [{"site_id": 8199, "dist_deg2": 1e-8}],
    )

    def _fetch_json_with_ttl(*, key: str, url: str, ttl_s: int) -> tuple[Any, str | None]:
        _ = ttl_s
        if key == "scenario:webtris:sites":
            return _live_diag_payload(url, extra={"sites": [{"site_id": 8199}]}), None
        if key.startswith("scenario:webtris:daily:"):
            return _live_diag_payload(url, extra={"rows": []}), None
        if key.startswith("scenario:trafficengland:"):
            return _live_diag_payload(url, extra={"events": []}), None
        if key.startswith("scenario:meteo:"):
            return _live_diag_payload(
                url,
                extra={
                    "current": {
                        "temperature_2m": 10.0,
                        "wind_speed_10m": 3.0,
                        "precipitation": 0.0,
                        "weather_code": 0,
                    }
                },
            ), None
        raise AssertionError(f"unexpected source key {key}")

    monkeypatch.setattr(live_data_sources, "_fetch_json_with_ttl", _fetch_json_with_ttl)
    monkeypatch.setattr(
        live_data_sources,
        "_fetch_dft_rows_paginated",
        lambda **_kwargs: (
            [{"all_motor_vehicles": 120.0, "latitude": 10.0, "longitude": 10.0}],
            {"page_diagnostics": []},
            None,
        ),
    )

    payload = live_data_sources.live_scenario_context(
        _base_route_context(),
        allow_partial_sources=True,
        min_source_count=3,
        skip_dft_in_partial_mode=False,
    )
    assert isinstance(payload, dict)
    assert "_live_error" not in payload
    coverage_gate = payload.get("coverage_gate")
    assert isinstance(coverage_gate, dict)
    assert bool(coverage_gate.get("waiver_applied")) is True
    assert int(coverage_gate.get("required_source_count_configured", 0)) == 3
    assert int(coverage_gate.get("required_source_count_effective", 0)) == 2
    assert int(coverage_gate.get("source_ok_count", 0)) == 2
    assert str(coverage_gate.get("resolved_road_hint_value", "")) == "M6"
    assert str(coverage_gate.get("resolved_road_hint_source", "")) == "centroid_region"


def test_live_scenario_context_sparse_observations_fail_when_webtris_unreachable(monkeypatch) -> None:
    monkeypatch.setattr(settings, "live_runtime_data_enabled", True)
    monkeypatch.setattr(live_data_sources, "_utc_now", lambda: datetime(2026, 2, 26, 12, 0, tzinfo=UTC))
    monkeypatch.setattr(live_data_sources, "_cache_get", lambda _key: None)
    monkeypatch.setattr(live_data_sources, "_cache_put", lambda _key, _payload: None)
    monkeypatch.setattr(live_data_sources, "_scenario_url_allowed", lambda _url: True)
    monkeypatch.setattr(
        live_data_sources,
        "_nearest_webtris_sites",
        lambda *_args, **_kwargs: [{"site_id": 8199, "dist_deg2": 1e-8}],
    )

    def _fetch_json_with_ttl(*, key: str, url: str, ttl_s: int) -> tuple[Any, str | None]:
        _ = ttl_s
        if key == "scenario:webtris:sites":
            return _live_diag_payload(url, extra={"sites": [{"site_id": 8199}]}), "webtris_sites_unavailable"
        if key.startswith("scenario:webtris:daily:"):
            return _live_diag_payload(url, extra={"rows": []}), None
        if key.startswith("scenario:trafficengland:"):
            return _live_diag_payload(url, extra={"events": []}), None
        if key.startswith("scenario:meteo:"):
            return _live_diag_payload(
                url,
                extra={
                    "current": {
                        "temperature_2m": 10.0,
                        "wind_speed_10m": 3.0,
                        "precipitation": 0.0,
                        "weather_code": 0,
                    }
                },
            ), None
        raise AssertionError(f"unexpected source key {key}")

    monkeypatch.setattr(live_data_sources, "_fetch_json_with_ttl", _fetch_json_with_ttl)
    monkeypatch.setattr(
        live_data_sources,
        "_fetch_dft_rows_paginated",
        lambda **_kwargs: (
            [{"all_motor_vehicles": 120.0, "latitude": 10.0, "longitude": 10.0}],
            {"page_diagnostics": []},
            None,
        ),
    )

    payload = live_data_sources.live_scenario_context(
        _base_route_context(),
        allow_partial_sources=True,
        min_source_count=3,
        skip_dft_in_partial_mode=False,
    )
    assert isinstance(payload, dict)
    assert "_live_error" in payload
    err = payload["_live_error"]
    assert isinstance(err, dict)
    diagnostics = err.get("diagnostics", {})
    assert isinstance(diagnostics, dict)
    coverage_gate = diagnostics.get("coverage_gate", {})
    assert isinstance(coverage_gate, dict)
    assert bool(coverage_gate.get("waiver_applied")) is False
    assert int(coverage_gate.get("required_source_count_effective", 0)) == 3
    assert int(coverage_gate.get("source_ok_count", 0)) == 2


def test_live_scenario_context_full_strict_mode_uses_reachability_gate(monkeypatch) -> None:
    monkeypatch.setattr(settings, "live_runtime_data_enabled", True)
    monkeypatch.setattr(settings, "live_scenario_dft_min_station_count", 1)
    monkeypatch.setattr(live_data_sources, "_utc_now", lambda: datetime(2026, 2, 26, 12, 0, tzinfo=UTC))
    monkeypatch.setattr(live_data_sources, "_cache_get", lambda _key: None)
    monkeypatch.setattr(live_data_sources, "_cache_put", lambda _key, _payload: None)
    monkeypatch.setattr(live_data_sources, "_scenario_url_allowed", lambda _url: True)
    monkeypatch.setattr(
        live_data_sources,
        "_nearest_webtris_sites",
        lambda *_args, **_kwargs: [{"site_id": 8199, "dist_deg2": 1e-8}],
    )

    def _fetch_json_with_ttl(*, key: str, url: str, ttl_s: int) -> tuple[Any, str | None]:
        _ = ttl_s
        if key == "scenario:webtris:sites":
            return _live_diag_payload(url, extra={"sites": [{"site_id": 8199}]}), None
        if key.startswith("scenario:webtris:daily:"):
            return _live_diag_payload(url, extra={"rows": []}), None
        if key.startswith("scenario:trafficengland:"):
            return _live_diag_payload(url, extra={"events": []}), None
        if key.startswith("scenario:meteo:"):
            return _live_diag_payload(
                url,
                extra={
                    "current": {
                        "temperature_2m": 10.0,
                        "wind_speed_10m": 3.0,
                        "precipitation": 0.0,
                        "weather_code": 0,
                    }
                },
            ), None
        raise AssertionError(f"unexpected source key {key}")

    monkeypatch.setattr(live_data_sources, "_fetch_json_with_ttl", _fetch_json_with_ttl)
    monkeypatch.setattr(
        live_data_sources,
        "_fetch_dft_rows_paginated",
        lambda **_kwargs: (
            [{"all_motor_vehicles": 120.0, "latitude": 52.39, "longitude": -2.46}],
            {"page_diagnostics": []},
            None,
        ),
    )

    payload = live_data_sources.live_scenario_context(
        _base_route_context(),
        allow_partial_sources=False,
        min_source_count=None,
        skip_dft_in_partial_mode=False,
    )
    assert isinstance(payload, dict)
    assert "_live_error" not in payload
    coverage_gate = payload.get("coverage_gate", {})
    assert isinstance(coverage_gate, dict)
    assert int(coverage_gate.get("required_source_count_effective", 0)) == 4
    assert int(coverage_gate.get("source_ok_count", 0)) == 4
    assert str(coverage_gate.get("source_gate_basis", "")).strip().lower() == "reachability"


@pytest.mark.parametrize(
    ("centroid_lat", "centroid_lon", "expected_road_hint"),
    [
        (52.4862, -1.8904, "M6"),
        (51.5074, -0.1278, "M25"),
        (51.4816, -3.1791, "M4"),
        (55.9533, -3.1883, "M8"),
    ],
)
def test_resolve_scenario_road_hint_uses_centroid_mapping(
    centroid_lat: float,
    centroid_lon: float,
    expected_road_hint: str,
) -> None:
    road_hint, source = live_data_sources._resolve_scenario_road_hint(
        {"corridor_bucket": "gcpvj", "road_hint": ""},
        centroid_lat=centroid_lat,
        centroid_lon=centroid_lon,
    )
    assert road_hint == expected_road_hint
    assert source == "centroid_region"


def test_prefetch_failed_details_text_includes_scenario_gate_summary() -> None:
    text = main_module._prefetch_failed_details_text(
        {
            "failed_source_details": [
                {
                    "source_key": "scenario_live_context",
                    "reason_code": "scenario_profile_unavailable",
                    "detail": "Scenario live context incomplete",
                    "detail_data": {
                        "coverage_gate": {
                            "source_ok_count": 2,
                            "required_source_count_configured": 3,
                            "required_source_count_effective": 2,
                            "waiver_applied": True,
                            "waiver_reason": "sparse_local_observations_control_plane_healthy",
                        },
                        "resolved_road_hint_value": "M6",
                        "webtris_used_site_count": 0,
                        "dft_selected_station_count": 0,
                    },
                }
            ]
        }
    )
    assert "source_ok=2/4" in text
    assert "required_configured=3/4" in text
    assert "required_effective=2/4" in text
    assert "road_hint=M6" in text


def test_prefetch_failure_text_stale_scenario_omits_coverage_gate_defaults() -> None:
    error = ModelDataError(
        reason_code="scenario_profile_unavailable",
        message="Live scenario coefficient payload is stale for strict runtime policy.",
        details={
            "as_of_utc": "2026-02-25T23:26:54+00:00",
            "max_age_minutes": 4320,
        },
    )

    reason, detail = main_module._prefetch_failure_text(error)
    assert reason == "scenario_profile_unavailable"
    assert "source_ok=" not in detail
    assert "required_effective=" not in detail
    assert "as_of_utc=2026-02-25T23:26:54+00:00" in detail
    assert "age_minutes=" in detail
    assert "max_age_minutes=4320" in detail


def test_prefetch_failed_details_text_stale_scenario_omits_coverage_gate_defaults() -> None:
    text = main_module._prefetch_failed_details_text(
        {
            "failed_source_details": [
                {
                    "source_key": "scenario_coefficients",
                    "reason_code": "scenario_profile_unavailable",
                    "detail": "Live scenario coefficient payload is stale for strict runtime policy.",
                    "detail_data": {
                        "as_of_utc": "2026-02-25T23:26:54+00:00",
                        "max_age_minutes": 4320,
                    },
                }
            ]
        }
    )
    assert "source_ok=" not in text
    assert "required_effective=" not in text
    assert "as_of_utc=2026-02-25T23:26:54+00:00" in text
    assert "age_minutes=" in text
    assert "max_age_minutes=4320" in text
