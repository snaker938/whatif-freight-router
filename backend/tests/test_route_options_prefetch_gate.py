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
    EmissionsContext,
    IncidentSimulatorConfig,
    LatLng,
    StochasticConfig,
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


def test_live_scenario_context_full_strict_mode_remains_fail_closed(monkeypatch) -> None:
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
    assert "_live_error" in payload
    err = payload["_live_error"]
    assert isinstance(err, dict)
    diagnostics = err.get("diagnostics", {})
    assert isinstance(diagnostics, dict)
    coverage_gate = diagnostics.get("coverage_gate", {})
    assert isinstance(coverage_gate, dict)
    assert int(coverage_gate.get("required_source_count_effective", 0)) == 4


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
