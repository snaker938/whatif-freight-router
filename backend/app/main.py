from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import json
import math
import os
import time
import uuid
import httpx
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from .calibration_loader import (
    load_departure_profile,
    load_fuel_price_snapshot,
    load_fuel_consumption_calibration,
    load_live_scenario_context,
    load_risk_normalization_reference,
    load_scenario_profiles,
    load_stochastic_regimes,
    load_stochastic_residual_prior,
    load_toll_segments_seed,
    load_toll_tariffs,
    load_uk_bank_holidays,
    refresh_live_runtime_route_caches,
)
from .carbon_model import apply_scope_emissions_adjustment, resolve_carbon_price
from .departure_profile import time_of_day_multiplier_uk
from .decision_critical import (
    DCCSConfig,
    DCCSCandidateRecord,
    record_refine_outcome,
    select_candidates,
    stable_candidate_id,
)
from .evidence_certification import (
    active_evidence_families,
    compute_certificate,
    compute_fragility_maps,
    sample_world_manifest,
)
from .experiment_store import (
    create_experiment,
    delete_experiment,
    get_experiment,
    list_experiments,
    update_experiment,
)
from .fuel_energy_model import segment_energy_and_emissions
from .incident_simulator import simulate_incident_events
from .logging_utils import log_event
from .live_call_trace import (
    current_trace_request_id as current_live_trace_request_id,
    finish_trace as finish_live_call_trace,
    get_trace as get_live_call_trace,
    mark_expected_calls_blocked as mark_live_expected_calls_blocked,
    reset_trace as reset_live_call_trace,
    start_trace as start_live_call_trace,
)
from .metrics_store import metrics_snapshot, record_request
from .model_data_errors import ModelDataError, normalize_reason_code
from .models import (
    BatchCSVImportRequest,
    BatchParetoRequest,
    BatchParetoResponse,
    BatchParetoResult,
    CostToggles,
    CustomVehicleListResponse,
    DepartureOptimizeCandidate,
    DepartureOptimizeRequest,
    DepartureOptimizeResponse,
    DutyChainLegResult,
    DutyChainRequest,
    DutyChainResponse,
    DutyChainStop,
    EvidenceProvenance,
    EvidenceSourceRecord,
    EmissionsContext,
    EpsilonConstraints,
    ExperimentBundle,
    ExperimentBundleInput,
    ExperimentCompareRequest,
    ExperimentListResponse,
    GeoJSONLineString,
    IncidentSimulatorConfig,
    LatLng,
    ODPair,
    OptimizationMode,
    OracleFeedCheckInput,
    OracleFeedCheckRecord,
    OracleQualityDashboardResponse,
    ParetoMethod,
    ParetoRequest,
    ParetoResponse,
    RouteMetrics,
    RouteOption,
    RouteRequest,
    RouteBaselineResponse,
    RouteCertificationSummary,
    RouteResponse,
    ScenarioCompareDelta,
    ScenarioCompareRequest,
    ScenarioCompareResponse,
    ScenarioCompareResult,
    ScenarioSummary,
    SignatureVerificationRequest,
    SignatureVerificationResponse,
    StochasticConfig,
    TerrainProfile,
    TerrainSummaryPayload,
    VoiStopSummary,
    VehicleDeleteResponse,
    VehicleListResponse,
    VehicleMutationResponse,
    Waypoint,
    WeatherImpactConfig,
)
from .multileg_engine import compose_multileg_route_options
from .objectives_emissions import route_emissions_kg
from .objectives_selection import normalise_weights
from .oracle_quality_store import (
    append_check_record,
    checks_path,
    compute_dashboard_payload,
    load_check_records,
    write_summary_artifacts,
)
from .pareto_methods import filter_by_epsilon, select_pareto_routes
from .provenance_store import provenance_event, provenance_path_for_run, write_provenance
from .rbac import require_role
from .reporting import write_report_pdf
from .risk_model import robust_objective
from .route_cache import clear_route_cache, get_cached_routes, route_cache_stats, set_cached_routes
from .routing_graph import (
    GraphCandidateDiagnostics,
    begin_route_graph_warmup,
    route_graph_candidate_routes,
    route_graph_od_feasibility,
    route_graph_status,
    route_graph_via_paths,
    route_graph_warmup_status,
)
from .routing_osrm import OSRMClient, OSRMError, extract_segment_annotations
from .run_store import (
    ARTIFACT_FILES,
    artifact_dir_for_run,
    artifact_path_for_name,
    artifact_paths_for_run,
    list_artifact_paths_for_run,
    write_csv_artifact,
    write_json_artifact,
    write_jsonl_artifact,
    write_manifest,
    write_run_artifacts,
    write_scenario_manifest,
    write_text_artifact,
)
from .scenario import (
    ScenarioMode,
    ScenarioRouteContext,
    build_scenario_route_context,
    resolve_scenario_profile,
)
from .settings import settings
from .signatures import SIGNATURE_ALGORITHM, verify_payload_signature
from .terrain_dem import (
    TerrainCoverageError,
    estimate_terrain_summary,
    segment_grade_profile,
    terrain_begin_route_run,
    terrain_live_diagnostics,
)
from .terrain_dem_index import sample_elevation_m
from .terrain_physics import params_for_vehicle, segment_duration_multiplier
from .toll_engine import compute_toll_cost
from .uncertainty_model import compute_uncertainty_summary, resolve_stochastic_regime
from .vehicles import (
    VehicleProfile,
    all_vehicles,
    create_custom_vehicle,
    delete_custom_vehicle,
    list_custom_vehicles,
    resolve_vehicle_profile,
    update_custom_vehicle,
)
from .voi_controller import VOIConfig, VOIControllerState, build_action_menu
from .weather_adapter import weather_incident_multiplier, weather_speed_multiplier, weather_summary

try:
    UK_TZ = ZoneInfo("Europe/London")
except ZoneInfoNotFoundError:
    UK_TZ = UTC


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.osrm = OSRMClient(base_url=settings.osrm_base_url, profile=settings.osrm_profile)
    if bool(settings.route_graph_warmup_on_startup) and "PYTEST_CURRENT_TEST" not in os.environ:
        begin_route_graph_warmup()
    yield
    await app.state.osrm.aclose()


app = FastAPI(title="Carbon‑Aware Freight Router (v0)", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def osrm_client(request: Request) -> OSRMClient:
    osrm: OSRMClient | None = getattr(request.app.state, "osrm", None)  # type: ignore[attr-defined]
    if osrm is None:
        raise HTTPException(status_code=503, detail="OSRM client not initialised")
    return osrm


OSRMDep = Annotated[OSRMClient, Depends(osrm_client)]


def require_user_access(request: Request) -> None:
    require_role(request, "user")


def require_admin_access(request: Request) -> None:
    require_role(request, "admin")


UserAccessDep = Annotated[None, Depends(require_user_access)]
AdminAccessDep = Annotated[None, Depends(require_admin_access)]


@app.get("/")
async def root() -> dict[str, str]:
    # Avoid confusion when opening the backend base URL directly.
    return {"message": "Backend is running. Visit /docs for the API UI.", "docs": "/docs"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


async def _route_graph_status_async() -> tuple[bool, str]:
    timeout_ms = max(100, int(settings.route_graph_status_check_timeout_ms))
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(route_graph_status),
            timeout=float(timeout_ms) / 1000.0,
        )
    except asyncio.TimeoutError:
        return False, "status_check_timeout"


async def _route_graph_od_feasibility_async(
    *,
    origin: LatLng,
    destination: LatLng,
) -> dict[str, Any]:
    timeout_ms = max(100, int(settings.route_graph_od_feasibility_timeout_ms))
    started = time.perf_counter()
    request_id = current_live_trace_request_id()
    log_event(
        "route_graph_precheck_start",
        request_id=request_id,
        origin_lat=float(origin.lat),
        origin_lon=float(origin.lon),
        destination_lat=float(destination.lat),
        destination_lon=float(destination.lon),
        timeout_ms=timeout_ms,
    )
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                route_graph_od_feasibility,
                origin_lat=float(origin.lat),
                origin_lon=float(origin.lon),
                destination_lat=float(destination.lat),
                destination_lon=float(destination.lon),
            ),
            timeout=float(timeout_ms) / 1000.0,
        )
    except asyncio.TimeoutError:
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
        log_event(
            "route_graph_precheck_timeout",
            request_id=request_id,
            origin_lat=float(origin.lat),
            origin_lon=float(origin.lon),
            destination_lat=float(destination.lat),
            destination_lon=float(destination.lon),
            timeout_ms=timeout_ms,
            elapsed_ms=elapsed_ms,
            reason_code="routing_graph_precheck_timeout",
        )
        return {
            "ok": False,
            "reason_code": "routing_graph_precheck_timeout",
            "message": (
                "Route graph OD feasibility check timed out before completion. "
                f"(timeout_ms={timeout_ms})"
            ),
            "timeout_ms": timeout_ms,
            "elapsed_ms": elapsed_ms,
            "stage": "collecting_candidates",
            "stage_detail": "route_graph_feasibility_check_timeout",
        }
    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
    ok = bool(result.get("ok"))
    reason_code = (
        "ok"
        if ok
        else normalize_reason_code(
            str(result.get("reason_code", "routing_graph_unavailable")).strip(),
            default="routing_graph_unavailable",
        )
    )
    log_event(
        "route_graph_precheck_complete",
        request_id=request_id,
        origin_lat=float(origin.lat),
        origin_lon=float(origin.lon),
        destination_lat=float(destination.lat),
        destination_lon=float(destination.lon),
        timeout_ms=timeout_ms,
        elapsed_ms=elapsed_ms,
        ok=bool(result.get("ok")),
        reason_code=reason_code,
    )
    enriched = dict(result)
    enriched["reason_code"] = reason_code
    enriched.setdefault("timeout_ms", timeout_ms)
    enriched.setdefault("elapsed_ms", elapsed_ms)
    return enriched


def _route_compute_expected_live_calls() -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def _add(
        *,
        source_key: str,
        component: str,
        url: str,
        description: str,
        phase: str = "prefetch",
        gate: str = "hard_refresh",
    ) -> None:
        text = str(url or "").strip()
        if not text:
            return
        calls.append(
            {
                "source_key": source_key,
                "component": component,
                "url": text,
                "method": "GET",
                "required": True,
                "description": description,
                "phase": str(phase).strip() or "prefetch",
                "gate": str(gate).strip() or "hard_refresh",
            }
        )

    _add(
        source_key="scenario_coefficients",
        component="scenario",
        url=str(settings.live_scenario_coefficient_url or ""),
        description="Scenario coefficient payload",
    )
    _add(
        source_key="scenario_webtris_sites",
        component="scenario",
        url=str(settings.live_scenario_webtris_sites_url or ""),
        description="WebTRIS site index",
    )
    _add(
        source_key="scenario_webtris_daily",
        component="scenario",
        url=str(settings.live_scenario_webtris_daily_url or ""),
        description="WebTRIS daily report data",
    )
    _add(
        source_key="scenario_traffic_england",
        component="scenario",
        url=str(settings.live_scenario_traffic_england_url or ""),
        description="Traffic England incidents feed",
    )
    _add(
        source_key="scenario_dft_counts",
        component="scenario",
        url=str(settings.live_scenario_dft_counts_url or ""),
        description="DfT raw traffic counts API",
    )
    _add(
        source_key="scenario_open_meteo",
        component="scenario",
        url=str(settings.live_scenario_open_meteo_forecast_url or ""),
        description="Open-Meteo weather forecast feed",
    )
    _add(
        source_key="departure_profiles",
        component="calibration",
        url=str(settings.live_departure_profile_url or ""),
        description="Departure profile artifact",
    )
    _add(
        source_key="stochastic_regimes",
        component="calibration",
        url=str(settings.live_stochastic_regimes_url or ""),
        description="Stochastic regime artifact",
    )
    _add(
        source_key="toll_topology",
        component="pricing",
        url=str(settings.live_toll_topology_url or ""),
        description="Toll topology artifact",
    )
    _add(
        source_key="toll_tariffs",
        component="pricing",
        url=str(settings.live_toll_tariffs_url or ""),
        description="Toll tariff artifact",
    )
    _add(
        source_key="fuel_prices",
        component="pricing",
        url=str(settings.live_fuel_price_url or ""),
        description="Fuel price artifact/source",
    )
    _add(
        source_key="carbon_schedule",
        component="pricing",
        url=str(settings.live_carbon_schedule_url or ""),
        description="Carbon schedule artifact/source",
    )
    _add(
        source_key="bank_holidays",
        component="calendar",
        url=str(settings.live_bank_holidays_url or ""),
        description="UK bank holiday calendar",
    )
    if _should_prefetch_terrain_source():
        _add(
            source_key="terrain_live_tile",
            component="terrain",
            url=str(settings.live_terrain_dem_url_template or ""),
            description="Live terrain tile template",
        )
    return calls


def _record_expected_calls_blocked(
    *,
    reason_code: str,
    stage: str,
    detail: str,
) -> None:
    mark_live_expected_calls_blocked(
        reason_code=reason_code,
        stage=stage,
        detail=detail,
    )


def _prefetch_failure_text(exc: Exception) -> tuple[str, str]:
    if isinstance(exc, ModelDataError):
        code = normalize_reason_code(str(exc.reason_code or "model_asset_unavailable"))
        detail = str(exc.message).strip() or code
        details_payload = exc.details if isinstance(exc.details, dict) else {}
        if code == "terrain_dem_coverage_insufficient":
            try:
                probe_count = int(details_payload.get("probe_count", 0))
            except (TypeError, ValueError):
                probe_count = 0
            try:
                covered_count = int(details_payload.get("covered_count", 0))
            except (TypeError, ValueError):
                covered_count = 0
            try:
                min_required = int(details_payload.get("min_covered_points_required", 1))
            except (TypeError, ValueError):
                min_required = 1
            if probe_count > 0:
                detail = (
                    f"{detail} (covered_probes={covered_count}/{probe_count}; "
                    f"min_required={max(1, min_required)})."
                )
        if code == "scenario_profile_unavailable":
            coverage_gate_raw = details_payload.get("coverage_gate")
            coverage_gate = coverage_gate_raw if isinstance(coverage_gate_raw, dict) else {}
            coverage_keys = (
                "source_ok_count",
                "required_source_count",
                "required_source_count_configured",
                "required_source_count_effective",
                "waiver_applied",
                "waiver_reason",
            )
            has_coverage_gate_data = any(key in coverage_gate for key in coverage_keys) or any(
                key in details_payload for key in coverage_keys
            )

            def _int_or_default(value: Any, default: int = 0) -> int:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return int(default)

            if has_coverage_gate_data:
                source_ok_count = _int_or_default(
                    coverage_gate.get("source_ok_count", details_payload.get("source_ok_count", 0)),
                    default=0,
                )
                required_configured = _int_or_default(
                    coverage_gate.get(
                        "required_source_count_configured",
                        details_payload.get("required_source_count", 0),
                    ),
                    default=0,
                )
                required_effective = _int_or_default(
                    coverage_gate.get(
                        "required_source_count_effective",
                        details_payload.get("required_source_count", 0),
                    ),
                    default=0,
                )
                waiver_applied = bool(coverage_gate.get("waiver_applied", False))
                waiver_reason = str(coverage_gate.get("waiver_reason", "")).strip()
                road_hint = str(
                    details_payload.get(
                        "resolved_road_hint_value",
                        (
                            details_payload.get("route_context", {}).get("road_hint")
                            if isinstance(details_payload.get("route_context"), dict)
                            else ""
                        ),
                    )
                    or ""
                ).strip()
                webtris_used_sites = _int_or_default(
                    details_payload.get("webtris_used_site_count", 0),
                    default=0,
                )
                dft_selected_station_count = _int_or_default(
                    details_payload.get("dft_selected_station_count", details_payload.get("dft_selected_count", 0)),
                    default=0,
                )
                waiver_text = "yes" if waiver_applied else "no"
                if waiver_reason:
                    waiver_text = f"{waiver_text}:{waiver_reason}"
                detail = (
                    f"{detail} (source_ok={source_ok_count}/4; "
                    f"required_configured={required_configured}/4; "
                    f"required_effective={required_effective}/4; "
                    f"waiver={waiver_text}; "
                    f"road_hint={road_hint or 'unknown'}; "
                    f"webtris_used_sites={webtris_used_sites}; "
                    f"dft_selected_station_count={dft_selected_station_count})."
                )
            else:
                as_of_text = str(
                    details_payload.get("as_of_utc", details_payload.get("as_of", ""))
                ).strip()
                max_age_minutes = _int_or_default(
                    details_payload.get("max_age_minutes", settings.live_scenario_coefficient_max_age_minutes),
                    default=int(settings.live_scenario_coefficient_max_age_minutes),
                )
                age_minutes_text = ""
                if as_of_text:
                    try:
                        as_of_dt = datetime.fromisoformat(as_of_text.replace("Z", "+00:00"))
                        if as_of_dt.tzinfo is None:
                            as_of_dt = as_of_dt.replace(tzinfo=UTC)
                        else:
                            as_of_dt = as_of_dt.astimezone(UTC)
                        age_minutes_text = f"{max(0.0, (datetime.now(UTC) - as_of_dt).total_seconds() / 60.0):.2f}"
                    except ValueError:
                        age_minutes_text = ""
                freshness_parts: list[str] = []
                if as_of_text:
                    freshness_parts.append(f"as_of_utc={as_of_text}")
                if age_minutes_text:
                    freshness_parts.append(f"age_minutes={age_minutes_text}")
                if max_age_minutes > 0:
                    freshness_parts.append(f"max_age_minutes={max_age_minutes}")
                if freshness_parts:
                    detail = f"{detail} ({'; '.join(freshness_parts)})."
        return code, detail
    text = str(exc).strip() or type(exc).__name__
    lowered = text.lower()
    if "timeout" in lowered:
        return "route_compute_timeout", text
    return "model_asset_unavailable", text


def _prefetch_failure_detail_data(exc: Exception) -> dict[str, Any] | None:
    if not isinstance(exc, ModelDataError):
        return None
    if not isinstance(exc.details, dict):
        return None
    return dict(exc.details)


def _prefetch_route_context(
    *,
    origin: LatLng,
    destination: LatLng,
    vehicle_class: str,
    departure_time_utc: datetime | None,
    weather_bucket: str,
) -> ScenarioRouteContext:
    return build_scenario_route_context(
        route_points=[
            (float(origin.lat), float(origin.lon)),
            (float(destination.lat), float(destination.lon)),
        ],
        road_class_counts=None,
        vehicle_class=vehicle_class,
        departure_time_utc=departure_time_utc,
        weather_bucket=weather_bucket,
    )


def _should_prefetch_terrain_source() -> bool:
    if bool(settings.live_route_compute_probe_terrain):
        return True
    if bool(settings.strict_live_data_required):
        return True
    return bool(settings.live_route_compute_require_all_expected)


def _terrain_prefetch_probe_fractions() -> list[float]:
    defaults: list[float] = [0.5, 0.35, 0.65, 0.2, 0.8]
    raw = str(settings.live_terrain_prefetch_probe_fractions or "").strip()
    if not raw:
        return defaults
    fractions: list[float] = []
    seen: set[float] = set()
    for token in raw.split(","):
        text = str(token).strip()
        if not text:
            continue
        try:
            value = float(text)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        clamped = max(0.0, min(1.0, value))
        key = round(clamped, 6)
        if key in seen:
            continue
        seen.add(key)
        fractions.append(clamped)
    return fractions or defaults


def _missing_expected_sources_from_trace() -> list[str]:
    request_id = current_live_trace_request_id()
    if not request_id:
        return []
    trace = get_live_call_trace(request_id)
    if not isinstance(trace, dict):
        return []
    rollup = trace.get("expected_rollup", [])
    if not isinstance(rollup, list):
        return []
    missing: list[str] = []
    for row in rollup:
        if not isinstance(row, dict):
            continue
        if not bool(row.get("required", True)):
            continue
        observed_calls = _as_int_or_zero(row.get("observed_calls"))
        if observed_calls > 0:
            continue
        source_key = str(row.get("source_key", "")).strip()
        if source_key:
            missing.append(source_key)
    deduped: list[str] = []
    seen: set[str] = set()
    for key in missing:
        if key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return deduped


def _call_uncached(fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    uncached = getattr(fn, "__wrapped__", None)
    target = uncached if callable(uncached) else fn
    return target(*args, **kwargs)


def _call_prefetch_loader(fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """
    Use cached loader paths for per-request strict prefetch to avoid re-running
    expensive remote fanouts on every compute call.

    Strict fail-closed behavior is still enforced by:
    1) source-level exceptions (stale/unavailable data still raises),
    2) prefetch row success/failure accounting,
    3) expected-source gate reconciliation after prefetch.

    `live_route_compute_force_uncached` defaults to strict compatibility. Runtime
    optimization can disable it in env to favor cached prefetch loader paths.
    """
    if bool(getattr(settings, "live_route_compute_force_uncached", True)):
        return _call_uncached(fn, *args, **kwargs)
    return fn(*args, **kwargs)


async def _prefetch_expected_live_sources(
    *,
    origin: LatLng,
    destination: LatLng,
    vehicle_class: str,
    departure_time_utc: datetime | None,
    weather_bucket: str,
    cost_toggles: CostToggles,
) -> dict[str, Any]:
    timeout_ms = max(1_000, int(settings.live_route_compute_prefetch_timeout_ms))
    started = time.perf_counter()
    deadline = started + (float(timeout_ms) / 1000.0)
    route_context = _prefetch_route_context(
        origin=origin,
        destination=destination,
        vehicle_class=vehicle_class,
        departure_time_utc=departure_time_utc,
        weather_bucket=weather_bucket,
    )
    source_rows: list[dict[str, Any]] = []

    async def _run_prefetch_call(
        call_index: int,
        source_key: str,
        fn: Callable[[], Any],
    ) -> tuple[int, dict[str, Any]]:
        async with prefetch_semaphore:
            call_started = time.perf_counter()
            remaining_s = deadline - call_started
            if remaining_s <= 0.0:
                return (
                    call_index,
                    {
                        "source_key": source_key,
                        "ok": False,
                        "reason_code": "route_compute_timeout",
                        "detail": (
                            f"Prefetch timeout budget exhausted before source call "
                            f"(timeout_ms={timeout_ms})."
                        ),
                        "duration_ms": 0.0,
                    },
                )
            try:
                await asyncio.wait_for(asyncio.to_thread(fn), timeout=remaining_s)
                return (
                    call_index,
                    {
                        "source_key": source_key,
                        "ok": True,
                        "reason_code": "ok",
                        "detail": "prefetch_ok",
                        "duration_ms": round((time.perf_counter() - call_started) * 1000.0, 2),
                    },
                )
            except asyncio.TimeoutError:
                return (
                    call_index,
                    {
                        "source_key": source_key,
                        "ok": False,
                        "reason_code": "route_compute_timeout",
                        "detail": (
                            f"Prefetch source timed out "
                            f"(source={source_key}; timeout_ms={timeout_ms})."
                        ),
                        "duration_ms": round((time.perf_counter() - call_started) * 1000.0, 2),
                    },
                )
            except Exception as exc:  # pragma: no cover - defensive boundary
                reason_code, detail = _prefetch_failure_text(exc)
                detail_data = _prefetch_failure_detail_data(exc)
                return (
                    call_index,
                    {
                        "source_key": source_key,
                        "ok": False,
                        "reason_code": reason_code,
                        "detail": detail,
                        "detail_data": detail_data if isinstance(detail_data, dict) else None,
                        "duration_ms": round((time.perf_counter() - call_started) * 1000.0, 2),
                    },
                )

    def _prefetch_scenario_context() -> ScenarioRouteContext:
        profiles = _call_prefetch_loader(load_scenario_profiles)
        transform_params = profiles.transform_params if isinstance(profiles.transform_params, dict) else None
        if bool(settings.strict_live_data_required) and transform_params is None:
            raise ModelDataError(
                reason_code="scenario_profile_invalid",
                message=(
                    "Scenario transform parameters are required to prefetch live scenario context in strict runtime."
                ),
            )
        transform_params_json = (
            json.dumps(
                transform_params,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
            if isinstance(transform_params, dict)
            else None
        )
        _call_prefetch_loader(
            load_live_scenario_context,
            corridor_bucket=route_context.corridor_geohash5,
            road_mix_bucket=route_context.road_mix_bucket,
            vehicle_class=route_context.vehicle_class,
            day_kind=route_context.day_kind,
            hour_slot_local=route_context.hour_slot_local,
            weather_bucket=route_context.weather_regime,
            centroid_lat=route_context.centroid_lat,
            centroid_lon=route_context.centroid_lon,
            road_hint=route_context.road_hint,
            transform_params_json=transform_params_json,
        )
        return route_context

    def _prefetch_terrain_probe() -> None:
        terrain_begin_route_run()
        origin_lat = float(origin.lat)
        origin_lon = float(origin.lon)
        delta_lat = float(destination.lat) - origin_lat
        delta_lon = float(destination.lon) - origin_lon
        fractions = _terrain_prefetch_probe_fractions()
        min_required = max(1, int(settings.live_terrain_prefetch_min_covered_points))
        min_required = min(min_required, len(fractions))
        probe_rows: list[dict[str, Any]] = []
        covered_count = 0
        seen_sources: set[str] = set()
        for fraction in fractions:
            point_lat = origin_lat + (delta_lat * float(fraction))
            point_lon = origin_lon + (delta_lon * float(fraction))
            elevation_m, covered, source = sample_elevation_m(point_lat, point_lon)
            covered_bool = bool(covered)
            source_text = str(source or "").strip()
            if source_text:
                seen_sources.add(source_text)
            if covered_bool:
                covered_count += 1
            probe_rows.append(
                {
                    "fraction": round(float(fraction), 6),
                    "lat": round(point_lat, 6),
                    "lon": round(point_lon, 6),
                    "covered": covered_bool,
                    "source": source_text,
                    "elevation_m": (
                        round(float(elevation_m), 3)
                        if isinstance(elevation_m, (int, float)) and math.isfinite(float(elevation_m))
                        else None
                    ),
                }
            )
        if covered_count >= min_required:
            return
        failed_points = [
            {
                "fraction": row.get("fraction"),
                "lat": row.get("lat"),
                "lon": row.get("lon"),
                "source": row.get("source"),
            }
            for row in probe_rows
            if not bool(row.get("covered"))
        ]
        raise ModelDataError(
            reason_code="terrain_dem_coverage_insufficient",
            message=(
                "Terrain live probe did not return sufficient covered elevation along the OD corridor "
                f"(covered={covered_count}/{len(probe_rows)}; required={min_required})."
            ),
            details={
                "probe_count": int(len(probe_rows)),
                "covered_count": int(covered_count),
                "min_covered_points_required": int(min_required),
                "sources_seen": sorted(seen_sources),
                "probe_points": probe_rows,
                "failed_points": failed_points,
            },
        )

    calls: list[tuple[str, Callable[[], Any]]] = [
        ("scenario_coefficients", lambda: _call_prefetch_loader(load_scenario_profiles)),
        ("scenario_live_context", _prefetch_scenario_context),
        ("departure_profiles", lambda: _call_prefetch_loader(load_departure_profile)),
        ("stochastic_regimes", lambda: _call_prefetch_loader(load_stochastic_regimes)),
        ("toll_topology", lambda: _call_prefetch_loader(load_toll_segments_seed)),
        ("toll_tariffs", lambda: _call_prefetch_loader(load_toll_tariffs)),
        (
            "fuel_prices",
            lambda: _call_prefetch_loader(load_fuel_price_snapshot, as_of_utc=departure_time_utc),
        ),
        (
            "carbon_schedule",
            lambda: resolve_carbon_price(
                request_price_per_kg=max(0.0, float(cost_toggles.carbon_price_per_kg)),
                departure_time_utc=departure_time_utc,
            ),
        ),
        ("bank_holidays", lambda: _call_prefetch_loader(load_uk_bank_holidays)),
    ]
    if _should_prefetch_terrain_source():
        calls.append(("terrain_live_tile", _prefetch_terrain_probe))

    prefetch_max_concurrency = max(
        1,
        min(int(settings.live_route_compute_prefetch_max_concurrency), len(calls)),
    )
    prefetch_semaphore = asyncio.Semaphore(prefetch_max_concurrency)

    prefetch_results = await asyncio.gather(
        *[
            _run_prefetch_call(call_index, source_key, fn)
            for call_index, (source_key, fn) in enumerate(calls)
        ]
    )
    prefetch_results.sort(key=lambda item: item[0])
    source_rows = [row for _call_index, row in prefetch_results]

    failed_rows = [row for row in source_rows if not bool(row.get("ok"))]
    failed_source_details = [
        {
            "source_key": str(row.get("source_key", "")).strip(),
            "reason_code": str(row.get("reason_code", "unknown")).strip() or "unknown",
            "detail": str(row.get("detail", "")).strip(),
            "detail_data": (row.get("detail_data") if isinstance(row.get("detail_data"), dict) else None),
            "duration_ms": _as_float_or_zero(row.get("duration_ms")),
        }
        for row in failed_rows
    ]
    missing_expected_sources = _missing_expected_sources_from_trace()
    if not bool(getattr(settings, "live_route_compute_force_uncached", True)):
        successful_prefetch_keys = {
            str(row.get("source_key", "")).strip()
            for row in source_rows
            if bool(row.get("ok"))
        }
        missing_expected_sources = [
            key
            for key in missing_expected_sources
            if key not in successful_prefetch_keys
        ]
    expected_required_total = 0
    expected_observed_total = 0
    request_id = current_live_trace_request_id()
    if request_id:
        trace_snapshot = get_live_call_trace(request_id)
        if isinstance(trace_snapshot, dict):
            rollup = trace_snapshot.get("expected_rollup", [])
            if isinstance(rollup, list):
                expected_required_total = sum(
                    1
                    for row in rollup
                    if isinstance(row, dict) and bool(row.get("required", True))
                )
                expected_observed_total = sum(
                    1
                    for row in rollup
                    if isinstance(row, dict)
                    and bool(row.get("required", True))
                    and _as_int_or_zero(row.get("observed_calls")) > 0
                )
    summary = {
        "timeout_ms": int(timeout_ms),
        "prefetch_max_concurrency": int(prefetch_max_concurrency),
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 2),
        "source_total": int(len(source_rows)),
        "source_success": int(len(source_rows) - len(failed_rows)),
        "source_failed": int(len(failed_rows)),
        "failed_source_keys": [str(row.get("source_key", "")) for row in failed_rows],
        "failed_source_details": failed_source_details,
        "missing_expected_sources": missing_expected_sources,
        "expected_required_total": int(expected_required_total),
        "expected_observed_total": int(expected_observed_total),
        "rows": source_rows,
    }
    log_event(
        "route_live_prefetch_summary",
        timeout_ms=summary["timeout_ms"],
        prefetch_max_concurrency=summary["prefetch_max_concurrency"],
        elapsed_ms=summary["elapsed_ms"],
        source_total=summary["source_total"],
        source_success=summary["source_success"],
        source_failed=summary["source_failed"],
        failed_source_keys=summary["failed_source_keys"],
        failed_source_details=summary["failed_source_details"],
        missing_expected_sources=summary["missing_expected_sources"],
        expected_required_total=summary["expected_required_total"],
        expected_observed_total=summary["expected_observed_total"],
    )
    strict_prefetch_gate = bool(settings.live_route_compute_require_all_expected) or bool(
        settings.strict_live_data_required
    )
    if strict_prefetch_gate and (failed_rows or missing_expected_sources):
        failed_keys = ",".join(
            str(row.get("source_key", "")).strip() for row in failed_rows if str(row.get("source_key", "")).strip()
        )
        failed_details_inline = "; ".join(
            (
                f"{str(row.get('source_key', '')).strip()}:"
                f"{str(row.get('reason_code', 'unknown')).strip() or 'unknown'}"
                f" ({str(row.get('detail', '')).strip()})"
            )
            for row in failed_source_details
            if str(row.get("source_key", "")).strip()
        )
        if len(failed_details_inline) > 1500:
            failed_details_inline = f"{failed_details_inline[:1497]}..."
        missing_keys = ",".join(str(key).strip() for key in missing_expected_sources if str(key).strip())
        raise ModelDataError(
            reason_code="live_source_refresh_failed",
            message=(
                "Live-source prefetch failed strict freshness gate "
                f"(failed_sources={failed_keys or 'none'}; "
                f"failed_source_details={failed_details_inline or 'none'}; "
                f"missing_expected_sources={missing_keys or 'none'})."
            ),
            details=summary,
        )
    return summary


def _routing_graph_warmup_failfast_detail() -> dict[str, Any] | None:
    if not bool(settings.strict_live_data_required):
        return None
    if not bool(settings.route_graph_warmup_failfast):
        return None
    warmup = route_graph_warmup_status()
    state = str(warmup.get("state", "")).strip().lower()
    if state == "loading":
        elapsed_ms = warmup.get("elapsed_ms")
        retry_after_seconds = 3
        if isinstance(elapsed_ms, (int, float)) and elapsed_ms >= 0:
            retry_after_seconds = max(1, min(30, int(round(float(elapsed_ms) / 1000.0)) + 2))
        return {
            "reason_code": "routing_graph_warming_up",
            "message": "Routing graph warmup is still in progress; retry shortly.",
            "warnings": [],
            "stage": "collecting_candidates",
            "stage_detail": "routing_graph_warming_up",
            "warmup": warmup,
            "retry_after_seconds": retry_after_seconds,
            "phase": warmup.get("phase"),
            "elapsed_ms": warmup.get("elapsed_ms"),
            "asset_path": warmup.get("asset_path"),
            "asset_size_mb": warmup.get("asset_size_mb"),
        }
    if state == "failed":
        last_error = str(warmup.get("last_error", "")).strip() or "unknown warmup failure"
        is_fragmented = "routing_graph_fragmented" in last_error.lower()
        return {
            "reason_code": "routing_graph_fragmented" if is_fragmented else "routing_graph_warmup_failed",
            "message": (
                "Routing graph failed strict connectivity quality checks."
                if is_fragmented
                else "Routing graph warmup failed. Rebuild the graph asset and restart backend."
            ),
            "warnings": [],
            "stage": "collecting_candidates",
            "stage_detail": "routing_graph_fragmented" if is_fragmented else "routing_graph_warmup_failed",
            "warmup": warmup,
            "phase": warmup.get("phase"),
            "elapsed_ms": warmup.get("elapsed_ms"),
            "asset_path": warmup.get("asset_path"),
            "asset_size_mb": warmup.get("asset_size_mb"),
            "last_error": last_error,
            "retry_hint": (
                "Rebuild routing_graph_uk.json with full UK topology coverage and verified connectivity, "
                "restart backend, then wait for GET /health/ready strict_route_ready=true."
            ),
        }
    return None


def _strict_live_readiness_status() -> dict[str, Any]:
    checked_at = datetime.now(UTC)
    strict_required = bool(settings.strict_live_data_required) or bool(settings.live_route_compute_require_all_expected)
    if not strict_required:
        return {
            "ok": True,
            "status": "disabled",
            "reason_code": "ok",
            "message": "Strict live readiness checks are disabled.",
            "as_of_utc": None,
            "age_minutes": None,
            "max_age_minutes": int(settings.live_scenario_coefficient_max_age_minutes),
            "checked_at_utc": checked_at.isoformat(),
        }
    if not bool(settings.live_runtime_data_enabled):
        return {
            "ok": False,
            "status": "unavailable",
            "reason_code": "scenario_profile_unavailable",
            "message": "Live runtime data is disabled while strict live checks are required.",
            "as_of_utc": None,
            "age_minutes": None,
            "max_age_minutes": int(settings.live_scenario_coefficient_max_age_minutes),
            "checked_at_utc": checked_at.isoformat(),
        }

    max_age_minutes = int(settings.live_scenario_coefficient_max_age_minutes)

    def _parse_iso_utc(raw: Any) -> datetime | None:
        text = str(raw or "").strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        else:
            parsed = parsed.astimezone(UTC)
        return parsed

    def _age_minutes(as_of_dt: datetime | None) -> float | None:
        if as_of_dt is None:
            return None
        return round(max(0.0, (checked_at - as_of_dt).total_seconds() / 60.0), 2)

    try:
        profiles = _call_uncached(load_scenario_profiles)
        as_of_text = str(
            getattr(profiles, "as_of_utc", None)
            or getattr(profiles, "generated_at_utc", None)
            or ""
        ).strip()
        as_of_dt = _parse_iso_utc(as_of_text)
        return {
            "ok": True,
            "status": "ok",
            "reason_code": "ok",
            "message": "Strict live scenario coefficients are ready.",
            "as_of_utc": as_of_dt.isoformat() if as_of_dt is not None else (as_of_text or None),
            "age_minutes": _age_minutes(as_of_dt),
            "max_age_minutes": int(max_age_minutes),
            "checked_at_utc": checked_at.isoformat(),
        }
    except ModelDataError as exc:
        reason_code = normalize_reason_code(str(exc.reason_code or "scenario_profile_unavailable"))
        detail_data = exc.details if isinstance(exc.details, dict) else {}
        as_of_text = str(
            detail_data.get("as_of_utc", detail_data.get("as_of", ""))
        ).strip()
        as_of_dt = _parse_iso_utc(as_of_text)
        detail_max_age = detail_data.get("max_age_minutes", max_age_minutes)
        try:
            effective_max_age = int(detail_max_age)
        except (TypeError, ValueError):
            effective_max_age = int(max_age_minutes)
        status = "stale" if as_of_dt is not None and effective_max_age > 0 else "unavailable"
        return {
            "ok": False,
            "status": status,
            "reason_code": reason_code,
            "message": str(exc.message).strip() or reason_code,
            "as_of_utc": as_of_dt.isoformat() if as_of_dt is not None else (as_of_text or None),
            "age_minutes": _age_minutes(as_of_dt),
            "max_age_minutes": int(effective_max_age),
            "checked_at_utc": checked_at.isoformat(),
        }
    except Exception as exc:  # pragma: no cover - defensive boundary
        return {
            "ok": False,
            "status": "unavailable",
            "reason_code": "model_asset_unavailable",
            "message": str(exc).strip() or type(exc).__name__,
            "as_of_utc": None,
            "age_minutes": None,
            "max_age_minutes": int(max_age_minutes),
            "checked_at_utc": checked_at.isoformat(),
        }


@app.get("/health/ready")
async def health_ready() -> dict[str, Any]:
    warmup = route_graph_warmup_status()
    warmup_state = str(warmup.get("state", "")).strip().lower()
    if warmup_state == "loading":
        graph_ok, graph_status = False, "warming_up"
        recommended_action_graph = "wait"
    elif warmup_state == "failed":
        graph_ok, graph_status = False, "warmup_failed"
        if bool(warmup.get("graph_fragmented")):
            graph_status = "fragmented"
        recommended_action_graph = "rebuild_graph"
    else:
        graph_ok, graph_status = await _route_graph_status_async()
        if graph_ok:
            recommended_action_graph = "ready"
        elif graph_status == "fragmented":
            recommended_action_graph = "rebuild_graph"
        else:
            recommended_action_graph = "retry"
    strict_live = _strict_live_readiness_status()
    strict_required = bool(settings.strict_live_data_required) or bool(settings.live_route_compute_require_all_expected)
    strict_live_ok = bool(strict_live.get("ok", False))
    strict_route_ready = bool(graph_ok) and (strict_live_ok if strict_required else True)
    recommended_action = recommended_action_graph
    if bool(graph_ok) and strict_required and not strict_live_ok:
        recommended_action = "refresh_live_sources"
    return {
        "status": "ready" if strict_route_ready else "not_ready",
        "strict_route_ready": bool(strict_route_ready),
        "recommended_action": recommended_action,
        "route_graph": {
            "ok": bool(graph_ok),
            "status": graph_status,
            **warmup,
        },
        "strict_live": strict_live,
    }


@app.get("/debug/live-calls/{request_id}")
async def debug_live_calls(request_id: str, _: UserAccessDep) -> dict[str, Any]:
    if not bool(settings.dev_route_debug_console_enabled):
        raise HTTPException(status_code=404, detail="Debug live-call diagnostics are disabled.")
    payload = get_live_call_trace(request_id)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"No live-call trace found for request_id={request_id}.")
    return payload


@app.get("/vehicles", response_model=VehicleListResponse)
async def list_vehicles() -> VehicleListResponse:
    merged = all_vehicles()
    return VehicleListResponse(vehicles=[merged[key] for key in sorted(merged)])


@app.get("/vehicles/custom", response_model=CustomVehicleListResponse)
async def list_custom_vehicle_profiles() -> CustomVehicleListResponse:
    return CustomVehicleListResponse(vehicles=list_custom_vehicles())


@app.post("/vehicles/custom", response_model=VehicleMutationResponse)
async def create_vehicle_profile(vehicle: VehicleProfile, _: AdminAccessDep) -> VehicleMutationResponse:
    t0 = time.perf_counter()
    has_error = False
    try:
        created, path = create_custom_vehicle(vehicle)
        log_event("vehicle_custom_create", vehicle_id=created.id, path=str(path))
        return VehicleMutationResponse(vehicle=created)
    except ValueError as e:
        has_error = True
        msg = str(e)
        status = 409 if "exists" in msg or "collides" in msg else 400
        raise HTTPException(status_code=status, detail=msg) from e
    except Exception:
        has_error = True
        raise
    finally:
        _record_endpoint_metric("vehicles_custom_post", t0, error=has_error)


@app.put("/vehicles/custom/{vehicle_id}", response_model=VehicleMutationResponse)
async def update_vehicle_profile(
    vehicle_id: str,
    vehicle: VehicleProfile,
    _: AdminAccessDep,
) -> VehicleMutationResponse:
    t0 = time.perf_counter()
    has_error = False
    try:
        updated, path = update_custom_vehicle(vehicle_id, vehicle)
        log_event("vehicle_custom_update", vehicle_id=updated.id, path=str(path))
        return VehicleMutationResponse(vehicle=updated)
    except KeyError as e:
        has_error = True
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        has_error = True
        msg = str(e)
        status = 400
        if "cannot be updated" in msg:
            status = 403
        raise HTTPException(status_code=status, detail=msg) from e
    except Exception:
        has_error = True
        raise
    finally:
        _record_endpoint_metric("vehicles_custom_put", t0, error=has_error)


@app.delete("/vehicles/custom/{vehicle_id}", response_model=VehicleDeleteResponse)
async def delete_vehicle_profile(vehicle_id: str, _: AdminAccessDep) -> VehicleDeleteResponse:
    t0 = time.perf_counter()
    has_error = False
    try:
        deleted_id, path = delete_custom_vehicle(vehicle_id)
        log_event("vehicle_custom_delete", vehicle_id=deleted_id, path=str(path))
        return VehicleDeleteResponse(vehicle_id=deleted_id, deleted=True)
    except KeyError as e:
        has_error = True
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        has_error = True
        msg = str(e)
        status = 400
        if "cannot be deleted" in msg:
            status = 403
        raise HTTPException(status_code=status, detail=msg) from e
    except Exception:
        has_error = True
        raise
    finally:
        _record_endpoint_metric("vehicles_custom_delete", t0, error=has_error)


@app.get("/metrics")
async def get_metrics() -> dict[str, object]:
    payload = metrics_snapshot()
    payload["route_cache"] = route_cache_stats()
    return payload


@app.get("/cache/stats")
async def get_cache_stats() -> dict[str, int]:
    return route_cache_stats()


@app.delete("/cache")
async def delete_cache(_: AdminAccessDep) -> dict[str, int]:
    t0 = time.perf_counter()
    has_error = False
    try:
        cleared = clear_route_cache()
        log_event("route_cache_clear", cleared=cleared)
        return {"cleared": cleared}
    except Exception:
        has_error = True
        raise
    finally:
        _record_endpoint_metric("cache_delete", t0, error=has_error)


def _datetime_to_utc_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    else:
        value = value.astimezone(UTC)
    return value.isoformat()


def _to_oracle_check_record(payload: OracleFeedCheckInput) -> OracleFeedCheckRecord:
    has_signature_failure = payload.signature_valid is False
    passed = bool(payload.schema_valid) and (not has_signature_failure) and not bool(payload.error)
    return OracleFeedCheckRecord(
        check_id=str(uuid.uuid4()),
        source=payload.source.strip(),
        schema_valid=payload.schema_valid,
        signature_valid=payload.signature_valid,
        freshness_s=payload.freshness_s,
        latency_ms=payload.latency_ms,
        record_count=payload.record_count,
        observed_at_utc=_datetime_to_utc_iso(payload.observed_at_utc),
        error=payload.error.strip() if payload.error else None,
        passed=passed,
        ingested_at_utc=datetime.now(UTC).isoformat(),
    )


@app.post("/oracle/quality/check", response_model=OracleFeedCheckRecord)
async def post_oracle_quality_check(
    payload: OracleFeedCheckInput,
    _: UserAccessDep,
) -> OracleFeedCheckRecord:
    t0 = time.perf_counter()
    has_error = False
    try:
        record = _to_oracle_check_record(payload)
        append_check_record(record.model_dump(mode="json"))
        records = load_check_records()
        dashboard = compute_dashboard_payload(records)
        summary, csv_path = write_summary_artifacts(dashboard)

        log_event(
            "oracle_quality_check_ingested",
            check_id=record.check_id,
            source=record.source,
            passed=record.passed,
            schema_valid=record.schema_valid,
            signature_valid=record.signature_valid,
            total_checks=dashboard["total_checks"],
            summary_path=str(summary),
            csv_path=str(csv_path),
        )
        return record
    except Exception:
        has_error = True
        raise
    finally:
        _record_endpoint_metric("oracle_quality_check_post", t0, error=has_error)


@app.get("/oracle/quality/dashboard", response_model=OracleQualityDashboardResponse)
async def get_oracle_quality_dashboard(_: UserAccessDep) -> OracleQualityDashboardResponse:
    t0 = time.perf_counter()
    has_error = False
    try:
        records = load_check_records()
        dashboard = compute_dashboard_payload(records)
        summary, csv_path = write_summary_artifacts(dashboard)
        log_event(
            "oracle_quality_dashboard_get",
            total_checks=dashboard["total_checks"],
            source_count=dashboard["source_count"],
            summary_path=str(summary),
            csv_path=str(csv_path),
            checks_path=str(checks_path()),
        )
        return OracleQualityDashboardResponse.model_validate(dashboard)
    except Exception:
        has_error = True
        raise
    finally:
        _record_endpoint_metric("oracle_quality_dashboard_get", t0, error=has_error)


@app.get("/oracle/quality/dashboard.csv")
async def get_oracle_quality_dashboard_csv(_: UserAccessDep) -> FileResponse:
    t0 = time.perf_counter()
    has_error = False
    try:
        records = load_check_records()
        dashboard = compute_dashboard_payload(records)
        _summary_path, csv_path = write_summary_artifacts(dashboard)
        if not csv_path.exists():
            raise HTTPException(status_code=404, detail="oracle quality dashboard CSV not found")
        return FileResponse(str(csv_path), media_type="text/csv", filename="oracle_quality_dashboard.csv")
    except HTTPException:
        has_error = True
        raise
    except Exception:
        has_error = True
        raise
    finally:
        _record_endpoint_metric("oracle_quality_dashboard_csv_get", t0, error=has_error)


def _record_endpoint_metric(endpoint: str, t0: float, *, error: bool = False) -> None:
    record_request(
        endpoint=endpoint,
        duration_ms=(time.perf_counter() - t0) * 1000.0,
        error=error,
    )


def _validate_osrm_geometry(route: dict[str, Any]) -> list[tuple[float, float]]:
    geom = route.get("geometry")
    if not isinstance(geom, dict):
        raise OSRMError("OSRM route missing geometry")

    coords = geom.get("coordinates")
    if not isinstance(coords, list) or len(coords) < 2:
        raise OSRMError("OSRM geometry missing coordinates")

    out: list[tuple[float, float]] = []
    for pt in coords:
        if (
            isinstance(pt, (list, tuple))
            and len(pt) == 2
            and isinstance(pt[0], (int, float))
            and isinstance(pt[1], (int, float))
        ):
            out.append((float(pt[0]), float(pt[1])))
    if len(out) < 2:
        raise OSRMError("OSRM geometry invalid")
    return out


# Monetary cost model: weight the time-based component so "£" doesn't perfectly track "time".
# If cost is dominated by time, one route often dominates and the Pareto set collapses to 1.
DRIVER_TIME_COST_WEIGHT: float = 0.35
MAX_GEOMETRY_POINTS: int = 1200
MAX_CANDIDATE_FETCH_CONCURRENCY: int = 6
CANDIDATE_CACHE_SCHEMA_VERSION: int = 2
ProgressCallback = Callable[[dict[str, Any]], Awaitable[None]]


@dataclass(frozen=True)
class CandidateFetchSpec:
    label: str
    alternatives: bool | int
    exclude: str | None = None
    via: list[tuple[float, float]] | None = None


@dataclass(frozen=True)
class CandidateFetchResult:
    spec: CandidateFetchSpec
    routes: list[dict[str, Any]]
    error: str | None = None


@dataclass(frozen=True)
class CandidateProgress:
    done: int
    total: int
    result: CandidateFetchResult


@dataclass(frozen=True)
class TerrainDiagnostics:
    fail_closed_count: int = 0
    unsupported_region_count: int = 0
    asset_unavailable_count: int = 0
    dem_version: str = "unknown"
    coverage_min_observed: float = 1.0


@dataclass(frozen=True)
class CandidateDiagnostics:
    raw_count: int = 0
    deduped_count: int = 0
    graph_explored_states: int = 0
    graph_generated_paths: int = 0
    graph_emitted_paths: int = 0
    candidate_budget: int = 0
    graph_effective_max_hops: int = 0
    graph_effective_hops_floor: int = 0
    graph_effective_state_budget_initial: int = 0
    graph_effective_state_budget: int = 0
    graph_no_path_reason: str = ""
    graph_no_path_detail: str = ""
    prefetch_total_sources: int = 0
    prefetch_success_sources: int = 0
    prefetch_failed_sources: int = 0
    prefetch_failed_keys: str = ""
    prefetch_failed_details: str = ""
    prefetch_missing_expected_sources: str = ""
    prefetch_rows_json: str = ""
    scenario_gate_required_configured: int = 0
    scenario_gate_required_effective: int = 0
    scenario_gate_source_ok_count: int = 0
    scenario_gate_waiver_applied: bool = False
    scenario_gate_waiver_reason: str = ""
    scenario_gate_source_signal_json: str = ""
    scenario_gate_source_reachability_json: str = ""
    scenario_gate_road_hint: str = ""
    scenario_candidate_family_count: int = 0
    scenario_candidate_jaccard_vs_baseline: float = 1.0
    scenario_candidate_jaccard_threshold: float = 1.0
    scenario_candidate_stress_score: float = 0.0
    scenario_candidate_gate_action: str = "not_applicable"
    scenario_edge_scaling_version: str = "v3_live_transform"
    precheck_reason_code: str = ""
    precheck_message: str = ""
    precheck_elapsed_ms: float = 0.0
    precheck_origin_node_id: str = ""
    precheck_destination_node_id: str = ""
    precheck_origin_nearest_m: float = 0.0
    precheck_destination_nearest_m: float = 0.0
    precheck_origin_selected_m: float = 0.0
    precheck_destination_selected_m: float = 0.0
    precheck_origin_candidate_count: int = 0
    precheck_destination_candidate_count: int = 0
    precheck_selected_component: int = 0
    precheck_selected_component_size: int = 0
    precheck_gate_action: str = ""
    graph_retry_attempted: bool = False
    graph_retry_state_budget: int = 0
    graph_retry_outcome: str = ""
    graph_rescue_attempted: bool = False
    graph_rescue_mode: str = "not_applicable"
    graph_rescue_state_budget: int = 0
    graph_rescue_outcome: str = "not_applicable"
    prefetch_ms: float = 0.0
    scenario_context_ms: float = 0.0
    graph_search_ms_initial: float = 0.0
    graph_search_ms_retry: float = 0.0
    graph_search_ms_rescue: float = 0.0
    osrm_refine_ms: float = 0.0
    build_options_ms: float = 0.0


def _is_toll_exclusion_label(label: str) -> bool:
    return label == "exclude:toll"


def _annotate_route_candidate_meta(
    route: dict[str, Any],
    *,
    source_labels: set[str],
    toll_exclusion_available: bool,
) -> None:
    ordered_labels = sorted(source_labels)
    seen_by_exclude_toll = any(_is_toll_exclusion_label(label) for label in ordered_labels)
    seen_by_non_exclude_toll = any(not _is_toll_exclusion_label(label) for label in ordered_labels)

    route["_candidate_meta"] = {
        "source_labels": ordered_labels,
        "seen_by_exclude_toll": seen_by_exclude_toll,
        "seen_by_non_exclude_toll": seen_by_non_exclude_toll,
        "toll_exclusion_available": bool(toll_exclusion_available),
    }


def _ndjson_line(event: dict[str, Any]) -> bytes:
    return (json.dumps(event, separators=(",", ":")) + "\n").encode("utf-8")


async def _emit_progress(
    progress_cb: ProgressCallback | None,
    payload: dict[str, Any],
) -> None:
    if progress_cb is None:
        return
    try:
        await progress_cb(payload)
    except Exception:
        # Progress callbacks are best-effort and must never break route compute.
        return


def _downsample_coords(
    coords: list[tuple[float, float]],
    *,
    max_points: int = MAX_GEOMETRY_POINTS,
) -> list[tuple[float, float]]:
    """Clamp route point count while preserving endpoints."""
    if max_points < 2 or len(coords) <= max_points:
        return coords

    last_idx = len(coords) - 1
    stride = last_idx / float(max_points - 1)
    out: list[tuple[float, float]] = [coords[0]]
    prev_idx = 0

    for i in range(1, max_points - 1):
        idx = int(round(i * stride))
        idx = min(last_idx - 1, max(prev_idx + 1, idx))
        out.append(coords[idx])
        prev_idx = idx

    out.append(coords[last_idx])
    return out


def _counterfactual_shift_scenario(mode: ScenarioMode) -> ScenarioMode:
    if mode == ScenarioMode.NO_SHARING:
        return ScenarioMode.PARTIAL_SHARING
    if mode == ScenarioMode.PARTIAL_SHARING:
        return ScenarioMode.FULL_SHARING
    return ScenarioMode.PARTIAL_SHARING


def _road_class_key(road_class_counts: dict[str, int] | None) -> tuple[tuple[str, int], ...]:
    if not road_class_counts:
        return ()
    return tuple(
        sorted(
            (
                str(key),
                int(value),
            )
            for key, value in road_class_counts.items()
        )
    )


@lru_cache(maxsize=1024)
def _deterministic_uncertainty_cached(
    *,
    base_duration_s: float,
    base_monetary_cost: float,
    base_emissions_kg: float,
    base_distance_km: float,
    route_signature: str,
    departure_time_iso: str,
    sigma: float,
    samples: int,
    utility_weight_time: float,
    utility_weight_money: float,
    utility_weight_co2: float,
    risk_aversion: float,
    road_class_key: tuple[tuple[str, int], ...],
    weather_profile: str,
    vehicle_type: str,
    vehicle_bucket: str,
    corridor_bucket: str,
) -> dict[str, float]:
    departure_time_utc: datetime | None = (
        datetime.fromisoformat(departure_time_iso) if departure_time_iso else None
    )
    road_class_counts = {key: int(value) for key, value in road_class_key}
    summary = compute_uncertainty_summary(
        base_duration_s=base_duration_s,
        base_monetary_cost=base_monetary_cost,
        base_emissions_kg=base_emissions_kg,
        base_distance_km=base_distance_km,
        route_signature=route_signature,
        departure_time_utc=departure_time_utc,
        user_seed=0,
        sigma=sigma,
        samples=samples,
        cvar_alpha=settings.risk_cvar_alpha,
        utility_weights=(
            utility_weight_time,
            utility_weight_money,
            utility_weight_co2,
        ),
        risk_aversion=risk_aversion,
        road_class_counts=road_class_counts,
        weather_profile=weather_profile,
        vehicle_type=vehicle_type or None,
        vehicle_bucket=vehicle_bucket or None,
        corridor_bucket=corridor_bucket or None,
    )
    return summary.as_dict()


def _route_stochastic_uncertainty(
    option: RouteOption,
    *,
    stochastic: StochasticConfig,
    route_signature: str,
    departure_time_utc: datetime | None,
    utility_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    risk_aversion: float = 1.0,
    road_class_counts: dict[str, int] | None = None,
    weather_profile: str | None = None,
    vehicle_type: str | None = None,
    vehicle_profile: VehicleProfile | None = None,
    corridor_bucket: str | None = None,
    scenario_mode: ScenarioMode | None = None,
    scenario_profile_version: str | None = None,
    scenario_sigma_multiplier: float = 1.0,
) -> tuple[dict[str, float] | None, dict[str, str | float | int | bool] | None]:
    sigma_multiplier = min(2.0, max(0.5, float(scenario_sigma_multiplier)))
    if not stochastic.enabled:
        base_duration = max(float(option.metrics.duration_s), 1e-6)
        base_monetary = max(float(option.metrics.monetary_cost), 0.0)
        base_emissions = max(float(option.metrics.emissions_kg), 0.0)
        weather_key = (weather_profile or "clear").strip().lower()
        road_bucket = _dominant_road_bucket(road_class_counts)
        day_kind = _day_kind_uk(departure_time_utc)
        regime_id, copula_id, calibration_version, calibration_as_of_utc, regime = resolve_stochastic_regime(
            departure_time_utc,
            road_class_counts=road_class_counts,
            weather_profile=weather_profile,
            corridor_bucket=corridor_bucket,
            vehicle_bucket=(
                str(vehicle_profile.stochastic_bucket)
                if vehicle_profile is not None
                else vehicle_type
            ),
        )
        calibration_as_of = calibration_as_of_utc or "unknown"
        prior = load_stochastic_residual_prior(
            day_kind=day_kind,
            road_bucket=road_bucket,
            weather_profile=weather_key,
            vehicle_type=vehicle_type,
            vehicle_bucket=(
                str(vehicle_profile.stochastic_bucket)
                if vehicle_profile is not None
                else vehicle_type
            ),
            corridor_bucket=corridor_bucket,
            local_time_slot=_local_time_slot_uk(departure_time_utc),
        )
        sigma_floor = min(
            0.35,
            max(0.02, float(prior.sigma_floor) * max(0.75, float(regime.sigma_scale))),
        )
        effective_sigma = min(0.6, max(0.0, sigma_floor * sigma_multiplier))
        deterministic_samples = int(max(24, min(196, max(int(prior.sample_count), (stochastic.samples * 2)))))
        deterministic = _deterministic_uncertainty_cached(
            base_duration_s=base_duration,
            base_monetary_cost=base_monetary,
            base_emissions_kg=base_emissions,
            base_distance_km=max(float(option.metrics.distance_km), 0.0),
            route_signature=route_signature,
            departure_time_iso=departure_time_utc.isoformat() if departure_time_utc else "",
            sigma=effective_sigma,
            samples=deterministic_samples,
            utility_weight_time=float(utility_weights[0]),
            utility_weight_money=float(utility_weights[1]),
            utility_weight_co2=float(utility_weights[2]),
            risk_aversion=float(risk_aversion),
            road_class_key=_road_class_key(road_class_counts),
            weather_profile=weather_profile or "clear",
            vehicle_type=vehicle_type or "",
            vehicle_bucket=(
                str(vehicle_profile.stochastic_bucket)
                if vehicle_profile is not None
                else (vehicle_type or "")
            ),
            corridor_bucket=corridor_bucket or "uk_default",
        )
        seed_material = f"{route_signature}|{departure_time_utc.isoformat() if departure_time_utc else 'none'}|deterministic|{deterministic_samples}|{sigma_floor:.6f}"
        seed_hash = hashlib.sha1(seed_material.encode("utf-8")).hexdigest()[:16]
        return (
            deterministic,
            {
                "sample_count": deterministic_samples,
                "seed": seed_hash,
                "sigma": round(effective_sigma, 6),
                "regime_id": regime_id,
                "copula_id": copula_id,
                "calibration_version": calibration_version,
                "calibration_as_of_utc": calibration_as_of,
                "as_of_utc": calibration_as_of,
                "prior_id": prior.prior_id,
                "prior_source": prior.source,
                "prior_sample_count": int(prior.sample_count),
                "seed_strategy": "route_signature+departure_slot+deterministic_seed",
                "stochastic_enabled": False,
                "deterministic_uncertainty_policy": "calibrated_residual_envelope",
                "scenario_mode": (
                    scenario_mode.value if scenario_mode is not None else ScenarioMode.NO_SHARING.value
                ),
                "scenario_profile_version": scenario_profile_version or "unknown",
                "scenario_sigma_multiplier": round(sigma_multiplier, 6),
                "utility_weight_time": float(utility_weights[0]),
                "utility_weight_money": float(utility_weights[1]),
                "utility_weight_co2": float(utility_weights[2]),
                "sample_count_requested": int(deterministic.get("sample_count_requested", deterministic_samples)),
                "sample_count_used": int(deterministic.get("sample_count_used", deterministic_samples)),
                "sample_count_clip_ratio": float(deterministic.get("sample_count_clip_ratio", 0.0)),
                "sigma_requested": float(deterministic.get("sigma_requested", effective_sigma)),
                "sigma_used": float(deterministic.get("sigma_used", effective_sigma)),
                "sigma_clip_ratio": float(deterministic.get("sigma_clip_ratio", 0.0)),
                "factor_clip_rate": float(deterministic.get("factor_clip_rate", 0.0)),
                "risk_family": str(settings.risk_family),
                "risk_family_theta": float(settings.risk_family_theta),
            },
        )

    effective_stochastic_sigma = min(0.6, max(0.0, float(stochastic.sigma) * sigma_multiplier))
    summary = compute_uncertainty_summary(
        base_duration_s=max(float(option.metrics.duration_s), 1e-6),
        base_monetary_cost=max(float(option.metrics.monetary_cost), 0.0),
        base_emissions_kg=max(float(option.metrics.emissions_kg), 0.0),
        base_distance_km=max(float(option.metrics.distance_km), 0.0),
        route_signature=route_signature,
        departure_time_utc=departure_time_utc,
        user_seed=stochastic.seed,
        sigma=effective_stochastic_sigma,
        samples=stochastic.samples,
        cvar_alpha=settings.risk_cvar_alpha,
        utility_weights=utility_weights,
        risk_aversion=risk_aversion,
        road_class_counts=road_class_counts,
        weather_profile=weather_profile,
        vehicle_type=vehicle_type,
        vehicle_bucket=(
            str(vehicle_profile.stochastic_bucket)
            if vehicle_profile is not None
            else vehicle_type
        ),
        corridor_bucket=corridor_bucket,
    )
    regime_id, copula_id, calibration_version, calibration_as_of_utc, _regime = resolve_stochastic_regime(
        departure_time_utc,
        road_class_counts=road_class_counts,
        weather_profile=weather_profile,
        corridor_bucket=corridor_bucket,
        vehicle_bucket=(
            str(vehicle_profile.stochastic_bucket)
            if vehicle_profile is not None
            else vehicle_type
        ),
    )
    calibration_as_of = calibration_as_of_utc or "unknown"
    seed_material = f"{route_signature}|{departure_time_utc.isoformat() if departure_time_utc else 'none'}|{stochastic.seed}|{stochastic.samples}|{stochastic.sigma}"
    seed_hash = hashlib.sha1(seed_material.encode("utf-8")).hexdigest()[:16]
    summary_dict_raw = summary.as_dict()
    summary_dict: dict[str, float] = {key: float(value) for key, value in summary_dict_raw.items()}
    objective_samples_json: str | None = None
    if summary.objective_samples:
        objective_samples_json = json.dumps(
            [
                [float(dur), float(mon), float(emi), float(util)]
                for dur, mon, emi, util in summary.objective_samples
            ],
            separators=(",", ":"),
            ensure_ascii=True,
        )
    quantile_invariants_ok = (
        summary_dict["q50_duration_s"] <= summary_dict["q90_duration_s"] <= summary_dict["q95_duration_s"] <= summary_dict["cvar95_duration_s"]
        and summary_dict["q50_monetary_cost"] <= summary_dict["q90_monetary_cost"] <= summary_dict["q95_monetary_cost"] <= summary_dict["cvar95_monetary_cost"]
        and summary_dict["q50_emissions_kg"] <= summary_dict["q90_emissions_kg"] <= summary_dict["q95_emissions_kg"] <= summary_dict["cvar95_emissions_kg"]
        and summary_dict["utility_q95"] <= summary_dict["utility_cvar95"]
    )
    return (
        summary_dict,
        {
            "sample_count": int(max(8, min(int(stochastic.samples), 600))),
            "seed": seed_hash,
            "sigma": effective_stochastic_sigma,
            "regime_id": regime_id,
            "copula_id": copula_id,
            "calibration_version": calibration_version,
            "calibration_as_of_utc": calibration_as_of,
            "as_of_utc": calibration_as_of,
            "seed_strategy": "route_signature+departure_slot+user_seed",
            "quantile_invariants_ok": quantile_invariants_ok,
            "stochastic_enabled": True,
            "scenario_mode": (
                scenario_mode.value if scenario_mode is not None else ScenarioMode.NO_SHARING.value
            ),
            "scenario_profile_version": scenario_profile_version or "unknown",
            "scenario_sigma_multiplier": round(sigma_multiplier, 6),
            "utility_weight_time": float(utility_weights[0]),
            "utility_weight_money": float(utility_weights[1]),
            "utility_weight_co2": float(utility_weights[2]),
            "sample_count_requested": int(summary_dict.get("sample_count_requested", int(stochastic.samples))),
            "sample_count_used": int(summary_dict.get("sample_count_used", max(8, min(int(stochastic.samples), 600)))),
            "sample_count_clip_ratio": float(summary_dict.get("sample_count_clip_ratio", 0.0)),
            "sigma_requested": float(summary_dict.get("sigma_requested", float(stochastic.sigma))),
            "sigma_used": float(summary_dict.get("sigma_used", effective_stochastic_sigma)),
            "sigma_clip_ratio": float(summary_dict.get("sigma_clip_ratio", 0.0)),
            "factor_clip_rate": float(summary_dict.get("factor_clip_rate", 0.0)),
            "risk_family": str(settings.risk_family),
            "risk_family_theta": float(settings.risk_family_theta),
            "objective_samples_json": objective_samples_json or "",
        },
    )


def _route_road_class_counts(route: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {"motorway": 0, "trunk": 0, "primary": 0, "secondary": 0, "local": 0}
    legs = route.get("legs", [])
    if not isinstance(legs, list):
        return counts
    for leg in legs:
        if not isinstance(leg, dict):
            continue
        steps = leg.get("steps", [])
        if not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, dict):
                continue
            classes = step.get("classes", [])
            class_set = {str(item).strip().lower() for item in classes} if isinstance(classes, list) else set()
            if "motorway" in class_set:
                counts["motorway"] += 1
            elif "trunk" in class_set:
                counts["trunk"] += 1
            elif "primary" in class_set:
                counts["primary"] += 1
            elif "secondary" in class_set:
                counts["secondary"] += 1
            else:
                counts["local"] += 1
    return counts


def _dominant_road_bucket(road_class_counts: dict[str, int] | None) -> str:
    if not road_class_counts:
        return "mixed"
    total = max(sum(max(0, int(v)) for v in road_class_counts.values()), 1)
    motorway_share = max(0, int(road_class_counts.get("motorway", 0))) / total
    trunk_share = max(0, int(road_class_counts.get("trunk", 0))) / total
    if motorway_share >= 0.55:
        return "motorway_heavy"
    if trunk_share >= 0.45:
        return "trunk_heavy"
    return "mixed"


def _day_kind_uk(departure_time_utc: datetime | None) -> str:
    if departure_time_utc is None:
        return "weekday"
    aware = departure_time_utc if departure_time_utc.tzinfo is not None else departure_time_utc.replace(
        tzinfo=UTC
    )
    local = aware.astimezone(UK_TZ)
    if local.weekday() >= 5:
        return "weekend"
    return "weekday"


def _local_time_slot_uk(departure_time_utc: datetime | None) -> str:
    if departure_time_utc is None:
        return "h12"
    aware = departure_time_utc if departure_time_utc.tzinfo is not None else departure_time_utc.replace(
        tzinfo=UTC
    )
    local = aware.astimezone(UK_TZ)
    return f"h{int(local.hour):02d}"


async def _scenario_context_from_od(
    *,
    origin: LatLng,
    destination: LatLng,
    vehicle_class: str,
    departure_time_utc: datetime | None,
    weather_bucket: str,
    road_class_counts: dict[str, int] | None = None,
    progress_cb: ProgressCallback | None = None,
) -> ScenarioRouteContext:
    od_road_counts = road_class_counts
    if (
        not od_road_counts
        and bool(settings.route_graph_enabled)
        and bool(settings.route_context_probe_enabled)
    ):
        probe_max_paths = max(1, int(settings.route_context_probe_max_paths))
        probe_timeout_ms = max(100, int(settings.route_context_probe_timeout_ms))
        probe_deadline_s = time.monotonic() + (float(probe_timeout_ms) / 1000.0)
        await _emit_progress(
            progress_cb,
            {
                "stage": "collecting_candidates",
                "stage_detail": "scenario_context_probe_start",
                "candidate_done": 0,
                "candidate_total": probe_max_paths,
            },
        )
        log_event(
            "scenario_context_probe_start",
            origin_lat=float(origin.lat),
            origin_lon=float(origin.lon),
            destination_lat=float(destination.lat),
            destination_lon=float(destination.lon),
            max_paths=probe_max_paths,
            timeout_ms=probe_timeout_ms,
        )
        try:
            graph_routes, _graph_diag = await asyncio.wait_for(
                _route_graph_candidate_routes_async(
                    origin_lat=float(origin.lat),
                    origin_lon=float(origin.lon),
                    destination_lat=float(destination.lat),
                    destination_lon=float(destination.lon),
                    max_paths=probe_max_paths,
                    scenario_edge_modifiers={
                        "mode": "full_sharing",
                        "duration_multiplier": 1.0,
                        "incident_rate_multiplier": 1.0,
                        "incident_delay_multiplier": 1.0,
                        "stochastic_sigma_multiplier": 1.0,
                        "traffic_pressure": 1.0,
                        "incident_pressure": 1.0,
                        "weather_pressure": 1.0,
                        "scenario_edge_scaling_version": "v3_live_transform",
                    },
                    max_hops_override=max(8, int(settings.route_context_probe_max_hops)),
                    max_state_budget_override=max(
                        1000,
                        int(settings.route_context_probe_max_state_budget),
                    ),
                    search_deadline_s=probe_deadline_s,
                ),
                timeout=float(probe_timeout_ms) / 1000.0,
            )
            route_count = len(graph_routes) if isinstance(graph_routes, list) else 0
            await _emit_progress(
                progress_cb,
                {
                    "stage": "collecting_candidates",
                    "stage_detail": "scenario_context_probe_complete",
                    "candidate_done": int(max(0, route_count)),
                    "candidate_total": probe_max_paths,
                },
            )
            log_event(
                "scenario_context_probe_complete",
                route_count=route_count,
                max_paths=probe_max_paths,
                timeout_ms=probe_timeout_ms,
            )
            if graph_routes:
                aggregate_counts: dict[str, float] = {}
                for rank, route in enumerate(graph_routes, start=1):
                    meta = route.get("_graph_meta", {})
                    raw_counts = meta.get("road_mix_counts", {})
                    if not isinstance(raw_counts, dict):
                        continue
                    # Weighted aggregate to avoid single-route context collapse.
                    rank_weight = 1.0 / max(1.0, float(rank))
                    for k, v in raw_counts.items():
                        key = str(k).strip()
                        if not key:
                            continue
                        aggregate_counts[key] = aggregate_counts.get(key, 0.0) + (
                            max(0.0, float(v)) * rank_weight
                        )
                if aggregate_counts:
                    od_road_counts = {
                        key: int(round(value))
                        for key, value in aggregate_counts.items()
                    }
        except asyncio.TimeoutError:
            od_road_counts = road_class_counts
            await _emit_progress(
                progress_cb,
                {
                    "stage": "collecting_candidates",
                    "stage_detail": "scenario_context_probe_timeout_fallback",
                    "candidate_done": 0,
                    "candidate_total": probe_max_paths,
                },
            )
            log_event(
                "scenario_context_probe_timeout_fallback",
                timeout_ms=probe_timeout_ms,
                max_paths=probe_max_paths,
                fallback="mixed_context",
            )
        except Exception as exc:
            od_road_counts = road_class_counts
            await _emit_progress(
                progress_cb,
                {
                    "stage": "collecting_candidates",
                    "stage_detail": "scenario_context_probe_error_fallback",
                    "candidate_done": 0,
                    "candidate_total": probe_max_paths,
                },
            )
            log_event(
                "scenario_context_probe_error_fallback",
                error_type=type(exc).__name__,
                error_message=str(exc).strip() or type(exc).__name__,
                max_paths=probe_max_paths,
                timeout_ms=probe_timeout_ms,
                fallback="mixed_context",
            )
    return build_scenario_route_context(
        route_points=[(float(origin.lat), float(origin.lon)), (float(destination.lat), float(destination.lon))],
        road_class_counts=od_road_counts,
        vehicle_class=vehicle_class,
        departure_time_utc=departure_time_utc,
        weather_bucket=weather_bucket,
    )


def _scenario_candidate_modifiers(
    *,
    scenario_mode: ScenarioMode,
    context: ScenarioRouteContext,
) -> dict[str, Any]:
    policy = resolve_scenario_profile(scenario_mode, context=context)
    weather_key = str(context.weather_regime or "clear").strip().lower()
    weather_regime_factor = {
        "clear": 1.00,
        "rain": 1.08,
        "fog": 1.06,
        "storm": 1.15,
        "snow": 1.18,
    }.get(weather_key, 1.03)
    hour = int(max(0, min(23, int(context.hour_slot_local))))
    if 7 <= hour <= 10 or 16 <= hour <= 19:
        hour_bucket_factor = 1.10
    elif 0 <= hour <= 5:
        hour_bucket_factor = 0.94
    else:
        hour_bucket_factor = 1.0
    duration_excess = max(0.0, float(policy.duration_multiplier) - 1.0)
    road_mix = context.road_mix_vector or {}
    road_class_factors: dict[str, float] = {}
    for road_class in (
        "motorway",
        "motorway_link",
        "trunk",
        "trunk_link",
        "primary",
        "primary_link",
        "secondary",
        "secondary_link",
        "tertiary",
        "tertiary_link",
        "unclassified",
        "residential",
    ):
        if road_class.startswith("motorway"):
            share = float(road_mix.get("motorway", 0.0))
        elif road_class.startswith("trunk"):
            share = float(road_mix.get("trunk", 0.0))
        elif road_class.startswith("primary"):
            share = float(road_mix.get("primary", 0.0))
        elif road_class.startswith("secondary") or road_class.startswith("tertiary"):
            share = float(road_mix.get("secondary", 0.0))
        else:
            share = float(road_mix.get("local", 0.0))
        road_class_factors[road_class] = max(
            0.75,
            min(1.55, 1.0 + (duration_excess * share * 0.42)),
        )
    return {
        "mode": scenario_mode.value,
        "duration_multiplier": float(policy.duration_multiplier),
        "incident_rate_multiplier": float(policy.incident_rate_multiplier),
        "incident_delay_multiplier": float(policy.incident_delay_multiplier),
        "stochastic_sigma_multiplier": float(policy.stochastic_sigma_multiplier),
        "traffic_pressure": float(policy.live_traffic_pressure),
        "incident_pressure": float(policy.live_incident_pressure),
        "weather_pressure": float(policy.live_weather_pressure),
        "weather_regime_factor": float(weather_regime_factor),
        "hour_bucket_factor": float(hour_bucket_factor),
        "road_class_factors": road_class_factors,
        "scenario_edge_scaling_version": str(policy.scenario_edge_scaling_version),
    }


async def _scenario_candidate_modifiers_async(
    *,
    scenario_mode: ScenarioMode,
    context: ScenarioRouteContext,
) -> dict[str, Any]:
    # Scenario policy resolution touches strict live feeds and is synchronous; keep it off the event loop.
    return await asyncio.to_thread(
        _scenario_candidate_modifiers,
        scenario_mode=scenario_mode,
        context=context,
    )


def _aggregate_route_segments(
    *,
    segment_distances_m: list[float],
    segment_durations_s: list[float],
    segment_grades: list[float] | None = None,
    max_segments: int,
) -> tuple[list[float], list[float], list[float]]:
    grades = list(segment_grades or [])
    if len(grades) < len(segment_distances_m):
        grades.extend([0.0] * (len(segment_distances_m) - len(grades)))
    if max_segments <= 0 or len(segment_distances_m) <= max_segments:
        return list(segment_distances_m), list(segment_durations_s), grades[: len(segment_distances_m)]
    bucket_size = max(1, int(math.ceil(len(segment_distances_m) / float(max_segments))))
    out_d: list[float] = []
    out_t: list[float] = []
    out_g: list[float] = []
    for start in range(0, len(segment_distances_m), bucket_size):
        end = min(len(segment_distances_m), start + bucket_size)
        dist_sum = sum(max(0.0, float(value)) for value in segment_distances_m[start:end])
        dur_sum = sum(max(0.0, float(value)) for value in segment_durations_s[start:end])
        weighted_grade = 0.0
        total_weight = 0.0
        for idx in range(start, end):
            weight = max(1.0, max(0.0, float(segment_distances_m[idx])))
            weighted_grade += float(grades[idx]) * weight
            total_weight += weight
        avg_grade = (weighted_grade / total_weight) if total_weight > 0 else 0.0
        out_d.append(dist_sum)
        out_t.append(dur_sum)
        out_g.append(avg_grade)
    return out_d, out_t, out_g


def build_option(
    route: dict[str, Any],
    *,
    option_id: str,
    vehicle_type: str,
    scenario_mode: ScenarioMode,
    cost_toggles: CostToggles,
    terrain_profile: TerrainProfile = "flat",
    stochastic: StochasticConfig | None = None,
    emissions_context: EmissionsContext | None = None,
    weather: WeatherImpactConfig | None = None,
    incident_simulation: IncidentSimulatorConfig | None = None,
    departure_time_utc: datetime | None = None,
    utility_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    risk_aversion: float = 1.0,
    scenario_policy_cache: dict[tuple[str, str], Any] | None = None,
    reset_terrain_route_run: bool = True,
) -> RouteOption:
    if reset_terrain_route_run:
        terrain_begin_route_run()
    vehicle = resolve_vehicle_profile(vehicle_type)
    ctx = emissions_context or EmissionsContext()
    weather_cfg = weather or WeatherImpactConfig()
    incident_cfg = incident_simulation or IncidentSimulatorConfig()
    if bool(settings.strict_live_data_required) and bool(incident_cfg.enabled):
        # Strict live runtime disallows synthetic incident generation.
        incident_cfg = incident_cfg.model_copy(update={"enabled": False})
    weather_speed = weather_speed_multiplier(weather_cfg)
    weather_incident = weather_incident_multiplier(weather_cfg)
    is_ev_mode = ctx.fuel_type == "ev" or vehicle.powertrain == "ev"
    ev_kwh_per_km = (
        float(vehicle.ev_kwh_per_km)
        if vehicle.ev_kwh_per_km is not None and vehicle.ev_kwh_per_km > 0
        else 1.25
    )
    grid_co2_kg_per_kwh = (
        float(vehicle.grid_co2_kg_per_kwh)
        if vehicle.grid_co2_kg_per_kwh is not None and vehicle.grid_co2_kg_per_kwh >= 0
        else 0.20
    )

    coords = _downsample_coords(_validate_osrm_geometry(route))
    raw_seg_d_m, raw_seg_t_s = extract_segment_annotations(route)
    seg_d_m = [max(0.0, float(seg)) for seg in raw_seg_d_m]
    seg_t_s = [max(0.0, float(seg)) for seg in raw_seg_t_s]
    try:
        route_signature = _route_signature(route)
    except OSRMError:
        route_signature = option_id

    distance_m = float(route.get("distance", 0.0))
    duration_s = float(route.get("duration", 0.0))

    if distance_m <= 0 or duration_s <= 0:
        # If OSRM omits these top-level fields, compute them from segments.
        distance_m = sum(seg_d_m)
        duration_s = sum(seg_t_s)

    distance_km = distance_m / 1000.0
    base_duration_s = duration_s
    route_points_lat_lon = [(lat, lon) for lon, lat in coords]
    route_centroid_lat = (
        sum(lat for lat, _lon in route_points_lat_lon) / len(route_points_lat_lon)
        if route_points_lat_lon
        else None
    )
    route_centroid_lon = (
        sum(lon for _lat, lon in route_points_lat_lon) / len(route_points_lat_lon)
        if route_points_lat_lon
        else None
    )
    road_class_counts = _route_road_class_counts(route)
    scenario_context = build_scenario_route_context(
        route_points=route_points_lat_lon,
        road_class_counts=road_class_counts,
        vehicle_class=str(vehicle.vehicle_class),
        departure_time_utc=departure_time_utc,
        weather_bucket=(weather_cfg.profile if weather_cfg.enabled else "clear"),
    )
    shared_policy_key = ("__shared__", scenario_mode.value)
    scenario_key = (scenario_mode.value, scenario_context.context_key)
    scenario_policy = None
    if scenario_policy_cache is not None:
        scenario_policy = scenario_policy_cache.get(shared_policy_key)
        if scenario_policy is None:
            scenario_policy = scenario_policy_cache.get(scenario_key)
    if scenario_policy is None:
        scenario_policy = resolve_scenario_profile(
            scenario_mode,
            context=scenario_context,
        )
        if scenario_policy_cache is not None:
            scenario_policy_cache[scenario_key] = scenario_policy
    scenario_multiplier = float(scenario_policy.duration_multiplier)
    scenario_incident_rate_multiplier = float(scenario_policy.incident_rate_multiplier)
    scenario_incident_delay_multiplier = float(scenario_policy.incident_delay_multiplier)
    scenario_fuel_multiplier = float(scenario_policy.fuel_consumption_multiplier)
    scenario_emissions_multiplier = float(scenario_policy.emissions_multiplier)
    scenario_sigma_multiplier = float(scenario_policy.stochastic_sigma_multiplier)
    departure_multiplier = time_of_day_multiplier_uk(
        departure_time_utc,
        route_points=route_points_lat_lon,
        road_class_counts=road_class_counts,
    )
    tod_multiplier = departure_multiplier.multiplier
    duration_after_tod_s = base_duration_s * tod_multiplier
    duration_after_scenario_s = duration_after_tod_s * scenario_multiplier
    duration_after_weather_s = duration_after_scenario_s * weather_speed
    avg_base_speed_kmh = distance_km / (base_duration_s / 3600.0) if base_duration_s > 0 else 0.0
    terrain_summary = estimate_terrain_summary(
        coordinates_lon_lat=coords,
        terrain_profile=terrain_profile,
        avg_speed_kmh=avg_base_speed_kmh,
        distance_km=distance_km,
        vehicle_type=vehicle_type,
        vehicle_profile=vehicle,
    )
    terrain_confidence = max(0.0, min(1.0, float(terrain_summary.confidence)))
    # Low DEM confidence widens downstream cost/emissions uncertainty to avoid
    # overconfident outputs on weak elevation coverage.
    terrain_uncertainty_scale = 1.0 + ((1.0 - terrain_confidence) * 0.35)
    gradient_duration_multiplier = terrain_summary.duration_multiplier
    gradient_emissions_multiplier = terrain_summary.emissions_multiplier
    segment_grades = segment_grade_profile(
        coordinates_lon_lat=coords,
        segment_distances_m=seg_d_m,
    )
    max_segments = max(32, int(settings.route_option_segment_cap))
    long_threshold_km = max(20.0, float(settings.route_option_long_distance_threshold_km))
    if distance_km >= long_threshold_km:
        max_segments = min(
            max_segments,
            max(32, int(settings.route_option_segment_cap_long)),
        )
    seg_d_m, seg_t_s, segment_grades = _aggregate_route_segments(
        segment_distances_m=seg_d_m,
        segment_durations_s=seg_t_s,
        segment_grades=segment_grades,
        max_segments=max_segments,
    )
    non_terrain_multiplier = tod_multiplier * scenario_multiplier * weather_speed
    terrain_vehicle_params = params_for_vehicle(vehicle)
    raw_terrain_multipliers: list[float] = []
    for idx, (d_m, t_s) in enumerate(zip(seg_d_m, seg_t_s, strict=True)):
        base_speed_kmh = (d_m / t_s) * 3.6 if t_s > 0 else max(8.0, avg_base_speed_kmh)
        seg_grade = segment_grades[idx] if idx < len(segment_grades) else 0.0
        raw_mult = segment_duration_multiplier(
            grade=seg_grade,
            speed_kmh=base_speed_kmh,
            terrain_profile=terrain_profile,
            params=terrain_vehicle_params,
        )
        raw_terrain_multipliers.append(raw_mult)
    if raw_terrain_multipliers:
        weighted_raw = 0.0
        weighted_base = 0.0
        for idx, raw_mult in enumerate(raw_terrain_multipliers):
            base_seg = max(float(seg_t_s[idx]), 0.0)
            weighted_raw += raw_mult * base_seg
            weighted_base += base_seg
        weighted_raw_avg = weighted_raw / max(1e-6, weighted_base)
        scale = gradient_duration_multiplier / max(1e-6, weighted_raw_avg)
        scale = min(1.5, max(0.7, scale))
        terrain_duration_multipliers = [
            min(1.85, max(0.75, raw_mult * scale)) for raw_mult in raw_terrain_multipliers
        ]
    else:
        terrain_duration_multipliers = raw_terrain_multipliers

    pre_incident_segment_durations_s = [
        max(float(base_seg), 0.0)
        * non_terrain_multiplier
        * (terrain_duration_multipliers[idx] if idx < len(terrain_duration_multipliers) else 1.0)
        for idx, base_seg in enumerate(seg_t_s)
    ]
    duration_after_gradient_s = sum(pre_incident_segment_durations_s)
    duration_s = duration_after_gradient_s
    if duration_after_weather_s > 0:
        gradient_duration_multiplier = duration_after_gradient_s / duration_after_weather_s
    else:
        gradient_duration_multiplier = 1.0
    route_key = option_id
    if incident_cfg.enabled:
        try:
            route_key = _route_signature(route)
        except OSRMError:
            route_key = option_id
    incident_config = incident_cfg.model_copy(
        update={
            "dwell_rate_per_100km": float(incident_cfg.dwell_rate_per_100km)
            * scenario_incident_rate_multiplier,
            "accident_rate_per_100km": float(incident_cfg.accident_rate_per_100km)
            * scenario_incident_rate_multiplier,
            "closure_rate_per_100km": float(incident_cfg.closure_rate_per_100km)
            * scenario_incident_rate_multiplier,
            "dwell_delay_s": float(incident_cfg.dwell_delay_s) * scenario_incident_delay_multiplier,
            "accident_delay_s": float(incident_cfg.accident_delay_s) * scenario_incident_delay_multiplier,
            "closure_delay_s": float(incident_cfg.closure_delay_s) * scenario_incident_delay_multiplier,
        }
    )
    incident_events = simulate_incident_events(
        config=incident_config,
        segment_distances_m=[float(seg) for seg in seg_d_m],
        segment_durations_s=pre_incident_segment_durations_s,
        weather_incident_multiplier=weather_incident,
        route_key=route_key,
    )
    incident_delay_by_segment: dict[int, float] = {}
    for event in incident_events:
        incident_delay_by_segment[event.segment_index] = (
            incident_delay_by_segment.get(event.segment_index, 0.0) + float(event.delay_s)
        )
    total_incident_delay_s = sum(incident_delay_by_segment.values())

    time_of_day_delta_s = max(duration_after_tod_s - base_duration_s, 0.0)
    scenario_delta_s = max(duration_after_scenario_s - duration_after_tod_s, 0.0)
    weather_delta_s = max(duration_after_weather_s - duration_after_scenario_s, 0.0)
    gradient_delta_s = max(duration_after_gradient_s - duration_after_weather_s, 0.0)
    fuel_multiplier = max(cost_toggles.fuel_price_multiplier, 0.0)
    carbon_context = resolve_carbon_price(
        request_price_per_kg=max(cost_toggles.carbon_price_per_kg, 0.0),
        departure_time_utc=departure_time_utc,
    )
    carbon_price = carbon_context.price_per_kg

    toll_result = compute_toll_cost(
        route=route,
        distance_km=distance_km,
        vehicle_type=vehicle_type,
        vehicle_profile=vehicle,
        departure_time_utc=departure_time_utc,
        use_tolls=cost_toggles.use_tolls,
        fallback_toll_cost_per_km=cost_toggles.toll_cost_per_km,
    )
    contains_toll = toll_result.contains_toll
    toll_distance_km = toll_result.toll_distance_km
    toll_component_total = toll_result.toll_cost_gbp
    legacy_contains_toll_hint = bool(route.get("contains_toll", False))
    strict_toll_pricing_required = True
    if (
        (strict_toll_pricing_required or legacy_contains_toll_hint)
        and toll_result.contains_toll
        and bool(toll_result.details.get("pricing_unresolved", False))
    ):
        raise ValueError("tolled route classification succeeded but tariff resolution is unavailable")

    segment_breakdown: list[dict[str, float | int]] = []
    total_emissions = 0.0
    total_energy_kwh = 0.0
    total_monetary = 0.0
    total_distance_km = 0.0
    total_duration_s = 0.0
    total_fuel_cost = 0.0
    total_fuel_liters = 0.0
    total_fuel_liters_p10 = 0.0
    total_fuel_liters_p50 = 0.0
    total_fuel_liters_p90 = 0.0
    total_fuel_cost_p10 = 0.0
    total_fuel_cost_p50 = 0.0
    total_fuel_cost_p90 = 0.0
    total_fuel_cost_uncertainty_low = 0.0
    total_fuel_cost_uncertainty_high = 0.0
    total_emissions_uncertainty_low = 0.0
    total_emissions_uncertainty_high = 0.0
    total_time_cost = 0.0
    total_toll_cost = 0.0
    total_carbon_cost = 0.0
    elapsed_route_s = 0.0
    fuel_price_source: str | None = None
    fuel_price_as_of: str | None = None
    tod_bucket_s = max(0, int(settings.route_option_tod_bucket_s))
    tod_ratio_cache: dict[int, float] = {}
    energy_speed_bin_kph = max(0.0, float(settings.route_option_energy_speed_bin_kph))
    energy_grade_bin_pct = max(0.0, float(settings.route_option_energy_grade_bin_pct))
    energy_rate_cache: dict[tuple[float, float], Any] = {}

    def _quantize_energy_key(speed_kmh: float, grade_ratio: float) -> tuple[float, float]:
        speed_value = max(1.0, float(speed_kmh))
        grade_pct = float(grade_ratio) * 100.0
        speed_key = (
            round(speed_value / energy_speed_bin_kph) * energy_speed_bin_kph
            if energy_speed_bin_kph > 0.0
            else round(speed_value, 3)
        )
        grade_key_pct = (
            round(grade_pct / energy_grade_bin_pct) * energy_grade_bin_pct
            if energy_grade_bin_pct > 0.0
            else round(grade_pct, 4)
        )
        return max(1.0, float(speed_key)), float(grade_key_pct) / 100.0

    for idx, (d_m, t_s) in enumerate(zip(seg_d_m, seg_t_s, strict=True)):
        d_m = max(float(d_m), 0.0)
        t_s = max(float(t_s), 0.0)

        seg_distance_km = d_m / 1000.0
        total_distance_km += seg_distance_km

        base_speed_kmh = (d_m / t_s) * 3.6 if t_s > 0 else 0.0
        base_pre_incident_s = (
            pre_incident_segment_durations_s[idx]
            if idx < len(pre_incident_segment_durations_s)
            else max(float(t_s), 0.0) * non_terrain_multiplier
        )
        seg_duration_s = base_pre_incident_s
        if departure_time_utc is not None:
            bucket_key = int(max(0.0, elapsed_route_s) // tod_bucket_s) if tod_bucket_s > 0 else idx
            tod_ratio = tod_ratio_cache.get(bucket_key)
            if tod_ratio is None:
                seg_departure_utc = departure_time_utc + timedelta(seconds=max(0.0, elapsed_route_s))
                seg_dep = time_of_day_multiplier_uk(
                    seg_departure_utc,
                    route_points=route_points_lat_lon,
                    road_class_counts=road_class_counts,
                )
                tod_ratio = min(1.35, max(0.75, seg_dep.multiplier / max(tod_multiplier, 1e-6)))
                tod_ratio_cache[bucket_key] = tod_ratio
            seg_duration_s *= float(tod_ratio)
        seg_grade = segment_grades[idx] if idx < len(segment_grades) else 0.0
        seg_incident_delay_s = incident_delay_by_segment.get(idx, 0.0)
        seg_duration_s += seg_incident_delay_s
        elapsed_route_s += seg_duration_s
        seg_energy_kwh = 0.0
        seg_fuel_liters = 0.0
        seg_fuel_liters_p10 = 0.0
        seg_fuel_liters_p50 = 0.0
        seg_fuel_liters_p90 = 0.0
        seg_fuel_cost_p10 = 0.0
        seg_fuel_cost_p50 = 0.0
        seg_fuel_cost_p90 = 0.0
        seg_fuel_cost_low = 0.0
        seg_fuel_cost_high = 0.0
        seg_emissions_low = 0.0
        seg_emissions_high = 0.0
        seg_emissions_raw = 0.0
        seg_fuel_cost = 0.0
        if seg_distance_km > 0.0:
            seg_speed_kmh = seg_distance_km / (seg_duration_s / 3600.0) if seg_duration_s > 0 else max(
                8.0, avg_base_speed_kmh
            )
            speed_key_kph, grade_key_ratio = _quantize_energy_key(seg_speed_kmh, seg_grade)
            energy_rate_key = (speed_key_kph, grade_key_ratio)
            energy_rate = energy_rate_cache.get(energy_rate_key)
            if energy_rate is None:
                energy_rate = segment_energy_and_emissions(
                    vehicle=vehicle,
                    emissions_context=ctx,
                    distance_km=1.0,
                    duration_s=(3600.0 / max(1.0, speed_key_kph)),
                    grade=grade_key_ratio,
                    fuel_price_multiplier=fuel_multiplier,
                    departure_time_utc=departure_time_utc,
                    route_centroid_lat=route_centroid_lat,
                    route_centroid_lon=route_centroid_lon,
                )
                energy_rate_cache[energy_rate_key] = energy_rate
            seg_scale = seg_distance_km * scenario_fuel_multiplier
            seg_fuel_cost = float(energy_rate.fuel_cost_gbp) * seg_scale
            seg_fuel_liters_p10 = float(energy_rate.fuel_liters_p10) * seg_scale
            seg_fuel_liters_p50 = float(energy_rate.fuel_liters_p50) * seg_scale
            seg_fuel_liters_p90 = float(energy_rate.fuel_liters_p90) * seg_scale
            seg_fuel_cost_p10 = float(energy_rate.fuel_cost_p10_gbp) * seg_scale
            seg_fuel_cost_p50 = float(energy_rate.fuel_cost_p50_gbp) * seg_scale
            seg_fuel_cost_p90 = float(energy_rate.fuel_cost_p90_gbp) * seg_scale
            seg_fuel_cost_low = float(energy_rate.fuel_cost_uncertainty_low_gbp) * seg_scale
            seg_fuel_cost_high = float(energy_rate.fuel_cost_uncertainty_high_gbp) * seg_scale
            seg_energy_kwh = float(energy_rate.energy_kwh) * seg_scale
            seg_emissions_raw = float(energy_rate.emissions_kg) * seg_scale
            seg_emissions = seg_emissions_raw
            seg_emissions_low = float(energy_rate.emissions_uncertainty_low_kg) * seg_scale
            seg_emissions_high = float(energy_rate.emissions_uncertainty_high_kg) * seg_scale
            seg_fuel_liters = float(energy_rate.fuel_liters) * seg_scale
            if fuel_price_source is None and energy_rate.price_source:
                fuel_price_source = energy_rate.price_source
            if fuel_price_as_of is None and energy_rate.price_as_of:
                fuel_price_as_of = energy_rate.price_as_of
        else:
            seg_emissions = 0.0
        seg_emissions = apply_scope_emissions_adjustment(
            emissions_kg=seg_emissions,
            is_ev_mode=is_ev_mode,
            scope_mode=carbon_context.scope_mode,
            departure_time_utc=departure_time_utc,
            route_centroid_lat=route_centroid_lat,
            route_centroid_lon=route_centroid_lon,
        )
        seg_emissions *= max(0.0, float(gradient_emissions_multiplier))
        seg_emissions *= scenario_emissions_multiplier
        if seg_emissions_raw > 1e-9:
            scope_scale = seg_emissions / max(1e-9, seg_emissions_raw)
            seg_emissions_low *= scope_scale
            seg_emissions_high *= scope_scale
        if terrain_uncertainty_scale > 1.0:
            fuel_low_span = max(0.0, seg_fuel_cost - seg_fuel_cost_low)
            fuel_high_span = max(0.0, seg_fuel_cost_high - seg_fuel_cost)
            seg_fuel_cost_low = max(0.0, seg_fuel_cost - (fuel_low_span * terrain_uncertainty_scale))
            seg_fuel_cost_high = max(seg_fuel_cost_low, seg_fuel_cost + (fuel_high_span * terrain_uncertainty_scale))
            em_low_span = max(0.0, seg_emissions - seg_emissions_low)
            em_high_span = max(0.0, seg_emissions_high - seg_emissions)
            seg_emissions_low = max(0.0, seg_emissions - (em_low_span * terrain_uncertainty_scale))
            seg_emissions_high = max(seg_emissions_low, seg_emissions + (em_high_span * terrain_uncertainty_scale))

        toll_share = (seg_distance_km / max(distance_km, 1e-6)) if distance_km > 0 else 0.0
        seg_toll_cost = toll_component_total * toll_share
        seg_time_cost = (seg_duration_s / 3600.0) * vehicle.cost_per_hour * DRIVER_TIME_COST_WEIGHT
        seg_carbon_cost = seg_emissions * carbon_price
        seg_monetary = seg_fuel_cost + seg_time_cost + seg_toll_cost + seg_carbon_cost

        seg_speed_kmh = seg_distance_km / (seg_duration_s / 3600.0) if seg_duration_s > 0 else 0.0
        seg_grade_pct = seg_grade * 100.0
        segment_breakdown.append(
            {
                "segment_index": idx,
                "distance_km": round(seg_distance_km, 6),
                "duration_s": round(seg_duration_s, 6),
                "incident_delay_s": round(seg_incident_delay_s, 6),
                "avg_speed_kmh": round(seg_speed_kmh, 6),
                "emissions_kg": round(seg_emissions, 6),
                "monetary_cost": round(seg_monetary, 6),
                "time_cost": round(seg_time_cost, 6),
                "toll_cost": round(seg_toll_cost, 6),
                "fuel_cost": round(seg_fuel_cost, 6),
                "fuel_liters_p10": round(seg_fuel_liters_p10, 6),
                "fuel_liters_p50": round(seg_fuel_liters_p50, 6),
                "fuel_liters_p90": round(seg_fuel_liters_p90, 6),
                "fuel_cost_p10_gbp": round(seg_fuel_cost_p10, 6),
                "fuel_cost_p50_gbp": round(seg_fuel_cost_p50, 6),
                "fuel_cost_p90_gbp": round(seg_fuel_cost_p90, 6),
                "fuel_cost_uncertainty_low": round(seg_fuel_cost_low, 6),
                "fuel_cost_uncertainty_high": round(seg_fuel_cost_high, 6),
                "carbon_cost": round(seg_carbon_cost, 6),
                "energy_kwh": round(seg_energy_kwh, 6),
                "fuel_liters": round(seg_fuel_liters, 6),
                "grade_pct": round(seg_grade_pct, 6),
                "terrain_confidence": round(terrain_confidence, 6),
                "terrain_uncertainty_scale": round(terrain_uncertainty_scale, 6),
                "emissions_uncertainty_low_kg": round(max(0.0, seg_emissions_low), 6),
                "emissions_uncertainty_high_kg": round(max(max(0.0, seg_emissions_low), seg_emissions_high), 6),
            }
        )

        total_duration_s += seg_duration_s
        total_emissions += seg_emissions
        total_energy_kwh += seg_energy_kwh
        total_monetary += seg_monetary
        total_fuel_cost += seg_fuel_cost
        total_fuel_liters += max(0.0, seg_fuel_liters)
        total_fuel_liters_p10 += max(0.0, seg_fuel_liters_p10)
        total_fuel_liters_p50 += max(max(0.0, seg_fuel_liters_p10), seg_fuel_liters_p50)
        total_fuel_liters_p90 += max(max(0.0, seg_fuel_liters_p50), seg_fuel_liters_p90)
        total_fuel_cost_p10 += max(0.0, seg_fuel_cost_p10)
        total_fuel_cost_p50 += max(max(0.0, seg_fuel_cost_p10), seg_fuel_cost_p50)
        total_fuel_cost_p90 += max(max(0.0, seg_fuel_cost_p50), seg_fuel_cost_p90)
        total_fuel_cost_uncertainty_low += max(0.0, seg_fuel_cost_low)
        total_fuel_cost_uncertainty_high += max(max(0.0, seg_fuel_cost_low), seg_fuel_cost_high)
        total_emissions_uncertainty_low += max(0.0, seg_emissions_low)
        total_emissions_uncertainty_high += max(max(0.0, seg_emissions_low), seg_emissions_high)
        total_time_cost += seg_time_cost
        total_toll_cost += seg_toll_cost
        total_carbon_cost += seg_carbon_cost

    avg_speed_kmh = total_distance_km / (total_duration_s / 3600.0) if total_duration_s > 0 else 0.0

    eta_explanations: list[str] = [f"Baseline ETA {base_duration_s / 60.0:.1f} min."]
    if time_of_day_delta_s > 0.5:
        eta_explanations.append(
            f"Time-of-day profile added {time_of_day_delta_s / 60.0:.1f} min "
            f"(x{tod_multiplier:.2f})."
        )
    else:
        eta_explanations.append("Time-of-day profile added no material delay.")
    if departure_multiplier is not None and departure_multiplier.local_time_iso is not None:
        eta_explanations.append(
            "UK local-time profile "
            f"({departure_multiplier.profile_day}, {departure_multiplier.local_time_iso}) applied."
        )
    if scenario_delta_s > 0.5:
        eta_explanations.append(
            f"Scenario mode added {scenario_delta_s / 60.0:.1f} min "
            f"(x{scenario_multiplier:.2f})."
        )
    else:
        eta_explanations.append("Scenario mode added no material delay.")
    if weather_cfg.enabled:
        if weather_delta_s > 0.5:
            eta_explanations.append(
                f"Weather profile '{weather_cfg.profile}' added {weather_delta_s / 60.0:.1f} min "
                f"(x{weather_speed:.2f})."
            )
        else:
            eta_explanations.append(
                f"Weather profile '{weather_cfg.profile}' added no material delay."
            )
    if incident_cfg.enabled:
        if total_incident_delay_s > 0.5:
            eta_explanations.append(
                f"Synthetic incidents added {total_incident_delay_s / 60.0:.1f} min "
                f"across {len(incident_events)} events."
            )
        else:
            eta_explanations.append("Synthetic incidents added no material delay.")
    if gradient_delta_s > 0.5:
        eta_explanations.append(
            f"Terrain profile '{terrain_profile}' added {gradient_delta_s / 60.0:.1f} min "
            f"(x{gradient_duration_multiplier:.2f})."
        )
    else:
        eta_explanations.append(f"Terrain profile '{terrain_profile}' added no material delay.")
    if terrain_summary is not None:
        eta_explanations.append(
            f"Terrain model ({terrain_summary.source}, cov={terrain_summary.coverage_ratio:.2f}, "
            f"v={terrain_summary.version}) estimated ascent {terrain_summary.ascent_m:.0f} m and "
            f"descent {terrain_summary.descent_m:.0f} m."
        )
    if is_ev_mode:
        eta_explanations.append(
            f"EV energy model active ({ev_kwh_per_km:.2f} kWh/km, grid {grid_co2_kg_per_kwh:.2f} kg/kWh)."
        )
    elif ctx.fuel_type != "diesel" or ctx.euro_class != "euro6" or abs(ctx.ambient_temp_c - 15.0) > 1e-6:
        eta_explanations.append(
            "Emissions context adjusted for "
            f"fuel={ctx.fuel_type}, euro={ctx.euro_class}, temp={ctx.ambient_temp_c:.1f}C."
        )
    eta_explanations.append(
        f"Toll engine classified {'tolled' if contains_toll else 'toll-free'} route "
        f"(distance {toll_distance_km:.2f} km)."
    )

    eta_timeline: list[dict[str, float | str]] = [
        {"stage": "baseline", "duration_s": round(base_duration_s, 2), "delta_s": 0.0},
        {
            "stage": "time_of_day",
            "duration_s": round(duration_after_tod_s, 2),
            "delta_s": round(time_of_day_delta_s, 2),
            "multiplier": round(tod_multiplier, 3),
            "profile_day": (
                departure_multiplier.profile_day
                if departure_multiplier is not None
                else "legacy"
            ),
        },
        {
            "stage": "scenario",
            "duration_s": round(duration_after_scenario_s, 2),
            "delta_s": round(scenario_delta_s, 2),
            "multiplier": round(scenario_multiplier, 3),
        },
    ]
    if weather_cfg.enabled:
        eta_timeline.append(
            {
                "stage": "weather",
                "duration_s": round(duration_after_weather_s, 2),
                "delta_s": round(weather_delta_s, 2),
                "multiplier": round(weather_speed, 3),
                "profile": weather_cfg.profile,
            }
        )
    gradient_stage_duration_s = max(total_duration_s - total_incident_delay_s, 0.0)
    gradient_stage_delta_s = max(gradient_stage_duration_s - duration_after_weather_s, 0.0)
    eta_timeline.append(
        {
            "stage": "gradient",
            "duration_s": round(gradient_stage_duration_s, 2),
            "delta_s": round(gradient_stage_delta_s, 2),
            "multiplier": round(gradient_duration_multiplier, 3),
            "profile": terrain_profile,
            "source": terrain_summary.source if terrain_summary is not None else "gradient_profile",
        }
    )
    if incident_cfg.enabled:
        eta_timeline.append(
            {
                "stage": "incidents",
                "duration_s": round(total_duration_s, 2),
                "delta_s": round(total_incident_delay_s, 2),
                "event_count": len(incident_events),
            }
        )

    shifted_departure_duration = total_duration_s
    if departure_time_utc is not None:
        shifted_time = departure_time_utc + timedelta(hours=2)
        shifted_tod_multiplier = time_of_day_multiplier_uk(shifted_time).multiplier
        shifted_after_tod = base_duration_s * shifted_tod_multiplier
        shifted_after_scenario = shifted_after_tod * scenario_multiplier
        shifted_after_weather = shifted_after_scenario * weather_speed
        shifted_departure_duration = shifted_after_weather * gradient_duration_multiplier

    shifted_mode = _counterfactual_shift_scenario(scenario_mode)
    shifted_key = (shifted_mode.value, scenario_context.context_key)
    shifted_shared_key = ("__shared__", shifted_mode.value)
    shifted_mode_policy = None
    if scenario_policy_cache is not None:
        shifted_mode_policy = scenario_policy_cache.get(shifted_shared_key)
        if shifted_mode_policy is None:
            shifted_mode_policy = scenario_policy_cache.get(shifted_key)
    if shifted_mode_policy is None:
        shifted_mode_policy = resolve_scenario_profile(shifted_mode, context=scenario_context)
        if scenario_policy_cache is not None:
            scenario_policy_cache.setdefault(shifted_shared_key, shifted_mode_policy)
            scenario_policy_cache[shifted_key] = shifted_mode_policy
    shifted_mode_duration = (
        duration_after_tod_s
        * float(shifted_mode_policy.duration_multiplier)
        * weather_speed
        * gradient_duration_multiplier
    )
    duration_ratio = shifted_mode_duration / total_duration_s if total_duration_s > 0 else 1.0
    emissions_policy_ratio = (
        (float(shifted_mode_policy.fuel_consumption_multiplier) * float(shifted_mode_policy.emissions_multiplier))
        / max(1e-9, scenario_fuel_multiplier * scenario_emissions_multiplier)
    )
    shifted_mode_emissions = total_emissions * duration_ratio * emissions_policy_ratio
    shifted_mode_time_cost = total_time_cost * duration_ratio
    fuel_policy_ratio = float(shifted_mode_policy.fuel_consumption_multiplier) / max(
        1e-9, scenario_fuel_multiplier
    )
    shifted_mode_fuel_cost = total_fuel_cost * fuel_policy_ratio
    shifted_mode_monetary = (
        shifted_mode_fuel_cost
        + shifted_mode_time_cost
        + total_toll_cost
        + (shifted_mode_emissions * carbon_price)
    )

    counterfactuals: list[dict[str, str | float | bool]] = [
        {
            "id": "fuel_plus_10pct",
            "label": "Fuel +10%",
            "metric": "monetary_cost",
            "baseline": round(total_monetary, 4),
            "counterfactual": round(total_monetary + (total_fuel_cost * 0.10), 4),
            "delta": round(total_fuel_cost * 0.10, 4),
            "improves": False,
        },
        {
            "id": "carbon_price_plus_0_10",
            "label": "Carbon price +0.10/kg",
            "metric": "monetary_cost",
            "baseline": round(total_monetary, 4),
            "counterfactual": round(total_monetary + (total_emissions * 0.10), 4),
            "delta": round(total_emissions * 0.10, 4),
            "improves": False,
        },
        {
            "id": "departure_plus_2h",
            "label": "Departure +2h",
            "metric": "duration_s",
            "baseline": round(total_duration_s, 4),
            "counterfactual": round(shifted_departure_duration, 4),
            "delta": round(shifted_departure_duration - total_duration_s, 4),
            "improves": shifted_departure_duration < total_duration_s,
        },
        {
            "id": "scenario_shift",
            "label": f"Scenario shift to {shifted_mode.value}",
            "metric": "duration_s",
            "baseline": round(total_duration_s, 4),
            "counterfactual": round(shifted_mode_duration, 4),
            "delta": round(shifted_mode_duration - total_duration_s, 4),
            "improves": shifted_mode_duration < total_duration_s,
            "emissions_delta": round(shifted_mode_emissions - total_emissions, 4),
            "monetary_delta": round(shifted_mode_monetary - total_monetary, 4),
        },
    ]

    option_weather_summary = weather_summary(weather_cfg) if weather_cfg.enabled else {}
    scenario_summary = ScenarioSummary(
        mode=scenario_mode,
        context_key=scenario_policy.context_key,
        duration_multiplier=round(scenario_multiplier, 6),
        incident_rate_multiplier=round(scenario_incident_rate_multiplier, 6),
        incident_delay_multiplier=round(scenario_incident_delay_multiplier, 6),
        fuel_consumption_multiplier=round(scenario_fuel_multiplier, 6),
        emissions_multiplier=round(scenario_emissions_multiplier, 6),
        stochastic_sigma_multiplier=round(scenario_sigma_multiplier, 6),
        source=scenario_policy.source,
        version=scenario_policy.version,
        calibration_basis=scenario_policy.calibration_basis,
        as_of_utc=scenario_policy.as_of_utc,
        live_as_of_utc=scenario_policy.live_as_of_utc,
        live_sources=(
            "|".join(sorted(scenario_policy.live_source_set.keys()))
            if scenario_policy.live_source_set
            else None
        ),
        live_coverage_overall=(
            round(float(scenario_policy.live_coverage.get("overall", 0.0)), 6)
            if scenario_policy.live_coverage
            else None
        ),
        live_traffic_pressure=round(float(scenario_policy.live_traffic_pressure), 6),
        live_incident_pressure=round(float(scenario_policy.live_incident_pressure), 6),
        live_weather_pressure=round(float(scenario_policy.live_weather_pressure), 6),
        scenario_edge_scaling_version=str(scenario_policy.scenario_edge_scaling_version),
        mode_observation_source=scenario_policy.mode_observation_source,
        mode_projection_ratio=(
            round(float(scenario_policy.mode_projection_ratio), 6)
            if scenario_policy.mode_projection_ratio is not None
            else None
        ),
    )
    if option_weather_summary is not None:
        option_weather_summary["weather_delay_s"] = round(weather_delta_s, 6)
        option_weather_summary["incident_rate_multiplier"] = round(weather_incident, 6)
        option_weather_summary["scenario_mode"] = scenario_mode.value
        option_weather_summary["scenario_profile_source"] = scenario_policy.source
        option_weather_summary["scenario_profile_version"] = scenario_policy.version
        option_weather_summary["scenario_context_key"] = scenario_policy.context_key
        option_weather_summary["scenario_duration_multiplier"] = round(scenario_multiplier, 6)
        option_weather_summary["scenario_incident_rate_multiplier"] = round(
            scenario_incident_rate_multiplier, 6
        )
        option_weather_summary["scenario_incident_delay_multiplier"] = round(
            scenario_incident_delay_multiplier, 6
        )
        option_weather_summary["scenario_fuel_consumption_multiplier"] = round(
            scenario_fuel_multiplier, 6
        )
        option_weather_summary["scenario_emissions_multiplier"] = round(
            scenario_emissions_multiplier, 6
        )
        option_weather_summary["scenario_sigma_multiplier"] = round(
            scenario_sigma_multiplier, 6
        )
        if scenario_policy.as_of_utc is not None:
            option_weather_summary["scenario_profile_as_of_utc"] = scenario_policy.as_of_utc
        if scenario_policy.live_as_of_utc is not None:
            option_weather_summary["scenario_live_as_of_utc"] = scenario_policy.live_as_of_utc
        option_weather_summary["scenario_live_traffic_pressure"] = round(
            float(scenario_policy.live_traffic_pressure),
            6,
        )
        option_weather_summary["scenario_live_incident_pressure"] = round(
            float(scenario_policy.live_incident_pressure),
            6,
        )
        option_weather_summary["scenario_live_weather_pressure"] = round(
            float(scenario_policy.live_weather_pressure),
            6,
        )
        if scenario_policy.mode_observation_source is not None:
            option_weather_summary["scenario_mode_observation_source"] = str(
                scenario_policy.mode_observation_source
            )
        if scenario_policy.mode_projection_ratio is not None:
            option_weather_summary["scenario_mode_projection_ratio"] = round(
                float(scenario_policy.mode_projection_ratio),
                6,
            )
    if option_weather_summary is not None and toll_result is not None:
        option_weather_summary["toll_model_source"] = toll_result.source
        option_weather_summary["toll_confidence"] = round(toll_result.confidence, 6)
        option_weather_summary["toll_fallback_policy_used"] = bool(
            toll_result.details.get("fallback_policy_used", False)
        )
    if option_weather_summary is not None and departure_multiplier is not None:
        option_weather_summary["departure_profile_source"] = departure_multiplier.profile_source
        option_weather_summary["departure_profile_day"] = departure_multiplier.profile_day
        option_weather_summary["departure_profile_key"] = departure_multiplier.profile_key
        option_weather_summary["departure_profile_version"] = departure_multiplier.profile_version
        if departure_multiplier.profile_as_of_utc is not None:
            option_weather_summary["departure_profile_as_of_utc"] = departure_multiplier.profile_as_of_utc
        if departure_multiplier.profile_refreshed_at_utc is not None:
            option_weather_summary["departure_profile_refreshed_at_utc"] = (
                departure_multiplier.profile_refreshed_at_utc
            )
        option_weather_summary["departure_applied_multiplier"] = round(departure_multiplier.multiplier, 6)
        if departure_multiplier.confidence_low is not None:
            option_weather_summary["departure_confidence_low"] = round(
                departure_multiplier.confidence_low, 6
            )
        if departure_multiplier.confidence_high is not None:
            option_weather_summary["departure_confidence_high"] = round(
                departure_multiplier.confidence_high, 6
            )
    if option_weather_summary is not None and terrain_summary is not None:
        option_weather_summary["terrain_source"] = terrain_summary.source
        option_weather_summary["terrain_ascent_m"] = terrain_summary.ascent_m
        option_weather_summary["terrain_descent_m"] = terrain_summary.descent_m
        option_weather_summary["terrain_coverage_ratio"] = round(terrain_summary.coverage_ratio, 6)
        option_weather_summary["terrain_confidence"] = round(terrain_summary.confidence, 6)
        option_weather_summary["terrain_dem_version"] = terrain_summary.version
        terrain_diag = terrain_live_diagnostics()
        option_weather_summary["terrain_live_source"] = str(terrain_diag.get("mode", "manifest_asset"))
        option_weather_summary["terrain_live_tile_zoom"] = float(terrain_diag.get("tile_zoom", 0.0))
        option_weather_summary["terrain_live_cache_hit_rate"] = round(
            float(terrain_diag.get("cache_hit_rate", 0.0)),
            6,
        )
        option_weather_summary["terrain_live_fetch_failures"] = float(
            terrain_diag.get("fetch_failures", 0.0)
        )
        option_weather_summary["terrain_live_stale_cache_used"] = bool(
            terrain_diag.get("stale_cache_used", False)
        )
        option_weather_summary["terrain_live_remote_fetches"] = float(
            terrain_diag.get("remote_fetches", 0.0)
        )
        option_weather_summary["terrain_live_circuit_breaker_open"] = bool(
            terrain_diag.get("circuit_breaker_open", False)
        )
    if option_weather_summary is not None:
        option_weather_summary["carbon_source"] = carbon_context.source
        option_weather_summary["carbon_schedule_year"] = carbon_context.schedule_year
        option_weather_summary["carbon_scope_mode"] = carbon_context.scope_mode
        option_weather_summary["carbon_policy_scenario"] = settings.carbon_policy_scenario
        option_weather_summary["carbon_price_uncertainty_low"] = round(carbon_context.uncertainty_low, 6)
        option_weather_summary["carbon_price_uncertainty_high"] = round(carbon_context.uncertainty_high, 6)
        if fuel_price_source is not None:
            option_weather_summary["fuel_price_source"] = fuel_price_source
        if fuel_price_as_of is not None:
            option_weather_summary["fuel_price_as_of"] = fuel_price_as_of
        option_weather_summary["fuel_cost_uncertainty_low"] = round(total_fuel_cost_uncertainty_low, 6)
        option_weather_summary["fuel_cost_uncertainty_high"] = round(total_fuel_cost_uncertainty_high, 6)
        option_weather_summary["fuel_liters_total"] = round(total_fuel_liters, 6)
        option_weather_summary["fuel_liters_p10"] = round(total_fuel_liters_p10, 6)
        option_weather_summary["fuel_liters_p50"] = round(total_fuel_liters_p50, 6)
        option_weather_summary["fuel_liters_p90"] = round(total_fuel_liters_p90, 6)
        option_weather_summary["fuel_cost_p10_gbp"] = round(total_fuel_cost_p10, 6)
        option_weather_summary["fuel_cost_p50_gbp"] = round(total_fuel_cost_p50, 6)
        option_weather_summary["fuel_cost_p90_gbp"] = round(total_fuel_cost_p90, 6)
        option_weather_summary["emissions_uncertainty_low_kg"] = round(total_emissions_uncertainty_low, 6)
        option_weather_summary["emissions_uncertainty_high_kg"] = round(total_emissions_uncertainty_high, 6)
        fuel_cal = load_fuel_consumption_calibration()
        option_weather_summary["consumption_model_source"] = fuel_cal.source
        option_weather_summary["consumption_model_version"] = fuel_cal.version
        if fuel_cal.as_of_utc is not None:
            option_weather_summary["consumption_model_as_of_utc"] = fuel_cal.as_of_utc
        option_weather_summary["vehicle_profile_id"] = vehicle.id
        option_weather_summary["vehicle_profile_version"] = int(vehicle.schema_version)
        option_weather_summary["vehicle_profile_source"] = vehicle.profile_source

    evidence_records: list[EvidenceSourceRecord] = []

    evidence_records.append(
        EvidenceSourceRecord(
            family="scenario",
            source=str(scenario_policy.source or "unknown"),
            active=True,
            freshness_timestamp_utc=scenario_policy.live_as_of_utc or scenario_policy.as_of_utc,
            max_age_minutes=float(getattr(settings, "live_scenario_coefficient_max_age_minutes", 0)),
            signature=str(scenario_policy.version or ""),
            confidence=max(0.0, min(1.0, float(scenario_policy.live_coverage.get("overall", 1.0))))
            if scenario_policy.live_coverage
            else None,
            fallback_used=bool("fallback" in str(scenario_policy.source or "").lower()),
            details={
                "duration_multiplier": round(scenario_multiplier, 6),
                "incident_rate_multiplier": round(scenario_incident_rate_multiplier, 6),
                "incident_delay_multiplier": round(scenario_incident_delay_multiplier, 6),
            },
        )
    )
    evidence_records.append(
        EvidenceSourceRecord(
            family="toll",
            source=str(toll_result.source or "unknown"),
            active=bool(cost_toggles.use_tolls),
            confidence=float(toll_result.confidence),
            coverage_ratio=(
                round(float(toll_distance_km / max(total_distance_km, 1e-9)), 6)
                if total_distance_km > 0.0
                else 0.0
            ),
            fallback_used=bool(toll_result.details.get("fallback_policy_used", False)),
            fallback_source=str(toll_result.details.get("fallback_reason", "") or "") or None,
            details={
                "contains_toll": bool(contains_toll),
                "toll_distance_km": round(float(toll_distance_km), 6),
            },
        )
    )
    evidence_records.append(
        EvidenceSourceRecord(
            family="terrain",
            source=str(terrain_summary.source if terrain_summary is not None else "missing"),
            active=terrain_summary is not None,
            confidence=float(terrain_summary.confidence) if terrain_summary is not None else None,
            coverage_ratio=float(terrain_summary.coverage_ratio) if terrain_summary is not None else None,
            signature=str(terrain_summary.version if terrain_summary is not None else ""),
            fallback_used=bool(
                terrain_summary is not None and terrain_summary.source in {"missing", "unsupported_region"}
            ),
            details={
                "ascent_m": round(float(terrain_summary.ascent_m), 3) if terrain_summary is not None else 0.0,
                "descent_m": round(float(terrain_summary.descent_m), 3) if terrain_summary is not None else 0.0,
            },
        )
    )
    evidence_records.append(
        EvidenceSourceRecord(
            family="fuel",
            source=str(fuel_price_source or "unknown"),
            active=True,
            freshness_timestamp_utc=fuel_price_as_of,
            max_age_minutes=float(getattr(settings, "live_fuel_max_age_minutes", 0)),
            confidence=None,
            fallback_used=bool("fallback" in str(fuel_price_source or "").lower()),
            details={
                "fuel_cost_gbp": round(float(total_fuel_cost), 6),
                "fuel_liters": round(float(total_fuel_liters), 6),
            },
        )
    )
    evidence_records.append(
        EvidenceSourceRecord(
            family="carbon",
            source=str(carbon_context.source or "unknown"),
            active=True,
            freshness_timestamp_utc=getattr(carbon_context, "as_of_utc", None),
            max_age_minutes=float(getattr(settings, "live_carbon_max_age_minutes", 0)),
            confidence=None,
            fallback_used=bool("fallback" in str(carbon_context.source or "").lower()),
            details={
                "price_per_kg": round(float(carbon_price), 6),
                "scope_mode": str(carbon_context.scope_mode),
            },
        )
    )
    evidence_records.append(
        EvidenceSourceRecord(
            family="weather",
            source=str(option_weather_summary.get("profile", "clear") if option_weather_summary else "clear"),
            active=bool(weather_cfg.enabled),
            freshness_timestamp_utc=(
                str(option_weather_summary.get("observed_at_utc")) if option_weather_summary else None
            ),
            confidence=max(0.0, min(1.0, 1.0 - (float(weather_cfg.intensity) * 0.15)))
            if weather_cfg.enabled
            else None,
            fallback_used=False,
            details={
                "speed_multiplier": round(float(weather_speed), 6),
                "incident_multiplier": round(float(weather_incident), 6),
            },
        )
    )
    evidence_records.append(
        EvidenceSourceRecord(
            family="stochastic",
            source="uncertainty_model",
            active=True,
            freshness_timestamp_utc=None,
            confidence=None,
            fallback_used=False,
            details={
                "samples": int((stochastic or StochasticConfig()).samples),
                "sigma": round(float((stochastic or StochasticConfig()).sigma), 6),
            },
        )
    )
    active_families = [record.family for record in evidence_records if record.active]
    evidence_provenance = EvidenceProvenance(
        active_families=active_families,
        families=evidence_records,
    )

    option = RouteOption(
        id=option_id,
        geometry=GeoJSONLineString(type="LineString", coordinates=coords),
        metrics=RouteMetrics(
            distance_km=round(total_distance_km, 3),
            duration_s=round(total_duration_s, 2),
            monetary_cost=round(total_monetary, 2),
            emissions_kg=round(total_emissions, 3),
            avg_speed_kmh=round(avg_speed_kmh, 2),
            energy_kwh=round(total_energy_kwh, 3) if is_ev_mode else None,
            weather_delay_s=round(weather_delta_s, 2),
            incident_delay_s=round(total_incident_delay_s, 2),
        ),
        eta_explanations=eta_explanations,
        eta_timeline=eta_timeline,
        segment_breakdown=segment_breakdown,
        counterfactuals=counterfactuals,
        vehicle_profile_id=vehicle.id,
        vehicle_profile_version=int(vehicle.schema_version),
        vehicle_profile_source=vehicle.profile_source,
        scenario_summary=scenario_summary,
        weather_summary=option_weather_summary,
        terrain_summary=(
            TerrainSummaryPayload.model_validate(terrain_summary.as_route_option_payload())
            if terrain_summary is not None
            else None
        ),
        incident_events=incident_events,
        evidence_provenance=evidence_provenance,
    )
    corridor_bucket = (
        departure_multiplier.profile_key.split(".", 1)[0]
        if departure_multiplier is not None and departure_multiplier.profile_key
        else "uk_default"
    )
    day_kind = (
        departure_multiplier.profile_day
        if departure_multiplier is not None and departure_multiplier.profile_day
        else _day_kind_uk(departure_time_utc)
    )

    option_uncertainty, option_uncertainty_meta = _route_stochastic_uncertainty(
        option,
        stochastic=stochastic or StochasticConfig(),
        route_signature=route_signature,
        departure_time_utc=departure_time_utc,
        utility_weights=utility_weights,
        risk_aversion=risk_aversion,
        road_class_counts=road_class_counts,
        weather_profile=(weather_cfg.profile if weather_cfg.enabled else "clear"),
        vehicle_type=vehicle_type,
        vehicle_profile=vehicle,
        corridor_bucket=corridor_bucket,
        scenario_mode=scenario_mode,
        scenario_profile_version=scenario_policy.version,
        scenario_sigma_multiplier=scenario_sigma_multiplier,
    )
    if option_uncertainty is None:
        raise ModelDataError(
            reason_code="risk_prior_unavailable",
            message="Route uncertainty payload is unavailable for strict runtime.",
        )
    if option_uncertainty_meta is None:
        raise ModelDataError(
            reason_code="risk_prior_unavailable",
            message="Route uncertainty sample metadata is unavailable for strict runtime.",
        )

    # Attach utility normalization provenance for reproducibility diagnostics.
    norm_ref = load_risk_normalization_reference(
        vehicle_type=vehicle_type,
        vehicle_bucket=str(vehicle.risk_bucket),
        corridor_bucket=corridor_bucket,
        day_kind=day_kind,
        local_time_slot=_local_time_slot_uk(departure_time_utc),
    )
    option_uncertainty_meta["normalization_ref_source"] = norm_ref.source
    option_uncertainty_meta["normalization_ref_version"] = norm_ref.version
    option_uncertainty_meta["normalization_ref_as_of_utc"] = norm_ref.as_of_utc or ""
    option_uncertainty_meta["normalization_corridor_bucket"] = norm_ref.corridor_bucket
    option_uncertainty_meta["normalization_day_kind"] = norm_ref.day_kind
    option_uncertainty_meta["normalization_local_time_slot"] = norm_ref.local_time_slot
    option_uncertainty_meta["scenario_context_key"] = scenario_policy.context_key
    if scenario_policy.live_as_of_utc is not None:
        option_uncertainty_meta["scenario_live_as_of_utc"] = scenario_policy.live_as_of_utc
    option_uncertainty_meta["scenario_live_traffic_pressure"] = round(
        float(scenario_policy.live_traffic_pressure),
        6,
    )
    option_uncertainty_meta["scenario_live_incident_pressure"] = round(
        float(scenario_policy.live_incident_pressure),
        6,
    )
    option_uncertainty_meta["scenario_live_weather_pressure"] = round(
        float(scenario_policy.live_weather_pressure),
        6,
    )
    terrain_diag_meta = terrain_live_diagnostics()
    option_uncertainty_meta["terrain_live_source"] = str(
        terrain_diag_meta.get("mode", "manifest_asset")
    )
    option_uncertainty_meta["terrain_live_tile_zoom"] = int(
        terrain_diag_meta.get("tile_zoom", 0)
    )
    option_uncertainty_meta["terrain_live_cache_hit_rate"] = round(
        float(terrain_diag_meta.get("cache_hit_rate", 0.0)),
        6,
    )
    option_uncertainty_meta["terrain_live_fetch_failures"] = int(
        terrain_diag_meta.get("fetch_failures", 0)
    )
    option_uncertainty_meta["terrain_live_stale_cache_used"] = bool(
        terrain_diag_meta.get("stale_cache_used", False)
    )
    option_uncertainty_meta["terrain_live_remote_fetches"] = int(
        terrain_diag_meta.get("remote_fetches", 0)
    )
    option_uncertainty_meta["terrain_live_circuit_breaker_open"] = bool(
        terrain_diag_meta.get("circuit_breaker_open", False)
    )
    option_uncertainty_meta["vehicle_profile_id"] = vehicle.id
    option_uncertainty_meta["vehicle_profile_version"] = int(vehicle.schema_version)
    option_uncertainty_meta["vehicle_profile_source"] = vehicle.profile_source

    option.uncertainty = option_uncertainty
    option.uncertainty_samples_meta = option_uncertainty_meta
    if toll_result is not None:
        option.toll_confidence = round(float(toll_result.confidence), 6)
        matched_asset_ids = (
            [item for item in str(toll_result.details.get("matched_asset_ids", "")).split("|") if item]
            if toll_result.details.get("matched_asset_ids")
            else []
        )
        option.toll_metadata = {
            "matched_assets": matched_asset_ids,
            "tariff_rule_ids": [
                item
                for item in str(toll_result.details.get("tariff_rule_ids", "")).split("|")
                if item
            ],
            "fallback_policy_used": bool(toll_result.details.get("fallback_policy_used", False)),
            "classification_source": str(toll_result.source),
        }
    return option


def _build_osrm_baseline_option(
    route: dict[str, Any],
    *,
    option_id: str,
    vehicle: VehicleProfile,
    scenario_mode: ScenarioMode | None = None,
    cost_toggles: CostToggles | None = None,
    emissions_context: EmissionsContext | None = None,
    weather: WeatherImpactConfig | None = None,
    departure_time_utc: datetime | None = None,
    baseline_duration_multiplier_override: float | None = None,
    baseline_distance_multiplier_override: float | None = None,
) -> RouteOption:
    coords_raw = _validate_osrm_geometry(route)
    coords = _downsample_coords(coords_raw)
    seg_d_m, seg_t_s = extract_segment_annotations(route)
    baseline_distance_multiplier = max(
        1.0,
        float(
            settings.route_baseline_distance_multiplier
            if baseline_distance_multiplier_override is None
            else baseline_distance_multiplier_override
        ),
    )
    seg_d_m_scaled = [max(0.0, float(seg)) * baseline_distance_multiplier for seg in seg_d_m]
    seg_t_s_scaled = [max(0.0, float(seg)) for seg in seg_t_s]

    distance_m = float(route.get("distance", 0.0))
    duration_s = float(route.get("duration", 0.0))
    if distance_m <= 0 or duration_s <= 0:
        distance_m = sum(seg_d_m_scaled)
        duration_s = sum(seg_t_s_scaled)
    else:
        distance_m = max(0.0, distance_m) * baseline_distance_multiplier
        duration_s = max(0.0, duration_s)

    distance_km = max(0.0, distance_m / 1000.0)
    duration_s = max(0.0, duration_s)
    avg_speed_kmh = distance_km / (duration_s / 3600.0) if duration_s > 0 else 0.0
    legacy_monetary_cost = (distance_km * float(vehicle.cost_per_km)) + (
        (duration_s / 3600.0) * float(vehicle.cost_per_hour)
    )

    toggles = cost_toggles or CostToggles()
    ctx = emissions_context or EmissionsContext()
    weather_cfg = weather or WeatherImpactConfig()
    route_points_lat_lon = [(float(lat), float(lon)) for lon, lat in coords_raw]
    road_class_counts = _route_road_class_counts(route)
    route_centroid_lat = (
        sum(point[0] for point in route_points_lat_lon) / len(route_points_lat_lon)
        if route_points_lat_lon
        else None
    )
    route_centroid_lon = (
        sum(point[1] for point in route_points_lat_lon) / len(route_points_lat_lon)
        if route_points_lat_lon
        else None
    )
    tod_multiplier = 1.0
    if departure_time_utc is not None:
        try:
            tod_multiplier = float(
                time_of_day_multiplier_uk(
                    departure_time_utc,
                    route_points=route_points_lat_lon,
                    road_class_counts=road_class_counts,
                ).multiplier
            )
        except Exception:
            tod_multiplier = 1.0
    scenario_duration_multiplier = 1.0
    scenario_fuel_multiplier = 1.0
    scenario_emissions_multiplier = 1.0
    if scenario_mode is not None:
        try:
            scenario_context = build_scenario_route_context(
                route_points=route_points_lat_lon,
                road_class_counts=road_class_counts,
                vehicle_class=str(vehicle.vehicle_class),
                departure_time_utc=departure_time_utc,
                weather_bucket=(weather_cfg.profile if weather_cfg.enabled else "clear"),
            )
            scenario_policy = resolve_scenario_profile(
                scenario_mode,
                context=scenario_context,
            )
            scenario_duration_multiplier = float(scenario_policy.duration_multiplier)
            scenario_fuel_multiplier = float(scenario_policy.fuel_consumption_multiplier)
            scenario_emissions_multiplier = float(scenario_policy.emissions_multiplier)
        except Exception:
            scenario_duration_multiplier = 1.0
            scenario_fuel_multiplier = 1.0
            scenario_emissions_multiplier = 1.0
    weather_duration_multiplier = weather_speed_multiplier(weather_cfg) if weather_cfg.enabled else 1.0
    baseline_duration_multiplier = max(
        1.0,
        float(
            settings.route_baseline_duration_multiplier
            if baseline_duration_multiplier_override is None
            else baseline_duration_multiplier_override
        ),
    )
    duration_scale = max(
        0.55,
        min(
            3.2,
            float(tod_multiplier)
            * float(scenario_duration_multiplier)
            * float(weather_duration_multiplier),
        ),
    )
    duration_scale *= baseline_duration_multiplier
    duration_s *= duration_scale
    avg_speed_kmh = distance_km / (duration_s / 3600.0) if duration_s > 0 else 0.0

    energy_kwh: float | None = None
    proxy_emissions_kg = route_emissions_kg(
        vehicle=vehicle,
        segment_distances_m=seg_d_m_scaled,
        segment_durations_s=seg_t_s_scaled,
    )
    proxy_emissions_kg *= max(
        0.65,
        min(
            3.0,
            duration_scale * scenario_fuel_multiplier * scenario_emissions_multiplier,
        ),
    )
    emissions_kg = proxy_emissions_kg
    monetary_cost = legacy_monetary_cost * max(
        0.65,
        min(3.0, duration_scale * scenario_fuel_multiplier),
    )

    # Keep baseline route generation fast, but use cost primitives consistent with
    # smart-route scoring when live runtime artifacts are available.
    try:
        fuel_multiplier = max(float(toggles.fuel_price_multiplier), 0.0)
        carbon_context = resolve_carbon_price(
            request_price_per_kg=max(float(toggles.carbon_price_per_kg), 0.0),
            departure_time_utc=departure_time_utc,
        )
        carbon_price = float(carbon_context.price_per_kg)
        toll_result = compute_toll_cost(
            route=route,
            distance_km=distance_km,
            vehicle_type=vehicle.id,
            vehicle_profile=vehicle,
            departure_time_utc=departure_time_utc,
            use_tolls=bool(toggles.use_tolls),
            fallback_toll_cost_per_km=float(toggles.toll_cost_per_km),
        )
        toll_component_total = float(toll_result.toll_cost_gbp)
        is_ev_mode = str(vehicle.powertrain).lower() == "ev"
        total_distance_km_for_share = max(distance_km, 1e-6)
        total_emissions = 0.0
        total_energy_kwh = 0.0
        total_monetary = 0.0

        for d_m, t_s in zip(seg_d_m_scaled, seg_t_s_scaled):
            seg_distance_km = max(float(d_m), 0.0) / 1000.0
            seg_duration_s = max(float(t_s), 0.0) * duration_scale
            seg_energy = segment_energy_and_emissions(
                vehicle=vehicle,
                emissions_context=ctx,
                distance_km=seg_distance_km,
                duration_s=seg_duration_s,
                grade=0.0,
                fuel_price_multiplier=fuel_multiplier,
                departure_time_utc=departure_time_utc,
                route_centroid_lat=route_centroid_lat,
                route_centroid_lon=route_centroid_lon,
            )
            seg_emissions = apply_scope_emissions_adjustment(
                emissions_kg=float(seg_energy.emissions_kg) * scenario_fuel_multiplier,
                is_ev_mode=is_ev_mode,
                scope_mode=carbon_context.scope_mode,
                departure_time_utc=departure_time_utc,
                route_centroid_lat=route_centroid_lat,
                route_centroid_lon=route_centroid_lon,
            )
            seg_emissions *= scenario_emissions_multiplier
            seg_toll_cost = toll_component_total * (seg_distance_km / total_distance_km_for_share)
            seg_time_cost = (seg_duration_s / 3600.0) * float(vehicle.cost_per_hour) * DRIVER_TIME_COST_WEIGHT
            seg_carbon_cost = seg_emissions * carbon_price
            seg_monetary = (
                (float(seg_energy.fuel_cost_gbp) * scenario_fuel_multiplier)
                + seg_time_cost
                + seg_toll_cost
                + seg_carbon_cost
            )
            total_emissions += seg_emissions
            total_energy_kwh += max(0.0, float(seg_energy.energy_kwh))
            total_monetary += max(0.0, seg_monetary)

        if proxy_emissions_kg > total_emissions:
            total_monetary += (proxy_emissions_kg - total_emissions) * carbon_price
        emissions_kg = max(0.0, proxy_emissions_kg, total_emissions)
        if is_ev_mode:
            energy_kwh = max(0.0, total_energy_kwh)
        monetary_cost = max(0.0, total_monetary)
    except Exception:
        # Keep baseline comparator available even if a live artifact is transiently missing.
        if str(vehicle.powertrain).lower() == "ev":
            ev_kwh_per_km = (
                float(vehicle.ev_kwh_per_km)
                if vehicle.ev_kwh_per_km is not None and float(vehicle.ev_kwh_per_km) > 0
                else 1.25
            )
            energy_kwh = distance_km * ev_kwh_per_km
            grid_co2_kg_per_kwh = (
                float(vehicle.grid_co2_kg_per_kwh)
                if vehicle.grid_co2_kg_per_kwh is not None and float(vehicle.grid_co2_kg_per_kwh) >= 0
                else 0.20
            )
            emissions_kg = max(0.0, energy_kwh * grid_co2_kg_per_kwh)
        monetary_cost = legacy_monetary_cost * max(
            0.65,
            min(3.0, duration_scale * scenario_fuel_multiplier),
        )

    vehicle_profile_version = (
        int(vehicle.schema_version) if getattr(vehicle, "schema_version", None) is not None else None
    )

    return RouteOption(
        id=option_id,
        geometry=GeoJSONLineString(type="LineString", coordinates=coords),
        metrics=RouteMetrics(
            distance_km=round(distance_km, 3),
            duration_s=round(duration_s, 2),
            monetary_cost=round(monetary_cost, 2),
            emissions_kg=round(max(0.0, float(emissions_kg)), 3),
            avg_speed_kmh=round(avg_speed_kmh, 2),
            energy_kwh=round(energy_kwh, 3) if energy_kwh is not None else None,
        ),
        vehicle_profile_id=vehicle.id,
        vehicle_profile_version=vehicle_profile_version,
        vehicle_profile_source=vehicle.profile_source,
    )


def _parse_route_duration_seconds(raw_duration: Any) -> float:
    text = str(raw_duration or "").strip()
    if not text:
        return 0.0
    if text.endswith("s"):
        text = text[:-1]
    try:
        value = float(text)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value):
        return 0.0
    return max(0.0, value)


def _decode_encoded_polyline(encoded: str) -> list[tuple[float, float]]:
    text = str(encoded or "").strip()
    if not text:
        return []
    coords: list[tuple[float, float]] = []
    index = 0
    lat = 0
    lon = 0
    length = len(text)
    while index < length:
        shift = 0
        result = 0
        while True:
            if index >= length:
                return coords
            b = ord(text[index]) - 63
            index += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        lat += ~(result >> 1) if (result & 1) else (result >> 1)

        shift = 0
        result = 0
        while True:
            if index >= length:
                return coords
            b = ord(text[index]) - 63
            index += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        lon += ~(result >> 1) if (result & 1) else (result >> 1)
        coords.append((lon / 1e5, lat / 1e5))
    return coords


def _haversine_segment_m(
    *,
    lat_a: float,
    lon_a: float,
    lat_b: float,
    lon_b: float,
) -> float:
    radius_m = 6_371_000.0
    phi_1 = math.radians(lat_a)
    phi_2 = math.radians(lat_b)
    dphi = math.radians(lat_b - lat_a)
    dlambda = math.radians(lon_b - lon_a)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi_1) * math.cos(phi_2) * math.sin(dlambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(1e-12, 1.0 - a)))
    return radius_m * c


def _build_osrm_like_route_from_polyline(
    *,
    coordinates_lon_lat: list[tuple[float, float]],
    distance_m: float,
    duration_s: float,
) -> dict[str, Any]:
    coords = list(coordinates_lon_lat)
    if len(coords) < 2:
        raise ValueError("Polyline must contain at least two coordinates.")

    segment_distances_m: list[float] = []
    for idx in range(1, len(coords)):
        lon_a, lat_a = coords[idx - 1]
        lon_b, lat_b = coords[idx]
        segment_distances_m.append(
            _haversine_segment_m(lat_a=lat_a, lon_a=lon_a, lat_b=lat_b, lon_b=lon_b)
        )

    geometric_total_m = max(1.0, float(sum(segment_distances_m)))
    reported_distance_m = max(0.0, float(distance_m))
    if reported_distance_m <= 0.0:
        reported_distance_m = geometric_total_m
    scale = reported_distance_m / geometric_total_m
    segment_distances_scaled = [max(0.0, seg * scale) for seg in segment_distances_m]

    reported_duration_s = max(0.0, float(duration_s))
    if reported_duration_s <= 0.0:
        # Keep fallback deterministic for provider payloads that omit duration.
        reported_duration_s = reported_distance_m / max(3.0, (65.0 / 3.6))
    distance_total_for_share = max(1e-6, float(sum(segment_distances_scaled)))
    segment_durations_s = [
        reported_duration_s * (seg_m / distance_total_for_share) for seg_m in segment_distances_scaled
    ]

    return {
        "distance": reported_distance_m,
        "duration": reported_duration_s,
        "geometry": {"type": "LineString", "coordinates": coords},
        "legs": [
            {
                "annotation": {
                    "distance": segment_distances_scaled,
                    "duration": segment_durations_s,
                }
            }
        ],
    }


async def _fetch_ors_reference_route_seed(
    *,
    req: RouteRequest,
) -> dict[str, Any]:
    api_key = str(settings.ors_directions_api_key or "").strip()
    if not api_key:
        raise RuntimeError("ORS_DIRECTIONS_API_KEY is not configured.")
    url_template = str(settings.ors_directions_url_template or "").strip()
    if not url_template:
        raise RuntimeError("ORS_DIRECTIONS_URL_TEMPLATE is empty.")

    vehicle_hint = str(req.vehicle_type or "").strip().lower()
    hgv_profile = str(settings.ors_directions_profile_hgv or "driving-hgv").strip() or "driving-hgv"
    default_profile = str(settings.ors_directions_profile_default or "driving-car").strip() or "driving-car"
    profile = (
        hgv_profile
        if any(token in vehicle_hint for token in ("hgv", "truck", "artic", "rigid"))
        else default_profile
    )
    url = url_template.format(profile=profile) if "{profile}" in url_template else url_template
    if not url:
        raise RuntimeError("ORS_DIRECTIONS_URL_TEMPLATE resolved to an empty URL.")

    coordinates: list[list[float]] = [
        [float(req.origin.lon), float(req.origin.lat)],
        *[[float(point.lon), float(point.lat)] for point in (req.waypoints or [])],
        [float(req.destination.lon), float(req.destination.lat)],
    ]
    body: dict[str, Any] = {
        "coordinates": coordinates,
        "instructions": False,
        "elevation": False,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": api_key,
    }

    timeout_s = max(1.0, float(settings.ors_directions_timeout_ms) / 1000.0)
    timeout = httpx.Timeout(timeout_s)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, json=body, headers=headers)

    if response.status_code >= 400:
        response_text = response.text.strip()
        raise RuntimeError(
            f"OpenRouteService returned HTTP {response.status_code}: {response_text[:300]}"
        )

    payload = response.json()
    distance_m = 0.0
    duration_s = 0.0
    coordinates_lon_lat: list[tuple[float, float]] = []

    routes = payload.get("routes")
    if isinstance(routes, list) and routes and isinstance(routes[0], dict):
        route0 = routes[0]
        summary = route0.get("summary") if isinstance(route0.get("summary"), dict) else {}
        distance_m = float((summary or {}).get("distance", route0.get("distance", 0.0)) or 0.0)
        duration_s = _parse_route_duration_seconds((summary or {}).get("duration", route0.get("duration")))
        geometry_payload = route0.get("geometry")
        if isinstance(geometry_payload, str):
            coordinates_lon_lat = _decode_encoded_polyline(geometry_payload)
        elif (
            isinstance(geometry_payload, dict)
            and isinstance(geometry_payload.get("coordinates"), list)
        ):
            coordinates_lon_lat = [
                (float(pair[0]), float(pair[1]))
                for pair in geometry_payload.get("coordinates", [])
                if isinstance(pair, (list, tuple)) and len(pair) >= 2
            ]

    if len(coordinates_lon_lat) < 2:
        features = payload.get("features")
        if isinstance(features, list) and features and isinstance(features[0], dict):
            feature0 = features[0]
            feature_geometry = feature0.get("geometry")
            if (
                isinstance(feature_geometry, dict)
                and isinstance(feature_geometry.get("coordinates"), list)
            ):
                coordinates_lon_lat = [
                    (float(pair[0]), float(pair[1]))
                    for pair in feature_geometry.get("coordinates", [])
                    if isinstance(pair, (list, tuple)) and len(pair) >= 2
                ]
            feature_props = feature0.get("properties")
            if isinstance(feature_props, dict) and isinstance(feature_props.get("summary"), dict):
                feature_summary = feature_props.get("summary", {})
                distance_m = float(feature_summary.get("distance", distance_m) or distance_m or 0.0)
                duration_s = _parse_route_duration_seconds(feature_summary.get("duration", duration_s))

    if len(coordinates_lon_lat) < 2:
        raise RuntimeError("OpenRouteService route geometry is missing or invalid.")

    return _build_osrm_like_route_from_polyline(
        coordinates_lon_lat=coordinates_lon_lat,
        distance_m=distance_m,
        duration_s=duration_s,
    )


def _route_signature(route: dict[str, Any]) -> str:
    coords = _validate_osrm_geometry(route)
    n = len(coords)
    step = max(1, n // 30)
    sample = coords[::step][:40]

    # round for stability; avoid huge hash variability
    parts = [f"{lon:.4f},{lat:.4f}" for lon, lat in sample]
    s = "|".join(parts)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _candidate_cache_key(
    *,
    origin: LatLng,
    destination: LatLng,
    waypoints: list[Waypoint] | None = None,
    vehicle_type: str,
    scenario_mode: ScenarioMode,
    max_routes: int,
    cost_toggles: CostToggles,
    terrain_profile: TerrainProfile = "flat",
    departure_time_utc: datetime | None = None,
    scenario_cache_token: str | None = None,
) -> str:
    payload = {
        "schema_version": CANDIDATE_CACHE_SCHEMA_VERSION,
        "origin": {"lat": round(origin.lat, 6), "lon": round(origin.lon, 6)},
        "destination": {"lat": round(destination.lat, 6), "lon": round(destination.lon, 6)},
        "waypoints": [
            {
                "lat": round(float(waypoint.lat), 6),
                "lon": round(float(waypoint.lon), 6),
                "label": (waypoint.label or "").strip(),
            }
            for waypoint in (waypoints or [])
        ],
        "vehicle_type": vehicle_type,
        "scenario_mode": scenario_mode.value,
        "max_routes": max_routes,
        "cost_toggles": cost_toggles.model_dump(mode="json"),
        "terrain_profile": terrain_profile,
        "departure_time_utc": departure_time_utc.isoformat() if departure_time_utc else None,
        "scenario_cache_token": scenario_cache_token or "",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()


def _candidate_via_points(origin: LatLng, destination: LatLng) -> list[tuple[float, float]]:
    """Generate bounded corridor via points for deterministic candidate expansion."""
    d_lat = destination.lat - origin.lat
    d_lon = destination.lon - origin.lon
    span = max(abs(d_lat), abs(d_lon))
    if span <= 1e-9:
        return []

    # Unit perpendicular vector in lat/lon plane.
    norm = max((d_lat * d_lat + d_lon * d_lon) ** 0.5, 1e-9)
    p_lat = -d_lon / norm
    p_lon = d_lat / norm

    base_offset = max(0.04, min(span * 0.22, 0.78))
    frac_points = (0.2, 0.33, 0.5, 0.67, 0.8)
    lateral_scales = (-1.0, -0.55, 0.55, 1.0)

    candidates: list[tuple[float, float]] = []
    for frac in frac_points:
        c_lat = origin.lat + (d_lat * frac)
        c_lon = origin.lon + (d_lon * frac)
        for scale in lateral_scales:
            lat = c_lat + (p_lat * base_offset * scale)
            lon = c_lon + (p_lon * base_offset * scale)
            if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                candidates.append((lat, lon))

    # Add short-axis perturbations near midline for mild detours.
    mid_lat = (origin.lat + destination.lat) / 2.0
    mid_lon = (origin.lon + destination.lon) / 2.0
    q_offset = max(0.025, min(span * 0.08, 0.25))
    candidates.extend(
        [
            (mid_lat + q_offset, mid_lon + q_offset),
            (mid_lat - q_offset, mid_lon - q_offset),
            (mid_lat + q_offset, mid_lon - q_offset),
            (mid_lat - q_offset, mid_lon + q_offset),
        ]
    )

    dedup: list[tuple[float, float]] = []
    seen: set[tuple[int, int]] = set()
    for lat, lon in candidates:
        key = (int(round(lat * 20_000)), int(round(lon * 20_000)))
        if key in seen:
            continue
        seen.add(key)
        dedup.append((lat, lon))
        if len(dedup) >= int(settings.route_candidate_via_budget):
            break
    return dedup


def _graph_family_via_points(
    route: dict[str, Any],
    *,
    max_landmarks: int = 2,
) -> list[tuple[float, float]]:
    """Extract deterministic via landmarks from a graph-family geometry.

    Returns (lat, lon) points suitable for OSRM refinement requests.
    """
    geom = route.get("geometry")
    if not isinstance(geom, dict):
        return []
    coords = geom.get("coordinates")
    if not isinstance(coords, list) or len(coords) < 4:
        return []
    usable = max(1, min(int(max_landmarks), 3))
    points: list[tuple[float, float]] = []
    seen: set[tuple[int, int]] = set()
    for idx in range(1, usable + 1):
        ratio = idx / float(usable + 1)
        coord_idx = int(round((len(coords) - 1) * ratio))
        coord_idx = max(1, min(len(coords) - 2, coord_idx))
        raw = coords[coord_idx]
        if not isinstance(raw, (list, tuple)) or len(raw) < 2:
            continue
        try:
            lon = float(raw[0])
            lat = float(raw[1])
        except (TypeError, ValueError):
            continue
        key = (int(round(lat * 20_000)), int(round(lon * 20_000)))
        if key in seen:
            continue
        seen.add(key)
        points.append((lat, lon))
    return points


def _candidate_fetch_specs(
    *,
    origin: LatLng,
    destination: LatLng,
    max_routes: int,
) -> list[CandidateFetchSpec]:
    max_routes = max(1, min(max_routes, int(settings.route_candidate_alternatives_max)))
    strict_graph_required = True
    specs: list[CandidateFetchSpec] = []

    graph_via_paths = route_graph_via_paths(
        origin_lat=float(origin.lat),
        origin_lon=float(origin.lon),
        destination_lat=float(destination.lat),
        destination_lon=float(destination.lon),
        max_paths=max(4, max_routes * 2),
    )
    if strict_graph_required:
        graph_ok, graph_status = route_graph_status()
        if not graph_ok:
            raise ModelDataError(
                reason_code="routing_graph_unavailable",
                message=f"Route graph is required but unavailable ({graph_status}).",
            )
        if not graph_via_paths:
            raise ModelDataError(
                reason_code="routing_graph_unavailable",
                message="Route graph is required but produced no feasible corridor paths for this OD pair.",
            )

    if not strict_graph_required:
        specs.append(CandidateFetchSpec(label="alternatives", alternatives=max_routes))
        for ex in (
            "motorway",
            "trunk",
            "toll",
            "ferry",
            "motorway,toll",
            "trunk,toll",
            "motorway,trunk",
            "motorway,ferry",
            "trunk,ferry",
            "motorway,toll,ferry",
            "trunk,toll,ferry",
        ):
            specs.append(CandidateFetchSpec(label=f"exclude:{ex}", alternatives=False, exclude=ex))

    via_source: tuple[tuple[tuple[float, float], ...], ...] = graph_via_paths
    via_prefix = "graph"
    if not via_source and strict_graph_required:
        raise ModelDataError(
            reason_code="routing_graph_unavailable",
            message="Route graph did not provide viable corridor hints for strict candidate fetch.",
        )
    if not via_source and not strict_graph_required:
        via_source = tuple((pt,) for pt in _candidate_via_points(origin, destination))
        via_prefix = "via"

    for idx, via_path in enumerate(via_source, start=1):
        via_points = [(float(lat), float(lon)) for lat, lon in via_path]
        specs.append(CandidateFetchSpec(label=f"{via_prefix}:{idx}", alternatives=False, via=via_points))

    return specs


def _long_corridor_fallback_specs(
    *,
    origin: LatLng,
    destination: LatLng,
    max_routes: int,
) -> list[CandidateFetchSpec]:
    """Build a bounded OSRM fallback family set for long-corridor recoveries."""
    alt_budget = max(2, min(int(settings.route_candidate_alternatives_max), max_routes * 2))
    specs: list[CandidateFetchSpec] = [CandidateFetchSpec(label="fallback:alternatives", alternatives=alt_budget)]
    via_candidates = _candidate_via_points(origin, destination)
    via_limit = max(2, min(len(via_candidates), max(4, max_routes)))
    for idx, via in enumerate(via_candidates[:via_limit], start=1):
        specs.append(CandidateFetchSpec(label=f"fallback:via:{idx}", alternatives=False, via=[via]))
    for ex in ("motorway", "toll", "ferry"):
        specs.append(CandidateFetchSpec(label=f"fallback:exclude:{ex}", alternatives=False, exclude=ex))
    return specs


def _candidate_fetch_concurrency(total_specs: int) -> int:
    capped = min(settings.batch_concurrency, MAX_CANDIDATE_FETCH_CONCURRENCY)
    return max(1, min(total_specs, capped))


async def _run_candidate_fetch(
    *,
    osrm: OSRMClient,
    origin: LatLng,
    destination: LatLng,
    spec: CandidateFetchSpec,
    sem: asyncio.Semaphore,
) -> CandidateFetchResult:
    try:
        async with sem:
            routes = await osrm.fetch_routes(
                origin_lat=origin.lat,
                origin_lon=origin.lon,
                dest_lat=destination.lat,
                dest_lon=destination.lon,
                alternatives=spec.alternatives,
                exclude=spec.exclude,
                via=spec.via,
            )
        return CandidateFetchResult(spec=spec, routes=routes)
    except OSRMError as e:
        return CandidateFetchResult(spec=spec, routes=[], error=str(e))
    except Exception as e:  # pragma: no cover - defensive fallback for unexpected failures
        msg = str(e).strip()
        detail = f"{type(e).__name__}: {msg}" if msg else type(e).__name__
        return CandidateFetchResult(spec=spec, routes=[], error=detail)


async def _iter_candidate_fetches(
    *,
    osrm: OSRMClient,
    origin: LatLng,
    destination: LatLng,
    specs: list[CandidateFetchSpec],
) -> AsyncIterator[CandidateProgress]:
    total = len(specs)
    if total == 0:
        return

    sem = asyncio.Semaphore(_candidate_fetch_concurrency(total))
    tasks = [
        asyncio.create_task(
            _run_candidate_fetch(
                osrm=osrm,
                origin=origin,
                destination=destination,
                spec=spec,
                sem=sem,
            )
        )
        for spec in specs
    ]

    done = 0
    try:
        for task in asyncio.as_completed(tasks):
            result = await task
            done += 1
            yield CandidateProgress(done=done, total=total, result=result)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def _route_duration_s(route: dict[str, Any]) -> float:
    duration = route.get("duration")
    if isinstance(duration, (int, float)) and duration > 0:
        return float(duration)
    try:
        _, seg_t_s = extract_segment_annotations(route)
    except OSRMError:
        return float("inf")
    return float(sum(seg_t_s))


def _graph_family_signature(route: dict[str, Any]) -> str:
    geom = route.get("geometry")
    if not isinstance(geom, dict):
        return ""
    coords = geom.get("coordinates")
    if not isinstance(coords, list) or len(coords) < 2:
        return ""
    sample: list[str] = []
    step = max(1, len(coords) // 24)
    for idx in range(0, len(coords), step):
        point = coords[idx]
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            lon = float(point[0])
            lat = float(point[1])
        except (TypeError, ValueError):
            continue
        sample.append(f"{lon:.4f},{lat:.4f}")
        if len(sample) >= 32:
            break
    if not sample:
        return ""
    return hashlib.sha1("|".join(sample).encode("utf-8")).hexdigest()


def _route_distance_km(route: dict[str, Any]) -> float:
    raw = route.get("distance")
    if isinstance(raw, (int, float)) and raw > 0:
        return float(raw) / 1000.0
    try:
        seg_d_m, _seg_t_s = extract_segment_annotations(route)
        if seg_d_m:
            return float(sum(seg_d_m)) / 1000.0
    except OSRMError:
        pass
    return 0.0


def _od_haversine_km(origin: LatLng, destination: LatLng) -> float:
    lat1 = math.radians(float(origin.lat))
    lon1 = math.radians(float(origin.lon))
    lat2 = math.radians(float(destination.lat))
    lon2 = math.radians(float(destination.lon))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * (math.sin(dlon / 2.0) ** 2)
    )
    return 2.0 * 6371.0 * math.asin(min(1.0, math.sqrt(max(0.0, a))))


def _search_deadline_s(timeout_ms: int) -> float | None:
    timeout = int(timeout_ms)
    if timeout <= 0:
        return None
    return time.monotonic() + (timeout / 1000.0)


def _route_corridor_signature(route: dict[str, Any]) -> str:
    meta = route.get("_graph_meta")
    if not isinstance(meta, dict):
        return "corridor:unknown"
    road_mix = meta.get("road_mix_counts")
    dominant_class = "unknown"
    if isinstance(road_mix, dict) and road_mix:
        dominant_class = str(
            max(
                road_mix.items(),
                key=lambda item: (int(item[1]) if isinstance(item[1], (int, float)) else 0, str(item[0])),
            )[0]
        ).strip() or "unknown"
    toll_edges = int(meta.get("toll_edges", 0)) if isinstance(meta.get("toll_edges", 0), (int, float)) else 0
    toll_bucket = "tolled" if toll_edges > 0 else "free"
    distance_bucket = int(_route_distance_km(route) // 25.0)
    return f"{dominant_class}:{toll_bucket}:{distance_bucket}"


def _neutral_scenario_edge_modifiers() -> dict[str, float | str]:
    return {
        "mode": "full_sharing",
        "duration_multiplier": 1.0,
        "incident_rate_multiplier": 1.0,
        "incident_delay_multiplier": 1.0,
        "stochastic_sigma_multiplier": 1.0,
        "traffic_pressure": 1.0,
        "incident_pressure": 1.0,
        "weather_pressure": 1.0,
        "scenario_edge_scaling_version": "v3_live_transform",
    }


def _is_neutral_scenario_modifiers(modifiers: dict[str, float | str] | None) -> bool:
    if not isinstance(modifiers, dict):
        return True
    neutral = _neutral_scenario_edge_modifiers()
    for key, neutral_value in neutral.items():
        value = modifiers.get(key, neutral_value)
        if isinstance(neutral_value, str):
            if str(value).strip().lower() != str(neutral_value).strip().lower():
                return False
            continue
        if not isinstance(value, (int, float, str)):
            return False
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return False
        if abs(numeric_value - float(neutral_value)) > 1e-9:
            return False
    return True


def _is_stressed_scenario_modifiers(modifiers: dict[str, float | str] | None) -> bool:
    if not isinstance(modifiers, dict):
        return False
    checks = (
        "duration_multiplier",
        "incident_rate_multiplier",
        "incident_delay_multiplier",
        "traffic_pressure",
        "incident_pressure",
        "weather_pressure",
    )
    for key in checks:
        try:
            if abs(float(modifiers.get(key, 1.0)) - 1.0) >= 0.10:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _scenario_stress_score(modifiers: dict[str, float | str] | None) -> float:
    if not isinstance(modifiers, dict):
        return 0.0
    checks = (
        "duration_multiplier",
        "incident_rate_multiplier",
        "incident_delay_multiplier",
        "traffic_pressure",
        "incident_pressure",
        "weather_pressure",
        "stochastic_sigma_multiplier",
    )
    deviations: list[float] = []
    for key in checks:
        try:
            deviations.append(abs(float(modifiers.get(key, 1.0)) - 1.0))
        except (TypeError, ValueError):
            continue
    if not deviations:
        return 0.0
    # Blend worst-case and mean stress to avoid overreacting to a single noisy factor.
    return float((0.6 * max(deviations)) + (0.4 * (sum(deviations) / len(deviations))))


def _adaptive_scenario_jaccard_threshold(modifiers: dict[str, float | str] | None) -> tuple[float, float]:
    base = max(0.0, min(1.0, float(settings.route_graph_scenario_jaccard_max)))
    floor = max(0.0, min(base, float(settings.route_graph_scenario_jaccard_floor)))
    stress = _scenario_stress_score(modifiers)
    # Start adaptive tightening once stress moves beyond mild perturbation.
    # This drops from base toward floor as stress rises.
    stress_excess = max(0.0, stress - 0.10)
    drop_cap = max(0.0, base - floor)
    adaptive_drop = min(drop_cap, stress_excess * 0.50)
    threshold = max(floor, base - adaptive_drop)
    return threshold, stress


def _select_ranked_candidate_routes(
    routes: list[dict[str, Any]],
    *,
    max_routes: int,
) -> list[dict[str, Any]]:
    max_routes = max(1, min(max_routes, int(settings.route_candidate_alternatives_max)))
    prefilter_multiplier = max(1, int(settings.route_candidate_prefilter_multiplier))
    long_prefilter_multiplier = max(1, int(settings.route_candidate_prefilter_multiplier_long))
    long_distance_threshold_km = max(
        0.0,
        float(settings.route_candidate_prefilter_long_distance_threshold_km),
    )
    if long_distance_threshold_km > 0.0 and any(
        _route_distance_km(route) >= long_distance_threshold_km for route in routes
    ):
        prefilter_multiplier = min(prefilter_multiplier, long_prefilter_multiplier)
    prefilter_target = max(
        max_routes,
        min(len(routes), max_routes * prefilter_multiplier),
    )

    scored: list[tuple[float, str, str, str, dict[str, Any]]] = []
    for route in routes:
        sig = _route_signature(route)
        family_sig = _graph_family_signature(route) or sig
        corridor_sig = _route_corridor_signature(route)
        scored.append((_route_duration_s(route), sig, family_sig, corridor_sig, route))
    scored.sort(key=lambda item: (item[0], item[1]))

    if len(scored) <= prefilter_target:
        return [route for _, _, _, _, route in scored]

    selected: list[tuple[float, str, str, str, dict[str, Any]]] = []
    family_counts: dict[str, int] = {}
    corridor_counts: dict[str, int] = {}
    remaining = list(scored)

    while remaining and len(selected) < prefilter_target:
        best_idx = 0
        best_score = float("inf")
        best_duration = float("inf")
        best_sig = ""
        for idx, (duration_s, sig, family_sig, corridor_sig, _route) in enumerate(remaining):
            family_count = family_counts.get(family_sig, 0)
            corridor_count = corridor_counts.get(corridor_sig, 0)
            diversity_multiplier = 1.0 + (0.18 * family_count) + (0.12 * corridor_count)
            blended_score = duration_s * diversity_multiplier
            if (
                blended_score < best_score
                or (
                    math.isclose(blended_score, best_score)
                    and (
                        duration_s < best_duration
                        or (math.isclose(duration_s, best_duration) and sig < best_sig)
                    )
                )
            ):
                best_idx = idx
                best_score = blended_score
                best_duration = duration_s
                best_sig = sig
        duration_s, sig, family_sig, corridor_sig, route = remaining.pop(best_idx)
        selected.append((duration_s, sig, family_sig, corridor_sig, route))
        family_counts[family_sig] = family_counts.get(family_sig, 0) + 1
        corridor_counts[corridor_sig] = corridor_counts.get(corridor_sig, 0) + 1

    selected.sort(key=lambda item: (item[0], item[1]))
    return [route for _, _, _, _, route in selected]


def _build_options(
    routes: list[dict[str, Any]],
    *,
    vehicle_type: str,
    scenario_mode: ScenarioMode,
    cost_toggles: CostToggles,
    terrain_profile: TerrainProfile = "flat",
    stochastic: StochasticConfig | None = None,
    emissions_context: EmissionsContext | None = None,
    weather: WeatherImpactConfig | None = None,
    incident_simulation: IncidentSimulatorConfig | None = None,
    departure_time_utc: datetime | None,
    utility_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    risk_aversion: float = 1.0,
    option_prefix: str,
) -> tuple[list[RouteOption], list[str], TerrainDiagnostics]:
    options: list[RouteOption] = []
    warnings: list[str] = []
    fail_closed_count = 0
    unsupported_region_count = 0
    asset_unavailable_count = 0
    coverage_min_observed = 1.0
    dem_version = "unknown"
    scenario_policy_cache: dict[tuple[str, str], Any] = {}
    terrain_begin_route_run()
    if bool(settings.route_option_reuse_scenario_policy) and routes:
        try:
            shared_route = routes[0]
            shared_coords = _downsample_coords(_validate_osrm_geometry(shared_route))
            shared_points_lat_lon = [(lat, lon) for lon, lat in shared_coords]
            shared_road_class_counts = _route_road_class_counts(shared_route)
            shared_context = build_scenario_route_context(
                route_points=shared_points_lat_lon,
                road_class_counts=shared_road_class_counts,
                vehicle_class=str(resolve_vehicle_profile(vehicle_type).vehicle_class),
                departure_time_utc=departure_time_utc,
                weather_bucket=(weather.profile if weather is not None and weather.enabled else "clear"),
            )
            scenario_policy_cache[("__shared__", scenario_mode.value)] = resolve_scenario_profile(
                scenario_mode,
                context=shared_context,
            )
            shifted_mode = _counterfactual_shift_scenario(scenario_mode)
            scenario_policy_cache[("__shared__", shifted_mode.value)] = resolve_scenario_profile(
                shifted_mode,
                context=shared_context,
            )
        except Exception:
            # Fall back to per-route scenario context if shared bootstrap fails.
            pass

    for route in routes:
        option_id = f"{option_prefix}_{len(options)}"
        try:
            option = build_option(
                route,
                option_id=option_id,
                vehicle_type=vehicle_type,
                scenario_mode=scenario_mode,
                cost_toggles=cost_toggles,
                terrain_profile=terrain_profile,
                stochastic=stochastic,
                emissions_context=emissions_context,
                weather=weather,
                incident_simulation=incident_simulation,
                departure_time_utc=departure_time_utc,
                utility_weights=utility_weights,
                risk_aversion=risk_aversion,
                scenario_policy_cache=scenario_policy_cache,
                reset_terrain_route_run=False,
            )
            options.append(option)
            if option.terrain_summary is not None:
                source = str(option.terrain_summary.source)
                if source == "missing":
                    warnings.append(
                        f"{option_id}: terrain_missing "
                        f"(coverage={float(option.terrain_summary.coverage_ratio):.4f}, "
                        f"dem={option.terrain_summary.version})"
                    )
                coverage_min_observed = min(
                    coverage_min_observed,
                    float(option.terrain_summary.coverage_ratio),
                )
                dem_version = str(option.terrain_summary.version)
        except TerrainCoverageError as e:
            reason = getattr(e, "reason_code", "terrain_dem_coverage_insufficient")
            if reason == "terrain_region_unsupported":
                unsupported_region_count += 1
                warnings.append(f"{option_id}: terrain_region_unsupported ({e})")
            elif reason == "terrain_dem_asset_unavailable":
                asset_unavailable_count += 1
                warnings.append(f"{option_id}: terrain_dem_asset_unavailable ({e})")
            else:
                fail_closed_count += 1
                warnings.append(f"{option_id}: terrain_fail_closed ({e})")
            coverage_min_observed = min(coverage_min_observed, float(e.coverage_ratio))
            dem_version = str(e.version)
        except ModelDataError as e:
            warnings.append(f"{option_id}: {e.reason_code} ({e})")
        except FileNotFoundError as e:
            msg = str(e).strip()
            warnings.append(f"{option_id}: model_asset_unavailable ({msg})")
        except (OSRMError, ValueError) as e:
            msg = str(e).strip()
            if "tariff resolution" in msg or "toll" in msg.lower():
                warnings.append(f"{option_id}: toll_pricing_unresolved ({msg})")
            else:
                warnings.append(f"{option_id}: {msg}")

    return (
        options,
        warnings,
        TerrainDiagnostics(
            fail_closed_count=fail_closed_count,
            unsupported_region_count=unsupported_region_count,
            asset_unavailable_count=asset_unavailable_count,
            dem_version=dem_version,
            coverage_min_observed=(
                coverage_min_observed
                if options or fail_closed_count or unsupported_region_count or asset_unavailable_count
                else 1.0
            ),
        ),
    )


def _has_warning_code(warnings: list[str], code: str) -> bool:
    token = f": {code}"
    return any(token in warning for warning in warnings)


def _precheck_timeout_fail_closed_enabled() -> bool:
    return bool(settings.route_graph_precheck_timeout_fail_closed)


def _strict_error_detail(
    *,
    reason_code: str,
    message: str,
    warnings: list[str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    detail: dict[str, Any] = {
        "reason_code": normalize_reason_code(reason_code),
        "message": message,
        "warnings": warnings,
    }
    if extra:
        detail.update(extra)
    return detail


def _strict_failure_detail_from_outcome(
    *,
    warnings: list[str],
    terrain_diag: TerrainDiagnostics,
    epsilon_requested: bool,
    terrain_message: str,
) -> dict[str, Any] | None:
    if terrain_diag.asset_unavailable_count > 0 or _has_warning_code(warnings, "terrain_dem_asset_unavailable"):
        return _strict_error_detail(
            reason_code="terrain_dem_asset_unavailable",
            message="Terrain DEM assets are unavailable for strict routing policy.",
            warnings=warnings,
            extra={
                "terrain_dem_version": terrain_diag.dem_version,
            },
        )
    if terrain_diag.unsupported_region_count > 0 or _has_warning_code(warnings, "terrain_region_unsupported"):
        return _strict_error_detail(
            reason_code="terrain_region_unsupported",
            message="Terrain model currently supports UK routes only in strict mode.",
            warnings=warnings,
            extra={
                "terrain_dem_version": terrain_diag.dem_version,
            },
        )
    if terrain_diag.fail_closed_count > 0:
        return _strict_error_detail(
            reason_code="terrain_dem_coverage_insufficient",
            message=terrain_message,
            warnings=warnings,
            extra={
                "terrain_coverage_min_observed": round(terrain_diag.coverage_min_observed, 6),
                "terrain_coverage_required": round(settings.terrain_dem_coverage_min_uk, 6),
                "terrain_dem_version": terrain_diag.dem_version,
            },
        )
    if _has_warning_code(warnings, "toll_pricing_unresolved"):
        return _strict_error_detail(
            reason_code="toll_tariff_unresolved",
            message="Tolled route was detected but no tariff rule could be resolved.",
            warnings=warnings,
        )
    if _has_warning_code(warnings, "routing_graph_fragmented"):
        return _strict_error_detail(
            reason_code="routing_graph_fragmented",
            message="Routing graph is fragmented under current strict runtime policy.",
            warnings=warnings,
        )
    if _has_warning_code(warnings, "routing_graph_disconnected_od"):
        return _strict_error_detail(
            reason_code="routing_graph_disconnected_od",
            message="Origin and destination are disconnected in the loaded route graph.",
            warnings=warnings,
        )
    if _has_warning_code(warnings, "routing_graph_coverage_gap"):
        return _strict_error_detail(
            reason_code="routing_graph_coverage_gap",
            message="Route graph coverage near origin/destination is insufficient.",
            warnings=warnings,
        )
    if _has_warning_code(warnings, "routing_graph_precheck_timeout") and _precheck_timeout_fail_closed_enabled():
        return _strict_error_detail(
            reason_code="routing_graph_precheck_timeout",
            message="Route graph feasibility precheck timed out before route compute could continue.",
            warnings=warnings,
        )
    if _has_warning_code(warnings, "routing_graph_no_path"):
        return _strict_error_detail(
            reason_code="routing_graph_no_path",
            message="Routing graph search exhausted without a feasible path under strict runtime policy.",
            warnings=warnings,
        )
    if _has_warning_code(warnings, "live_source_refresh_failed"):
        return _strict_error_detail(
            reason_code="live_source_refresh_failed",
            message="Live source refresh gate failed before route candidate search could continue.",
            warnings=warnings,
        )
    if _has_warning_code(warnings, "routing_graph_unavailable"):
        return _strict_error_detail(
            reason_code="routing_graph_unavailable",
            message="Routing graph assets are unavailable for strict routing policy.",
            warnings=warnings,
        )
    if _has_warning_code(warnings, "toll_tariff_unavailable"):
        return _strict_error_detail(
            reason_code="toll_tariff_unavailable",
            message="Live toll tariff catalog is unavailable for strict routing policy.",
            warnings=warnings,
        )
    if _has_warning_code(warnings, "departure_profile_unavailable"):
        return _strict_error_detail(
            reason_code="departure_profile_unavailable",
            message="Departure profile assets are unavailable for strict routing policy.",
            warnings=warnings,
        )
    if _has_warning_code(warnings, "holiday_data_unavailable"):
        return _strict_error_detail(
            reason_code="holiday_data_unavailable",
            message="Live holiday data is unavailable for strict routing policy.",
            warnings=warnings,
        )
    if _has_warning_code(warnings, "stochastic_calibration_unavailable"):
        return _strict_error_detail(
            reason_code="stochastic_calibration_unavailable",
            message="Live stochastic calibration data is unavailable for strict routing policy.",
            warnings=warnings,
        )
    if _has_warning_code(warnings, "scenario_profile_unavailable"):
        return _strict_error_detail(
            reason_code="scenario_profile_unavailable",
            message="Scenario profile data is unavailable for strict routing policy.",
            warnings=warnings,
        )
    if _has_warning_code(warnings, "risk_normalization_unavailable"):
        return _strict_error_detail(
            reason_code="risk_normalization_unavailable",
            message="Risk normalization references are unavailable for strict routing policy.",
            warnings=warnings,
        )
    if _has_warning_code(warnings, "risk_prior_unavailable"):
        return _strict_error_detail(
            reason_code="risk_prior_unavailable",
            message="Risk prior calibration is unavailable for strict routing policy.",
            warnings=warnings,
        )
    if _has_warning_code(warnings, "vehicle_profile_unavailable"):
        return _strict_error_detail(
            reason_code="vehicle_profile_unavailable",
            message="Vehicle profile is unavailable for strict routing policy.",
            warnings=warnings,
        )
    if _has_warning_code(warnings, "vehicle_profile_invalid"):
        return _strict_error_detail(
            reason_code="vehicle_profile_invalid",
            message="Vehicle profile data is invalid for strict routing policy.",
            warnings=warnings,
        )
    if _has_warning_code(warnings, "toll_topology_unavailable"):
        return _strict_error_detail(
            reason_code="toll_topology_unavailable",
            message="Live toll topology data is unavailable for strict routing policy.",
            warnings=warnings,
        )
    if _has_warning_code(warnings, "fuel_price_auth_unavailable"):
        return _strict_error_detail(
            reason_code="fuel_price_auth_unavailable",
            message="Fuel price source authentication failed or credentials are missing.",
            warnings=warnings,
        )
    if _has_warning_code(warnings, "fuel_price_source_unavailable"):
        return _strict_error_detail(
            reason_code="fuel_price_source_unavailable",
            message="Fuel price source is unavailable for strict routing policy.",
            warnings=warnings,
        )
    if _has_warning_code(warnings, "carbon_policy_unavailable"):
        return _strict_error_detail(
            reason_code="carbon_policy_unavailable",
            message="Carbon policy source is unavailable for strict routing policy.",
            warnings=warnings,
        )
    if _has_warning_code(warnings, "carbon_intensity_unavailable"):
        return _strict_error_detail(
            reason_code="carbon_intensity_unavailable",
            message="Carbon intensity source is unavailable for strict routing policy.",
            warnings=warnings,
        )
    if _has_warning_code(warnings, "model_asset_unavailable"):
        return _strict_error_detail(
            reason_code="model_asset_unavailable",
            message="Required model assets are unavailable for strict routing policy.",
            warnings=warnings,
        )
    if epsilon_requested:
        return _strict_error_detail(
            reason_code="epsilon_infeasible",
            message="No routes satisfy epsilon constraints for this request.",
            warnings=warnings,
        )
    return None


def _strict_error_text(detail: dict[str, Any]) -> str:
    reason_code = normalize_reason_code(str(detail.get("reason_code", "unknown_error")).strip())
    message = str(detail.get("message", "Request failed")).strip() or "Request failed"
    warning_summary = ""
    warnings_value = detail.get("warnings")
    if isinstance(warnings_value, list) and warnings_value:
        warning_text = str(warnings_value[0]).strip()
        if warning_text:
            warning_summary = f"; warning={warning_text}"
    return f"reason_code:{reason_code}; message:{message}{warning_summary}"


def _stream_fatal_event_from_detail(detail: dict[str, Any]) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": "fatal",
        "reason_code": normalize_reason_code(str(detail.get("reason_code", "internal_error"))),
        "message": str(detail.get("message", "Request failed")),
        "warnings": [],
    }
    warnings_value = detail.get("warnings")
    if isinstance(warnings_value, list):
        event["warnings"] = warnings_value
    for key in (
        "terrain_coverage_min_observed",
        "terrain_coverage_required",
        "terrain_dem_version",
        "warmup",
        "retry_after_seconds",
        "retry_hint",
        "phase",
        "last_error",
        "asset_path",
        "asset_size_mb",
        "stage",
        "stage_detail",
    ):
        if key in detail:
            event[key] = detail[key]
    return event


def _stream_fatal_event_from_exception(exc: Exception) -> dict[str, Any]:
    detail = getattr(exc, "detail", None)
    if isinstance(detail, dict):
        return _stream_fatal_event_from_detail(detail)
    message = str(detail if detail is not None else exc).strip()
    lowered = message.lower()
    reason_code = "internal_error"
    if "epsilon" in lowered and "infeasible" in lowered:
        reason_code = "epsilon_infeasible"
    elif "route_compute_timeout" in lowered or ("timeout" in lowered and "route compute" in lowered):
        reason_code = "route_compute_timeout"
    elif ("terrain" in lowered and "unsupported" in lowered) or "terrain_region_unsupported" in lowered:
        reason_code = "terrain_region_unsupported"
    elif ("terrain" in lowered and "asset" in lowered) or "terrain_dem_asset_unavailable" in lowered:
        reason_code = "terrain_dem_asset_unavailable"
    elif "terrain" in lowered and ("coverage" in lowered or "dem" in lowered):
        reason_code = "terrain_dem_coverage_insufficient"
    elif "toll_tariff_unavailable" in lowered:
        reason_code = "toll_tariff_unavailable"
    elif "routing_graph_fragmented" in lowered:
        reason_code = "routing_graph_fragmented"
    elif "routing_graph_disconnected_od" in lowered:
        reason_code = "routing_graph_disconnected_od"
    elif "routing_graph_coverage_gap" in lowered:
        reason_code = "routing_graph_coverage_gap"
    elif "routing_graph_precheck_timeout" in lowered:
        reason_code = "routing_graph_precheck_timeout"
    elif "routing_graph_no_path" in lowered:
        reason_code = "routing_graph_no_path"
    elif "live_source_refresh_failed" in lowered:
        reason_code = "live_source_refresh_failed"
    elif "routing_graph_unavailable" in lowered:
        reason_code = "routing_graph_unavailable"
    elif "routing_graph_warming_up" in lowered or ("graph" in lowered and "warming" in lowered):
        reason_code = "routing_graph_warming_up"
    elif "routing_graph_warmup_failed" in lowered or ("graph" in lowered and "warmup" in lowered and "failed" in lowered):
        reason_code = "routing_graph_warmup_failed"
    elif "tariff" in lowered or "toll" in lowered:
        reason_code = "toll_tariff_unresolved"
    elif "departure_profile" in lowered or "departure profile" in lowered:
        reason_code = "departure_profile_unavailable"
    elif "holiday_data_unavailable" in lowered or "holiday data" in lowered:
        reason_code = "holiday_data_unavailable"
    elif "stochastic_calibration_unavailable" in lowered or "stochastic calibration" in lowered:
        reason_code = "stochastic_calibration_unavailable"
    elif "scenario_profile_unavailable" in lowered or "scenario profile" in lowered and "unavailable" in lowered:
        reason_code = "scenario_profile_unavailable"
    elif "scenario_profile_invalid" in lowered or "scenario profile" in lowered and "invalid" in lowered:
        reason_code = "scenario_profile_invalid"
    elif "risk_normalization_unavailable" in lowered:
        reason_code = "risk_normalization_unavailable"
    elif "risk_prior_unavailable" in lowered:
        reason_code = "risk_prior_unavailable"
    elif "vehicle_profile_unavailable" in lowered:
        reason_code = "vehicle_profile_unavailable"
    elif "vehicle_profile_invalid" in lowered:
        reason_code = "vehicle_profile_invalid"
    elif "fuel_price_auth_unavailable" in lowered:
        reason_code = "fuel_price_auth_unavailable"
    elif "fuel_price_source_unavailable" in lowered:
        reason_code = "fuel_price_source_unavailable"
    elif "carbon_policy_unavailable" in lowered:
        reason_code = "carbon_policy_unavailable"
    elif "carbon_intensity_unavailable" in lowered:
        reason_code = "carbon_intensity_unavailable"
    elif "toll_topology_unavailable" in lowered:
        reason_code = "toll_topology_unavailable"
    elif "model_asset_unavailable" in lowered:
        reason_code = "model_asset_unavailable"
    return {
        "type": "fatal",
        "reason_code": normalize_reason_code(reason_code),
        "message": message or type(exc).__name__,
        "warnings": [],
    }


def _option_joint_utility_stats(option: RouteOption) -> tuple[float, float]:
    utility_mean, _utility_q95, utility_cvar95 = _option_joint_utility_distribution(option)
    return utility_mean, utility_cvar95


def _option_objective_distribution(
    option: RouteOption,
    objective: str,
    *,
    optimization_mode: OptimizationMode = "expected_value",
    risk_aversion: float = 1.0,
) -> tuple[float, float, float]:
    deterministic = _option_objective_value(
        option,
        objective,
        optimization_mode=optimization_mode,
        risk_aversion=risk_aversion,
    )
    uncertainty = option.uncertainty or {}
    if objective == "duration":
        mean_key = "mean_duration_s"
        q95_key = "q95_duration_s"
        cvar_key = "cvar95_duration_s"
    elif objective == "money":
        mean_key = "mean_monetary_cost"
        q95_key = "q95_monetary_cost"
        cvar_key = "cvar95_monetary_cost"
    else:
        mean_key = "mean_emissions_kg"
        q95_key = "q95_emissions_kg"
        cvar_key = "cvar95_emissions_kg"
    mean_value = float(uncertainty.get(mean_key, deterministic))
    q95_value = max(mean_value, float(uncertainty.get(q95_key, mean_value)))
    cvar_value = max(q95_value, float(uncertainty.get(cvar_key, q95_value)))
    return mean_value, q95_value, cvar_value


def _option_joint_utility_distribution(option: RouteOption) -> tuple[float, float, float]:
    uncertainty = option.uncertainty or {}
    utility_mean = uncertainty.get("utility_mean")
    utility_q95 = uncertainty.get("utility_q95")
    utility_cvar95 = uncertainty.get("utility_cvar95")
    if utility_mean is None or utility_q95 is None or utility_cvar95 is None:
        raise ModelDataError(
            reason_code="risk_prior_unavailable",
            message=(
                "Canonical uncertainty utility fields are required in strict runtime "
                "(utility_mean, utility_q95, and utility_cvar95)."
            ),
        )
    mean_value = float(utility_mean)
    q95_value = max(mean_value, float(utility_q95))
    cvar_value = max(q95_value, float(utility_cvar95))
    return mean_value, q95_value, cvar_value


def _option_joint_robust_utility(option: RouteOption, *, risk_aversion: float) -> float:
    utility_mean, utility_cvar95 = _option_joint_utility_stats(option)
    return robust_objective(
        mean_value=utility_mean,
        cvar_value=utility_cvar95,
        risk_aversion=risk_aversion,
        risk_family=settings.risk_family,
        risk_theta=settings.risk_family_theta,
    )


def _robust_utility_vector(
    option: RouteOption,
    *,
    risk_aversion: float,
) -> tuple[float, float, float, float, float, float, float, float, float, float, float, float]:
    utility_mean, utility_q95, utility_cvar95 = _option_joint_utility_distribution(option)
    dur_mean, dur_q95, dur_cvar95 = _option_objective_distribution(
        option,
        "duration",
        optimization_mode="expected_value",
        risk_aversion=risk_aversion,
    )
    money_mean, money_q95, money_cvar95 = _option_objective_distribution(
        option,
        "money",
        optimization_mode="expected_value",
        risk_aversion=risk_aversion,
    )
    co2_mean, co2_q95, co2_cvar95 = _option_objective_distribution(
        option,
        "co2",
        optimization_mode="expected_value",
        risk_aversion=risk_aversion,
    )
    return (
        utility_cvar95,
        utility_q95,
        utility_mean,
        dur_cvar95,
        money_cvar95,
        co2_cvar95,
        dur_q95,
        money_q95,
        co2_q95,
        dur_mean,
        money_mean,
        co2_mean,
    )


def _strictly_dominates(lhs: tuple[float, ...], rhs: tuple[float, ...]) -> bool:
    if len(lhs) != len(rhs):
        return False
    le_all = True
    lt_any = False
    for l_item, r_item in zip(lhs, rhs, strict=True):
        if l_item > r_item:
            le_all = False
            break
        if l_item < r_item:
            lt_any = True
    return le_all and lt_any


def _nondominated_indices(vectors: list[tuple[float, ...]]) -> list[int]:
    out: list[int] = []
    for idx, candidate in enumerate(vectors):
        dominated = False
        for jdx, other in enumerate(vectors):
            if idx == jdx:
                continue
            if _strictly_dominates(other, candidate):
                dominated = True
                break
        if not dominated:
            out.append(idx)
    return out


def _option_objective_samples(option: RouteOption) -> tuple[tuple[float, float, float, float], ...]:
    raw: Any = None
    sample_meta = option.uncertainty_samples_meta
    if isinstance(sample_meta, dict):
        raw_json = sample_meta.get("objective_samples_json")
        if isinstance(raw_json, str) and raw_json.strip():
            try:
                parsed = json.loads(raw_json)
                if isinstance(parsed, list):
                    raw = parsed
            except Exception:
                raw = None
    if raw is None:
        uncertainty = option.uncertainty
        if not isinstance(uncertainty, dict):
            return ()
        raw = uncertainty.get("objective_samples")
    if not isinstance(raw, list):
        return ()
    out: list[tuple[float, float, float, float]] = []
    for row in raw:
        if not isinstance(row, list | tuple) or len(row) < 4:
            continue
        try:
            out.append((float(row[0]), float(row[1]), float(row[2]), float(row[3])))
        except (TypeError, ValueError):
            continue
    return tuple(out)


def _sample_vector_dominates(lhs: tuple[float, float, float, float], rhs: tuple[float, float, float, float]) -> bool:
    le_all = True
    lt_any = False
    for l_val, r_val in zip(lhs, rhs, strict=True):
        if l_val > (r_val + 1e-9):
            le_all = False
            break
        if l_val + 1e-9 < r_val:
            lt_any = True
    return le_all and lt_any


def _stochastic_dominance_probability(
    lhs_samples: tuple[tuple[float, float, float, float], ...],
    rhs_samples: tuple[tuple[float, float, float, float], ...],
    *,
    pair_samples: int,
) -> float:
    if not lhs_samples or not rhs_samples:
        return 0.0
    lhs_n = len(lhs_samples)
    rhs_n = len(rhs_samples)
    pair_budget = max(16, min(int(pair_samples), lhs_n * rhs_n))
    dominate_hits = 0
    # Deterministic stratified quasi-random pairing to avoid modular-index bias.
    golden = 0.6180339887498949
    silver = 0.4142135623730950
    offset_seed = ((lhs_n * 1315423911) ^ (rhs_n * 2654435761) ^ int(pair_budget * 11400714819323198485)) & 0xFFFFFFFF
    base_offset = float(offset_seed) / float(0xFFFFFFFF)
    for idx in range(pair_budget):
        lhs_u = (base_offset + ((float(idx) + 0.5) / float(pair_budget)) + (golden * 0.5)) % 1.0
        rhs_u = (base_offset + (float(idx) * silver) + (golden * 0.25)) % 1.0
        lhs_idx = min(lhs_n - 1, max(0, int(lhs_u * float(lhs_n))))
        rhs_idx = min(rhs_n - 1, max(0, int(rhs_u * float(rhs_n))))
        if _sample_vector_dominates(lhs_samples[lhs_idx], rhs_samples[rhs_idx]):
            dominate_hits += 1
    return float(dominate_hits) / float(pair_budget)


def _robust_pairwise_dominance_matrix(
    options: list[RouteOption],
    *,
    risk_aversion: float,
) -> tuple[list[list[float]], list[tuple[float, ...]], list[tuple[tuple[float, float, float, float], ...]]]:
    n = len(options)
    robust_vectors = [_robust_utility_vector(option, risk_aversion=risk_aversion) for option in options]
    objective_samples = [_option_objective_samples(option) for option in options]
    pair_samples = int(max(16, min(int(settings.risk_dominance_pair_samples), 2048)))
    matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            lhs_samples = objective_samples[i]
            rhs_samples = objective_samples[j]
            if lhs_samples and rhs_samples:
                p_ij = _stochastic_dominance_probability(lhs_samples, rhs_samples, pair_samples=pair_samples)
                p_ji = _stochastic_dominance_probability(rhs_samples, lhs_samples, pair_samples=pair_samples)
            else:
                p_ij = 1.0 if _strictly_dominates(robust_vectors[i], robust_vectors[j]) else 0.0
                p_ji = 1.0 if _strictly_dominates(robust_vectors[j], robust_vectors[i]) else 0.0
            matrix[i][j] = p_ij
            matrix[j][i] = p_ji
    return matrix, robust_vectors, objective_samples


def _is_probabilistically_dominant(
    p_ab: float,
    p_ba: float,
    *,
    threshold: float,
) -> bool:
    return (p_ab >= threshold) and ((p_ab - p_ba) >= 0.02)


def _robust_nondominated_indices(
    options: list[RouteOption],
    *,
    risk_aversion: float,
) -> tuple[list[int], list[list[float]], list[tuple[float, ...]]]:
    matrix, robust_vectors, _samples = _robust_pairwise_dominance_matrix(options, risk_aversion=risk_aversion)
    threshold = float(max(0.50, min(0.99, settings.risk_dominance_min_probability)))
    out: list[int] = []
    for idx in range(len(options)):
        dominated = False
        for jdx in range(len(options)):
            if idx == jdx:
                continue
            if _is_probabilistically_dominant(matrix[jdx][idx], matrix[idx][jdx], threshold=threshold):
                dominated = True
                break
        if not dominated:
            out.append(idx)
    return out, matrix, robust_vectors


def _robust_rank_key(
    *,
    idx: int,
    options: list[RouteOption],
    dominance_matrix: list[list[float]],
    robust_vectors: list[tuple[float, ...]],
    risk_aversion: float,
) -> tuple[float | str, ...]:
    threshold = float(max(0.50, min(0.99, settings.risk_dominance_min_probability)))
    n = len(options)
    if n <= 1:
        loss_score = 0.0
        gain_score = 0.0
    else:
        loss_score = 0.0
        gain_acc = 0.0
        for jdx in range(n):
            if jdx == idx:
                continue
            p_ij = dominance_matrix[idx][jdx]
            p_ji = dominance_matrix[jdx][idx]
            loss_score = max(loss_score, max(0.0, p_ji - p_ij))
            if _is_probabilistically_dominant(p_ij, p_ji, threshold=threshold):
                gain_acc += (p_ij - p_ji)
        gain_score = gain_acc / max(1.0, float(n - 1))
    vector = robust_vectors[idx]
    option = options[idx]
    return (
        float(loss_score),
        -float(gain_score),
        *vector,
        _option_joint_robust_utility(option, risk_aversion=risk_aversion),
        option.id,
    )


def _option_objective_value(
    option: RouteOption,
    objective: str,
    *,
    optimization_mode: OptimizationMode = "expected_value",
    risk_aversion: float = 1.0,
) -> float:
    if objective == "duration":
        deterministic = float(option.metrics.duration_s)
        mean_key = "mean_duration_s"
    elif objective == "money":
        deterministic = float(option.metrics.monetary_cost)
        mean_key = "mean_monetary_cost"
    else:
        deterministic = float(option.metrics.emissions_kg)
        mean_key = "mean_emissions_kg"

    if not option.uncertainty:
        return deterministic

    if optimization_mode == "robust":
        # Robust mode uses joint utility for ranking/frontier. Objective-level values remain
        # expected-value metrics for diagnostics/tie-breaks.
        return float(option.uncertainty.get(mean_key, deterministic))

    mean_value = float(option.uncertainty.get(mean_key, deterministic))
    return mean_value


def _pick_best_option(
    options: list[RouteOption],
    *,
    w_time: float,
    w_money: float,
    w_co2: float,
    optimization_mode: OptimizationMode = "expected_value",
    risk_aversion: float = 1.0,
) -> RouteOption:
    wt, wm, we = normalise_weights(w_time, w_money, w_co2)
    if optimization_mode == "robust":
        nondominated, dominance_matrix, robust_vectors = _robust_nondominated_indices(
            options,
            risk_aversion=risk_aversion,
        )
        contenders = (
            [options[idx] for idx in nondominated]
            if nondominated
            else list(options)
        )
        index_by_id = {option.id: idx for idx, option in enumerate(options)}
        contenders.sort(
            key=lambda option: _robust_rank_key(
                idx=int(index_by_id.get(option.id, 0)),
                options=options,
                dominance_matrix=dominance_matrix,
                robust_vectors=robust_vectors,
                risk_aversion=risk_aversion,
            )
        )
        if contenders:
            return contenders[0]

    times = [
        _option_objective_value(
            option,
            "duration",
            optimization_mode=optimization_mode,
            risk_aversion=risk_aversion,
        )
        for option in options
    ]
    moneys = [
        _option_objective_value(
            option,
            "money",
            optimization_mode=optimization_mode,
            risk_aversion=risk_aversion,
        )
        for option in options
    ]
    emissions = [
        _option_objective_value(
            option,
            "co2",
            optimization_mode=optimization_mode,
            risk_aversion=risk_aversion,
        )
        for option in options
    ]
    distances = [float(max(0.0, option.metrics.distance_km)) for option in options]

    t_min, t_max = min(times), max(times)
    m_min, m_max = min(moneys), max(moneys)
    e_min, e_max = min(emissions), max(emissions)
    d_min, d_max = min(distances), max(distances)

    def _norm(value: float, min_value: float, max_value: float) -> float:
        return 0.0 if max_value <= min_value else (value - min_value) / (max_value - min_value)

    math_profile = str(settings.route_selection_math_profile or "modified_vikor_distance").strip().lower()
    regret_weight = max(0.0, float(settings.route_selection_modified_regret_weight))
    balance_weight = max(0.0, float(settings.route_selection_modified_balance_weight))
    distance_weight = max(0.0, float(settings.route_selection_modified_distance_weight))
    eta_distance_weight = max(0.0, float(settings.route_selection_modified_eta_distance_weight))
    entropy_weight = max(0.0, float(settings.route_selection_modified_entropy_weight))
    knee_weight = max(0.0, float(settings.route_selection_modified_knee_weight))
    tchebycheff_rho = max(0.0, float(settings.route_selection_tchebycheff_rho))
    vikor_v = min(1.0, max(0.0, float(settings.route_selection_vikor_v)))

    norm_rows = [
        (
            _norm(time_v, t_min, t_max),
            _norm(money_v, m_min, m_max),
            _norm(co2_v, e_min, e_max),
            _norm(distance_v, d_min, d_max),
        )
        for time_v, money_v, co2_v, distance_v in zip(
            times,
            moneys,
            emissions,
            distances,
            strict=True,
        )
    ]

    weighted_sum_rows = [
        (wt * n_time) + (wm * n_money) + (we * n_co2)
        for n_time, n_money, n_co2, _ in norm_rows
    ]
    weighted_regret_rows = [
        max(wt * n_time, wm * n_money, we * n_co2)
        for n_time, n_money, n_co2, _ in norm_rows
    ]
    vikor_s_min = min(weighted_sum_rows)
    vikor_s_max = max(weighted_sum_rows)
    vikor_r_min = min(weighted_regret_rows)
    vikor_r_max = max(weighted_regret_rows)

    def _safe_scale(value: float, min_value: float, max_value: float) -> float:
        return 0.0 if max_value <= min_value else (value - min_value) / (max_value - min_value)

    best = options[0]
    best_tuple = (float("inf"), float("inf"), "")
    for option, time_v, money_v, co2_v, distance_v, norms, weighted_sum, weighted_regret in zip(
        options,
        times,
        moneys,
        emissions,
        distances,
        norm_rows,
        weighted_sum_rows,
        weighted_regret_rows,
        strict=True,
    ):
        n_time, n_money, n_co2, n_distance = norms

        # Academic and modified route-selection formulas used after strict feasibility and Pareto:
        #
        # Academic references (unchanged baseline formulas):
        # 1) Weighted-sum scalarisation:
        #    Marler & Arora (2010) https://doi.org/10.1007/s00158-009-0460-7
        # 2) Augmented Tchebycheff scalarisation (L-infinity regret + epsilon term):
        #    Steuer & Choo (1983) https://doi.org/10.1007/BF02591962
        # 3) VIKOR compromise score (utility/regret mixture using group utility S and
        #    individual regret R, normalised over the candidate set):
        #    Opricovic & Tzeng (2004) https://doi.org/10.1016/S0377-2217(03)00020-1
        #
        # Modified engineering profiles (deliberate practical extension):
        # 4) Distance-as-objective influence for route choice in multi-criteria routing:
        #    Martins (1984) https://doi.org/10.1016/0377-2217(84)90202-2
        # 5) Knee-oriented preference signal (approximation of "balanced compromise"
        #    behavior near high-curvature Pareto regions):
        #    Branke et al. (2004) https://doi.org/10.1007/978-3-540-30217-9_73
        # 6) Entropy reward to prefer routes improving multiple objectives together:
        #    Shannon (1948) https://doi.org/10.1002/j.1538-7305.1948.tb01338.x
        #
        # Implementation note:
        # - The strict fail-closed gates, live-source validation, and Pareto filtering are
        #   unchanged.
        # - The formulas below only pick one highlighted route from an already-feasible set.
        # - `modified_*` profiles are intentionally not claimed as novel theory; they are
        #   transparent engineering blends of known multi-objective building blocks.
        mean_norm = (n_time + n_money + n_co2) / 3.0
        balance_penalty = math.sqrt(
            max(
                0.0,
                ((n_time - mean_norm) ** 2 + (n_money - mean_norm) ** 2 + (n_co2 - mean_norm) ** 2)
                / 3.0,
            )
        )
        knee_penalty = (
            abs(n_time - n_money)
            + abs(n_time - n_co2)
            + abs(n_money - n_co2)
        ) / 3.0
        eta_distance_penalty = math.sqrt(max(0.0, n_time * n_distance))
        improve_time = max(1e-6, 1.0 - n_time)
        improve_money = max(1e-6, 1.0 - n_money)
        improve_co2 = max(1e-6, 1.0 - n_co2)
        improve_sum = improve_time + improve_money + improve_co2
        p_time = improve_time / improve_sum
        p_money = improve_money / improve_sum
        p_co2 = improve_co2 / improve_sum
        entropy_reward = -(
            (p_time * math.log(p_time))
            + (p_money * math.log(p_money))
            + (p_co2 * math.log(p_co2))
        ) / math.log(3.0)
        vikor_q = (vikor_v * _safe_scale(weighted_sum, vikor_s_min, vikor_s_max)) + (
            (1.0 - vikor_v) * _safe_scale(weighted_regret, vikor_r_min, vikor_r_max)
        )

        if math_profile == "academic_reference":
            score = weighted_sum
        elif math_profile == "academic_tchebycheff":
            score = weighted_regret + (tchebycheff_rho * weighted_sum)
        elif math_profile == "academic_vikor":
            score = vikor_q
        elif math_profile == "modified_hybrid":
            score = weighted_sum + (regret_weight * weighted_regret) + (balance_weight * balance_penalty)
        elif math_profile == "modified_vikor_distance":
            score = (
                vikor_q
                + (balance_weight * balance_penalty)
                + (distance_weight * n_distance)
                + (eta_distance_weight * eta_distance_penalty)
                + (knee_weight * knee_penalty)
                - (entropy_weight * entropy_reward)
            )
        else:
            score = (
                weighted_sum
                + (regret_weight * weighted_regret)
                + (balance_weight * balance_penalty)
                + (distance_weight * n_distance)
                + (eta_distance_weight * eta_distance_penalty)
                + (knee_weight * knee_penalty)
                - (entropy_weight * entropy_reward)
            )
        tie_break = (
            score,
            distance_v,
            _option_objective_value(
                option,
                "duration",
                optimization_mode=optimization_mode,
                risk_aversion=risk_aversion,
            ),
            option.id,
        )
        if tie_break < best_tuple:
            best = option
            best_tuple = tie_break
    return best


def _sort_options_by_mode(
    options: list[RouteOption],
    *,
    optimization_mode: OptimizationMode = "expected_value",
    risk_aversion: float = 1.0,
) -> list[RouteOption]:
    if optimization_mode != "robust":
        return options

    nondominated, dominance_matrix, robust_vectors = _robust_nondominated_indices(
        options,
        risk_aversion=risk_aversion,
    )
    ordered_pool = (
        [options[idx] for idx in nondominated]
        if nondominated
        else list(options)
    )
    index_by_id = {option.id: idx for idx, option in enumerate(options)}
    return sorted(
        ordered_pool,
        key=lambda option: _robust_rank_key(
            idx=int(index_by_id.get(option.id, 0)),
            options=options,
            dominance_matrix=dominance_matrix,
            robust_vectors=robust_vectors,
            risk_aversion=risk_aversion,
        ),
    )


def _pareto_objective_key(
    option: RouteOption,
    *,
    optimization_mode: OptimizationMode = "expected_value",
    risk_aversion: float = 1.0,
) -> tuple[float, ...]:
    if optimization_mode == "robust":
        return _robust_utility_vector(option, risk_aversion=risk_aversion)
    return (
        _option_objective_value(
            option,
            "duration",
            optimization_mode=optimization_mode,
            risk_aversion=risk_aversion,
        ),
        _option_objective_value(
            option,
            "money",
            optimization_mode=optimization_mode,
            risk_aversion=risk_aversion,
        ),
        _option_objective_value(
            option,
            "co2",
            optimization_mode=optimization_mode,
            risk_aversion=risk_aversion,
        ),
    )


def _finalize_pareto_options(
    options: list[RouteOption],
    *,
    max_alternatives: int,
    pareto_method: ParetoMethod = "dominance",
    epsilon: EpsilonConstraints | None = None,
    optimization_mode: OptimizationMode = "expected_value",
    risk_aversion: float = 1.0,
) -> list[RouteOption]:
    objective_fn = lambda option: _pareto_objective_key(
        option,
        optimization_mode=optimization_mode,
        risk_aversion=risk_aversion,
    )
    pareto_input = list(options)
    if optimization_mode == "robust" and pareto_input:
        robust_nondominated, _dom_matrix, _vectors = _robust_nondominated_indices(
            pareto_input,
            risk_aversion=risk_aversion,
        )
        if robust_nondominated:
            pareto_input = [pareto_input[idx] for idx in robust_nondominated]
    pareto = select_pareto_routes(
        pareto_input,
        max_alternatives=max_alternatives,
        pareto_method=pareto_method,
        epsilon=epsilon,
        strict_frontier=True,
        objective_key=objective_fn,
    )
    sorted_pareto = _sort_options_by_mode(
        pareto,
        optimization_mode=optimization_mode,
        risk_aversion=risk_aversion,
    )
    if not bool(settings.route_pareto_backfill_enabled):
        return sorted_pareto
    target_count = max(
        1,
        min(
            max(1, int(max_alternatives)),
            max(
                int(len(sorted_pareto)),
                int(settings.route_pareto_backfill_min_alternatives),
            ),
        ),
    )
    if len(sorted_pareto) >= target_count:
        return sorted_pareto
    backfill_pool = list(options)
    if pareto_method == "epsilon_constraint" and epsilon is not None:
        backfill_pool = filter_by_epsilon(backfill_pool, epsilon, objective_key=objective_fn)
    if not backfill_pool:
        return sorted_pareto
    ranked_pool = _sort_options_by_mode(
        backfill_pool,
        optimization_mode=optimization_mode,
        risk_aversion=risk_aversion,
    )
    seen_ids = {option.id for option in sorted_pareto}
    for option in ranked_pool:
        if option.id in seen_ids:
            continue
        sorted_pareto.append(option)
        seen_ids.add(option.id)
        if len(sorted_pareto) >= target_count:
            break
    return sorted_pareto


async def _route_graph_candidate_routes_async(
    *,
    origin_lat: float,
    origin_lon: float,
    destination_lat: float,
    destination_lon: float,
    max_paths: int,
    scenario_edge_modifiers: dict[str, Any] | None = None,
    max_hops_override: int | None = None,
    max_state_budget_override: int | None = None,
    use_transition_state: bool = True,
    search_deadline_s: float | None = None,
    start_node_id: str | None = None,
    goal_node_id: str | None = None,
) -> tuple[list[dict[str, Any]], Any]:
    # Route-graph path enumeration can be CPU heavy for long corridors. Running it in a worker
    # thread keeps the event loop responsive so streaming heartbeats and health checks continue.
    kwargs: dict[str, Any] = {
        "origin_lat": origin_lat,
        "origin_lon": origin_lon,
        "destination_lat": destination_lat,
        "destination_lon": destination_lon,
        "max_paths": max_paths,
        "scenario_edge_modifiers": scenario_edge_modifiers,
        "max_hops_override": max_hops_override,
        "max_state_budget_override": max_state_budget_override,
        "use_transition_state": bool(use_transition_state),
        "search_deadline_s": search_deadline_s,
    }
    if start_node_id is not None:
        kwargs["start_node_id"] = start_node_id
    if goal_node_id is not None:
        kwargs["goal_node_id"] = goal_node_id
    return await asyncio.to_thread(route_graph_candidate_routes, **kwargs)


_ROUTE_GRAPH_SEARCH_NO_PATH_REASONS: frozenset[str] = frozenset(
    {
        "no_path",
        "path_search_exhausted",
        "start_or_goal_blocked",
        "candidate_pool_exhausted",
        "state_budget_exceeded",
    }
)


def _normalize_route_graph_search_reason(reason: str) -> str:
    code = str(reason or "").strip()
    if code == "routing_graph_no_path" or code in _ROUTE_GRAPH_SEARCH_NO_PATH_REASONS:
        return "routing_graph_no_path"
    return normalize_reason_code(code, default="routing_graph_unavailable")


def _route_graph_precheck_warning(result: dict[str, Any]) -> str:
    reason_code = normalize_reason_code(
        str(result.get("reason_code", "routing_graph_unavailable")).strip(),
        default="routing_graph_unavailable",
    )
    message = str(result.get("message", "Route graph feasibility check failed.")).strip()
    if reason_code == "routing_graph_coverage_gap":
        origin_dist = result.get("origin_nearest_distance_m")
        destination_dist = result.get("destination_nearest_distance_m")
        max_dist = result.get("max_nearest_node_distance_m")
        message = (
            f"{message} "
            f"(origin_nearest_m={origin_dist}, destination_nearest_m={destination_dist}, max_nearest_m={max_dist})."
        )
    elif reason_code == "routing_graph_disconnected_od":
        message = (
            f"{message} "
            f"(origin_component={result.get('origin_component')}, "
            f"destination_component={result.get('destination_component')}, "
            f"origin_node_id={result.get('origin_node_id')}, "
            f"destination_node_id={result.get('destination_node_id')}, "
            f"origin_nearest_m={result.get('origin_nearest_distance_m')}, "
            f"destination_nearest_m={result.get('destination_nearest_distance_m')}, "
            f"origin_candidates={result.get('origin_candidate_count')}, "
            f"destination_candidates={result.get('destination_candidate_count')}, "
            f"candidate_search_radius={result.get('candidate_search_radius')}, "
            f"candidate_radius_cap={result.get('candidate_search_radius_cap')})."
        )
        selected_component = result.get("selected_component")
        if selected_component is not None:
            message = (
                f"{message} "
                f"(selected_component={selected_component}, "
                f"selected_component_size={result.get('selected_component_size')}, "
                f"origin_selected_m={result.get('origin_selected_distance_m')}, "
                f"destination_selected_m={result.get('destination_selected_distance_m')})."
            )
    elif reason_code == "routing_graph_fragmented":
        message = (
            f"{message} "
            f"(component_count={result.get('component_count')}, "
            f"largest_component_nodes={result.get('largest_component_nodes')}, "
            f"largest_component_ratio={result.get('largest_component_ratio')})."
        )
    elif reason_code == "routing_graph_precheck_timeout":
        message = (
            f"{message} "
            f"(timeout_ms={result.get('timeout_ms')}, elapsed_ms={result.get('elapsed_ms')})."
        )
    return f"route_graph: {reason_code} ({message})"


def _as_float_or_zero(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(parsed):
        return 0.0
    return float(parsed)


def _as_int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _prefetch_failed_details_text(payload: dict[str, Any] | None) -> str:
    data = payload if isinstance(payload, dict) else {}
    rows = data.get("failed_source_details")
    if not isinstance(rows, list):
        return ""
    parts: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = str(row.get("source_key", "")).strip()
        reason = str(row.get("reason_code", "")).strip()
        detail = str(row.get("detail", "")).strip()
        detail_data = row.get("detail_data") if isinstance(row.get("detail_data"), dict) else {}
        if not key:
            continue
        section = key
        if reason:
            section = f"{section}:{reason}"
        if reason == "terrain_dem_coverage_insufficient" and isinstance(detail_data, dict):
            probe_count = _as_int_or_zero(detail_data.get("probe_count"))
            covered_count = _as_int_or_zero(detail_data.get("covered_count"))
            min_required = _as_int_or_zero(detail_data.get("min_covered_points_required"))
            if probe_count > 0:
                detail = (
                    f"{detail} "
                    f"[covered_probes={covered_count}/{probe_count}; min_required={max(1, min_required)}]"
                ).strip()
        if reason == "scenario_profile_unavailable" and isinstance(detail_data, dict):
            coverage_gate_raw = detail_data.get("coverage_gate")
            coverage_gate = coverage_gate_raw if isinstance(coverage_gate_raw, dict) else {}
            coverage_keys = (
                "source_ok_count",
                "required_source_count",
                "required_source_count_configured",
                "required_source_count_effective",
                "waiver_applied",
                "waiver_reason",
            )
            has_coverage_gate_data = any(key in coverage_gate for key in coverage_keys) or any(
                key in detail_data for key in coverage_keys
            )
            if has_coverage_gate_data:
                source_ok_count = _as_int_or_zero(
                    coverage_gate.get("source_ok_count", detail_data.get("source_ok_count", 0))
                )
                required_configured = _as_int_or_zero(
                    coverage_gate.get(
                        "required_source_count_configured",
                        detail_data.get("required_source_count", 0),
                    )
                )
                required_effective = _as_int_or_zero(
                    coverage_gate.get(
                        "required_source_count_effective",
                        detail_data.get("required_source_count", 0),
                    )
                )
                waiver_applied = bool(coverage_gate.get("waiver_applied", False))
                waiver_reason = str(coverage_gate.get("waiver_reason", "")).strip()
                road_hint = str(
                    detail_data.get(
                        "resolved_road_hint_value",
                        (
                            detail_data.get("route_context", {}).get("road_hint")
                            if isinstance(detail_data.get("route_context"), dict)
                            else ""
                        ),
                    )
                    or ""
                ).strip()
                webtris_used_sites = _as_int_or_zero(detail_data.get("webtris_used_site_count"))
                dft_selected_station_count = _as_int_or_zero(
                    detail_data.get("dft_selected_station_count", detail_data.get("dft_selected_count"))
                )
                waiver_text = "yes" if waiver_applied else "no"
                if waiver_reason:
                    waiver_text = f"{waiver_text}:{waiver_reason}"
                if "source_ok=" not in detail and "required_effective=" not in detail:
                    detail = (
                        f"{detail} "
                        f"[source_ok={source_ok_count}/4; "
                        f"required_configured={required_configured}/4; "
                        f"required_effective={required_effective}/4; "
                        f"waiver={waiver_text}; "
                        f"road_hint={road_hint or 'unknown'}; "
                        f"webtris_used_sites={webtris_used_sites}; "
                        f"dft_selected_station_count={dft_selected_station_count}]"
                    ).strip()
            else:
                as_of_text = str(
                    detail_data.get("as_of_utc", detail_data.get("as_of", ""))
                ).strip()
                max_age = _as_int_or_zero(
                    detail_data.get("max_age_minutes", settings.live_scenario_coefficient_max_age_minutes)
                )
                age_minutes_text = ""
                if as_of_text:
                    try:
                        as_of_dt = datetime.fromisoformat(as_of_text.replace("Z", "+00:00"))
                        if as_of_dt.tzinfo is None:
                            as_of_dt = as_of_dt.replace(tzinfo=UTC)
                        else:
                            as_of_dt = as_of_dt.astimezone(UTC)
                        age_minutes_text = f"{max(0.0, (datetime.now(UTC) - as_of_dt).total_seconds() / 60.0):.2f}"
                    except ValueError:
                        age_minutes_text = ""
                freshness_parts: list[str] = []
                if as_of_text:
                    freshness_parts.append(f"as_of_utc={as_of_text}")
                if age_minutes_text:
                    freshness_parts.append(f"age_minutes={age_minutes_text}")
                if max_age > 0:
                    freshness_parts.append(f"max_age_minutes={max_age}")
                if freshness_parts and "max_age_minutes=" not in detail and "age_minutes=" not in detail:
                    detail = f"{detail} [{'; '.join(freshness_parts)}]".strip()
        if detail:
            section = f"{section} ({detail})"
        parts.append(section)
    return "; ".join(parts)


def _prefetch_missing_expected_text(payload: dict[str, Any] | None) -> str:
    data = payload if isinstance(payload, dict) else {}
    raw = data.get("missing_expected_sources")
    if isinstance(raw, list):
        items = [str(item).strip() for item in raw if str(item).strip()]
        return ",".join(items)
    return str(raw or "").strip()


def _prefetch_rows_json_text(payload: dict[str, Any] | None) -> str:
    data = payload if isinstance(payload, dict) else {}
    rows = data.get("rows")
    if not isinstance(rows, list):
        return ""
    compact_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        compact_rows.append(
            {
                "source_key": str(row.get("source_key", "")).strip(),
                "ok": bool(row.get("ok")),
                "reason_code": str(row.get("reason_code", "")).strip(),
                "detail": str(row.get("detail", "")).strip(),
                "duration_ms": round(_as_float_or_zero(row.get("duration_ms")), 2),
            }
        )
    try:
        return json.dumps(compact_rows, separators=(",", ":"), ensure_ascii=True)
    except Exception:
        return ""


def _prefetch_candidate_diag_kwargs(payload: dict[str, Any] | None) -> dict[str, Any]:
    data = payload if isinstance(payload, dict) else {}
    failed_keys = data.get("failed_source_keys", [])
    failed_keys_text = (
        ",".join(str(key).strip() for key in failed_keys if str(key).strip())
        if isinstance(failed_keys, list)
        else str(failed_keys or "").strip()
    )
    scenario_gate_required_configured = 0
    scenario_gate_required_effective = 0
    scenario_gate_source_ok_count = 0
    scenario_gate_waiver_applied = False
    scenario_gate_waiver_reason = ""
    scenario_gate_source_signal_json = ""
    scenario_gate_source_reachability_json = ""
    scenario_gate_road_hint = ""
    failed_source_details = data.get("failed_source_details", [])
    if isinstance(failed_source_details, list):
        for row in failed_source_details:
            if not isinstance(row, dict):
                continue
            if str(row.get("source_key", "")).strip() != "scenario_live_context":
                continue
            detail_data = row.get("detail_data")
            if not isinstance(detail_data, dict):
                continue
            coverage_gate_raw = detail_data.get("coverage_gate")
            coverage_gate = coverage_gate_raw if isinstance(coverage_gate_raw, dict) else {}
            scenario_gate_required_configured = _as_int_or_zero(
                coverage_gate.get(
                    "required_source_count_configured",
                    detail_data.get("required_source_count_configured", detail_data.get("required_source_count", 0)),
                )
            )
            scenario_gate_required_effective = _as_int_or_zero(
                coverage_gate.get(
                    "required_source_count_effective",
                    detail_data.get("required_source_count_effective", detail_data.get("required_source_count", 0)),
                )
            )
            scenario_gate_source_ok_count = _as_int_or_zero(
                coverage_gate.get("source_ok_count", detail_data.get("source_ok_count", 0))
            )
            scenario_gate_waiver_applied = bool(coverage_gate.get("waiver_applied", False))
            scenario_gate_waiver_reason = str(coverage_gate.get("waiver_reason", "")).strip()
            signal_set = coverage_gate.get("source_signal_set", detail_data.get("source_signal_set", {}))
            reachability_set = coverage_gate.get(
                "source_reachability_set",
                detail_data.get("source_reachability_set", {}),
            )
            if isinstance(signal_set, dict):
                try:
                    scenario_gate_source_signal_json = json.dumps(
                        signal_set,
                        separators=(",", ":"),
                        ensure_ascii=True,
                        sort_keys=True,
                    )
                except Exception:
                    scenario_gate_source_signal_json = ""
            if isinstance(reachability_set, dict):
                try:
                    scenario_gate_source_reachability_json = json.dumps(
                        reachability_set,
                        separators=(",", ":"),
                        ensure_ascii=True,
                        sort_keys=True,
                    )
                except Exception:
                    scenario_gate_source_reachability_json = ""
            scenario_gate_road_hint = str(
                detail_data.get(
                    "resolved_road_hint_value",
                    (
                        detail_data.get("route_context", {}).get("road_hint")
                        if isinstance(detail_data.get("route_context"), dict)
                        else ""
                    ),
                )
                or ""
            ).strip()
            break
    return {
        "prefetch_total_sources": _as_int_or_zero(
            data.get("source_total", data.get("prefetch_total_sources", 0))
        ),
        "prefetch_success_sources": _as_int_or_zero(
            data.get("source_success", data.get("prefetch_success_sources", 0))
        ),
        "prefetch_failed_sources": _as_int_or_zero(
            data.get("source_failed", data.get("prefetch_failed_sources", 0))
        ),
        "prefetch_failed_keys": failed_keys_text,
        "prefetch_failed_details": _prefetch_failed_details_text(data),
        "prefetch_missing_expected_sources": _prefetch_missing_expected_text(data),
        "prefetch_rows_json": _prefetch_rows_json_text(data),
        "scenario_gate_required_configured": int(scenario_gate_required_configured),
        "scenario_gate_required_effective": int(scenario_gate_required_effective),
        "scenario_gate_source_ok_count": int(scenario_gate_source_ok_count),
        "scenario_gate_waiver_applied": bool(scenario_gate_waiver_applied),
        "scenario_gate_waiver_reason": str(scenario_gate_waiver_reason),
        "scenario_gate_source_signal_json": str(scenario_gate_source_signal_json),
        "scenario_gate_source_reachability_json": str(scenario_gate_source_reachability_json),
        "scenario_gate_road_hint": str(scenario_gate_road_hint),
    }


def _candidate_precheck_diag_kwargs(result: dict[str, Any] | None) -> dict[str, Any]:
    payload = result if isinstance(result, dict) else {}
    return {
        "precheck_reason_code": str(payload.get("reason_code", "")).strip(),
        "precheck_message": str(payload.get("message", "")).strip(),
        "precheck_elapsed_ms": _as_float_or_zero(payload.get("elapsed_ms")),
        "precheck_origin_node_id": str(payload.get("origin_node_id", "")).strip(),
        "precheck_destination_node_id": str(payload.get("destination_node_id", "")).strip(),
        "precheck_origin_nearest_m": _as_float_or_zero(payload.get("origin_nearest_distance_m")),
        "precheck_destination_nearest_m": _as_float_or_zero(payload.get("destination_nearest_distance_m")),
        "precheck_origin_selected_m": _as_float_or_zero(payload.get("origin_selected_distance_m")),
        "precheck_destination_selected_m": _as_float_or_zero(payload.get("destination_selected_distance_m")),
        "precheck_origin_candidate_count": _as_int_or_zero(payload.get("origin_candidate_count")),
        "precheck_destination_candidate_count": _as_int_or_zero(payload.get("destination_candidate_count")),
        "precheck_selected_component": _as_int_or_zero(payload.get("selected_component")),
        "precheck_selected_component_size": _as_int_or_zero(payload.get("selected_component_size")),
        "precheck_gate_action": str(payload.get("gate_action", "")).strip(),
    }


def _candidate_precheck_payload(candidate_diag: CandidateDiagnostics) -> dict[str, Any] | None:
    reason_code = str(candidate_diag.precheck_reason_code or "").strip()
    if not reason_code:
        return None
    return {
        "reason_code": reason_code,
        "message": str(candidate_diag.precheck_message or ""),
        "gate_action": str(candidate_diag.precheck_gate_action or ""),
        "elapsed_ms": float(candidate_diag.precheck_elapsed_ms),
        "origin_node_id": str(candidate_diag.precheck_origin_node_id or ""),
        "destination_node_id": str(candidate_diag.precheck_destination_node_id or ""),
        "origin_nearest_m": float(candidate_diag.precheck_origin_nearest_m),
        "destination_nearest_m": float(candidate_diag.precheck_destination_nearest_m),
        "origin_selected_m": float(candidate_diag.precheck_origin_selected_m),
        "destination_selected_m": float(candidate_diag.precheck_destination_selected_m),
        "origin_candidate_count": int(candidate_diag.precheck_origin_candidate_count),
        "destination_candidate_count": int(candidate_diag.precheck_destination_candidate_count),
        "selected_component": int(candidate_diag.precheck_selected_component),
        "selected_component_size": int(candidate_diag.precheck_selected_component_size),
    }


async def _collect_candidate_routes(
    *,
    osrm: OSRMClient,
    origin: LatLng,
    destination: LatLng,
    max_routes: int,
    cache_key: str | None = None,
    scenario_edge_modifiers: dict[str, Any] | None = None,
    start_node_id: str | None = None,
    goal_node_id: str | None = None,
    progress_cb: ProgressCallback | None = None,
) -> tuple[list[dict[str, Any]], list[str], int, CandidateDiagnostics]:
    warnings: list[str] = []
    await _emit_progress(
        progress_cb,
        {
            "stage": "collecting_candidates",
            "stage_detail": "candidate_cache_lookup_start",
            "candidate_done": 0,
            "candidate_total": 0,
        },
    )
    if cache_key:
        cached = get_cached_routes(cache_key)
        if cached is not None:
            if len(cached) == 4:
                routes, warnings, spec_count, diag = cached
            else:
                routes, warnings, spec_count = cached
                diag = {
                    "raw_count": len(routes),
                    "deduped_count": len(routes),
                    "graph_explored_states": 0,
                    "graph_generated_paths": 0,
                    "graph_emitted_paths": 0,
                    "candidate_budget": int(spec_count),
                }
            await _emit_progress(
                progress_cb,
                {
                    "stage": "refining_candidates",
                    "stage_detail": "candidate_cache_hit",
                    "candidate_done": int(spec_count),
                    "candidate_total": int(spec_count),
                },
            )
            return routes, warnings, spec_count, CandidateDiagnostics(
                raw_count=int(diag.get("raw_count", len(routes))),
                deduped_count=int(diag.get("deduped_count", len(routes))),
                graph_explored_states=int(diag.get("graph_explored_states", 0)),
                graph_generated_paths=int(diag.get("graph_generated_paths", 0)),
                graph_emitted_paths=int(diag.get("graph_emitted_paths", 0)),
                candidate_budget=int(diag.get("candidate_budget", spec_count)),
                graph_effective_max_hops=int(diag.get("graph_effective_max_hops", 0)),
                graph_effective_hops_floor=int(diag.get("graph_effective_hops_floor", 0)),
                graph_effective_state_budget_initial=int(
                    diag.get("graph_effective_state_budget_initial", 0)
                ),
                graph_effective_state_budget=int(diag.get("graph_effective_state_budget", 0)),
                graph_no_path_reason=str(diag.get("graph_no_path_reason", "")),
                graph_no_path_detail=str(diag.get("graph_no_path_detail", "")),
                prefetch_total_sources=int(diag.get("prefetch_total_sources", 0)),
                prefetch_success_sources=int(diag.get("prefetch_success_sources", 0)),
                prefetch_failed_sources=int(diag.get("prefetch_failed_sources", 0)),
                prefetch_failed_keys=str(diag.get("prefetch_failed_keys", "")),
                prefetch_failed_details=str(diag.get("prefetch_failed_details", "")),
                prefetch_missing_expected_sources=str(diag.get("prefetch_missing_expected_sources", "")),
                prefetch_rows_json=str(diag.get("prefetch_rows_json", "")),
                scenario_candidate_family_count=int(diag.get("scenario_candidate_family_count", 0)),
                scenario_candidate_jaccard_vs_baseline=float(
                    diag.get("scenario_candidate_jaccard_vs_baseline", 1.0)
                ),
                scenario_candidate_jaccard_threshold=float(
                    diag.get("scenario_candidate_jaccard_threshold", 1.0)
                ),
                scenario_candidate_stress_score=float(
                    diag.get("scenario_candidate_stress_score", 0.0)
                ),
                scenario_candidate_gate_action=str(
                    diag.get("scenario_candidate_gate_action", "not_applicable")
                ),
                scenario_edge_scaling_version=str(diag.get("scenario_edge_scaling_version", "v3_live_transform")),
                precheck_gate_action=str(diag.get("precheck_gate_action", "")),
                graph_retry_attempted=bool(diag.get("graph_retry_attempted", False)),
                graph_retry_state_budget=int(diag.get("graph_retry_state_budget", 0)),
                graph_retry_outcome=str(diag.get("graph_retry_outcome", "")),
                graph_rescue_attempted=bool(diag.get("graph_rescue_attempted", False)),
                graph_rescue_mode=str(diag.get("graph_rescue_mode", "not_applicable")),
                graph_rescue_state_budget=int(diag.get("graph_rescue_state_budget", 0)),
                graph_rescue_outcome=str(diag.get("graph_rescue_outcome", "not_applicable")),
                prefetch_ms=float(diag.get("prefetch_ms", 0.0)),
                scenario_context_ms=float(diag.get("scenario_context_ms", 0.0)),
                graph_search_ms_initial=float(diag.get("graph_search_ms_initial", 0.0)),
                graph_search_ms_retry=float(diag.get("graph_search_ms_retry", 0.0)),
                graph_search_ms_rescue=float(diag.get("graph_search_ms_rescue", 0.0)),
                osrm_refine_ms=float(diag.get("osrm_refine_ms", 0.0)),
                build_options_ms=float(diag.get("build_options_ms", 0.0)),
            )

    graph_ok, graph_status = await _route_graph_status_async()
    if not graph_ok:
        status_reason_map = {
            "fragmented": "routing_graph_fragmented",
            "status_check_timeout": "routing_graph_precheck_timeout",
            "insufficient_graph_coverage": "routing_graph_coverage_gap",
        }
        reason_code = normalize_reason_code(
            status_reason_map.get(graph_status, "routing_graph_unavailable"),
            default="routing_graph_unavailable",
        )
        detail = (
            f"Route graph status check timed out (timeout_ms={settings.route_graph_status_check_timeout_ms})."
            if graph_status == "status_check_timeout"
            else f"Route graph unavailable: {graph_status}"
        )
        _record_expected_calls_blocked(
            reason_code=reason_code,
            stage="collecting_candidates",
            detail=detail,
        )
        await _emit_progress(
            progress_cb,
            {
                "stage": "collecting_candidates",
                "stage_detail": reason_code,
            },
        )
        return (
            [],
            [f"route_graph: {reason_code} ({detail})"],
            0,
            CandidateDiagnostics(),
        )
    await _emit_progress(
        progress_cb,
        {
            "stage": "collecting_candidates",
            "stage_detail": "routing_graph_search_initial_start",
        },
    )
    request_id = current_live_trace_request_id()
    max_paths = max(4, max_routes * 2)
    configured_state_budget = max(1000, int(settings.route_graph_max_state_budget))
    corridor_distance_km = _od_haversine_km(origin, destination)
    long_corridor_threshold_km = max(10.0, float(settings.route_graph_long_corridor_threshold_km))
    long_corridor = corridor_distance_km >= long_corridor_threshold_km
    if long_corridor:
        long_corridor_path_cap = max(2, int(settings.route_graph_long_corridor_max_paths))
        max_paths = max(2, min(max_paths, max_routes, long_corridor_path_cap))
    reduced_initial_enabled = bool(settings.route_graph_reduced_initial_for_long_corridor)
    initial_use_transition_state = not (reduced_initial_enabled and long_corridor)
    long_corridor_fast_fallback_reasons = {
        "state_budget_exceeded",
        "path_search_exhausted",
        "no_path",
        "candidate_pool_exhausted",
        "skipped_long_corridor_graph_search",
    }
    skip_initial_graph_search = bool(
        long_corridor and settings.route_graph_skip_initial_search_long_corridor
    )
    skip_retry_rescue_for_long_corridor = bool(long_corridor)
    initial_max_hops_override: int | None = None
    if long_corridor:
        scaled_hops = int(math.ceil(max(1.0, corridor_distance_km) * 8.5))
        initial_max_hops_override = max(900, min(int(settings.route_graph_max_hops_cap), scaled_hops))
    initial_search_timeout_ms = max(0, int(settings.route_graph_search_initial_timeout_ms))
    retry_search_timeout_ms = max(0, int(settings.route_graph_search_retry_timeout_ms))
    rescue_search_timeout_ms = max(0, int(settings.route_graph_search_rescue_timeout_ms))
    graph_retry_attempted = False
    graph_retry_state_budget = 0
    graph_retry_outcome = "not_applicable"
    graph_rescue_attempted = False
    graph_rescue_mode = "not_applicable"
    graph_rescue_state_budget = 0
    graph_rescue_outcome = "not_applicable"
    graph_search_ms_initial = 0.0
    graph_search_ms_retry = 0.0
    graph_search_ms_rescue = 0.0
    osrm_refine_ms = 0.0
    if skip_initial_graph_search:
        log_event(
            "route_graph_search_budget",
            request_id=request_id,
            pass_name="initial_skipped",
            max_paths=max_paths,
            configured_state_budget=configured_state_budget,
            corridor_distance_km=round(float(corridor_distance_km), 3),
            long_corridor_threshold_km=round(float(long_corridor_threshold_km), 3),
            long_corridor=True,
            reason="long_corridor_skip_enabled",
        )
        graph_routes = []
        graph_diag = GraphCandidateDiagnostics(
            explored_states=0,
            generated_paths=0,
            emitted_paths=0,
            candidate_budget=max_paths,
            effective_max_hops=int(initial_max_hops_override or 0),
            effective_hops_floor=0,
            effective_state_budget_initial=int(configured_state_budget),
            effective_state_budget=int(configured_state_budget),
            no_path_reason="skipped_long_corridor_graph_search",
            no_path_detail=(
                "Skipped expensive long-corridor graph search and used OSRM family fallback."
            ),
        )
        graph_search_ms_initial = 0.0
    else:
        log_event(
            "route_graph_search_budget",
            request_id=request_id,
            pass_name="initial",
            max_paths=max_paths,
            configured_state_budget=configured_state_budget,
            retry_multiplier=float(settings.route_graph_state_budget_retry_multiplier),
            retry_cap=max(1000, int(settings.route_graph_state_budget_retry_cap)),
            corridor_distance_km=round(float(corridor_distance_km), 3),
            long_corridor_threshold_km=round(float(long_corridor_threshold_km), 3),
            long_corridor=bool(long_corridor),
            initial_use_transition_state=bool(initial_use_transition_state),
            max_hops_override=initial_max_hops_override,
            search_timeout_ms=initial_search_timeout_ms,
            start_node_id=start_node_id,
            goal_node_id=goal_node_id,
        )
        graph_initial_started = time.perf_counter()
        graph_routes, graph_diag = await _route_graph_candidate_routes_async(
            origin_lat=float(origin.lat),
            origin_lon=float(origin.lon),
            destination_lat=float(destination.lat),
            destination_lon=float(destination.lon),
            max_paths=max_paths,
            scenario_edge_modifiers=scenario_edge_modifiers,
            max_hops_override=initial_max_hops_override,
            max_state_budget_override=(configured_state_budget if long_corridor else None),
            use_transition_state=initial_use_transition_state,
            search_deadline_s=_search_deadline_s(initial_search_timeout_ms),
            start_node_id=start_node_id,
            goal_node_id=goal_node_id,
        )
        graph_search_ms_initial = round((time.perf_counter() - graph_initial_started) * 1000.0, 2)
    initial_no_path_reason = str(graph_diag.no_path_reason or "").strip()
    should_retry = not (
        skip_retry_rescue_for_long_corridor and initial_no_path_reason in long_corridor_fast_fallback_reasons
    )
    if not graph_routes and initial_no_path_reason in {"state_budget_exceeded", "path_search_exhausted"} and should_retry:
        base_state_budget = max(
            1000,
            int(getattr(graph_diag, "effective_state_budget", 0)) or configured_state_budget,
        )
        retry_multiplier = max(1.0, float(settings.route_graph_state_budget_retry_multiplier))
        retry_cap = max(base_state_budget + 1, int(settings.route_graph_state_budget_retry_cap))
        proposed_retry_budget = max(base_state_budget + 1, int(math.ceil(base_state_budget * retry_multiplier)))
        retry_state_budget = min(proposed_retry_budget, retry_cap)
        if retry_state_budget > base_state_budget:
            graph_retry_attempted = True
            graph_retry_state_budget = int(retry_state_budget)
            await _emit_progress(
                progress_cb,
                {
                    "stage": "collecting_candidates",
                    "stage_detail": "routing_graph_search_retry_start",
                },
            )
            log_event(
                "route_graph_search_budget",
                request_id=request_id,
                pass_name="retry",
                max_paths=max_paths,
                configured_state_budget=configured_state_budget,
                base_state_budget=base_state_budget,
                retry_state_budget=retry_state_budget,
                use_transition_state=bool(initial_use_transition_state),
                search_timeout_ms=retry_search_timeout_ms,
                start_node_id=start_node_id,
                goal_node_id=goal_node_id,
            )
            graph_retry_started = time.perf_counter()
            graph_routes, graph_diag = await _route_graph_candidate_routes_async(
                origin_lat=float(origin.lat),
                origin_lon=float(origin.lon),
                destination_lat=float(destination.lat),
                destination_lon=float(destination.lon),
                max_paths=max_paths,
                scenario_edge_modifiers=scenario_edge_modifiers,
                max_state_budget_override=retry_state_budget,
                use_transition_state=initial_use_transition_state,
                search_deadline_s=_search_deadline_s(retry_search_timeout_ms),
                start_node_id=start_node_id,
                goal_node_id=goal_node_id,
            )
            graph_search_ms_retry = round((time.perf_counter() - graph_retry_started) * 1000.0, 2)
            graph_retry_outcome = "succeeded" if graph_routes else "exhausted"
        else:
            graph_retry_attempted = True
            graph_retry_state_budget = int(base_state_budget)
            graph_retry_outcome = "skipped_budget_cap"
    elif not graph_routes and not should_retry and initial_no_path_reason in long_corridor_fast_fallback_reasons:
        graph_retry_attempted = True
        graph_retry_state_budget = int(getattr(graph_diag, "effective_state_budget", 0) or configured_state_budget)
        graph_retry_outcome = "skipped_long_corridor_fast_fallback"
    if not graph_routes:
        rescue_reason = str(graph_diag.no_path_reason or "").strip()
        rescue_mode_setting = str(settings.route_graph_state_space_rescue_mode or "reduced").strip().lower()
        rescue_mode = rescue_mode_setting if rescue_mode_setting in {"reduced", "full"} else "reduced"
        rescue_enabled = bool(settings.route_graph_state_space_rescue_enabled)
        rescue_candidate_reasons = {"state_budget_exceeded", "path_search_exhausted", "no_path"}
        should_rescue = not (
            skip_retry_rescue_for_long_corridor and rescue_reason in long_corridor_fast_fallback_reasons
        )
        if rescue_enabled and rescue_reason in rescue_candidate_reasons and should_rescue:
            graph_rescue_attempted = True
            graph_rescue_mode = rescue_mode
            rescue_budget_cap = max(1000, int(settings.route_graph_state_budget_retry_cap))
            rescue_base_budget = max(
                configured_state_budget,
                int(getattr(graph_diag, "effective_state_budget", 0)) or configured_state_budget,
                int(graph_retry_state_budget or 0),
            )
            graph_rescue_state_budget = int(min(rescue_base_budget, rescue_budget_cap))
            graph_rescue_state_budget = max(graph_rescue_state_budget, configured_state_budget)
            await _emit_progress(
                progress_cb,
                {
                    "stage": "collecting_candidates",
                    "stage_detail": "routing_graph_search_rescue_start",
                },
            )
            log_event(
                "route_graph_search_budget",
                request_id=request_id,
                pass_name="rescue",
                max_paths=max_paths,
                configured_state_budget=configured_state_budget,
                rescue_state_budget=graph_rescue_state_budget,
                rescue_mode=graph_rescue_mode,
                search_timeout_ms=rescue_search_timeout_ms,
                start_node_id=start_node_id,
                goal_node_id=goal_node_id,
            )
            graph_rescue_started = time.perf_counter()
            graph_routes, graph_diag = await _route_graph_candidate_routes_async(
                origin_lat=float(origin.lat),
                origin_lon=float(origin.lon),
                destination_lat=float(destination.lat),
                destination_lon=float(destination.lon),
                max_paths=max_paths,
                scenario_edge_modifiers=scenario_edge_modifiers,
                max_state_budget_override=graph_rescue_state_budget,
                use_transition_state=(graph_rescue_mode == "full"),
                search_deadline_s=_search_deadline_s(rescue_search_timeout_ms),
                start_node_id=start_node_id,
                goal_node_id=goal_node_id,
            )
            graph_search_ms_rescue = round((time.perf_counter() - graph_rescue_started) * 1000.0, 2)
            graph_rescue_outcome = "succeeded" if graph_routes else "exhausted"
            if graph_routes:
                warnings.append(
                    "route_graph: routing_graph_search_rescued "
                    f"(mode={graph_rescue_mode}, state_budget={graph_rescue_state_budget})."
                )
        elif rescue_enabled and rescue_reason in rescue_candidate_reasons and not should_rescue:
            graph_rescue_attempted = True
            graph_rescue_mode = "long_corridor_fast_fallback"
            graph_rescue_state_budget = int(getattr(graph_diag, "effective_state_budget", 0) or configured_state_budget)
            graph_rescue_outcome = "skipped_long_corridor_fast_fallback"
    scenario_family_count = 0
    scenario_jaccard = 1.0
    scenario_jaccard_threshold = 1.0
    scenario_stress_score = 0.0
    scenario_gate_action = "not_applicable"
    scenario_scaling_version = str(
        (scenario_edge_modifiers or {}).get("scenario_edge_scaling_version", "v3_live_transform")
    )
    scenario_signatures = {sig for sig in (_graph_family_signature(route) for route in graph_routes) if sig}
    scenario_family_count = len(scenario_signatures)
    await _emit_progress(
        progress_cb,
        {
            "stage": "refining_candidates",
            "stage_detail": "routing_graph_search_complete",
            "candidate_done": 0,
            "candidate_total": int(max(0, len(graph_routes))),
        },
    )
    separability_fail_enforced = bool(settings.route_graph_scenario_separability_fail)
    if (
        graph_routes
        and not _is_neutral_scenario_modifiers(scenario_edge_modifiers)
        and separability_fail_enforced
    ):
        baseline_routes, _baseline_diag = await _route_graph_candidate_routes_async(
            origin_lat=float(origin.lat),
            origin_lon=float(origin.lon),
            destination_lat=float(destination.lat),
            destination_lon=float(destination.lon),
            max_paths=max_paths,
            scenario_edge_modifiers=_neutral_scenario_edge_modifiers(),
            start_node_id=start_node_id,
            goal_node_id=goal_node_id,
        )
        baseline_signatures = {sig for sig in (_graph_family_signature(route) for route in baseline_routes) if sig}
        union = scenario_signatures | baseline_signatures
        if union:
            scenario_jaccard = len(scenario_signatures & baseline_signatures) / float(len(union))
        if _is_stressed_scenario_modifiers(scenario_edge_modifiers):
            scenario_jaccard_threshold, scenario_stress_score = _adaptive_scenario_jaccard_threshold(
                scenario_edge_modifiers
            )
            if scenario_jaccard > scenario_jaccard_threshold:
                message = (
                    "Scenario candidate-family separability below strict threshold "
                    f"(jaccard={scenario_jaccard:.4f} > {scenario_jaccard_threshold:.4f}, "
                    f"stress={scenario_stress_score:.4f})."
                )
                scenario_gate_action = "failed"
                return (
                    [],
                    [f"route_graph: scenario_profile_invalid ({message})"],
                    graph_diag.candidate_budget,
                    CandidateDiagnostics(
                        raw_count=len(graph_routes),
                        deduped_count=0,
                        graph_explored_states=graph_diag.explored_states,
                        graph_generated_paths=graph_diag.generated_paths,
                        graph_emitted_paths=graph_diag.emitted_paths,
                        candidate_budget=graph_diag.candidate_budget,
                        graph_effective_max_hops=int(getattr(graph_diag, "effective_max_hops", 0)),
                        graph_effective_hops_floor=int(getattr(graph_diag, "effective_hops_floor", 0)),
                        graph_effective_state_budget_initial=int(
                            getattr(graph_diag, "effective_state_budget_initial", 0)
                        ),
                        graph_effective_state_budget=int(getattr(graph_diag, "effective_state_budget", 0)),
                        graph_no_path_reason=str(graph_diag.no_path_reason or ""),
                        graph_no_path_detail=str(graph_diag.no_path_detail or ""),
                        scenario_candidate_family_count=scenario_family_count,
                        scenario_candidate_jaccard_vs_baseline=round(float(scenario_jaccard), 6),
                        scenario_candidate_jaccard_threshold=round(float(scenario_jaccard_threshold), 6),
                        scenario_candidate_stress_score=round(float(scenario_stress_score), 6),
                        scenario_candidate_gate_action=scenario_gate_action,
                        scenario_edge_scaling_version=scenario_scaling_version,
                        graph_retry_attempted=graph_retry_attempted,
                        graph_retry_state_budget=graph_retry_state_budget,
                        graph_retry_outcome=graph_retry_outcome,
                        graph_rescue_attempted=graph_rescue_attempted,
                        graph_rescue_mode=graph_rescue_mode,
                        graph_rescue_state_budget=graph_rescue_state_budget,
                        graph_rescue_outcome=graph_rescue_outcome,
                        graph_search_ms_initial=graph_search_ms_initial,
                        graph_search_ms_retry=graph_search_ms_retry,
                        graph_search_ms_rescue=graph_search_ms_rescue,
                    ),
                )
        else:
            scenario_jaccard_threshold = max(0.0, min(1.0, float(settings.route_graph_scenario_jaccard_max)))
            scenario_stress_score = _scenario_stress_score(scenario_edge_modifiers)
            scenario_gate_action = "not_stressed"
    elif graph_routes and not _is_neutral_scenario_modifiers(scenario_edge_modifiers):
        scenario_jaccard_threshold = max(0.0, min(1.0, float(settings.route_graph_scenario_jaccard_max)))
        scenario_stress_score = _scenario_stress_score(scenario_edge_modifiers)
        scenario_gate_action = "skipped_non_enforcing"
    fallback_routes_by_signature: dict[str, dict[str, Any]] = {}
    fallback_spec_count = 0
    osrm_family_fallback_reasons = {
        "routing_graph_deferred_load",
    }
    should_run_osrm_family_fallback = bool(
        not graph_routes
        and (
            (
                long_corridor
                and initial_no_path_reason in long_corridor_fast_fallback_reasons
            )
            or initial_no_path_reason in osrm_family_fallback_reasons
        )
    )
    if (
        should_run_osrm_family_fallback
    ):
        fallback_specs = _long_corridor_fallback_specs(
            origin=origin,
            destination=destination,
            max_routes=max_routes,
        )
        fallback_spec_count = int(len(fallback_specs))
        await _emit_progress(
            progress_cb,
            {
                "stage": "refining_candidates",
                "stage_detail": "osrm_long_corridor_fallback_start",
                "candidate_done": 0,
                "candidate_total": int(max(1, len(fallback_specs))),
            },
        )
        fallback_started = time.perf_counter()
        if fallback_specs:
            async for progress in _iter_candidate_fetches(
                osrm=osrm,
                origin=origin,
                destination=destination,
                specs=fallback_specs,
            ):
                await _emit_progress(
                    progress_cb,
                    {
                        "stage": "refining_candidates",
                        "stage_detail": "osrm_long_corridor_fallback_progress",
                        "candidate_done": int(progress.done),
                        "candidate_total": int(progress.total),
                    },
                )
                result = progress.result
                if result.error:
                    warnings.append(f"{result.spec.label}: {result.error}")
                    continue
                for route in result.routes:
                    try:
                        sig = _route_signature(route)
                    except OSRMError:
                        continue
                    if sig not in fallback_routes_by_signature:
                        fallback_routes_by_signature[sig] = route
                    _annotate_route_candidate_meta(
                        fallback_routes_by_signature[sig],
                        source_labels={f"{result.spec.label}:long_corridor_fallback"},
                        toll_exclusion_available=False,
                    )
        osrm_refine_ms = round((time.perf_counter() - fallback_started) * 1000.0, 2)
        if fallback_routes_by_signature:
            if long_corridor and initial_no_path_reason in long_corridor_fast_fallback_reasons:
                warnings.append(
                    "route_graph: routing_graph_long_corridor_osrm_fallback "
                    f"(distance_km={corridor_distance_km:.1f}, engine_reason={initial_no_path_reason or 'unknown'})."
                )
                scenario_gate_action = "long_corridor_osrm_fallback"
            else:
                warnings.append(
                    "route_graph: routing_graph_osrm_fallback "
                    f"(engine_reason={initial_no_path_reason or 'unknown'})."
                )
                scenario_gate_action = "graph_deferred_osrm_fallback"
    if not graph_routes and not fallback_routes_by_signature:
        engine_reason = str(graph_diag.no_path_reason or "").strip()
        no_path_reason = _normalize_route_graph_search_reason(engine_reason)
        no_path_detail = (
            str(graph_diag.no_path_detail or "").strip()
            or "Route graph search produced no candidate paths."
        )
        return (
            [],
            [
                (
                    f"route_graph: {no_path_reason} "
                    f"({no_path_detail}; explored_states={int(graph_diag.explored_states)}, "
                    f"generated_paths={int(graph_diag.generated_paths)}, emitted_paths={int(graph_diag.emitted_paths)}, "
                    f"engine_reason={engine_reason or 'unknown'})."
                )
            ],
            0,
            CandidateDiagnostics(
                raw_count=0,
                deduped_count=0,
                graph_explored_states=graph_diag.explored_states,
                graph_generated_paths=graph_diag.generated_paths,
                graph_emitted_paths=graph_diag.emitted_paths,
                candidate_budget=graph_diag.candidate_budget,
                graph_effective_max_hops=int(getattr(graph_diag, "effective_max_hops", 0)),
                graph_effective_hops_floor=int(getattr(graph_diag, "effective_hops_floor", 0)),
                graph_effective_state_budget_initial=int(
                    getattr(graph_diag, "effective_state_budget_initial", 0)
                ),
                graph_effective_state_budget=int(getattr(graph_diag, "effective_state_budget", 0)),
                graph_no_path_reason=no_path_reason,
                graph_no_path_detail=no_path_detail,
                scenario_candidate_family_count=scenario_family_count,
                scenario_candidate_jaccard_vs_baseline=round(float(scenario_jaccard), 6),
                scenario_candidate_jaccard_threshold=round(float(scenario_jaccard_threshold), 6),
                scenario_candidate_stress_score=round(float(scenario_stress_score), 6),
                scenario_candidate_gate_action=scenario_gate_action,
                scenario_edge_scaling_version=scenario_scaling_version,
                graph_retry_attempted=graph_retry_attempted,
                graph_retry_state_budget=graph_retry_state_budget,
                graph_retry_outcome=graph_retry_outcome,
                graph_rescue_attempted=graph_rescue_attempted,
                graph_rescue_mode=graph_rescue_mode,
                graph_rescue_state_budget=graph_rescue_state_budget,
                graph_rescue_outcome=graph_rescue_outcome,
                graph_search_ms_initial=graph_search_ms_initial,
                graph_search_ms_retry=graph_search_ms_retry,
                graph_search_ms_rescue=graph_search_ms_rescue,
            ),
        )

    total_candidate_fetches = int(max(0, fallback_spec_count))
    family_specs: list[CandidateFetchSpec] = []
    for idx, route in enumerate(graph_routes, start=1):
        via = _graph_family_via_points(
            route,
            max_landmarks=max(1, int(settings.route_graph_via_landmarks_per_path)),
        )
        family_specs.append(
            CandidateFetchSpec(
                label=f"graph_family:{idx}",
                alternatives=False,
                via=via if via else None,
            )
            )
    total_candidate_fetches += int(len(family_specs))

    routes_by_signature: dict[str, dict[str, Any]] = dict(fallback_routes_by_signature)
    if family_specs:
        await _emit_progress(
            progress_cb,
            {
                "stage": "refining_candidates",
                "stage_detail": "osrm_family_refine_start",
                "candidate_done": 0,
                "candidate_total": int(max(0, len(family_specs))),
            },
        )
        osrm_refine_started = time.perf_counter()
        async for progress in _iter_candidate_fetches(
            osrm=osrm,
            origin=origin,
            destination=destination,
            specs=family_specs,
        ):
            await _emit_progress(
                progress_cb,
                {
                    "stage": "refining_candidates",
                    "stage_detail": "osrm_family_refine_progress",
                    "candidate_done": int(progress.done),
                    "candidate_total": int(progress.total),
                },
            )
            result = progress.result
            if result.error:
                warnings.append(f"{result.spec.label}: {result.error}")
                continue
            for route in result.routes:
                try:
                    sig = _route_signature(route)
                except OSRMError:
                    continue
                if sig not in routes_by_signature:
                    routes_by_signature[sig] = route
                _annotate_route_candidate_meta(
                    routes_by_signature[sig],
                    source_labels={f"{result.spec.label}:osrm_refined"},
                    toll_exclusion_available=False,
                )
        osrm_refine_ms += round((time.perf_counter() - osrm_refine_started) * 1000.0, 2)

    # Graph-native strict runtime: if OSRM refinement fails, emit graph families directly.
    if not routes_by_signature:
        for idx, route in enumerate(graph_routes, start=1):
            try:
                sig = _route_signature(route)
            except OSRMError:
                continue
            if sig not in routes_by_signature:
                routes_by_signature[sig] = route
            _annotate_route_candidate_meta(
                routes_by_signature[sig],
                source_labels={f"graph_path:{idx}:strict_fallback"},
                toll_exclusion_available=False,
            )

    if not routes_by_signature:
        await _emit_progress(
            progress_cb,
            {
                "stage": "refining_candidates",
                "stage_detail": "osrm_family_refine_empty",
                "candidate_done": int(max(0, total_candidate_fetches)),
                "candidate_total": int(max(0, total_candidate_fetches)),
            },
        )
        return (
            [],
            [
                "route_graph: no_route_candidates (Graph families produced no OSRM-refined routes).",
                *warnings[:6],
            ],
            total_candidate_fetches,
            CandidateDiagnostics(
                raw_count=len(graph_routes),
                deduped_count=0,
                graph_explored_states=graph_diag.explored_states,
                graph_generated_paths=graph_diag.generated_paths,
                graph_emitted_paths=graph_diag.emitted_paths,
                candidate_budget=graph_diag.candidate_budget,
                graph_effective_max_hops=int(getattr(graph_diag, "effective_max_hops", 0)),
                graph_effective_hops_floor=int(getattr(graph_diag, "effective_hops_floor", 0)),
                graph_effective_state_budget_initial=int(
                    getattr(graph_diag, "effective_state_budget_initial", 0)
                ),
                graph_effective_state_budget=int(getattr(graph_diag, "effective_state_budget", 0)),
                graph_no_path_reason=str(graph_diag.no_path_reason or ""),
                graph_no_path_detail=str(graph_diag.no_path_detail or ""),
                graph_retry_attempted=graph_retry_attempted,
                graph_retry_state_budget=graph_retry_state_budget,
                graph_retry_outcome=graph_retry_outcome,
                graph_rescue_attempted=graph_rescue_attempted,
                graph_rescue_mode=graph_rescue_mode,
                graph_rescue_state_budget=graph_rescue_state_budget,
                graph_rescue_outcome=graph_rescue_outcome,
                graph_search_ms_initial=graph_search_ms_initial,
                graph_search_ms_retry=graph_search_ms_retry,
                graph_search_ms_rescue=graph_search_ms_rescue,
                osrm_refine_ms=osrm_refine_ms,
            ),
        )

    ranked_routes = _select_ranked_candidate_routes(
        list(routes_by_signature.values()),
        max_routes=max_routes,
    )
    if long_corridor and len(ranked_routes) > max_routes:
        ranked_routes = ranked_routes[:max_routes]
    await _emit_progress(
        progress_cb,
        {
            "stage": "refining_candidates",
            "stage_detail": "candidate_refine_complete",
            "candidate_done": int(max(0, total_candidate_fetches)),
            "candidate_total": int(max(0, total_candidate_fetches)),
        },
    )
    diag = CandidateDiagnostics(
        raw_count=len(graph_routes),
        deduped_count=len(routes_by_signature),
        graph_explored_states=graph_diag.explored_states,
        graph_generated_paths=graph_diag.generated_paths,
        graph_emitted_paths=graph_diag.emitted_paths,
        candidate_budget=graph_diag.candidate_budget,
        graph_effective_max_hops=int(getattr(graph_diag, "effective_max_hops", 0)),
        graph_effective_hops_floor=int(getattr(graph_diag, "effective_hops_floor", 0)),
        graph_effective_state_budget_initial=int(
            getattr(graph_diag, "effective_state_budget_initial", 0)
        ),
        graph_effective_state_budget=int(getattr(graph_diag, "effective_state_budget", 0)),
        graph_no_path_reason=str(graph_diag.no_path_reason or ""),
        graph_no_path_detail=str(graph_diag.no_path_detail or ""),
        scenario_candidate_family_count=scenario_family_count,
        scenario_candidate_jaccard_vs_baseline=round(float(scenario_jaccard), 6),
        scenario_candidate_jaccard_threshold=round(float(scenario_jaccard_threshold), 6),
        scenario_candidate_stress_score=round(float(scenario_stress_score), 6),
        scenario_candidate_gate_action=scenario_gate_action,
        scenario_edge_scaling_version=scenario_scaling_version,
        graph_retry_attempted=graph_retry_attempted,
        graph_retry_state_budget=graph_retry_state_budget,
        graph_retry_outcome=graph_retry_outcome,
        graph_rescue_attempted=graph_rescue_attempted,
        graph_rescue_mode=graph_rescue_mode,
        graph_rescue_state_budget=graph_rescue_state_budget,
        graph_rescue_outcome=graph_rescue_outcome,
        graph_search_ms_initial=graph_search_ms_initial,
        graph_search_ms_retry=graph_search_ms_retry,
        graph_search_ms_rescue=graph_search_ms_rescue,
        osrm_refine_ms=osrm_refine_ms,
    )
    if cache_key:
        set_cached_routes(
            cache_key,
            (
                ranked_routes,
                warnings,
                total_candidate_fetches,
                {
                    "raw_count": diag.raw_count,
                    "deduped_count": diag.deduped_count,
                    "graph_explored_states": diag.graph_explored_states,
                    "graph_generated_paths": diag.graph_generated_paths,
                    "graph_emitted_paths": diag.graph_emitted_paths,
                    "candidate_budget": diag.candidate_budget,
                    "graph_effective_max_hops": diag.graph_effective_max_hops,
                    "graph_effective_hops_floor": diag.graph_effective_hops_floor,
                    "graph_effective_state_budget_initial": diag.graph_effective_state_budget_initial,
                    "graph_effective_state_budget": diag.graph_effective_state_budget,
                    "graph_no_path_reason": diag.graph_no_path_reason,
                    "graph_no_path_detail": diag.graph_no_path_detail,
                    "prefetch_total_sources": diag.prefetch_total_sources,
                    "prefetch_success_sources": diag.prefetch_success_sources,
                    "prefetch_failed_sources": diag.prefetch_failed_sources,
                    "prefetch_failed_keys": diag.prefetch_failed_keys,
                    "prefetch_failed_details": diag.prefetch_failed_details,
                    "prefetch_missing_expected_sources": diag.prefetch_missing_expected_sources,
                    "prefetch_rows_json": diag.prefetch_rows_json,
                    "scenario_candidate_family_count": diag.scenario_candidate_family_count,
                    "scenario_candidate_jaccard_vs_baseline": diag.scenario_candidate_jaccard_vs_baseline,
                    "scenario_candidate_jaccard_threshold": diag.scenario_candidate_jaccard_threshold,
                    "scenario_candidate_stress_score": diag.scenario_candidate_stress_score,
                    "scenario_candidate_gate_action": diag.scenario_candidate_gate_action,
                    "scenario_edge_scaling_version": diag.scenario_edge_scaling_version,
                    "precheck_gate_action": diag.precheck_gate_action,
                    "graph_retry_attempted": diag.graph_retry_attempted,
                    "graph_retry_state_budget": diag.graph_retry_state_budget,
                    "graph_retry_outcome": diag.graph_retry_outcome,
                    "graph_rescue_attempted": diag.graph_rescue_attempted,
                    "graph_rescue_mode": diag.graph_rescue_mode,
                    "graph_rescue_state_budget": diag.graph_rescue_state_budget,
                    "graph_rescue_outcome": diag.graph_rescue_outcome,
                    "prefetch_ms": diag.prefetch_ms,
                    "scenario_context_ms": diag.scenario_context_ms,
                    "graph_search_ms_initial": diag.graph_search_ms_initial,
                    "graph_search_ms_retry": diag.graph_search_ms_retry,
                    "graph_search_ms_rescue": diag.graph_search_ms_rescue,
                    "osrm_refine_ms": diag.osrm_refine_ms,
                    "build_options_ms": diag.build_options_ms,
                },
            ),
        )
    return ranked_routes, warnings, total_candidate_fetches, diag


def _normalize_waypoints(waypoints: list[Waypoint] | None) -> list[Waypoint]:
    out: list[Waypoint] = []
    for waypoint in waypoints or []:
        lat = float(waypoint.lat)
        lon = float(waypoint.lon)
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            continue
        if out and math.isclose(out[-1].lat, lat, abs_tol=1e-9) and math.isclose(out[-1].lon, lon, abs_tol=1e-9):
            continue
        out.append(Waypoint(lat=lat, lon=lon, label=waypoint.label))
    return out


async def _collect_route_options(
    *,
    osrm: OSRMClient,
    origin: LatLng,
    destination: LatLng,
    waypoints: list[Waypoint] | None,
    max_alternatives: int,
    vehicle_type: str,
    scenario_mode: ScenarioMode,
    cost_toggles: CostToggles,
    terrain_profile: TerrainProfile,
    stochastic: StochasticConfig | None,
    emissions_context: EmissionsContext | None,
    weather: WeatherImpactConfig | None,
    incident_simulation: IncidentSimulatorConfig | None,
    departure_time_utc: datetime | None,
    pareto_method: ParetoMethod,
    epsilon: EpsilonConstraints | None,
    optimization_mode: OptimizationMode,
    risk_aversion: float,
    utility_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    option_prefix: str,
    progress_cb: ProgressCallback | None = None,
) -> tuple[list[RouteOption], list[str], int, TerrainDiagnostics, CandidateDiagnostics]:
    await _emit_progress(
        progress_cb,
        {
            "stage": "collecting_candidates",
            "stage_detail": "route_options_start",
            "candidate_done": 0,
            "candidate_total": int(max(1, max_alternatives)),
        },
    )
    refresh_mode = str(settings.live_route_compute_refresh_mode or "route_compute").strip().lower()
    refresh_live_runtime_route_caches(mode=refresh_mode)
    prefetch_elapsed_ms = 0.0
    scenario_context_elapsed_ms = 0.0
    build_options_elapsed_ms = 0.0
    scenario_resolution_cache: dict[str, tuple[Any, dict[str, Any], str]] = {}
    multileg_candidate_diags: list[CandidateDiagnostics] = []
    log_event(
        "route_compute_runtime_budgets",
        request_id=current_live_trace_request_id(),
        max_alternatives=int(max_alternatives),
        precheck_timeout_ms=int(settings.route_graph_od_feasibility_timeout_ms),
        precheck_timeout_fail_closed=bool(settings.route_graph_precheck_timeout_fail_closed),
        graph_max_state_budget=int(settings.route_graph_max_state_budget),
        graph_state_budget_retry_multiplier=float(settings.route_graph_state_budget_retry_multiplier),
        graph_state_budget_retry_cap=int(settings.route_graph_state_budget_retry_cap),
        scenario_separability_fail=bool(settings.route_graph_scenario_separability_fail),
    )
    prefetch_summary: dict[str, Any] | None = None
    selected_start_node_id: str | None = None
    selected_goal_node_id: str | None = None
    precheck_diag_kwargs: dict[str, Any] = {}
    precheck_warnings: list[str] = []
    try:
        vehicle = resolve_vehicle_profile(vehicle_type)
        weather_cfg = weather or WeatherImpactConfig()
        weather_bucket_value = weather_cfg.profile if weather_cfg.enabled else "clear"
        should_prefetch = bool(settings.strict_live_data_required) or bool(
            settings.live_route_compute_require_all_expected
        ) or refresh_mode in {
            "all_sources",
            "full",
        }
        if should_prefetch:
            await _emit_progress(
                progress_cb,
                {
                    "stage": "collecting_candidates",
                    "stage_detail": "route_live_prefetch_start",
                    "candidate_done": 0,
                    "candidate_total": int(max(1, max_alternatives)),
                },
            )
            prefetch_started = time.perf_counter()
            prefetch_summary = await _prefetch_expected_live_sources(
                origin=origin,
                destination=destination,
                vehicle_class=str(vehicle.vehicle_class),
                departure_time_utc=departure_time_utc,
                weather_bucket=weather_bucket_value,
                cost_toggles=cost_toggles,
            )
            prefetch_elapsed_ms = round((time.perf_counter() - prefetch_started) * 1000.0, 2)
            await _emit_progress(
                progress_cb,
                {
                    "stage": "collecting_candidates",
                    "stage_detail": "route_live_prefetch_complete",
                    "candidate_done": int(max(0, int(prefetch_summary.get("source_success", 0)))),
                    "candidate_total": int(max(1, int(prefetch_summary.get("source_total", 1)))),
                },
            )
        await _emit_progress(
            progress_cb,
            {
                "stage": "collecting_candidates",
                "stage_detail": "route_graph_feasibility_check_start",
                "candidate_done": 0,
                "candidate_total": int(max(1, max_alternatives)),
            },
        )
        graph_precheck = await _route_graph_od_feasibility_async(origin=origin, destination=destination)
        precheck_diag_kwargs = _candidate_precheck_diag_kwargs(graph_precheck)
        if not bool(graph_precheck.get("ok")):
            reason_code = normalize_reason_code(
                str(graph_precheck.get("reason_code", "routing_graph_unavailable")).strip(),
                default="routing_graph_unavailable",
            )
            warning_text = _route_graph_precheck_warning(graph_precheck)
            timeout_degraded_continue = (
                reason_code in {"routing_graph_precheck_timeout", "routing_graph_deferred_load"}
                and not _precheck_timeout_fail_closed_enabled()
            )
            if timeout_degraded_continue:
                precheck_diag_kwargs["precheck_gate_action"] = "degraded_continue"
                precheck_warnings.append(warning_text)
                await _emit_progress(
                    progress_cb,
                    {
                        "stage": "collecting_candidates",
                        "stage_detail": "route_graph_feasibility_check_timeout_degraded",
                        "candidate_done": 0,
                        "candidate_total": int(max(1, max_alternatives)),
                    },
                )
            else:
                precheck_diag_kwargs["precheck_gate_action"] = "fail_closed"
                _record_expected_calls_blocked(
                    reason_code=reason_code,
                    stage="collecting_candidates",
                    detail=str(graph_precheck.get("stage_detail", "route_graph_precheck_failed")),
                )
                await _emit_progress(
                    progress_cb,
                    {
                        "stage": "collecting_candidates",
                        "stage_detail": reason_code,
                        "candidate_done": 0,
                        "candidate_total": int(max(1, max_alternatives)),
                    },
                )
                return (
                    [],
                    [warning_text],
                    0,
                    TerrainDiagnostics(),
                    CandidateDiagnostics(
                        **_prefetch_candidate_diag_kwargs(prefetch_summary),
                        **precheck_diag_kwargs,
                        prefetch_ms=prefetch_elapsed_ms,
                        scenario_context_ms=scenario_context_elapsed_ms,
                        build_options_ms=build_options_elapsed_ms,
                    ),
                )
        else:
            precheck_diag_kwargs["precheck_gate_action"] = "ok"
            selected_start_node_id = str(graph_precheck.get("origin_node_id", "")).strip() or None
            selected_goal_node_id = str(graph_precheck.get("destination_node_id", "")).strip() or None
            await _emit_progress(
                progress_cb,
                {
                    "stage": "collecting_candidates",
                    "stage_detail": "route_graph_feasibility_check_complete",
                    "candidate_done": 0,
                    "candidate_total": int(max(1, max_alternatives)),
                },
            )
        await _emit_progress(
            progress_cb,
            {
                "stage": "collecting_candidates",
                "stage_detail": "scenario_context_resolve_start",
                "candidate_done": 0,
                "candidate_total": int(max(1, max_alternatives)),
            },
        )
        base_context_started = time.perf_counter()
        base_candidate_context = await _scenario_context_from_od(
            origin=origin,
            destination=destination,
            vehicle_class=str(vehicle.vehicle_class),
            departure_time_utc=departure_time_utc,
            weather_bucket=weather_bucket_value,
            progress_cb=progress_cb,
        )
        await _emit_progress(
            progress_cb,
            {
                "stage": "collecting_candidates",
                "stage_detail": "scenario_context_resolve_complete",
                "candidate_done": 0,
                "candidate_total": int(max(1, max_alternatives)),
            },
        )
        await _emit_progress(
            progress_cb,
            {
                "stage": "collecting_candidates",
                "stage_detail": "scenario_modifier_resolve_start",
                "candidate_done": 0,
                "candidate_total": int(max(1, max_alternatives)),
            },
        )
        base_scenario_modifiers = await _scenario_candidate_modifiers_async(
            scenario_mode=scenario_mode,
            context=base_candidate_context,
        )
        scenario_context_elapsed_ms = round(
            (time.perf_counter() - base_context_started) * 1000.0,
            2,
        )
        await _emit_progress(
            progress_cb,
            {
                "stage": "collecting_candidates",
                "stage_detail": "scenario_modifier_resolve_complete",
                "candidate_done": 0,
                "candidate_total": int(max(1, max_alternatives)),
            },
        )
        scenario_cache_token = hashlib.sha1(
            json.dumps(base_scenario_modifiers, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        base_resolution_key = (
            f"{origin.lat:.6f},{origin.lon:.6f}->{destination.lat:.6f},{destination.lon:.6f}"
            f"|{vehicle_type}|{scenario_mode}|{departure_time_utc.isoformat() if departure_time_utc else 'none'}"
            f"|{weather_bucket_value}"
        )
        scenario_resolution_cache[base_resolution_key] = (
            base_candidate_context,
            base_scenario_modifiers,
            scenario_cache_token,
        )
    except ModelDataError as exc:
        reason_code = normalize_reason_code(str(exc.reason_code or "model_asset_unavailable").strip())
        detail_dict = exc.details if isinstance(exc.details, dict) else {}
        if reason_code == "live_source_refresh_failed":
            _record_expected_calls_blocked(
                reason_code=reason_code,
                stage="collecting_candidates",
                detail="route_live_prefetch_failed",
            )
            await _emit_progress(
                progress_cb,
                {
                    "stage": "collecting_candidates",
                    "stage_detail": reason_code,
                    "candidate_done": 0,
                    "candidate_total": int(max(1, max_alternatives)),
                },
            )
            warning_text = f"route_live_prefetch: {reason_code} ({exc.message})"
        else:
            warning_text = f"route_0: {reason_code} ({exc.message})"
        return (
            [],
            [warning_text],
            0,
            TerrainDiagnostics(),
            CandidateDiagnostics(
                **_prefetch_candidate_diag_kwargs(detail_dict),
                **precheck_diag_kwargs,
                prefetch_ms=prefetch_elapsed_ms,
                scenario_context_ms=scenario_context_elapsed_ms,
                build_options_ms=build_options_elapsed_ms,
            ),
        )

    normalized_waypoints = _normalize_waypoints(waypoints)
    if not normalized_waypoints:
        cache_key = _candidate_cache_key(
            origin=origin,
            destination=destination,
            waypoints=[],
            vehicle_type=vehicle_type,
            scenario_mode=scenario_mode,
            max_routes=max_alternatives,
            cost_toggles=cost_toggles,
            terrain_profile=terrain_profile,
            departure_time_utc=departure_time_utc,
            scenario_cache_token=scenario_cache_token,
        )
        routes, warnings, candidate_fetches, candidate_diag = await _collect_candidate_routes(
            osrm=osrm,
            origin=origin,
            destination=destination,
            max_routes=max_alternatives,
            cache_key=cache_key,
            scenario_edge_modifiers=base_scenario_modifiers,
            start_node_id=selected_start_node_id,
            goal_node_id=selected_goal_node_id,
            progress_cb=progress_cb,
        )
        if precheck_warnings:
            warnings = [*precheck_warnings, *warnings]
        if prefetch_summary is not None:
            candidate_diag = replace(
                candidate_diag,
                **_prefetch_candidate_diag_kwargs(prefetch_summary),
                **precheck_diag_kwargs,
            )
        elif precheck_diag_kwargs:
            candidate_diag = replace(
                candidate_diag,
                **precheck_diag_kwargs,
            )
        await _emit_progress(
            progress_cb,
            {
                "stage": "building_options",
                "stage_detail": "route_option_build_start",
                "candidate_done": int(max(0, candidate_fetches)),
                "candidate_total": int(max(0, candidate_fetches)),
            },
        )
        build_started = time.perf_counter()
        options, build_warnings, terrain_diag = _build_options(
            routes,
            vehicle_type=vehicle_type,
            scenario_mode=scenario_mode,
            cost_toggles=cost_toggles,
            terrain_profile=terrain_profile,
            stochastic=stochastic,
            emissions_context=emissions_context,
            weather=weather,
            incident_simulation=incident_simulation,
            departure_time_utc=departure_time_utc,
            utility_weights=utility_weights,
            risk_aversion=risk_aversion,
            option_prefix=option_prefix,
        )
        build_options_elapsed_ms = round((time.perf_counter() - build_started) * 1000.0, 2)
        candidate_diag = replace(
            candidate_diag,
            prefetch_ms=prefetch_elapsed_ms,
            scenario_context_ms=scenario_context_elapsed_ms,
            build_options_ms=build_options_elapsed_ms,
        )
        warnings.extend(build_warnings)
        await _emit_progress(
            progress_cb,
            {
                "stage": "building_options",
                "stage_detail": "route_option_build_complete",
                "candidate_done": int(max(0, len(options))),
                "candidate_total": int(max(1, len(options))),
            },
        )
        return options, warnings, candidate_fetches, terrain_diag, candidate_diag

    async def _solve_leg(
        leg_index: int,
        leg_origin: LatLng,
        leg_destination: LatLng,
    ) -> tuple[list[RouteOption], list[str], int, TerrainDiagnostics, CandidateDiagnostics]:
        nonlocal scenario_context_elapsed_ms, build_options_elapsed_ms
        leg_precheck_warnings: list[str] = []
        await _emit_progress(
            progress_cb,
            {
                "stage": "collecting_candidates",
                "stage_detail": f"multileg_leg_{leg_index}_graph_feasibility_start",
            },
        )
        leg_precheck = await _route_graph_od_feasibility_async(
            origin=leg_origin,
            destination=leg_destination,
        )
        leg_precheck_diag_kwargs = _candidate_precheck_diag_kwargs(leg_precheck)
        if not bool(leg_precheck.get("ok")):
            reason_code = normalize_reason_code(
                str(leg_precheck.get("reason_code", "routing_graph_unavailable")).strip(),
                default="routing_graph_unavailable",
            )
            warning_text = _route_graph_precheck_warning(leg_precheck)
            timeout_degraded_continue = (
                reason_code in {"routing_graph_precheck_timeout", "routing_graph_deferred_load"}
                and not _precheck_timeout_fail_closed_enabled()
            )
            if timeout_degraded_continue:
                leg_precheck_diag_kwargs["precheck_gate_action"] = "degraded_continue"
                leg_precheck_warnings.append(f"leg_{leg_index}: {warning_text}")
                await _emit_progress(
                    progress_cb,
                    {
                        "stage": "collecting_candidates",
                        "stage_detail": f"multileg_leg_{leg_index}_routing_graph_precheck_timeout_degraded",
                    },
                )
            else:
                leg_precheck_diag_kwargs["precheck_gate_action"] = "fail_closed"
                _record_expected_calls_blocked(
                    reason_code=reason_code,
                    stage="collecting_candidates",
                    detail=f"multileg_leg_{leg_index}_graph_feasibility_failed",
                )
                await _emit_progress(
                    progress_cb,
                    {
                        "stage": "collecting_candidates",
                        "stage_detail": f"multileg_leg_{leg_index}_{reason_code}",
                    },
                )
                return (
                    [],
                    [f"leg_{leg_index}: {warning_text}"],
                    0,
                    TerrainDiagnostics(),
                    CandidateDiagnostics(
                        **_prefetch_candidate_diag_kwargs(prefetch_summary),
                        **leg_precheck_diag_kwargs,
                        prefetch_ms=prefetch_elapsed_ms,
                        scenario_context_ms=scenario_context_elapsed_ms,
                        build_options_ms=build_options_elapsed_ms,
                    ),
                )
        if bool(leg_precheck.get("ok")):
            leg_precheck_diag_kwargs["precheck_gate_action"] = "ok"
        await _emit_progress(
            progress_cb,
            {
                "stage": "collecting_candidates",
                "stage_detail": f"multileg_leg_{leg_index}_graph_feasibility_complete",
            },
        )
        leg_resolution_key = (
            f"{leg_origin.lat:.6f},{leg_origin.lon:.6f}->{leg_destination.lat:.6f},{leg_destination.lon:.6f}"
            f"|{vehicle_type}|{scenario_mode}|{departure_time_utc.isoformat() if departure_time_utc else 'none'}"
            f"|{weather_bucket_value}"
        )
        cached_leg_resolution = scenario_resolution_cache.get(leg_resolution_key)
        if cached_leg_resolution is not None:
            _leg_context, leg_modifiers, leg_scenario_cache_token = cached_leg_resolution
            await _emit_progress(
                progress_cb,
                {
                    "stage": "collecting_candidates",
                    "stage_detail": f"multileg_leg_{leg_index}_scenario_context_cache_hit",
                },
            )
        else:
            try:
                await _emit_progress(
                    progress_cb,
                    {
                        "stage": "collecting_candidates",
                        "stage_detail": f"multileg_leg_{leg_index}_scenario_context_start",
                    },
                )
                leg_context_started = time.perf_counter()
                leg_context = await _scenario_context_from_od(
                    origin=leg_origin,
                    destination=leg_destination,
                    vehicle_class=str(vehicle.vehicle_class),
                    departure_time_utc=departure_time_utc,
                    weather_bucket=weather_bucket_value,
                    progress_cb=progress_cb,
                )
                await _emit_progress(
                    progress_cb,
                    {
                        "stage": "collecting_candidates",
                        "stage_detail": f"multileg_leg_{leg_index}_scenario_context_complete",
                    },
                )
                leg_modifiers = await _scenario_candidate_modifiers_async(
                    scenario_mode=scenario_mode,
                    context=leg_context,
                )
                scenario_context_elapsed_ms += round(
                    (time.perf_counter() - leg_context_started) * 1000.0,
                    2,
                )
                leg_scenario_cache_token = hashlib.sha1(
                    json.dumps(leg_modifiers, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest()
                scenario_resolution_cache[leg_resolution_key] = (
                    leg_context,
                    leg_modifiers,
                    leg_scenario_cache_token,
                )
            except ModelDataError as exc:
                return (
                    [],
                    [f"leg_{leg_index}: {normalize_reason_code(exc.reason_code)} ({exc.message})"],
                    0,
                    TerrainDiagnostics(),
                    CandidateDiagnostics(
                        **_prefetch_candidate_diag_kwargs(prefetch_summary),
                        **leg_precheck_diag_kwargs,
                        prefetch_ms=prefetch_elapsed_ms,
                        scenario_context_ms=scenario_context_elapsed_ms,
                        build_options_ms=build_options_elapsed_ms,
                    ),
                )
        leg_cache_key = _candidate_cache_key(
            origin=leg_origin,
            destination=leg_destination,
            waypoints=[],
            vehicle_type=vehicle_type,
            scenario_mode=scenario_mode,
            max_routes=max_alternatives,
            cost_toggles=cost_toggles,
            terrain_profile=terrain_profile,
            departure_time_utc=departure_time_utc,
            scenario_cache_token=leg_scenario_cache_token,
        )
        leg_routes, leg_warnings, leg_candidate_fetches, leg_candidate_diag = await _collect_candidate_routes(
            osrm=osrm,
            origin=leg_origin,
            destination=leg_destination,
            max_routes=max_alternatives,
            cache_key=leg_cache_key,
            scenario_edge_modifiers=leg_modifiers,
            start_node_id=str(leg_precheck.get("origin_node_id", "")).strip() or None,
            goal_node_id=str(leg_precheck.get("destination_node_id", "")).strip() or None,
            progress_cb=progress_cb,
        )
        if leg_precheck_warnings:
            leg_warnings = [*leg_precheck_warnings, *leg_warnings]
        if prefetch_summary is not None:
            leg_candidate_diag = replace(
                leg_candidate_diag,
                **_prefetch_candidate_diag_kwargs(prefetch_summary),
                **leg_precheck_diag_kwargs,
            )
        elif leg_precheck_diag_kwargs:
            leg_candidate_diag = replace(
                leg_candidate_diag,
                **leg_precheck_diag_kwargs,
            )
        await _emit_progress(
            progress_cb,
            {
                "stage": "building_options",
                "stage_detail": f"multileg_leg_{leg_index}_build_start",
                "candidate_done": int(max(0, leg_candidate_fetches)),
                "candidate_total": int(max(0, leg_candidate_fetches)),
            },
        )
        leg_build_started = time.perf_counter()
        leg_options, leg_build_warnings, leg_diag = _build_options(
            leg_routes,
            vehicle_type=vehicle_type,
            scenario_mode=scenario_mode,
            cost_toggles=cost_toggles,
            terrain_profile=terrain_profile,
            stochastic=stochastic,
            emissions_context=emissions_context,
            weather=weather,
            incident_simulation=incident_simulation,
            departure_time_utc=departure_time_utc,
            utility_weights=utility_weights,
            risk_aversion=risk_aversion,
            option_prefix=f"{option_prefix}_leg{leg_index}",
        )
        build_options_elapsed_ms += round((time.perf_counter() - leg_build_started) * 1000.0, 2)
        leg_candidate_diag = replace(
            leg_candidate_diag,
            prefetch_ms=prefetch_elapsed_ms,
            scenario_context_ms=scenario_context_elapsed_ms,
            build_options_ms=build_options_elapsed_ms,
        )
        leg_warnings.extend(leg_build_warnings)
        leg_pareto = _finalize_pareto_options(
            leg_options,
            max_alternatives=max_alternatives,
            pareto_method="dominance",
            epsilon=None,
            optimization_mode=optimization_mode,
            risk_aversion=risk_aversion,
        )
        leg_chain_target = max(1, min(max_alternatives, 4))
        if len(leg_pareto) < leg_chain_target and len(leg_options) > len(leg_pareto):
            leg_sorted = _sort_options_by_mode(
                list(leg_options),
                optimization_mode=optimization_mode,
                risk_aversion=risk_aversion,
            )
            leg_seen_ids = {opt.id for opt in leg_pareto}
            for option in leg_sorted:
                if option.id in leg_seen_ids:
                    continue
                leg_pareto.append(option)
                leg_seen_ids.add(option.id)
                if len(leg_pareto) >= leg_chain_target:
                    break
        await _emit_progress(
            progress_cb,
            {
                "stage": "finalizing_pareto",
                "stage_detail": f"multileg_leg_{leg_index}_pareto_complete",
                "candidate_done": int(max(0, len(leg_pareto))),
                "candidate_total": int(max(1, len(leg_pareto))),
            },
        )
        multileg_candidate_diags.append(leg_candidate_diag)
        return leg_pareto, leg_warnings, leg_candidate_fetches, leg_diag, leg_candidate_diag

    composed = await compose_multileg_route_options(
        origin=origin,
        destination=destination,
        waypoints=normalized_waypoints,
        max_alternatives=max_alternatives,
        leg_solver=_solve_leg,
        pareto_selector=lambda routes, limit: _finalize_pareto_options(
            routes,
            max_alternatives=max(1, limit),
            pareto_method="dominance",
            epsilon=None,
            optimization_mode=optimization_mode,
            risk_aversion=risk_aversion,
        ),
        option_prefix=option_prefix,
    )
    coverage_values = [
        float(option.terrain_summary.coverage_ratio)
        for option in composed.options
        if option.terrain_summary is not None
    ]
    dem_versions = {
        str(option.terrain_summary.version)
        for option in composed.options
        if option.terrain_summary is not None and option.terrain_summary.version
    }
    fail_closed_count = sum(1 for warning in composed.warnings if "terrain_fail_closed" in warning)
    unsupported_region_count = sum(1 for warning in composed.warnings if "terrain_region_unsupported" in warning)
    asset_unavailable_count = sum(1 for warning in composed.warnings if "terrain_dem_asset_unavailable" in warning)
    precheck_diag = next(
        (
            row
            for row in multileg_candidate_diags
            if str(row.precheck_reason_code or "").strip()
        ),
        None,
    )
    graph_retry_outcomes = [str(row.graph_retry_outcome or "").strip() for row in multileg_candidate_diags]
    graph_rescue_outcomes = [str(row.graph_rescue_outcome or "").strip() for row in multileg_candidate_diags]
    terrain_diag = TerrainDiagnostics(
        fail_closed_count=fail_closed_count,
        unsupported_region_count=unsupported_region_count,
        asset_unavailable_count=asset_unavailable_count,
        dem_version="|".join(sorted(dem_versions)) if dem_versions else "unknown",
        coverage_min_observed=min(coverage_values) if coverage_values else 1.0,
    )
    candidate_diag = CandidateDiagnostics(
        **_prefetch_candidate_diag_kwargs(prefetch_summary),
        raw_count=len(composed.options),
        deduped_count=len(composed.options),
        graph_explored_states=sum(row.graph_explored_states for row in multileg_candidate_diags),
        graph_generated_paths=sum(row.graph_generated_paths for row in multileg_candidate_diags),
        graph_emitted_paths=sum(row.graph_emitted_paths for row in multileg_candidate_diags),
        candidate_budget=max(
            [max(0, int(composed.candidate_fetches))]
            + [int(row.candidate_budget) for row in multileg_candidate_diags],
        ),
        graph_effective_max_hops=max([0] + [int(row.graph_effective_max_hops) for row in multileg_candidate_diags]),
        graph_effective_hops_floor=max([0] + [int(row.graph_effective_hops_floor) for row in multileg_candidate_diags]),
        graph_effective_state_budget_initial=max(
            [0] + [int(row.graph_effective_state_budget_initial) for row in multileg_candidate_diags]
        ),
        graph_effective_state_budget=max(
            [0] + [int(row.graph_effective_state_budget) for row in multileg_candidate_diags]
        ),
        graph_no_path_reason=next(
            (
                str(row.graph_no_path_reason)
                for row in multileg_candidate_diags
                if str(row.graph_no_path_reason or "").strip()
            ),
            "",
        ),
        graph_no_path_detail=next(
            (
                str(row.graph_no_path_detail)
                for row in multileg_candidate_diags
                if str(row.graph_no_path_detail or "").strip()
            ),
            "",
        ),
        precheck_reason_code=str(precheck_diag.precheck_reason_code) if precheck_diag is not None else "",
        precheck_message=str(precheck_diag.precheck_message) if precheck_diag is not None else "",
        precheck_elapsed_ms=float(precheck_diag.precheck_elapsed_ms) if precheck_diag is not None else 0.0,
        precheck_origin_node_id=str(precheck_diag.precheck_origin_node_id) if precheck_diag is not None else "",
        precheck_destination_node_id=str(precheck_diag.precheck_destination_node_id) if precheck_diag is not None else "",
        precheck_origin_nearest_m=float(precheck_diag.precheck_origin_nearest_m) if precheck_diag is not None else 0.0,
        precheck_destination_nearest_m=float(precheck_diag.precheck_destination_nearest_m)
        if precheck_diag is not None
        else 0.0,
        precheck_origin_selected_m=float(precheck_diag.precheck_origin_selected_m) if precheck_diag is not None else 0.0,
        precheck_destination_selected_m=float(precheck_diag.precheck_destination_selected_m)
        if precheck_diag is not None
        else 0.0,
        precheck_origin_candidate_count=int(precheck_diag.precheck_origin_candidate_count)
        if precheck_diag is not None
        else 0,
        precheck_destination_candidate_count=int(precheck_diag.precheck_destination_candidate_count)
        if precheck_diag is not None
        else 0,
        precheck_selected_component=int(precheck_diag.precheck_selected_component) if precheck_diag is not None else 0,
        precheck_selected_component_size=int(precheck_diag.precheck_selected_component_size)
        if precheck_diag is not None
        else 0,
        precheck_gate_action=str(precheck_diag.precheck_gate_action) if precheck_diag is not None else "",
        scenario_candidate_family_count=max(
            [0] + [int(row.scenario_candidate_family_count) for row in multileg_candidate_diags]
        ),
        scenario_candidate_jaccard_vs_baseline=min(
            [1.0] + [float(row.scenario_candidate_jaccard_vs_baseline) for row in multileg_candidate_diags]
        ),
        scenario_candidate_jaccard_threshold=min(
            [1.0] + [float(row.scenario_candidate_jaccard_threshold) for row in multileg_candidate_diags]
        ),
        scenario_candidate_stress_score=max(
            [0.0] + [float(row.scenario_candidate_stress_score) for row in multileg_candidate_diags]
        ),
        scenario_candidate_gate_action=next(
            (
                str(row.scenario_candidate_gate_action or "")
                for row in multileg_candidate_diags
                if str(row.scenario_candidate_gate_action or "").strip()
                and str(row.scenario_candidate_gate_action or "").strip() != "not_applicable"
            ),
            "not_applicable",
        ),
        scenario_edge_scaling_version=next(
            (
                str(row.scenario_edge_scaling_version or "")
                for row in multileg_candidate_diags
                if str(row.scenario_edge_scaling_version or "").strip()
            ),
            "v3_live_transform",
        ),
        graph_retry_attempted=any(bool(row.graph_retry_attempted) for row in multileg_candidate_diags),
        graph_retry_state_budget=max([0] + [int(row.graph_retry_state_budget) for row in multileg_candidate_diags]),
        graph_retry_outcome=next(
            (outcome for outcome in graph_retry_outcomes if outcome and outcome != "not_applicable"),
            "not_applicable",
        ),
        graph_rescue_attempted=any(bool(row.graph_rescue_attempted) for row in multileg_candidate_diags),
        graph_rescue_mode=next(
            (
                str(row.graph_rescue_mode or "")
                for row in multileg_candidate_diags
                if str(row.graph_rescue_mode or "").strip()
                and str(row.graph_rescue_mode or "").strip() != "not_applicable"
            ),
            "not_applicable",
        ),
        graph_rescue_state_budget=max([0] + [int(row.graph_rescue_state_budget) for row in multileg_candidate_diags]),
        graph_rescue_outcome=next(
            (outcome for outcome in graph_rescue_outcomes if outcome and outcome != "not_applicable"),
            "not_applicable",
        ),
        prefetch_ms=prefetch_elapsed_ms,
        scenario_context_ms=scenario_context_elapsed_ms,
        graph_search_ms_initial=sum(float(row.graph_search_ms_initial) for row in multileg_candidate_diags),
        graph_search_ms_retry=sum(float(row.graph_search_ms_retry) for row in multileg_candidate_diags),
        graph_search_ms_rescue=sum(float(row.graph_search_ms_rescue) for row in multileg_candidate_diags),
        osrm_refine_ms=sum(float(row.osrm_refine_ms) for row in multileg_candidate_diags),
        build_options_ms=build_options_elapsed_ms,
    )
    await _emit_progress(
        progress_cb,
        {
            "stage": "building_options",
            "stage_detail": "multileg_option_build_complete",
            "candidate_done": int(max(0, len(composed.options))),
            "candidate_total": int(max(1, len(composed.options))),
        },
    )
    return (
        composed.options,
        composed.warnings,
        composed.candidate_fetches,
        terrain_diag,
        candidate_diag,
    )


def _normalize_collect_route_options_result(
    result: tuple[Any, ...],
) -> tuple[list[RouteOption], list[str], int, TerrainDiagnostics, CandidateDiagnostics]:
    if len(result) == 5:
        options, warnings, candidate_fetches, terrain_diag, candidate_diag = result
    elif len(result) == 4:
        # Compatibility shim for test monkeypatches only; strict runtime requires
        # full candidate diagnostics payload.
        if "PYTEST_CURRENT_TEST" not in os.environ:
            raise ValueError(
                "_collect_route_options compatibility tuple length is unsupported in strict runtime."
            )
        options, warnings, candidate_fetches, terrain_diag = result
        candidate_diag = CandidateDiagnostics(
            raw_count=len(options),
            deduped_count=len(options),
            graph_explored_states=0,
            graph_generated_paths=0,
            graph_emitted_paths=0,
            candidate_budget=max(0, int(candidate_fetches)),
        )
    else:
        raise ValueError(
            f"_collect_route_options returned unexpected tuple length: {len(result)}"
        )
    return (
        list(options),
        list(warnings),
        int(candidate_fetches),
        terrain_diag,
        candidate_diag,
    )


async def _collect_route_options_with_diagnostics(
    *,
    progress_cb: ProgressCallback | None = None,
    **kwargs: Any,
) -> tuple[
    list[RouteOption], list[str], int, TerrainDiagnostics, CandidateDiagnostics
]:
    result = await _collect_route_options(progress_cb=progress_cb, **kwargs)
    return _normalize_collect_route_options_result(result)


@app.post("/pareto", response_model=ParetoResponse)
async def compute_pareto(
    req: ParetoRequest,
    response: Response,
    osrm: OSRMDep,
    _: UserAccessDep,
) -> ParetoResponse:
    request_id = str(uuid.uuid4())
    t0 = time.perf_counter()
    has_error = False
    trace_status = "ok"
    trace_error: str | None = None
    trace_token = start_live_call_trace(
        request_id,
        endpoint="/pareto",
        expected_calls=_route_compute_expected_live_calls(),
    )
    response.headers["x-route-request-id"] = request_id
    try:
        warmup_detail = _routing_graph_warmup_failfast_detail()
        if warmup_detail is not None:
            _record_expected_calls_blocked(
                reason_code=str(warmup_detail.get("reason_code", "routing_graph_warming_up")),
                stage=str(warmup_detail.get("stage", "collecting_candidates")),
                detail=str(warmup_detail.get("stage_detail", "routing_graph_warming_up")),
            )
            raise HTTPException(status_code=503, detail=warmup_detail)
        timeout_s = max(1.0, float(settings.route_compute_attempt_timeout_s))
        try:
            options, warnings, candidate_fetches, terrain_diag, candidate_diag = await asyncio.wait_for(
                _collect_route_options_with_diagnostics(
                    osrm=osrm,
                    origin=req.origin,
                    destination=req.destination,
                    waypoints=req.waypoints,
                    max_alternatives=req.max_alternatives,
                    vehicle_type=req.vehicle_type,
                    scenario_mode=req.scenario_mode,
                    cost_toggles=req.cost_toggles,
                    terrain_profile=req.terrain_profile,
                    stochastic=req.stochastic,
                    emissions_context=req.emissions_context,
                    weather=req.weather,
                    incident_simulation=req.incident_simulation,
                    departure_time_utc=req.departure_time_utc,
                    pareto_method=req.pareto_method,
                    epsilon=req.epsilon,
                    optimization_mode=req.optimization_mode,
                    risk_aversion=req.risk_aversion,
                    utility_weights=(req.weights.time, req.weights.money, req.weights.co2),
                    option_prefix="route",
                ),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            _record_expected_calls_blocked(
                reason_code="route_compute_timeout",
                stage="collecting_candidates",
                detail="attempt_timeout_reached",
            )
            raise HTTPException(
                status_code=422,
                detail=_strict_error_detail(
                    reason_code="route_compute_timeout",
                    message="Route compute attempt exceeded timeout budget.",
                    warnings=[],
                    extra={
                        "stage": "collecting_candidates",
                        "stage_detail": "attempt_timeout_reached",
                        "timeout_s": timeout_s,
                    },
                ),
            ) from None
        strict_frontier_applied = True
        epsilon_feasible_count = (
            len(
                filter_by_epsilon(
                    options,
                    req.epsilon,
                    objective_key=lambda option: _pareto_objective_key(
                        option,
                        optimization_mode=req.optimization_mode,
                        risk_aversion=req.risk_aversion,
                    ),
                )
            )
            if req.pareto_method == "epsilon_constraint" and req.epsilon is not None
            else len(options)
        )
        diagnostics_base = {
            "candidate_count_raw": int(candidate_diag.raw_count),
            "candidate_count_deduped": int(candidate_diag.deduped_count),
            "epsilon_feasible_count": epsilon_feasible_count,
            "strict_frontier_applied": strict_frontier_applied,
            "frontier_certificate": "",
            "graph_explored_states": int(candidate_diag.graph_explored_states),
            "graph_generated_paths": int(candidate_diag.graph_generated_paths),
            "graph_emitted_paths": int(candidate_diag.graph_emitted_paths),
            "candidate_budget": int(candidate_diag.candidate_budget),
            "graph_effective_max_hops": int(candidate_diag.graph_effective_max_hops),
            "graph_effective_hops_floor": int(candidate_diag.graph_effective_hops_floor),
            "graph_effective_state_budget_initial": int(candidate_diag.graph_effective_state_budget_initial),
            "graph_effective_state_budget": int(candidate_diag.graph_effective_state_budget),
            "graph_no_path_reason": str(candidate_diag.graph_no_path_reason or ""),
            "graph_no_path_detail": str(candidate_diag.graph_no_path_detail or ""),
            "graph_retry_attempted": bool(candidate_diag.graph_retry_attempted),
            "graph_retry_state_budget": int(candidate_diag.graph_retry_state_budget),
            "graph_retry_outcome": str(candidate_diag.graph_retry_outcome or ""),
            "graph_rescue_attempted": bool(candidate_diag.graph_rescue_attempted),
            "graph_rescue_mode": str(candidate_diag.graph_rescue_mode or ""),
            "graph_rescue_state_budget": int(candidate_diag.graph_rescue_state_budget),
            "graph_rescue_outcome": str(candidate_diag.graph_rescue_outcome or ""),
            "prefetch_ms": float(candidate_diag.prefetch_ms),
            "scenario_context_ms": float(candidate_diag.scenario_context_ms),
            "graph_search_ms_initial": float(candidate_diag.graph_search_ms_initial),
            "graph_search_ms_retry": float(candidate_diag.graph_search_ms_retry),
            "graph_search_ms_rescue": float(candidate_diag.graph_search_ms_rescue),
            "osrm_refine_ms": float(candidate_diag.osrm_refine_ms),
            "build_options_ms": float(candidate_diag.build_options_ms),
            "prefetch_total_sources": int(candidate_diag.prefetch_total_sources),
            "prefetch_success_sources": int(candidate_diag.prefetch_success_sources),
            "prefetch_failed_sources": int(candidate_diag.prefetch_failed_sources),
            "prefetch_failed_keys": str(candidate_diag.prefetch_failed_keys or ""),
            "prefetch_failed_details": str(candidate_diag.prefetch_failed_details or ""),
            "prefetch_missing_expected_sources": str(candidate_diag.prefetch_missing_expected_sources or ""),
            "prefetch_rows_json": str(candidate_diag.prefetch_rows_json or ""),
            "scenario_gate_required_configured": int(candidate_diag.scenario_gate_required_configured),
            "scenario_gate_required_effective": int(candidate_diag.scenario_gate_required_effective),
            "scenario_gate_source_ok_count": int(candidate_diag.scenario_gate_source_ok_count),
            "scenario_gate_waiver_applied": bool(candidate_diag.scenario_gate_waiver_applied),
            "scenario_gate_waiver_reason": str(candidate_diag.scenario_gate_waiver_reason or ""),
            "scenario_gate_source_signal_json": str(candidate_diag.scenario_gate_source_signal_json or ""),
            "scenario_gate_source_reachability_json": str(
                candidate_diag.scenario_gate_source_reachability_json or ""
            ),
            "scenario_gate_road_hint": str(candidate_diag.scenario_gate_road_hint or ""),
            "precheck_reason_code": str(candidate_diag.precheck_reason_code or ""),
            "precheck_message": str(candidate_diag.precheck_message or ""),
            "precheck_elapsed_ms": float(candidate_diag.precheck_elapsed_ms),
            "precheck_origin_node_id": str(candidate_diag.precheck_origin_node_id or ""),
            "precheck_destination_node_id": str(candidate_diag.precheck_destination_node_id or ""),
            "precheck_origin_nearest_m": float(candidate_diag.precheck_origin_nearest_m),
            "precheck_destination_nearest_m": float(candidate_diag.precheck_destination_nearest_m),
            "precheck_origin_selected_m": float(candidate_diag.precheck_origin_selected_m),
            "precheck_destination_selected_m": float(candidate_diag.precheck_destination_selected_m),
            "precheck_origin_candidate_count": int(candidate_diag.precheck_origin_candidate_count),
            "precheck_destination_candidate_count": int(candidate_diag.precheck_destination_candidate_count),
            "precheck_selected_component": int(candidate_diag.precheck_selected_component),
            "precheck_selected_component_size": int(candidate_diag.precheck_selected_component_size),
            "precheck_gate_action": str(candidate_diag.precheck_gate_action or ""),
            "scenario_candidate_family_count": int(candidate_diag.scenario_candidate_family_count),
            "scenario_candidate_jaccard_vs_baseline": round(
                float(candidate_diag.scenario_candidate_jaccard_vs_baseline),
                6,
            ),
            "scenario_candidate_jaccard_threshold": round(
                float(candidate_diag.scenario_candidate_jaccard_threshold),
                6,
            ),
            "scenario_candidate_stress_score": round(
                float(candidate_diag.scenario_candidate_stress_score),
                6,
            ),
            "scenario_candidate_gate_action": str(candidate_diag.scenario_candidate_gate_action),
            "scenario_edge_scaling_version": str(candidate_diag.scenario_edge_scaling_version),
            "terrain_fail_closed_count": terrain_diag.fail_closed_count,
            "terrain_unsupported_region_count": terrain_diag.unsupported_region_count,
            "terrain_asset_unavailable_count": terrain_diag.asset_unavailable_count,
            "terrain_dem_version": terrain_diag.dem_version,
            "terrain_coverage_min_observed": round(terrain_diag.coverage_min_observed, 6),
        }

        if not options:
            strict_detail = _strict_failure_detail_from_outcome(
                warnings=warnings,
                terrain_diag=terrain_diag,
                epsilon_requested=False,
                terrain_message="All candidates were removed by terrain DEM strict coverage policy.",
            )
            if strict_detail is not None:
                _record_expected_calls_blocked(
                    reason_code=str(strict_detail.get("reason_code", "strict_runtime_error")),
                    stage=str(strict_detail.get("stage", "finalizing_pareto")),
                    detail=str(strict_detail.get("message", "strict_runtime_error")),
                )
                raise HTTPException(
                    status_code=422,
                    detail=strict_detail,
                )
            if req.pareto_method == "epsilon_constraint" and req.epsilon is not None:
                log_event(
                    "pareto_request",
                    request_id=request_id,
                    vehicle_type=req.vehicle_type,
                    scenario_mode=req.scenario_mode.value,
                    pareto_method=req.pareto_method,
                    epsilon=req.epsilon.model_dump(mode="json"),
                    origin=req.origin.model_dump(),
                    destination=req.destination.model_dump(),
                    waypoint_count=len(req.waypoints),
                    candidate_fetches=candidate_fetches,
                    candidate_count=0,
                    pareto_count=0,
                    warning_count=len(warnings),
                    duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                )
                warning_message = "No routes satisfy epsilon constraints for this request."
                response_warnings = [*warnings, warning_message] if warnings else [warning_message]
                return ParetoResponse(
                    routes=[],
                    warnings=response_warnings,
                    diagnostics={**diagnostics_base, "pareto_count": 0},
                )
            raise HTTPException(
                status_code=422,
                detail=_strict_error_detail(
                    reason_code="no_route_candidates",
                    message="No route candidates could be computed.",
                    warnings=warnings,
                ),
            )

        pareto_sorted = _finalize_pareto_options(
            options,
            max_alternatives=req.max_alternatives,
            pareto_method=req.pareto_method,
            epsilon=req.epsilon,
            optimization_mode=req.optimization_mode,
            risk_aversion=req.risk_aversion,
        )

        log_event(
            "pareto_request",
            request_id=request_id,
            vehicle_type=req.vehicle_type,
            scenario_mode=req.scenario_mode.value,
            weights=req.weights.model_dump(),
            cost_toggles=req.cost_toggles.model_dump(),
            terrain_profile=req.terrain_profile,
            stochastic=req.stochastic.model_dump(mode="json"),
            weather=req.weather.model_dump(mode="json"),
            incident_simulation=req.incident_simulation.model_dump(mode="json"),
            optimization_mode=req.optimization_mode,
            risk_aversion=req.risk_aversion,
            selection_math_profile=str(settings.route_selection_math_profile or "modified_vikor_distance"),
            emissions_context=req.emissions_context.model_dump(mode="json"),
            pareto_method=req.pareto_method,
            epsilon=req.epsilon.model_dump(mode="json") if req.epsilon is not None else None,
            departure_time_utc=(
                req.departure_time_utc.isoformat() if req.departure_time_utc is not None else None
            ),
            origin=req.origin.model_dump(),
            destination=req.destination.model_dump(),
            waypoint_count=len(req.waypoints),
            candidate_fetches=candidate_fetches,
            candidate_count=len(options),
            pareto_count=len(pareto_sorted),
            warning_count=len(warnings),
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
        )

        response_warnings = warnings
        if (
            req.pareto_method == "epsilon_constraint"
            and req.epsilon is not None
            and len(options) > 0
            and len(pareto_sorted) == 0
        ):
            response_warnings = [*warnings, "No routes satisfy epsilon constraints for this request."]
        frontier_material = "|".join(
            f"{route.id}:{route.metrics.duration_s:.3f}:{route.metrics.monetary_cost:.3f}:{route.metrics.emissions_kg:.3f}"
            for route in pareto_sorted
        )
        frontier_certificate = hashlib.sha1(frontier_material.encode("utf-8")).hexdigest()[:20]
        return ParetoResponse(
            routes=pareto_sorted,
            warnings=response_warnings,
            diagnostics={
                **diagnostics_base,
                "pareto_count": len(pareto_sorted),
                "dominated_count": max(0, len(options) - len(pareto_sorted)),
                "frontier_certificate": frontier_certificate,
            },
        )
    except HTTPException as e:
        has_error = True
        trace_status = "error"
        detail_obj = e.detail if isinstance(e.detail, dict) else {}
        trace_error = normalize_reason_code(
            str((detail_obj or {}).get("reason_code", "http_exception"))
        )
        headers = dict(getattr(e, "headers", {}) or {})
        headers["x-route-request-id"] = request_id
        e.headers = headers
        raise
    except Exception as exc:
        has_error = True
        trace_status = "error"
        trace_error = str(exc).strip() or type(exc).__name__
        raise
    except BaseException:
        has_error = True
        trace_status = "error"
        raise
    finally:
        finish_live_call_trace(
            request_id=request_id,
            endpoint="/pareto",
            status=trace_status,
            error_reason=trace_error,
        )
        reset_live_call_trace(trace_token)
        _record_endpoint_metric("pareto", t0, error=has_error)


@app.post("/pareto/stream")
@app.post("/api/pareto/stream")
async def compute_pareto_stream(
    req: ParetoRequest,
    request: Request,
    osrm: OSRMDep,
    _: UserAccessDep,
) -> StreamingResponse:
    request_id = str(uuid.uuid4())
    t0 = time.perf_counter()

    async def stream() -> AsyncIterator[bytes]:
        stream_has_error = False
        trace_status = "ok"
        trace_error: str | None = None
        trace_token = start_live_call_trace(
            request_id,
            endpoint="/api/pareto/stream",
            expected_calls=_route_compute_expected_live_calls(),
        )
        options_task: asyncio.Task[
            tuple[list[RouteOption], list[str], int, TerrainDiagnostics, CandidateDiagnostics]
        ] | None = None
        stream_started = time.perf_counter()
        stage = "collecting_candidates"
        stage_detail = "initialising"
        stage_started = stream_started
        candidate_done = 0
        candidate_total = int(max(1, req.max_alternatives))
        heartbeat = 0
        timeout_s = max(1.0, float(settings.route_compute_attempt_timeout_s))
        candidate_diag_payload: dict[str, Any] = {}

        def _elapsed_ms() -> float:
            return round((time.perf_counter() - stream_started) * 1000.0, 2)

        def _stage_elapsed_ms() -> float:
            return round((time.perf_counter() - stage_started) * 1000.0, 2)

        def _meta_event() -> dict[str, Any]:
            return {
                "type": "meta",
                "request_id": request_id,
                "total": int(max(1, req.max_alternatives)),
                "done": int(max(0, candidate_done)),
                "stage": stage,
                "stage_detail": stage_detail,
                "elapsed_ms": _elapsed_ms(),
                "stage_elapsed_ms": _stage_elapsed_ms(),
                "heartbeat": int(max(0, heartbeat)),
                "candidate_done": int(max(0, candidate_done)),
                "candidate_total": int(max(0, candidate_total)),
                "candidate_diagnostics": candidate_diag_payload or None,
            }

        def _set_stage(
            next_stage: str,
            *,
            detail: str | None = None,
            done: int | None = None,
            total: int | None = None,
            force_log: bool = False,
        ) -> None:
            nonlocal stage, stage_detail, stage_started, candidate_done, candidate_total
            changed = (next_stage != stage) or (detail is not None and detail != stage_detail)
            if changed:
                stage = next_stage
                stage_started = time.perf_counter()
            if detail is not None:
                stage_detail = detail
            if done is not None:
                candidate_done = int(max(0, done))
            if total is not None:
                candidate_total = int(max(0, total))
            should_log = force_log or changed
            if should_log:
                log_event(
                    "pareto_stream_stage",
                    request_id=request_id,
                    stage=stage,
                    stage_detail=stage_detail,
                    elapsed_ms=_elapsed_ms(),
                    stage_elapsed_ms=_stage_elapsed_ms(),
                    candidate_done=int(max(0, candidate_done)),
                    candidate_total=int(max(0, candidate_total)),
                )

        async def _progress_cb(payload: dict[str, Any]) -> None:
            next_stage = str(payload.get("stage", stage)).strip() or stage
            detail = str(payload.get("stage_detail", stage_detail)).strip() or stage_detail
            done_val = payload.get("candidate_done")
            total_val = payload.get("candidate_total")
            done = int(done_val) if isinstance(done_val, (int, float)) else None
            total = int(total_val) if isinstance(total_val, (int, float)) else None
            force_log = False
            if done is not None and total is not None and total > 0:
                if done in (1, total) or done % 4 == 0:
                    force_log = True
            _set_stage(next_stage, detail=detail, done=done, total=total, force_log=force_log)

        try:
            warmup_detail = _routing_graph_warmup_failfast_detail()
            if warmup_detail is not None:
                _record_expected_calls_blocked(
                    reason_code=str(warmup_detail.get("reason_code", "routing_graph_warming_up")),
                    stage=str(warmup_detail.get("stage", "collecting_candidates")),
                    detail=str(warmup_detail.get("stage_detail", "routing_graph_warming_up")),
                )
                stream_has_error = True
                trace_status = "error"
                trace_error = str(warmup_detail.get("reason_code", "routing_graph_warming_up"))
                warmup_reason = (
                    str(warmup_detail.get("reason_code", "routing_graph_warming_up")).strip()
                    or "routing_graph_warming_up"
                )
                _set_stage(
                    "collecting_candidates",
                    detail=warmup_reason,
                    done=0,
                    total=int(max(1, req.max_alternatives)),
                    force_log=True,
                )
                warmup_fatal = _stream_fatal_event_from_detail(warmup_detail)
                warmup_fatal["request_id"] = request_id
                warmup_fatal["stage"] = stage
                warmup_fatal["stage_detail"] = stage_detail
                warmup_fatal["elapsed_ms"] = _elapsed_ms()
                warmup_fatal["stage_elapsed_ms"] = _stage_elapsed_ms()
                warmup_fatal["candidate_done"] = int(max(0, candidate_done))
                warmup_fatal["candidate_total"] = int(max(0, candidate_total))
                warmup_fatal["failure_chain"] = {
                    "stage": stage,
                    "stage_detail": stage_detail,
                    "reason_code": str(warmup_detail.get("reason_code", "routing_graph_warming_up")),
                    "message": str(warmup_detail.get("message", "Routing graph warmup gating failure.")),
                    "warning_count": len(warmup_detail.get("warnings", []) or []),
                    "warnings": list((warmup_detail.get("warnings", []) or []))[:10],
                }
                yield _ndjson_line(warmup_fatal)
                return

            log_event(
                "pareto_stream_started",
                request_id=request_id,
                mode="multileg" if req.waypoints else "single_leg",
                waypoint_count=len(req.waypoints or []),
                max_alternatives=req.max_alternatives,
                timeout_s=timeout_s,
            )
            _set_stage(
                "collecting_candidates",
                detail="route_options_task_start",
                done=0,
                total=int(max(1, req.max_alternatives)),
                force_log=True,
            )
            options_task = asyncio.create_task(
                _collect_route_options_with_diagnostics(
                    osrm=osrm,
                    origin=req.origin,
                    destination=req.destination,
                    waypoints=req.waypoints,
                    max_alternatives=req.max_alternatives,
                    vehicle_type=req.vehicle_type,
                    scenario_mode=req.scenario_mode,
                    cost_toggles=req.cost_toggles,
                    terrain_profile=req.terrain_profile,
                    stochastic=req.stochastic,
                    emissions_context=req.emissions_context,
                    weather=req.weather,
                    incident_simulation=req.incident_simulation,
                    departure_time_utc=req.departure_time_utc,
                    pareto_method=req.pareto_method,
                    epsilon=req.epsilon,
                    optimization_mode=req.optimization_mode,
                    risk_aversion=req.risk_aversion,
                    utility_weights=(req.weights.time, req.weights.money, req.weights.co2),
                    option_prefix="route",
                    progress_cb=_progress_cb,
                )
            )
            yield _ndjson_line(_meta_event())

            while not options_task.done():
                elapsed_s = time.perf_counter() - stream_started
                if elapsed_s >= timeout_s:
                    stream_has_error = True
                    trace_status = "error"
                    trace_error = "route_compute_timeout"
                    _set_stage(
                        stage,
                        detail=f"{stage_detail}|attempt_timeout_reached",
                        force_log=True,
                    )
                    log_event(
                        "pareto_stream_timeout",
                        request_id=request_id,
                        timeout_s=timeout_s,
                        elapsed_ms=_elapsed_ms(),
                        stage=stage,
                        stage_detail=stage_detail,
                        stage_elapsed_ms=_stage_elapsed_ms(),
                        candidate_done=int(max(0, candidate_done)),
                        candidate_total=int(max(0, candidate_total)),
                    )
                    if options_task is not None and not options_task.done():
                        options_task.cancel()
                        await asyncio.gather(options_task, return_exceptions=True)
                    yield _ndjson_line(
                        {
                            "type": "fatal",
                            "reason_code": "route_compute_timeout",
                            "message": "Route compute attempt exceeded timeout budget.",
                            "warnings": [],
                            "request_id": request_id,
                            "stage": stage,
                            "stage_detail": stage_detail,
                            "elapsed_ms": _elapsed_ms(),
                            "stage_elapsed_ms": _stage_elapsed_ms(),
                            "candidate_done": int(max(0, candidate_done)),
                            "candidate_total": int(max(0, candidate_total)),
                        }
                    )
                    return
                if await request.is_disconnected():
                    _set_stage(
                        stage,
                        detail=f"{stage_detail}|client_disconnected",
                        force_log=True,
                    )
                    if options_task is not None and not options_task.done():
                        options_task.cancel()
                        await asyncio.gather(options_task, return_exceptions=True)
                    log_event(
                        "pareto_stream_cancelled",
                        request_id=request_id,
                        duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                        stage=stage,
                        stage_detail=stage_detail,
                        stage_elapsed_ms=_stage_elapsed_ms(),
                        candidate_done=int(max(0, candidate_done)),
                        candidate_total=int(max(0, candidate_total)),
                        reason="client_disconnected",
                    )
                    return
                try:
                    remaining_s = max(0.1, timeout_s - elapsed_s)
                    wait_window_s = max(0.1, min(10.0, remaining_s))
                    await asyncio.wait_for(asyncio.shield(options_task), timeout=wait_window_s)
                except asyncio.TimeoutError:
                    heartbeat += 1
                    yield _ndjson_line(_meta_event())

            options, warnings, candidate_fetches, terrain_diag, candidate_diag = await options_task
            candidate_diag_payload = {
                "raw_count": int(candidate_diag.raw_count),
                "deduped_count": int(candidate_diag.deduped_count),
                "graph_explored_states": int(candidate_diag.graph_explored_states),
                "graph_generated_paths": int(candidate_diag.graph_generated_paths),
                "graph_emitted_paths": int(candidate_diag.graph_emitted_paths),
                "candidate_budget": int(candidate_diag.candidate_budget),
                "graph_effective_max_hops": int(candidate_diag.graph_effective_max_hops),
                "graph_effective_hops_floor": int(candidate_diag.graph_effective_hops_floor),
                "graph_effective_state_budget_initial": int(candidate_diag.graph_effective_state_budget_initial),
                "graph_effective_state_budget": int(candidate_diag.graph_effective_state_budget),
                "graph_no_path_reason": str(candidate_diag.graph_no_path_reason or ""),
                "graph_no_path_detail": str(candidate_diag.graph_no_path_detail or ""),
                "prefetch_total_sources": int(candidate_diag.prefetch_total_sources),
                "prefetch_success_sources": int(candidate_diag.prefetch_success_sources),
                "prefetch_failed_sources": int(candidate_diag.prefetch_failed_sources),
                "prefetch_failed_keys": str(candidate_diag.prefetch_failed_keys or ""),
                "prefetch_failed_details": str(candidate_diag.prefetch_failed_details or ""),
                "prefetch_missing_expected_sources": str(candidate_diag.prefetch_missing_expected_sources or ""),
                "prefetch_rows_json": str(candidate_diag.prefetch_rows_json or ""),
                "scenario_gate_required_configured": int(candidate_diag.scenario_gate_required_configured),
                "scenario_gate_required_effective": int(candidate_diag.scenario_gate_required_effective),
                "scenario_gate_source_ok_count": int(candidate_diag.scenario_gate_source_ok_count),
                "scenario_gate_waiver_applied": bool(candidate_diag.scenario_gate_waiver_applied),
                "scenario_gate_waiver_reason": str(candidate_diag.scenario_gate_waiver_reason or ""),
                "scenario_gate_source_signal_json": str(candidate_diag.scenario_gate_source_signal_json or ""),
                "scenario_gate_source_reachability_json": str(
                    candidate_diag.scenario_gate_source_reachability_json or ""
                ),
                "scenario_gate_road_hint": str(candidate_diag.scenario_gate_road_hint or ""),
                "precheck_reason_code": str(candidate_diag.precheck_reason_code or ""),
                "precheck_message": str(candidate_diag.precheck_message or ""),
                "precheck_elapsed_ms": float(candidate_diag.precheck_elapsed_ms),
                "precheck_origin_node_id": str(candidate_diag.precheck_origin_node_id or ""),
                "precheck_destination_node_id": str(candidate_diag.precheck_destination_node_id or ""),
                "precheck_origin_nearest_m": float(candidate_diag.precheck_origin_nearest_m),
                "precheck_destination_nearest_m": float(candidate_diag.precheck_destination_nearest_m),
                "precheck_origin_selected_m": float(candidate_diag.precheck_origin_selected_m),
                "precheck_destination_selected_m": float(candidate_diag.precheck_destination_selected_m),
                "precheck_origin_candidate_count": int(candidate_diag.precheck_origin_candidate_count),
                "precheck_destination_candidate_count": int(candidate_diag.precheck_destination_candidate_count),
                "precheck_selected_component": int(candidate_diag.precheck_selected_component),
                "precheck_selected_component_size": int(candidate_diag.precheck_selected_component_size),
                "precheck_gate_action": str(candidate_diag.precheck_gate_action or ""),
                "scenario_candidate_family_count": int(candidate_diag.scenario_candidate_family_count),
                "scenario_candidate_jaccard_vs_baseline": round(
                    float(candidate_diag.scenario_candidate_jaccard_vs_baseline),
                    6,
                ),
                "scenario_candidate_jaccard_threshold": round(
                    float(candidate_diag.scenario_candidate_jaccard_threshold),
                    6,
                ),
                "scenario_candidate_stress_score": round(
                    float(candidate_diag.scenario_candidate_stress_score),
                    6,
                ),
                "scenario_candidate_gate_action": str(candidate_diag.scenario_candidate_gate_action),
                "scenario_edge_scaling_version": str(candidate_diag.scenario_edge_scaling_version),
                "graph_retry_attempted": bool(candidate_diag.graph_retry_attempted),
                "graph_retry_state_budget": int(candidate_diag.graph_retry_state_budget),
                "graph_retry_outcome": str(candidate_diag.graph_retry_outcome or ""),
                "graph_rescue_attempted": bool(candidate_diag.graph_rescue_attempted),
                "graph_rescue_mode": str(candidate_diag.graph_rescue_mode or ""),
                "graph_rescue_state_budget": int(candidate_diag.graph_rescue_state_budget),
                "graph_rescue_outcome": str(candidate_diag.graph_rescue_outcome or ""),
                "prefetch_ms": float(candidate_diag.prefetch_ms),
                "scenario_context_ms": float(candidate_diag.scenario_context_ms),
                "graph_search_ms_initial": float(candidate_diag.graph_search_ms_initial),
                "graph_search_ms_retry": float(candidate_diag.graph_search_ms_retry),
                "graph_search_ms_rescue": float(candidate_diag.graph_search_ms_rescue),
                "osrm_refine_ms": float(candidate_diag.osrm_refine_ms),
                "build_options_ms": float(candidate_diag.build_options_ms),
            }
            precheck_failure_payload = _candidate_precheck_payload(candidate_diag)
            _set_stage(
                "building_options",
                detail="candidate_collection_complete",
                done=int(max(0, candidate_fetches)),
                total=int(max(0, candidate_fetches)),
                force_log=True,
            )
            _set_stage(
                "finalizing_pareto",
                detail="pareto_finalize_start",
                done=int(max(0, len(options))),
                total=int(max(1, len(options))),
                force_log=True,
            )
            yield _ndjson_line(_meta_event())

            if not options:
                strict_detail = _strict_failure_detail_from_outcome(
                    warnings=warnings,
                    terrain_diag=terrain_diag,
                    epsilon_requested=req.pareto_method == "epsilon_constraint" and req.epsilon is not None,
                    terrain_message="All candidates were removed by terrain DEM strict coverage policy.",
                )
                if strict_detail is not None:
                    _record_expected_calls_blocked(
                        reason_code=str(strict_detail.get("reason_code", "strict_runtime_error")),
                        stage=str(strict_detail.get("stage", stage)),
                        detail=str(strict_detail.get("message", "strict_runtime_error")),
                    )
                    stream_has_error = True
                    trace_status = "error"
                    trace_error = str(strict_detail.get("reason_code", "strict_runtime_error"))
                    fatal_event = _stream_fatal_event_from_detail(strict_detail)
                    fatal_event["request_id"] = request_id
                    fatal_event["stage"] = stage
                    fatal_event["stage_detail"] = stage_detail
                    fatal_event["elapsed_ms"] = _elapsed_ms()
                    fatal_event["stage_elapsed_ms"] = _stage_elapsed_ms()
                    fatal_event["candidate_done"] = int(max(0, candidate_done))
                    fatal_event["candidate_total"] = int(max(0, candidate_total))
                    fatal_event["candidate_diagnostics"] = candidate_diag_payload
                    failure_chain_payload = {
                        "stage": stage,
                        "stage_detail": stage_detail,
                        "reason_code": str(strict_detail.get("reason_code", "strict_runtime_error")),
                        "message": str(strict_detail.get("message", "strict_runtime_error")),
                        "warning_count": len(warnings),
                        "warnings": warnings[:10],
                    }
                    if precheck_failure_payload is not None:
                        failure_chain_payload["precheck"] = precheck_failure_payload
                    fatal_event["failure_chain"] = failure_chain_payload
                    yield _ndjson_line(fatal_event)
                    return
                stream_has_error = True
                trace_status = "error"
                trace_error = "no_route_candidates"
                _record_expected_calls_blocked(
                    reason_code="no_route_candidates",
                    stage=stage,
                    detail="No route candidates were returned after strict candidate collection.",
                )
                message = "No route candidates could be computed."
                if warnings:
                    message = f"{message} {warnings[0]}"
                yield _ndjson_line(
                    {
                        "type": "fatal",
                        "reason_code": "no_route_candidates",
                        "message": message,
                        "warnings": [],
                        "request_id": request_id,
                        "stage": stage,
                        "stage_detail": stage_detail,
                        "elapsed_ms": _elapsed_ms(),
                        "stage_elapsed_ms": _stage_elapsed_ms(),
                        "candidate_done": int(max(0, candidate_done)),
                        "candidate_total": int(max(0, candidate_total)),
                        "candidate_diagnostics": candidate_diag_payload,
                        "failure_chain": {
                            "stage": stage,
                            "stage_detail": stage_detail,
                            "reason_code": "no_route_candidates",
                            "message": message,
                            "warning_count": len(warnings),
                            "warnings": warnings[:10],
                            **(
                                {"precheck": precheck_failure_payload}
                                if precheck_failure_payload is not None
                                else {}
                            ),
                        },
                    }
                )
                return

            pareto_sorted = _finalize_pareto_options(
                options,
                max_alternatives=req.max_alternatives,
                pareto_method=req.pareto_method,
                epsilon=req.epsilon,
                optimization_mode=req.optimization_mode,
                risk_aversion=req.risk_aversion,
            )
            if not pareto_sorted and req.pareto_method == "epsilon_constraint" and req.epsilon is not None:
                stream_has_error = True
                trace_status = "error"
                trace_error = "epsilon_infeasible"
                fatal_event = _stream_fatal_event_from_detail(
                    _strict_error_detail(
                        reason_code="epsilon_infeasible",
                        message="No routes satisfy epsilon constraints for this request.",
                        warnings=warnings,
                    )
                )
                fatal_event["request_id"] = request_id
                fatal_event["stage"] = stage
                fatal_event["stage_detail"] = stage_detail
                fatal_event["elapsed_ms"] = _elapsed_ms()
                fatal_event["stage_elapsed_ms"] = _stage_elapsed_ms()
                fatal_event["candidate_done"] = int(max(0, candidate_done))
                fatal_event["candidate_total"] = int(max(0, candidate_total))
                fatal_event["candidate_diagnostics"] = candidate_diag_payload
                yield _ndjson_line(fatal_event)
                return

            total_routes = len(pareto_sorted)
            _set_stage(
                "finalizing_pareto",
                detail="pareto_finalize_complete",
                done=total_routes,
                total=max(1, total_routes),
                force_log=True,
            )
            for idx, route in enumerate(pareto_sorted, start=1):
                yield _ndjson_line(
                    {
                        "type": "route",
                        "done": idx,
                        "total": total_routes,
                        "route": route.model_dump(mode="json"),
                    }
                )

            log_event(
                "pareto_stream_request",
                request_id=request_id,
                vehicle_type=req.vehicle_type,
                scenario_mode=req.scenario_mode.value,
                weights=req.weights.model_dump(),
                cost_toggles=req.cost_toggles.model_dump(),
                terrain_profile=req.terrain_profile,
                stochastic=req.stochastic.model_dump(mode="json"),
                weather=req.weather.model_dump(mode="json"),
                incident_simulation=req.incident_simulation.model_dump(mode="json"),
                optimization_mode=req.optimization_mode,
                risk_aversion=req.risk_aversion,
                emissions_context=req.emissions_context.model_dump(mode="json"),
                pareto_method=req.pareto_method,
                epsilon=req.epsilon.model_dump(mode="json") if req.epsilon is not None else None,
                departure_time_utc=(
                    req.departure_time_utc.isoformat() if req.departure_time_utc is not None else None
                ),
                origin=req.origin.model_dump(),
                destination=req.destination.model_dump(),
                candidate_fetches=candidate_fetches,
                candidate_count=len(options),
                pareto_count=len(pareto_sorted),
                warning_count=len(warnings),
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                stage=stage,
                stage_detail=stage_detail,
                candidate_done=int(max(0, candidate_done)),
                candidate_total=int(max(0, candidate_total)),
            )

            yield _ndjson_line(
                    {
                        "type": "done",
                        "done": total_routes,
                        "total": total_routes,
                        "routes": [route.model_dump(mode="json") for route in pareto_sorted],
                        "warning_count": len(warnings),
                        "warnings": warnings,
                        "candidate_diagnostics": candidate_diag_payload,
                    }
                )
        except asyncio.CancelledError:
            trace_status = "error"
            trace_error = "client_cancelled"
            log_event(
                "pareto_stream_cancelled",
                request_id=request_id,
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                stage=stage,
                stage_detail=stage_detail,
                stage_elapsed_ms=_stage_elapsed_ms(),
                candidate_done=int(max(0, candidate_done)),
                candidate_total=int(max(0, candidate_total)),
                reason="server_task_cancelled",
            )
            raise
        except HTTPException as e:
            stream_has_error = True
            detail_obj = e.detail if isinstance(e.detail, dict) else {}
            trace_status = "error"
            trace_error = normalize_reason_code(
                str((detail_obj or {}).get("reason_code", str(getattr(e, "detail", e))))
            )
            fatal_event = _stream_fatal_event_from_exception(e)
            fatal_event["request_id"] = request_id
            fatal_event["stage"] = stage
            fatal_event["stage_detail"] = stage_detail
            fatal_event["elapsed_ms"] = _elapsed_ms()
            fatal_event["stage_elapsed_ms"] = _stage_elapsed_ms()
            fatal_event["candidate_done"] = int(max(0, candidate_done))
            fatal_event["candidate_total"] = int(max(0, candidate_total))
            fatal_event["candidate_diagnostics"] = candidate_diag_payload
            log_event(
                "pareto_stream_fatal",
                request_id=request_id,
                message=str(getattr(e, "detail", e)).strip() or type(e).__name__,
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                stage=stage,
                stage_detail=stage_detail,
            )
            yield _ndjson_line(fatal_event)
        except Exception as e:
            stream_has_error = True
            msg = str(e).strip() or type(e).__name__
            trace_status = "error"
            trace_error = msg
            log_event(
                "pareto_stream_fatal",
                request_id=request_id,
                message=msg,
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                stage=stage,
                stage_detail=stage_detail,
            )
            fatal_event = _stream_fatal_event_from_exception(e)
            fatal_event["request_id"] = request_id
            fatal_event["stage"] = stage
            fatal_event["stage_detail"] = stage_detail
            fatal_event["elapsed_ms"] = _elapsed_ms()
            fatal_event["stage_elapsed_ms"] = _stage_elapsed_ms()
            fatal_event["candidate_done"] = int(max(0, candidate_done))
            fatal_event["candidate_total"] = int(max(0, candidate_total))
            fatal_event["candidate_diagnostics"] = candidate_diag_payload
            yield _ndjson_line(fatal_event)
        finally:
            if options_task is not None and not options_task.done():
                options_task.cancel()
                await asyncio.gather(options_task, return_exceptions=True)
            finish_live_call_trace(
                request_id=request_id,
                endpoint="/api/pareto/stream",
                status=trace_status,
                error_reason=trace_error,
            )
            reset_live_call_trace(trace_token)
            _record_endpoint_metric("pareto_stream", t0, error=stream_has_error)

    return StreamingResponse(
        stream(),
        media_type="application/x-ndjson",
        headers={
            "cache-control": "no-store",
            "x-route-request-id": request_id,
        },
    )


def _resolve_pipeline_mode(requested_mode: str | None) -> str:
    aliases = {
        "legacy": "legacy",
        "dccs": "dccs",
        "dccs_refc": "dccs_refc",
        "voi": "voi",
        "thesis_voi": "voi",
        "voi_ad2r": "voi",
        "dccs+refc": "dccs_refc",
    }
    env_mode = aliases.get(str(settings.route_pipeline_default_mode or "legacy").strip().lower(), "legacy")
    if env_mode not in {"legacy", "dccs", "dccs_refc", "voi"}:
        env_mode = "legacy"
    if not bool(settings.route_pipeline_request_override_enabled):
        return env_mode
    request_mode = aliases.get(str(requested_mode or "").strip().lower(), "")
    if request_mode in {"legacy", "dccs", "dccs_refc", "voi"}:
        return request_mode
    return env_mode


def _resolve_pipeline_seed(req: RouteRequest) -> int:
    if req.pipeline_seed is not None:
        return int(req.pipeline_seed)
    request_payload = req.model_dump(mode="json")
    request_payload.pop("pipeline_seed", None)
    encoded = json.dumps(request_payload, sort_keys=True, separators=(",", ":"))
    deterministic = int(hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:8], 16)
    return int(max(0, deterministic ^ int(settings.route_pipeline_default_seed)))


def _split_csv_config(raw: str, *, fallback: Sequence[str] = ()) -> list[str]:
    items = [str(item).strip() for item in str(raw or "").split(",")]
    cleaned = [item for item in items if item]
    return cleaned or [str(item).strip() for item in fallback if str(item).strip()]


def _strict_frontier_options(
    options: list[RouteOption],
    *,
    max_alternatives: int,
    pareto_method: ParetoMethod = "dominance",
    epsilon: EpsilonConstraints | None = None,
    optimization_mode: OptimizationMode = "expected_value",
    risk_aversion: float = 1.0,
) -> list[RouteOption]:
    objective_fn = lambda option: _pareto_objective_key(
        option,
        optimization_mode=optimization_mode,
        risk_aversion=risk_aversion,
    )
    pareto_input = list(options)
    if optimization_mode == "robust" and pareto_input:
        robust_nondominated, _dom_matrix, _vectors = _robust_nondominated_indices(
            pareto_input,
            risk_aversion=risk_aversion,
        )
        if robust_nondominated:
            pareto_input = [pareto_input[idx] for idx in robust_nondominated]
    pareto = select_pareto_routes(
        pareto_input,
        max_alternatives=max_alternatives,
        pareto_method=pareto_method,
        epsilon=epsilon,
        strict_frontier=True,
        objective_key=objective_fn,
    )
    return _sort_options_by_mode(
        pareto,
        optimization_mode=optimization_mode,
        risk_aversion=risk_aversion,
    )


def _route_selection_score_map(
    options: list[RouteOption],
    *,
    w_time: float,
    w_money: float,
    w_co2: float,
    optimization_mode: OptimizationMode = "expected_value",
    risk_aversion: float = 1.0,
) -> dict[str, float]:
    if not options:
        return {}
    wt, wm, we = normalise_weights(w_time, w_money, w_co2)
    if optimization_mode == "robust":
        return {
            option.id: float(_option_joint_robust_utility(option, risk_aversion=risk_aversion))
            for option in options
        }

    times = [
        _option_objective_value(
            option,
            "duration",
            optimization_mode=optimization_mode,
            risk_aversion=risk_aversion,
        )
        for option in options
    ]
    moneys = [
        _option_objective_value(
            option,
            "money",
            optimization_mode=optimization_mode,
            risk_aversion=risk_aversion,
        )
        for option in options
    ]
    emissions = [
        _option_objective_value(
            option,
            "co2",
            optimization_mode=optimization_mode,
            risk_aversion=risk_aversion,
        )
        for option in options
    ]
    distances = [float(max(0.0, option.metrics.distance_km)) for option in options]

    t_min, t_max = min(times), max(times)
    m_min, m_max = min(moneys), max(moneys)
    e_min, e_max = min(emissions), max(emissions)
    d_min, d_max = min(distances), max(distances)

    def _norm(value: float, min_value: float, max_value: float) -> float:
        return 0.0 if max_value <= min_value else (value - min_value) / (max_value - min_value)

    math_profile = str(settings.route_selection_math_profile or "modified_vikor_distance").strip().lower()
    regret_weight = max(0.0, float(settings.route_selection_modified_regret_weight))
    balance_weight = max(0.0, float(settings.route_selection_modified_balance_weight))
    distance_weight = max(0.0, float(settings.route_selection_modified_distance_weight))
    eta_distance_weight = max(0.0, float(settings.route_selection_modified_eta_distance_weight))
    entropy_weight = max(0.0, float(settings.route_selection_modified_entropy_weight))
    knee_weight = max(0.0, float(settings.route_selection_modified_knee_weight))
    tchebycheff_rho = max(0.0, float(settings.route_selection_tchebycheff_rho))
    vikor_v = min(1.0, max(0.0, float(settings.route_selection_vikor_v)))

    norm_rows = [
        (
            _norm(time_v, t_min, t_max),
            _norm(money_v, m_min, m_max),
            _norm(co2_v, e_min, e_max),
            _norm(distance_v, d_min, d_max),
        )
        for time_v, money_v, co2_v, distance_v in zip(
            times,
            moneys,
            emissions,
            distances,
            strict=True,
        )
    ]
    weighted_sum_rows = [
        (wt * n_time) + (wm * n_money) + (we * n_co2)
        for n_time, n_money, n_co2, _ in norm_rows
    ]
    weighted_regret_rows = [
        max(wt * n_time, wm * n_money, we * n_co2)
        for n_time, n_money, n_co2, _ in norm_rows
    ]
    vikor_s_min = min(weighted_sum_rows)
    vikor_s_max = max(weighted_sum_rows)
    vikor_r_min = min(weighted_regret_rows)
    vikor_r_max = max(weighted_regret_rows)

    def _safe_scale(value: float, min_value: float, max_value: float) -> float:
        return 0.0 if max_value <= min_value else (value - min_value) / (max_value - min_value)

    score_map: dict[str, float] = {}
    for option, n_row, weighted_sum, weighted_regret in zip(
        options,
        norm_rows,
        weighted_sum_rows,
        weighted_regret_rows,
        strict=True,
    ):
        n_time, n_money, n_co2, n_distance = n_row
        mean_norm = (n_time + n_money + n_co2) / 3.0
        balance_penalty = math.sqrt(
            max(
                0.0,
                ((n_time - mean_norm) ** 2 + (n_money - mean_norm) ** 2 + (n_co2 - mean_norm) ** 2) / 3.0,
            )
        )
        knee_penalty = (
            abs(n_time - n_money)
            + abs(n_time - n_co2)
            + abs(n_money - n_co2)
        ) / 3.0
        eta_distance_penalty = math.sqrt(max(0.0, n_time * n_distance))
        improve_time = max(1e-6, 1.0 - n_time)
        improve_money = max(1e-6, 1.0 - n_money)
        improve_co2 = max(1e-6, 1.0 - n_co2)
        improve_sum = improve_time + improve_money + improve_co2
        p_time = improve_time / improve_sum
        p_money = improve_money / improve_sum
        p_co2 = improve_co2 / improve_sum
        entropy_reward = -(
            (p_time * math.log(p_time))
            + (p_money * math.log(p_money))
            + (p_co2 * math.log(p_co2))
        ) / math.log(3.0)
        vikor_q = (vikor_v * _safe_scale(weighted_sum, vikor_s_min, vikor_s_max)) + (
            (1.0 - vikor_v) * _safe_scale(weighted_regret, vikor_r_min, vikor_r_max)
        )

        if math_profile == "academic_reference":
            score = weighted_sum
        elif math_profile == "academic_tchebycheff":
            score = weighted_regret + (tchebycheff_rho * weighted_sum)
        elif math_profile == "academic_vikor":
            score = vikor_q
        elif math_profile == "modified_hybrid":
            score = weighted_sum + (regret_weight * weighted_regret) + (balance_weight * balance_penalty)
        elif math_profile == "modified_vikor_distance":
            score = (
                vikor_q
                + (balance_weight * balance_penalty)
                + (distance_weight * n_distance)
                + (eta_distance_weight * eta_distance_penalty)
                + (knee_weight * knee_penalty)
                - (entropy_weight * entropy_reward)
            )
        else:
            score = (
                weighted_sum
                + (regret_weight * weighted_regret)
                + (balance_weight * balance_penalty)
                + (distance_weight * n_distance)
                + (eta_distance_weight * eta_distance_penalty)
                + (knee_weight * knee_penalty)
                - (entropy_weight * entropy_reward)
            )
        score_map[option.id] = float(score)
    return score_map


def _clone_option_with_objectives(
    option: RouteOption,
    objective_vector: tuple[float, float, float],
) -> RouteOption:
    metrics = option.metrics.model_copy(
        update={
            "duration_s": round(float(objective_vector[0]), 2),
            "monetary_cost": round(float(objective_vector[1]), 2),
            "emissions_kg": round(float(objective_vector[2]), 3),
        }
    )
    uncertainty = dict(option.uncertainty) if isinstance(option.uncertainty, dict) else None
    if uncertainty is not None:
        uncertainty["mean_duration_s"] = float(metrics.duration_s)
        uncertainty["mean_monetary_cost"] = float(metrics.monetary_cost)
        uncertainty["mean_emissions_kg"] = float(metrics.emissions_kg)
    return option.model_copy(update={"metrics": metrics, "uncertainty": uncertainty})


def _selector_score_map_callback(
    frontier_options: Sequence[RouteOption],
    *,
    w_time: float,
    w_money: float,
    w_co2: float,
    optimization_mode: OptimizationMode,
    risk_aversion: float,
) -> Callable[[Sequence[Mapping[str, Any]], Mapping[str, tuple[float, float, float]]], Mapping[str, float]]:
    option_by_id = {option.id: option for option in frontier_options}

    def _callback(
        routes: Sequence[Mapping[str, Any]],
        perturbed_by_id: Mapping[str, tuple[float, float, float]],
    ) -> Mapping[str, float]:
        cloned: list[RouteOption] = []
        for route in routes:
            route_id = str(route.get("route_id", route.get("id", ""))).strip()
            base = option_by_id.get(route_id)
            if base is None:
                continue
            objective = perturbed_by_id.get(
                route_id,
                (
                    float(base.metrics.duration_s),
                    float(base.metrics.monetary_cost),
                    float(base.metrics.emissions_kg),
                ),
            )
            cloned.append(_clone_option_with_objectives(base, objective))
        return _route_selection_score_map(
            cloned,
            w_time=w_time,
            w_money=w_money,
            w_co2=w_co2,
            optimization_mode=optimization_mode,
            risk_aversion=risk_aversion,
        )

    return _callback


def _route_option_dependency_weights(option: RouteOption) -> dict[str, dict[str, float]]:
    provenance = option.evidence_provenance
    if provenance is None:
        return {}
    base_weights: dict[str, tuple[float, float, float]] = {
        "scenario": (0.70, 0.30, 0.20),
        "toll": (0.05, 0.95, 0.10),
        "terrain": (0.45, 0.20, 0.70),
        "fuel": (0.10, 0.85, 0.25),
        "carbon": (0.00, 0.80, 0.65),
        "weather": (0.60, 0.15, 0.15),
        "stochastic": (0.55, 0.55, 0.45),
    }
    out: dict[str, dict[str, float]] = {}
    for record in provenance.families:
        if not bool(record.active):
            continue
        confidence = float(record.confidence) if record.confidence is not None else 0.8
        coverage = float(record.coverage_ratio) if record.coverage_ratio is not None else 1.0
        strength = max(0.15, min(1.0, (0.55 * confidence) + (0.45 * coverage)))
        time_w, money_w, co2_w = base_weights.get(record.family, (0.35, 0.35, 0.35))
        out[record.family] = {
            "time": round(float(time_w * strength), 6),
            "money": round(float(money_w * strength), 6),
            "co2": round(float(co2_w * strength), 6),
        }
    return out


def _route_option_certification_payload(option: RouteOption) -> dict[str, Any]:
    dependency_weights = _route_option_dependency_weights(option)
    provenance_payload = option.evidence_provenance.model_dump(mode="json") if option.evidence_provenance else {}
    provenance_payload["dependency_weights"] = dependency_weights
    return {
        "route_id": option.id,
        "id": option.id,
        "objective_vector": (
            float(option.metrics.duration_s),
            float(option.metrics.monetary_cost),
            float(option.metrics.emissions_kg),
        ),
        "distance_km": float(option.metrics.distance_km),
        "evidence_tensor": dependency_weights,
        "evidence_provenance": provenance_payload,
        "route_option": option,
    }


def _apply_world_state_overrides(
    worlds: Sequence[Mapping[str, Any]],
    *,
    forced_refreshed_families: set[str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for world in worlds:
        states = dict(world.get("states", {}))
        for family in forced_refreshed_families:
            if family in states:
                states[family] = "refreshed"
        out.append({"world_id": world.get("world_id"), "states": states})
    return out


def _compute_frontier_certification(
    *,
    frontier_options: Sequence[RouteOption],
    selected_route_id: str,
    run_seed: int,
    world_count: int,
    threshold: float,
    w_time: float,
    w_money: float,
    w_co2: float,
    optimization_mode: OptimizationMode,
    risk_aversion: float,
    forced_refreshed_families: set[str] | None = None,
) -> tuple[Any, Any, dict[str, Any], list[str]]:
    route_payloads = [_route_option_certification_payload(option) for option in frontier_options]
    configured_families = _split_csv_config(
        settings.route_refc_evidence_families,
        fallback=("scenario", "toll", "terrain", "fuel", "carbon", "weather", "stochastic"),
    )
    active_families = active_evidence_families(route_payloads, configured_families=configured_families)
    world_manifest = sample_world_manifest(
        active_families=active_families,
        seed=int(run_seed),
        world_count=int(world_count),
        state_catalog=tuple(
            _split_csv_config(
                settings.route_refc_state_catalog,
                fallback=("nominal", "mildly_stale", "severely_stale", "low_confidence", "proxy", "refreshed"),
            )
        ),
    )
    forced_refresh = set(forced_refreshed_families or set())
    worlds = _apply_world_state_overrides(
        world_manifest.get("worlds", []),
        forced_refreshed_families=forced_refresh,
    )
    selector_score_map_fn = _selector_score_map_callback(
        frontier_options,
        w_time=w_time,
        w_money=w_money,
        w_co2=w_co2,
        optimization_mode=optimization_mode,
        risk_aversion=risk_aversion,
    )
    certificate = compute_certificate(
        route_payloads,
        worlds=worlds,
        selector_weights=(w_time, w_money, w_co2),
        threshold=threshold,
        active_families=active_families,
        selector_score_map_fn=selector_score_map_fn,
    )
    fragility = compute_fragility_maps(
        route_payloads,
        worlds=worlds,
        selector_weights=(w_time, w_money, w_co2),
        active_families=active_families,
        selected_route_id=selected_route_id,
        selector_score_map_fn=selector_score_map_fn,
    )
    manifest_payload = dict(world_manifest)
    manifest_payload["worlds"] = worlds
    manifest_payload["forced_refreshed_families"] = sorted(forced_refresh)
    manifest_payload["selected_route_id"] = selected_route_id
    return certificate, fragility, manifest_payload, active_families


def _near_tie_mass_from_certificate(
    certificate_map: Mapping[str, float],
    *,
    winner_id: str,
    threshold: float,
) -> float:
    winner_value = float(certificate_map.get(winner_id, 0.0))
    if winner_value <= 0.0:
        return 0.0
    competitors = [
        float(value)
        for route_id, value in certificate_map.items()
        if route_id != winner_id and (winner_value - float(value)) <= threshold
    ]
    return float(len(competitors)) / float(max(1, len(certificate_map) - 1))


def _top_competitor_for_route(
    route_id: str,
    competitor_breakdown: Mapping[str, Mapping[str, Mapping[str, int]]],
) -> str | None:
    route_breakdown = competitor_breakdown.get(route_id, {})
    best_id: str | None = None
    best_score = -1
    for competitor_id, family_counts in route_breakdown.items():
        total = sum(max(0, int(value)) for value in family_counts.values())
        if total > best_score or (total == best_score and best_id is not None and competitor_id < best_id):
            best_id = competitor_id
            best_score = total
    return best_id


def _attach_route_certifications(
    options: Sequence[RouteOption],
    *,
    certificate_map: Mapping[str, float],
    threshold: float,
    active_families: Sequence[str],
    fragility_map: Mapping[str, Mapping[str, float]],
    competitor_breakdown: Mapping[str, Mapping[str, Mapping[str, int]]],
    top_refresh_family: str | None,
) -> tuple[list[RouteOption], dict[str, RouteCertificationSummary]]:
    summaries: dict[str, RouteCertificationSummary] = {}
    updated: list[RouteOption] = []
    for option in options:
        if option.id not in certificate_map:
            updated.append(option)
            continue
        fragility_scores = fragility_map.get(option.id, {})
        ordered_families = [
            family
            for family, value in sorted(
                fragility_scores.items(),
                key=lambda item: (-float(item[1]), str(item[0])),
            )
            if float(value) > 0.0
        ]
        summary = RouteCertificationSummary(
            route_id=option.id,
            certificate=float(certificate_map.get(option.id, 0.0)),
            certified=float(certificate_map.get(option.id, 0.0)) >= float(threshold),
            threshold=float(threshold),
            active_families=list(active_families),
            top_fragility_families=ordered_families[:3],
            top_competitor_route_id=_top_competitor_for_route(option.id, competitor_breakdown),
            top_value_of_refresh_family=top_refresh_family,
        )
        summaries[option.id] = summary
        updated.append(option.model_copy(update={"certification": summary}))
    return updated, summaries


def _graph_route_candidate_payload(
    route: dict[str, Any],
    *,
    origin: LatLng,
    destination: LatLng,
    vehicle: VehicleProfile,
    cost_toggles: CostToggles,
) -> dict[str, Any]:
    meta = route.get("_graph_meta")
    meta_dict = meta if isinstance(meta, dict) else {}
    road_mix_counts = meta_dict.get("road_mix_counts")
    counts: dict[str, int]
    if isinstance(road_mix_counts, dict) and road_mix_counts:
        counts = {
            str(key): max(0, int(value))
            for key, value in road_mix_counts.items()
            if isinstance(value, (int, float))
        }
    else:
        counts = _route_road_class_counts(route)
    total_counts = max(1, sum(max(0, int(value)) for value in counts.values()))
    motorway_share = sum(
        max(0, int(value))
        for key, value in counts.items()
        if str(key).startswith("motorway")
    ) / float(total_counts)
    a_road_share = sum(
        max(0, int(value))
        for key, value in counts.items()
        if str(key).startswith("trunk")
        or str(key).startswith("primary")
        or str(key).startswith("secondary")
    ) / float(total_counts)
    urban_share = sum(
        max(0, int(value))
        for key, value in counts.items()
        if str(key).startswith("residential")
        or str(key).startswith("local")
        or str(key).startswith("unclassified")
    ) / float(total_counts)
    other_share = max(0.0, 1.0 - motorway_share - a_road_share - urban_share)
    path_node_ids = meta_dict.get("path_node_ids", [])
    graph_path = [str(node_id) for node_id in path_node_ids] if isinstance(path_node_ids, list) else []
    graph_length_km = float(max(0.0, _route_distance_km(route)))
    duration_s = float(max(0.0, _route_duration_s(route)))
    path_nodes = max(2, int(meta_dict.get("path_nodes", len(graph_path) or 2)))
    toll_share = max(0.0, float(meta_dict.get("toll_edges", 0))) / float(max(1, path_nodes - 1))
    turn_burden = max(0.0, float(meta_dict.get("turn_burden", 0.0)))
    terrain_burden = min(1.0, turn_burden / float(max(1, path_nodes - 2)))
    straight_line_km = max(0.001, float(_od_haversine_km(origin, destination)))
    proxy_money = (
        (float(vehicle.cost_per_km) * graph_length_km)
        + (float(vehicle.cost_per_hour) * (duration_s / 3600.0))
    )
    if bool(cost_toggles.use_tolls):
        proxy_money += float(cost_toggles.toll_cost_per_km) * graph_length_km * toll_share
    proxy_co2 = max(0.0, float(vehicle.emission_factor_kg_per_tkm) * float(vehicle.mass_tonnes) * graph_length_km)
    if float(cost_toggles.carbon_price_per_kg) > 0.0:
        proxy_money += proxy_co2 * float(cost_toggles.carbon_price_per_kg)
    candidate = {
        "graph_path": graph_path,
        "node_ids": graph_path,
        "graph_length_km": round(graph_length_km, 6),
        "straight_line_km": round(straight_line_km, 6),
        "road_class_mix": {
            "motorway_share": round(motorway_share, 6),
            "a_road_share": round(a_road_share, 6),
            "urban_share": round(urban_share, 6),
            "other_share": round(other_share, 6),
        },
        "motorway_share": round(motorway_share, 6),
        "a_road_share": round(a_road_share, 6),
        "urban_share": round(urban_share, 6),
        "toll_share": round(toll_share, 6),
        "terrain_burden": round(terrain_burden, 6),
        "proxy_objective": (
            round(duration_s, 6),
            round(proxy_money, 6),
            round(proxy_co2, 6),
        ),
        "mechanism_descriptor": {
            "motorway_share": round(motorway_share, 6),
            "a_road_share": round(a_road_share, 6),
            "urban_share": round(urban_share, 6),
            "toll_share": round(toll_share, 6),
            "terrain_burden": round(terrain_burden, 6),
        },
        "proxy_confidence": {
            "time": round(max(0.45, min(0.98, 0.82 + (0.08 * motorway_share))), 6),
            "money": round(max(0.40, min(0.95, 0.70 + (0.20 * (1.0 - toll_share)))), 6),
            "co2": round(max(0.40, min(0.95, 0.66 + (0.18 * (1.0 - urban_share)))), 6),
        },
    }
    candidate["candidate_id"] = stable_candidate_id(candidate)
    return candidate


def _dccs_runtime_config(
    *,
    mode: str,
    search_budget: int,
) -> DCCSConfig:
    return DCCSConfig(
        mode=mode,
        search_budget=max(0, int(search_budget)),
        bootstrap_seed_size=max(1, int(settings.route_dccs_bootstrap_count)),
        near_duplicate_threshold=float(settings.route_dccs_overlap_threshold),
        flip_bias=float(settings.route_dccs_pflip_bias),
        flip_objective_weight=float(settings.route_dccs_pflip_gap_weight),
        flip_mechanism_weight=float(settings.route_dccs_pflip_mechanism_weight),
        flip_overlap_weight=float(settings.route_dccs_pflip_overlap_weight),
        flip_stretch_weight=float(settings.route_dccs_pflip_detour_weight),
    )


async def _refine_graph_candidate_batch(
    *,
    osrm: OSRMClient,
    origin: LatLng,
    destination: LatLng,
    selected_records: Sequence[DCCSCandidateRecord],
    raw_graph_routes_by_id: Mapping[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], dict[str, float], int, float]:
    warnings: list[str] = []
    routes_by_signature: dict[str, dict[str, Any]] = {}
    observed_costs: dict[str, float] = {}
    candidate_fetches = 0
    batch_started = time.perf_counter()
    sem = asyncio.Semaphore(1)

    for record in selected_records:
        raw_route = raw_graph_routes_by_id.get(record.candidate_id)
        if raw_route is None:
            warnings.append(f"graph_family:{record.candidate_id}: missing_raw_candidate")
            continue
        via = _graph_family_via_points(
            raw_route,
            max_landmarks=max(1, int(settings.route_graph_via_landmarks_per_path)),
        )
        spec = CandidateFetchSpec(
            label=f"graph_family:{record.candidate_id}",
            alternatives=False,
            via=via if via else None,
        )
        fetch_started = time.perf_counter()
        result = await _run_candidate_fetch(
            osrm=osrm,
            origin=origin,
            destination=destination,
            spec=spec,
            sem=sem,
        )
        elapsed_ms = round((time.perf_counter() - fetch_started) * 1000.0, 2)
        observed_costs[record.candidate_id] = float(elapsed_ms)
        candidate_fetches += 1
        candidate_routes = result.routes or []
        if result.error:
            warnings.append(f"{result.spec.label}: {result.error}")
        if not candidate_routes:
            try:
                candidate_routes = [json.loads(json.dumps(raw_route, separators=(",", ":"), default=str))]
            except Exception:
                candidate_routes = [dict(raw_route)]
        for route in candidate_routes:
            route["_dccs_candidate_id"] = record.candidate_id
            existing_ids = route.get("_dccs_candidate_ids")
            route["_dccs_candidate_ids"] = list(existing_ids) if isinstance(existing_ids, list) else [record.candidate_id]
            if record.candidate_id not in route["_dccs_candidate_ids"]:
                route["_dccs_candidate_ids"].append(record.candidate_id)
            try:
                sig = _route_signature(route)
            except OSRMError:
                continue
            existing = routes_by_signature.get(sig)
            if existing is None:
                routes_by_signature[sig] = route
                existing = route
            existing_candidate_ids = existing.get("_dccs_candidate_ids")
            if not isinstance(existing_candidate_ids, list):
                existing_candidate_ids = []
                existing["_dccs_candidate_ids"] = existing_candidate_ids
            for candidate_id in route["_dccs_candidate_ids"]:
                if candidate_id not in existing_candidate_ids:
                    existing_candidate_ids.append(candidate_id)
            _annotate_route_candidate_meta(
                existing,
                source_labels={f"{result.spec.label}:osrm_refined"},
                toll_exclusion_available=False,
            )
    elapsed_total_ms = round((time.perf_counter() - batch_started) * 1000.0, 2)
    return list(routes_by_signature.values()), warnings, observed_costs, candidate_fetches, elapsed_total_ms


async def _compute_direct_route_pipeline(
    *,
    req: RouteRequest,
    osrm: OSRMClient,
    max_alternatives: int,
    pipeline_mode: str,
    run_seed: int,
) -> dict[str, Any]:
    warnings: list[str] = []
    stage_timings: dict[str, float] = {}
    dccs_batches: list[dict[str, Any]] = []
    action_trace: list[dict[str, Any]] = []
    action_score_rows: list[dict[str, Any]] = []
    forced_refreshed_families: set[str] = set()
    candidate_records_by_id: dict[str, DCCSCandidateRecord] = {}
    vehicle = resolve_vehicle_profile(req.vehicle_type)
    total_search_budget = max(1, int(req.search_budget or settings.route_pipeline_search_budget))
    total_evidence_budget = max(
        0,
        int(
            req.evidence_budget
            if req.evidence_budget is not None
            else settings.route_pipeline_evidence_budget
        ),
    )
    current_world_count = max(
        10,
        int(
            req.cert_world_count
            if req.cert_world_count is not None
            else settings.route_pipeline_cert_world_count
        ),
    )
    certificate_threshold = float(
        req.certificate_threshold
        if req.certificate_threshold is not None
        else settings.route_pipeline_certificate_threshold
    )
    tau_stop = float(req.tau_stop if req.tau_stop is not None else settings.route_pipeline_tau_stop)
    world_increment = max(1, int(settings.route_pipeline_world_increment))
    weather_bucket = req.weather.profile if req.weather.enabled else "clear"
    search_used = 0
    evidence_used = 0
    candidate_fetches = 0

    refresh_live_runtime_route_caches(mode=str(settings.live_route_compute_refresh_mode or "route_compute").strip().lower())
    scenario_started = time.perf_counter()
    scenario_context = await _scenario_context_from_od(
        origin=req.origin,
        destination=req.destination,
        vehicle_class=str(vehicle.vehicle_class),
        departure_time_utc=req.departure_time_utc,
        weather_bucket=str(weather_bucket),
    )
    scenario_modifiers = await _scenario_candidate_modifiers_async(
        scenario_mode=req.scenario_mode,
        context=scenario_context,
    )
    stage_timings["scenario_context_ms"] = round((time.perf_counter() - scenario_started) * 1000.0, 2)

    preflight_started = time.perf_counter()
    precheck = route_graph_od_feasibility(
        origin_lat=float(req.origin.lat),
        origin_lon=float(req.origin.lon),
        destination_lat=float(req.destination.lat),
        destination_lon=float(req.destination.lon),
    )
    stage_timings["preflight_ms"] = round((time.perf_counter() - preflight_started) * 1000.0, 2)
    precheck_kwargs = _candidate_precheck_diag_kwargs(precheck)
    if not bool(precheck.get("ok")):
        message = _route_graph_precheck_warning(precheck)
        warnings.append(message)
        raise HTTPException(
            status_code=422,
            detail=_strict_error_detail(
                reason_code=str(precheck.get("reason_code", "routing_graph_unavailable")),
                message=str(precheck.get("message", "Graph feasibility check failed.")),
                warnings=warnings,
                extra={
                    "stage": "collecting_candidates",
                    "stage_detail": "route_graph_precheck_failed",
                },
            ),
        )
    start_node_id = str(precheck.get("origin_node_id", "")).strip() or None
    goal_node_id = str(precheck.get("destination_node_id", "")).strip() or None

    graph_started = time.perf_counter()
    graph_routes, graph_diag = await _route_graph_candidate_routes_async(
        origin_lat=float(req.origin.lat),
        origin_lon=float(req.origin.lon),
        destination_lat=float(req.destination.lat),
        destination_lon=float(req.destination.lon),
        max_paths=max(4, int(max_alternatives) * 2),
        scenario_edge_modifiers=scenario_modifiers,
        search_deadline_s=_search_deadline_s(int(settings.route_graph_search_initial_timeout_ms)),
        start_node_id=start_node_id,
        goal_node_id=goal_node_id,
    )
    stage_timings["k_raw_ms"] = round((time.perf_counter() - graph_started) * 1000.0, 2)
    if not graph_routes:
        no_path_reason = _normalize_route_graph_search_reason(str(graph_diag.no_path_reason or ""))
        no_path_detail = (
            str(graph_diag.no_path_detail or "").strip()
            or "Route graph search produced no candidate paths."
        )
        warnings.append(f"route_graph: {no_path_reason} ({no_path_detail})")
        raise HTTPException(
            status_code=422,
            detail=_strict_error_detail(
                reason_code=no_path_reason,
                message=no_path_detail,
                warnings=warnings,
                extra={
                    "stage": "collecting_candidates",
                    "stage_detail": "route_graph_search_empty",
                },
            ),
        )

    raw_candidate_payloads = [
        _graph_route_candidate_payload(
            route,
            origin=req.origin,
            destination=req.destination,
            vehicle=vehicle,
            cost_toggles=req.cost_toggles,
        )
        for route in graph_routes
    ]
    raw_graph_routes_by_id = {
        str(candidate["candidate_id"]): route
        for candidate, route in zip(raw_candidate_payloads, graph_routes, strict=True)
    }
    raw_candidate_by_id = {
        str(candidate["candidate_id"]): candidate
        for candidate in raw_candidate_payloads
    }
    initial_budget = total_search_budget
    if pipeline_mode == "voi":
        initial_budget = min(
            total_search_budget,
            max(1, int(settings.route_dccs_bootstrap_count)),
        )
    initial_dccs_started = time.perf_counter()
    initial_dccs = select_candidates(
        raw_candidate_payloads,
        config=_dccs_runtime_config(mode="bootstrap", search_budget=initial_budget),
    )
    stage_timings["dccs_ms"] = round((time.perf_counter() - initial_dccs_started) * 1000.0, 2)
    dccs_batches.append({"iteration": 0, **initial_dccs.summary})
    for record in [*initial_dccs.selected, *initial_dccs.skipped]:
        candidate_records_by_id[record.candidate_id] = record

    refined_routes: list[dict[str, Any]] = []
    refined_route_signatures: set[str] = set()
    refined_candidate_ids: set[str] = set()

    def _ingest_refined_routes(routes: Sequence[dict[str, Any]]) -> None:
        for route in routes:
            try:
                signature = _route_signature(route)
            except OSRMError:
                continue
            if signature in refined_route_signatures:
                continue
            refined_route_signatures.add(signature)
            refined_routes.append(route)

    batch_routes, batch_warnings, observed_costs, batch_fetches, batch_refine_ms = await _refine_graph_candidate_batch(
        osrm=osrm,
        origin=req.origin,
        destination=req.destination,
        selected_records=initial_dccs.selected,
        raw_graph_routes_by_id=raw_graph_routes_by_id,
    )
    warnings.extend(batch_warnings)
    _ingest_refined_routes(batch_routes)
    candidate_fetches += batch_fetches
    search_used += len(initial_dccs.selected)
    refined_candidate_ids.update(record.candidate_id for record in initial_dccs.selected)
    stage_timings["osrm_refine_ms"] = round(batch_refine_ms, 2)

    def _rebuild_route_state() -> tuple[
        list[RouteOption],
        list[RouteOption],
        list[RouteOption],
        RouteOption,
        TerrainDiagnostics,
        dict[str, list[str]],
        dict[str, float],
    ]:
        build_started = time.perf_counter()
        options, build_warnings, terrain_diag = _build_options(
            refined_routes,
            vehicle_type=req.vehicle_type,
            scenario_mode=req.scenario_mode,
            cost_toggles=req.cost_toggles,
            terrain_profile=req.terrain_profile,
            stochastic=req.stochastic,
            emissions_context=req.emissions_context,
            weather=req.weather,
            incident_simulation=req.incident_simulation,
            departure_time_utc=req.departure_time_utc,
            utility_weights=(req.weights.time, req.weights.money, req.weights.co2),
            risk_aversion=req.risk_aversion,
            option_prefix="route",
        )
        stage_timings["option_build_ms"] = stage_timings.get("option_build_ms", 0.0) + round(
            (time.perf_counter() - build_started) * 1000.0,
            2,
        )
        for warning in build_warnings:
            if warning not in warnings:
                warnings.append(warning)
        pareto_started = time.perf_counter()
        strict_frontier = _strict_frontier_options(
            options,
            max_alternatives=max_alternatives,
            pareto_method=req.pareto_method,
            epsilon=req.epsilon,
            optimization_mode=req.optimization_mode,
            risk_aversion=req.risk_aversion,
        )
        if not strict_frontier:
            strict_detail = _strict_failure_detail_from_outcome(
                warnings=warnings,
                terrain_diag=terrain_diag,
                epsilon_requested=req.pareto_method == "epsilon_constraint" and req.epsilon is not None,
                terrain_message="Terrain DEM coverage is insufficient for strict routing policy.",
            )
            if strict_detail is not None:
                raise HTTPException(status_code=422, detail=strict_detail)
            raise HTTPException(
                status_code=422,
                detail=_strict_error_detail(
                    reason_code="epsilon_infeasible",
                    message="No routes satisfy epsilon constraints for this request.",
                    warnings=warnings,
                ),
            )
        display_candidates = _finalize_pareto_options(
            options,
            max_alternatives=max_alternatives,
            pareto_method=req.pareto_method,
            epsilon=req.epsilon,
            optimization_mode=req.optimization_mode,
            risk_aversion=req.risk_aversion,
        )
        selected = _pick_best_option(
            strict_frontier,
            w_time=req.weights.time,
            w_money=req.weights.money,
            w_co2=req.weights.co2,
            optimization_mode=req.optimization_mode,
            risk_aversion=req.risk_aversion,
        )
        if all(option.id != selected.id for option in display_candidates):
            display_candidates = [
                selected,
                *[option for option in display_candidates if option.id != selected.id],
            ][: max_alternatives]
        stage_timings["pareto_ms"] = stage_timings.get("pareto_ms", 0.0) + round(
            (time.perf_counter() - pareto_started) * 1000.0,
            2,
        )
        option_candidate_ids = {
            option.id: [
                str(candidate_id)
                for candidate_id in (
                    route.get("_dccs_candidate_ids", [])
                    if isinstance(route.get("_dccs_candidate_ids"), list)
                    else []
                )
                if str(candidate_id).strip()
            ]
            for route, option in zip(refined_routes, options, strict=True)
        }
        selection_score_map = _route_selection_score_map(
            display_candidates,
            w_time=req.weights.time,
            w_money=req.weights.money,
            w_co2=req.weights.co2,
            optimization_mode=req.optimization_mode,
            risk_aversion=req.risk_aversion,
        )
        return (
            options,
            strict_frontier,
            display_candidates,
            selected,
            terrain_diag,
            option_candidate_ids,
            selection_score_map,
        )

    def _record_candidate_batch_outcomes(
        *,
        selected_records: Sequence[DCCSCandidateRecord],
        previous_frontier_ids: set[str],
        previous_selected_id: str | None,
        option_candidate_ids: Mapping[str, list[str]],
        current_frontier_ids: set[str],
        current_selected_id: str,
        selection_score_map: Mapping[str, float],
        observed_refine_costs: Mapping[str, float],
    ) -> None:
        selected_score = float(selection_score_map.get(current_selected_id, 0.0))
        for record in selected_records:
            candidate_option_ids = [
                option_id
                for option_id, candidate_ids in option_candidate_ids.items()
                if record.candidate_id in candidate_ids
            ]
            frontier_added = any(
                option_id in current_frontier_ids and option_id not in previous_frontier_ids
                for option_id in candidate_option_ids
            )
            decision_flip = (
                previous_selected_id is not None
                and previous_selected_id != current_selected_id
                and current_selected_id in candidate_option_ids
            )
            best_candidate_score = min(
                (float(selection_score_map.get(option_id, float("inf"))) for option_id in candidate_option_ids),
                default=float("inf"),
            )
            dominated_but_close = (
                bool(candidate_option_ids)
                and not frontier_added
                and not decision_flip
                and math.isfinite(best_candidate_score)
                and best_candidate_score <= (selected_score + (0.05 * max(1.0, abs(selected_score))))
            )
            redundant = not frontier_added and not decision_flip and not dominated_but_close
            candidate_records_by_id[record.candidate_id] = record_refine_outcome(
                record,
                observed_refine_cost=float(
                    observed_refine_costs.get(record.candidate_id, record.predicted_refine_cost)
                ),
                frontier_added=frontier_added,
                decision_flip=decision_flip,
                dominated_but_close=dominated_but_close,
                redundant=redundant,
            )

    options, strict_frontier, display_candidates, selected, terrain_diag, option_candidate_ids, selection_score_map = _rebuild_route_state()
    _record_candidate_batch_outcomes(
        selected_records=initial_dccs.selected,
        previous_frontier_ids=set(),
        previous_selected_id=None,
        option_candidate_ids=option_candidate_ids,
        current_frontier_ids={option.id for option in strict_frontier},
        current_selected_id=selected.id,
        selection_score_map=selection_score_map,
        observed_refine_costs=observed_costs,
    )

    certificate_result: Any | None = None
    fragility_result: Any | None = None
    world_manifest_payload: dict[str, Any] | None = None
    active_families: list[str] = []
    selected_certificate: RouteCertificationSummary | None = None
    stop_reason = "not_applicable"
    best_rejected_action_payload: dict[str, Any] | None = None

    def _route_option_dccs_payload(
        option: RouteOption,
        *,
        option_candidate_ids_map: Mapping[str, list[str]],
    ) -> dict[str, Any]:
        candidate_ids = [
            candidate_id
            for candidate_id in option_candidate_ids_map.get(option.id, [])
            if candidate_id in raw_candidate_by_id
        ]
        raw_candidate = raw_candidate_by_id.get(candidate_ids[0]) if candidate_ids else None
        road_mix = (
            dict(raw_candidate.get("road_class_mix", {}))
            if isinstance(raw_candidate, Mapping) and isinstance(raw_candidate.get("road_class_mix"), Mapping)
            else {}
        )
        mechanism = (
            dict(raw_candidate.get("mechanism_descriptor", {}))
            if isinstance(raw_candidate, Mapping) and isinstance(raw_candidate.get("mechanism_descriptor"), Mapping)
            else {}
        )
        return {
            "candidate_id": candidate_ids[0] if candidate_ids else option.id,
            "route_id": option.id,
            "id": option.id,
            "graph_path": list(raw_candidate.get("graph_path", [])) if isinstance(raw_candidate, Mapping) else [],
            "graph_length_km": float(
                raw_candidate.get("graph_length_km", option.metrics.distance_km)
            )
            if isinstance(raw_candidate, Mapping)
            else float(option.metrics.distance_km),
            "straight_line_km": max(0.001, float(_od_haversine_km(req.origin, req.destination))),
            "road_class_mix": road_mix,
            "motorway_share": float(road_mix.get("motorway_share", 0.0)),
            "a_road_share": float(road_mix.get("a_road_share", 0.0)),
            "urban_share": float(road_mix.get("urban_share", 0.0)),
            "toll_share": float(raw_candidate.get("toll_share", 0.0)) if isinstance(raw_candidate, Mapping) else 0.0,
            "terrain_burden": float(raw_candidate.get("terrain_burden", 0.0))
            if isinstance(raw_candidate, Mapping)
            else 0.0,
            "proxy_objective": (
                float(option.metrics.duration_s),
                float(option.metrics.monetary_cost),
                float(option.metrics.emissions_kg),
            ),
            "objective_vector": (
                float(option.metrics.duration_s),
                float(option.metrics.monetary_cost),
                float(option.metrics.emissions_kg),
            ),
            "mechanism_descriptor": mechanism,
            "proxy_confidence": (
                dict(raw_candidate.get("proxy_confidence", {}))
                if isinstance(raw_candidate, Mapping) and isinstance(raw_candidate.get("proxy_confidence"), Mapping)
                else {}
            ),
        }

    def _route_signature_map(current_options: Sequence[RouteOption]) -> dict[str, str]:
        signatures: dict[str, str] = {}
        for route, option in zip(refined_routes, current_options, strict=True):
            try:
                signatures[option.id] = _route_signature(route)
            except OSRMError:
                continue
        return signatures

    def _refresh_certification_views() -> None:
        nonlocal certificate_result
        nonlocal fragility_result
        nonlocal world_manifest_payload
        nonlocal active_families
        nonlocal selected_certificate
        nonlocal strict_frontier
        nonlocal display_candidates
        nonlocal selected
        if pipeline_mode not in {"dccs_refc", "voi"}:
            return
        refc_started = time.perf_counter()
        certificate_result, fragility_result, world_manifest_payload, active_families = _compute_frontier_certification(
            frontier_options=strict_frontier,
            selected_route_id=selected.id,
            run_seed=run_seed,
            world_count=current_world_count,
            threshold=certificate_threshold,
            w_time=req.weights.time,
            w_money=req.weights.money,
            w_co2=req.weights.co2,
            optimization_mode=req.optimization_mode,
            risk_aversion=req.risk_aversion,
            forced_refreshed_families=forced_refreshed_families,
        )
        stage_timings["refc_ms"] = stage_timings.get("refc_ms", 0.0) + round(
            (time.perf_counter() - refc_started) * 1000.0,
            2,
        )
        top_refresh_family = None
        if isinstance(fragility_result.value_of_refresh, Mapping):
            family_raw = str(fragility_result.value_of_refresh.get("top_refresh_family", "")).strip()
            top_refresh_family = family_raw or None
        strict_frontier, strict_summaries = _attach_route_certifications(
            strict_frontier,
            certificate_map=certificate_result.certificate,
            threshold=certificate_threshold,
            active_families=active_families,
            fragility_map=fragility_result.route_fragility_map,
            competitor_breakdown=fragility_result.competitor_fragility_breakdown,
            top_refresh_family=top_refresh_family,
        )
        display_candidates, display_summaries = _attach_route_certifications(
            display_candidates,
            certificate_map=certificate_result.certificate,
            threshold=certificate_threshold,
            active_families=active_families,
            fragility_map=fragility_result.route_fragility_map,
            competitor_breakdown=fragility_result.competitor_fragility_breakdown,
            top_refresh_family=top_refresh_family,
        )
        selected = next((option for option in display_candidates if option.id == selected.id), selected)
        selected_certificate = (
            display_summaries.get(selected.id)
            or strict_summaries.get(selected.id)
        )

    if pipeline_mode in {"dccs_refc", "voi"}:
        _refresh_certification_views()

    if pipeline_mode == "voi":
        voi_started = time.perf_counter()
        voi_config = VOIConfig(
            certificate_threshold=certificate_threshold,
            stop_threshold=tau_stop,
            search_budget=total_search_budget,
            evidence_budget=total_evidence_budget,
            max_iterations=max(1, total_search_budget + total_evidence_budget + 2),
            top_k_refine=min(2, max(1, total_search_budget)),
            resample_increment=world_increment,
        )

        while True:
            remaining_search_budget = max(0, total_search_budget - search_used)
            remaining_evidence_budget = max(0, total_evidence_budget - evidence_used)
            selected_cert_value = (
                float(certificate_result.certificate.get(selected.id, 0.0))
                if certificate_result is not None
                else 0.0
            )
            if selected_cert_value >= certificate_threshold:
                stop_reason = "certified"
                break
            if remaining_search_budget <= 0 and remaining_evidence_budget <= 0:
                stop_reason = "budget_exhausted"
                break
            if fragility_result is None or certificate_result is None:
                stop_reason = "error"
                break

            frontier_payloads = [
                _route_option_dccs_payload(option, option_candidate_ids_map=option_candidate_ids)
                for option in strict_frontier
            ]
            refined_payloads = [
                _route_option_dccs_payload(option, option_candidate_ids_map=option_candidate_ids)
                for option in options
            ]
            remaining_candidates = [
                raw_candidate_by_id[candidate_id]
                for candidate_id in sorted(raw_candidate_by_id)
                if candidate_id not in refined_candidate_ids
            ]
            challenger_started = time.perf_counter()
            challenger_dccs = select_candidates(
                remaining_candidates,
                frontier=frontier_payloads,
                refined=refined_payloads,
                config=_dccs_runtime_config(
                    mode="challenger",
                    search_budget=remaining_search_budget,
                ),
            )
            stage_timings["dccs_ms"] = stage_timings.get("dccs_ms", 0.0) + round(
                (time.perf_counter() - challenger_started) * 1000.0,
                2,
            )
            dccs_batches.append(
                {"iteration": len(action_trace) + 1, **challenger_dccs.summary}
            )
            for record in [*challenger_dccs.selected, *challenger_dccs.skipped]:
                candidate_records_by_id[record.candidate_id] = record

            near_tie_mass = _near_tie_mass_from_certificate(
                certificate_result.certificate,
                winner_id=selected.id,
                threshold=0.03,
            )
            controller_state = VOIControllerState(
                iteration_index=len(action_trace),
                frontier=frontier_payloads,
                certificate=dict(certificate_result.certificate),
                winner_id=selected.id,
                selected_route_id=selected.id,
                remaining_search_budget=remaining_search_budget,
                remaining_evidence_budget=remaining_evidence_budget,
                action_trace=list(action_trace),
                active_evidence_families=list(active_families),
                near_tie_mass=near_tie_mass,
            )
            actions = [
                action
                for action in build_action_menu(
                    controller_state,
                    dccs=challenger_dccs,
                    fragility=fragility_result,
                    config=voi_config,
                )
                if not (
                    action.kind == "refresh_top1_vor"
                    and action.target in forced_refreshed_families
                )
            ]
            feasible_actions = [action for action in actions if action.kind != "stop"]
            for action in actions:
                action_score_rows.append(
                    {
                        "iteration": len(action_trace),
                        "action_id": action.action_id,
                        "kind": action.kind,
                        "target": action.target,
                        "cost_search": action.cost_search,
                        "cost_evidence": action.cost_evidence,
                        "predicted_delta_certificate": round(float(action.predicted_delta_certificate), 6),
                        "predicted_delta_margin": round(float(action.predicted_delta_margin), 6),
                        "predicted_delta_frontier": round(float(action.predicted_delta_frontier), 6),
                        "q_score": round(float(action.q_score), 6),
                        "selected_route_id": selected.id,
                        "selected_certificate": round(float(selected_cert_value), 6),
                    }
                )
            if not feasible_actions:
                stop_reason = "no_action_worth_it"
                break
            chosen_action = feasible_actions[0]
            best_rejected_action_payload = (
                feasible_actions[1].as_dict() if len(feasible_actions) > 1 else chosen_action.as_dict()
            )
            trace_entry = {
                "iteration": len(action_trace),
                "selected_route_id": selected.id,
                "selected_certificate": round(float(selected_cert_value), 6),
                "remaining_search_budget": remaining_search_budget,
                "remaining_evidence_budget": remaining_evidence_budget,
                "frontier_size": len(strict_frontier),
                "feasible_actions": [action.as_dict() for action in actions],
            }
            if float(chosen_action.q_score) < tau_stop:
                trace_entry["chosen_action"] = {
                    "action_id": "stop",
                    "kind": "stop",
                    "target": "stop",
                    "q_score": 0.0,
                }
                action_trace.append(trace_entry)
                stop_reason = "no_action_worth_it"
                break

            trace_entry["chosen_action"] = chosen_action.as_dict()
            action_trace.append(trace_entry)

            if chosen_action.kind in {"refine_top1_dccs", "refine_topk_dccs"}:
                top_k = max(1, int(chosen_action.metadata.get("top_k", 1)))
                selected_records = challenger_dccs.selected[:top_k]
                if not selected_records:
                    stop_reason = "no_action_worth_it"
                    break
                previous_frontier_ids = {option.id for option in strict_frontier}
                previous_selected_id = selected.id
                batch_routes, batch_warnings, batch_observed_costs, batch_fetches, batch_refine_ms = await _refine_graph_candidate_batch(
                    osrm=osrm,
                    origin=req.origin,
                    destination=req.destination,
                    selected_records=selected_records,
                    raw_graph_routes_by_id=raw_graph_routes_by_id,
                )
                warnings.extend(batch_warnings)
                _ingest_refined_routes(batch_routes)
                candidate_fetches += batch_fetches
                search_used += len(selected_records)
                refined_candidate_ids.update(record.candidate_id for record in selected_records)
                stage_timings["osrm_refine_ms"] = stage_timings.get("osrm_refine_ms", 0.0) + round(
                    batch_refine_ms,
                    2,
                )
                options, strict_frontier, display_candidates, selected, terrain_diag, option_candidate_ids, selection_score_map = _rebuild_route_state()
                _record_candidate_batch_outcomes(
                    selected_records=selected_records,
                    previous_frontier_ids=previous_frontier_ids,
                    previous_selected_id=previous_selected_id,
                    option_candidate_ids=option_candidate_ids,
                    current_frontier_ids={option.id for option in strict_frontier},
                    current_selected_id=selected.id,
                    selection_score_map=selection_score_map,
                    observed_refine_costs=batch_observed_costs,
                )
                _refresh_certification_views()
                continue

            if chosen_action.kind == "refresh_top1_vor":
                forced_refreshed_families.add(chosen_action.target)
                evidence_used += max(1, int(chosen_action.cost_evidence))
                _refresh_certification_views()
                continue

            if chosen_action.kind == "increase_stochastic_samples":
                current_world_count += world_increment
                evidence_used += max(1, int(chosen_action.cost_evidence))
                _refresh_certification_views()
                continue

            stop_reason = "no_action_worth_it"
            break

        stage_timings["voi_ms"] = round((time.perf_counter() - voi_started) * 1000.0, 2)
    elif pipeline_mode in {"dccs", "dccs_refc"}:
        stop_reason = "single_pass"

    route_signature_map = _route_signature_map(options)
    final_selected_certificate = (
        float(certificate_result.certificate.get(selected.id, 0.0))
        if certificate_result is not None
        else None
    )
    if pipeline_mode == "voi":
        best_rejected_action = None
        best_rejected_q = None
        if best_rejected_action_payload is not None:
            best_rejected_action = str(
                best_rejected_action_payload.get(
                    "action_id",
                    best_rejected_action_payload.get("kind", ""),
                )
            ).strip() or None
            q_value = best_rejected_action_payload.get("q_score")
            best_rejected_q = float(q_value) if isinstance(q_value, (int, float)) else None
        voi_stop_summary = VoiStopSummary(
            final_route_id=selected.id,
            certificate=float(final_selected_certificate or 0.0),
            certified=bool(
                final_selected_certificate is not None
                and final_selected_certificate >= certificate_threshold
            ),
            iteration_count=len(action_trace),
            search_budget_used=int(search_used),
            evidence_budget_used=int(evidence_used),
            stop_reason=stop_reason,
            best_rejected_action=best_rejected_action,
            best_rejected_q=best_rejected_q,
        )
    else:
        voi_stop_summary = None

    candidate_diag = CandidateDiagnostics(
        raw_count=len(raw_candidate_payloads),
        deduped_count=len(refined_routes),
        graph_explored_states=int(graph_diag.explored_states),
        graph_generated_paths=int(graph_diag.generated_paths),
        graph_emitted_paths=int(graph_diag.emitted_paths),
        candidate_budget=int(graph_diag.candidate_budget),
        graph_effective_max_hops=int(getattr(graph_diag, "effective_max_hops", 0)),
        graph_effective_hops_floor=int(getattr(graph_diag, "effective_hops_floor", 0)),
        graph_effective_state_budget_initial=int(
            getattr(graph_diag, "effective_state_budget_initial", 0)
        ),
        graph_effective_state_budget=int(getattr(graph_diag, "effective_state_budget", 0)),
        graph_no_path_reason=str(graph_diag.no_path_reason or ""),
        graph_no_path_detail=str(graph_diag.no_path_detail or ""),
        scenario_context_ms=float(stage_timings.get("scenario_context_ms", 0.0)),
        graph_search_ms_initial=float(stage_timings.get("k_raw_ms", 0.0)),
        osrm_refine_ms=float(stage_timings.get("osrm_refine_ms", 0.0)),
        build_options_ms=float(stage_timings.get("option_build_ms", 0.0)),
        **precheck_kwargs,
    )

    dccs_rows = [
        candidate_records_by_id[candidate_id].as_dict()
        for candidate_id in sorted(candidate_records_by_id)
    ]
    refined_count = sum(1 for row in dccs_rows if row.get("observed_refine_cost") is not None)
    frontier_additions = sum(1 for row in dccs_rows if row.get("decision_reason") == "frontier_addition")
    decision_flips = sum(1 for row in dccs_rows if row.get("decision_reason") == "decision_flip")
    challenger_hits = sum(
        1
        for row in dccs_rows
        if row.get("decision_reason") in {"frontier_addition", "decision_flip", "challenger_but_not_added"}
    )
    dccs_summary = {
        "pipeline_mode": pipeline_mode,
        "search_budget_total": int(total_search_budget),
        "search_budget_used": int(search_used),
        "candidate_count_raw": len(raw_candidate_payloads),
        "refined_count": refined_count,
        "frontier_count": len(strict_frontier),
        "selected_route_id": selected.id,
        "selected_candidate_ids": option_candidate_ids.get(selected.id, []),
        "dc_yield": round(
            (frontier_additions + decision_flips) / float(max(1, refined_count)),
            6,
        ),
        "challenger_hit_rate": round(
            challenger_hits / float(max(1, refined_count)),
            6,
        ),
        "frontier_gain_per_refinement": round(
            frontier_additions / float(max(1, refined_count)),
            6,
        ),
        "decision_flips": decision_flips,
        "frontier_additions": frontier_additions,
        "candidate_fetches": int(candidate_fetches),
        "overlap_threshold": float(settings.route_dccs_overlap_threshold),
        "baseline_policy": str(settings.route_dccs_default_baseline_policy),
        "batches": dccs_batches,
    }

    refined_route_rows = []
    for route, option in zip(refined_routes, options, strict=True):
        refined_route_rows.append(
            {
                "route_id": option.id,
                "route_signature": route_signature_map.get(option.id, ""),
                "candidate_ids": option_candidate_ids.get(option.id, []),
                "distance_km": float(option.metrics.distance_km),
                "duration_s": float(option.metrics.duration_s),
                "monetary_cost": float(option.metrics.monetary_cost),
                "emissions_kg": float(option.metrics.emissions_kg),
                "selected": option.id == selected.id,
            }
        )
    strict_frontier_rows = [
        {
            "route_id": option.id,
            "route_signature": route_signature_map.get(option.id, ""),
            "candidate_ids": option_candidate_ids.get(option.id, []),
            "distance_km": float(option.metrics.distance_km),
            "duration_s": float(option.metrics.duration_s),
            "monetary_cost": float(option.metrics.monetary_cost),
            "emissions_kg": float(option.metrics.emissions_kg),
            "certificate": (
                float(certificate_result.certificate.get(option.id, 0.0))
                if certificate_result is not None
                else None
            ),
            "selected": option.id == selected.id,
        }
        for option in strict_frontier
    ]
    winner_summary = {
        "pipeline_mode": pipeline_mode,
        "route_id": selected.id,
        "route_signature": route_signature_map.get(selected.id, ""),
        "candidate_ids": option_candidate_ids.get(selected.id, []),
        "objective_vector": {
            "time": float(selected.metrics.duration_s),
            "money": float(selected.metrics.monetary_cost),
            "co2": float(selected.metrics.emissions_kg),
        },
        "selector_score": float(selection_score_map.get(selected.id, 0.0)),
        "certificate": final_selected_certificate,
        "certified": bool(
            final_selected_certificate is not None
            and final_selected_certificate >= certificate_threshold
        ),
    }

    if certificate_result is not None and fragility_result is not None and world_manifest_payload is not None:
        certificate_summary = {
            "pipeline_mode": pipeline_mode,
            "winner_route_id": certificate_result.winner_id,
            "selected_route_id": selected.id,
            "selected_certificate": float(certificate_result.certificate.get(selected.id, 0.0)),
            "certificate_threshold": float(certificate_threshold),
            "certified": bool(
                float(certificate_result.certificate.get(selected.id, 0.0)) >= certificate_threshold
            ),
            "route_certificates": dict(certificate_result.certificate),
            "frontier_route_ids": [option.id for option in strict_frontier],
            "world_count": int(world_manifest_payload.get("world_count", current_world_count)),
            "active_families": list(active_families),
            "selector_config": certificate_result.selector_config,
            "forced_refreshed_families": sorted(forced_refreshed_families),
        }
        route_fragility_map = fragility_result.route_fragility_map
        competitor_fragility_breakdown = fragility_result.competitor_fragility_breakdown
        value_of_refresh = fragility_result.value_of_refresh
        sampled_world_manifest = world_manifest_payload
    else:
        certificate_summary = {
            "pipeline_mode": pipeline_mode,
            "status": "not_requested",
            "selected_route_id": selected.id,
        }
        route_fragility_map = {}
        competitor_fragility_breakdown = {}
        value_of_refresh = {"ranking": [], "top_refresh_family": None}
        sampled_world_manifest = {
            "status": "not_requested",
            "world_count": 0,
            "active_families": [],
            "worlds": [],
        }

    voi_stop_certificate_payload = (
        {
            "final_winner_route_id": selected.id,
            "final_winner_objective_vector": {
                "time": float(selected.metrics.duration_s),
                "money": float(selected.metrics.monetary_cost),
                "co2": float(selected.metrics.emissions_kg),
            },
            "final_strict_frontier_size": len(strict_frontier),
            "certificate_value": float(final_selected_certificate or 0.0),
            "certified": bool(
                final_selected_certificate is not None
                and final_selected_certificate >= certificate_threshold
            ),
            "search_budget_used": int(search_used),
            "search_budget_remaining": int(max(0, total_search_budget - search_used)),
            "evidence_budget_used": int(evidence_used),
            "evidence_budget_remaining": int(max(0, total_evidence_budget - evidence_used)),
            "stop_reason": stop_reason,
            "action_trace": action_trace,
            "best_rejected_action": best_rejected_action_payload,
            "ambiguity_summary": {
                "top_fragility_families": (
                    selected_certificate.top_fragility_families if selected_certificate is not None else []
                ),
                "top_refresh_family": (
                    selected_certificate.top_value_of_refresh_family if selected_certificate is not None else None
                ),
                "top_competitor_route_id": (
                    selected_certificate.top_competitor_route_id if selected_certificate is not None else None
                ),
            },
        }
        if pipeline_mode == "voi"
        else {
            "pipeline_mode": pipeline_mode,
            "status": "not_requested",
            "selected_route_id": selected.id,
        }
    )
    final_route_trace = {
        "pipeline_mode": pipeline_mode,
        "run_seed": int(run_seed),
        "stage_timings_ms": {key: round(float(value), 2) for key, value in stage_timings.items()},
        "counts": {
            "k_raw": len(raw_candidate_payloads),
            "refined": len(refined_routes),
            "strict_frontier": len(strict_frontier),
        },
        "budgets": {
            "search_total": int(total_search_budget),
            "search_used": int(search_used),
            "evidence_total": int(total_evidence_budget),
            "evidence_used": int(evidence_used),
            "world_count": int(current_world_count),
        },
        "dccs_batches": dccs_batches,
        "selected_route_id": selected.id,
        "selected_route_signature": route_signature_map.get(selected.id, ""),
        "selected_candidate_ids": option_candidate_ids.get(selected.id, []),
        "selected_certificate": (
            selected_certificate.model_dump(mode="json") if selected_certificate is not None else None
        ),
        "voi": {
            "stop_reason": stop_reason,
            "action_trace": action_trace,
            "best_rejected_action": best_rejected_action_payload,
        },
        "artifact_pointers": {
            "dccs_candidates": "dccs_candidates.jsonl",
            "dccs_summary": "dccs_summary.json",
            "refined_routes": "refined_routes.jsonl",
            "strict_frontier": "strict_frontier.jsonl",
            "winner_summary": "winner_summary.json",
            "certificate_summary": "certificate_summary.json",
            "route_fragility_map": "route_fragility_map.json",
            "competitor_fragility_breakdown": "competitor_fragility_breakdown.json",
            "value_of_refresh": "value_of_refresh.json",
            "sampled_world_manifest": "sampled_world_manifest.json",
            "voi_action_trace": "voi_action_trace.json",
            "voi_stop_certificate": "voi_stop_certificate.json",
        },
    }

    return {
        "selected": selected,
        "candidates": display_candidates,
        "warnings": warnings,
        "candidate_fetches": int(candidate_fetches),
        "terrain_diag": terrain_diag,
        "candidate_diag": candidate_diag,
        "selected_certificate": selected_certificate,
        "voi_stop_summary": voi_stop_summary,
        "extra_json_artifacts": {
            "dccs_summary.json": dccs_summary,
            "winner_summary.json": winner_summary,
            "certificate_summary.json": certificate_summary,
            "route_fragility_map.json": route_fragility_map,
            "competitor_fragility_breakdown.json": competitor_fragility_breakdown,
            "value_of_refresh.json": value_of_refresh,
            "sampled_world_manifest.json": sampled_world_manifest,
            "voi_action_trace.json": {
                "pipeline_mode": pipeline_mode,
                "selected_route_id": selected.id,
                "actions": action_trace,
            },
            "voi_stop_certificate.json": voi_stop_certificate_payload,
            "final_route_trace.json": final_route_trace,
        },
        "extra_jsonl_artifacts": {
            "dccs_candidates.jsonl": dccs_rows,
            "refined_routes.jsonl": refined_route_rows,
            "strict_frontier.jsonl": strict_frontier_rows,
        },
        "extra_csv_artifacts": {
            "voi_action_scores.csv": (
                [
                    "iteration",
                    "action_id",
                    "kind",
                    "target",
                    "cost_search",
                    "cost_evidence",
                    "predicted_delta_certificate",
                    "predicted_delta_margin",
                    "predicted_delta_frontier",
                    "q_score",
                    "selected_route_id",
                    "selected_certificate",
                ],
                action_score_rows,
            )
        },
    }


@app.post("/route", response_model=RouteResponse)
async def compute_route(
    req: RouteRequest,
    response: Response,
    osrm: OSRMDep,
    _: UserAccessDep,
) -> RouteResponse:
    request_id = str(uuid.uuid4())
    t0 = time.perf_counter()
    has_error = False
    trace_status = "ok"
    trace_error: str | None = None
    trace_token = start_live_call_trace(
        request_id,
        endpoint="/route",
        expected_calls=_route_compute_expected_live_calls(),
    )
    response.headers["x-route-request-id"] = request_id
    requested_alternatives = max(1, int(req.max_alternatives))
    route_alternatives = max(
        1,
        min(requested_alternatives, int(settings.route_candidate_alternatives_max)),
    )
    effective_pipeline_mode = _resolve_pipeline_mode(req.pipeline_mode)
    actual_pipeline_mode = effective_pipeline_mode
    legacy_mode_warning: str | None = None
    if req.waypoints and effective_pipeline_mode != "legacy":
        actual_pipeline_mode = "legacy"
        legacy_mode_warning = (
            "VOI pipeline currently supports single-leg OD requests only; using legacy routing for waypoint requests."
        )
    run_seed = _resolve_pipeline_seed(req)
    log_event(
        "route_request_started",
        request_id=request_id,
        requested_max_alternatives=requested_alternatives,
        effective_max_alternatives=route_alternatives,
        waypoint_count=len(req.waypoints or []),
        pipeline_mode=actual_pipeline_mode,
        run_seed=int(run_seed),
    )
    try:
        warmup_detail = _routing_graph_warmup_failfast_detail()
        if warmup_detail is not None:
            _record_expected_calls_blocked(
                reason_code=str(warmup_detail.get("reason_code", "routing_graph_warming_up")),
                stage=str(warmup_detail.get("stage", "collecting_candidates")),
                detail=str(warmup_detail.get("stage_detail", "routing_graph_warming_up")),
            )
            raise HTTPException(status_code=503, detail=warmup_detail)
        timeout_s = max(1.0, float(settings.route_compute_single_attempt_timeout_s))
        extra_json_artifacts: dict[str, dict[str, Any] | list[Any]] | None = None
        extra_jsonl_artifacts: dict[str, list[dict[str, Any]]] | None = None
        extra_csv_artifacts: dict[str, tuple[list[str], list[dict[str, Any]]]] | None = None
        extra_text_artifacts: dict[str, str] | None = None
        selected_certificate: RouteCertificationSummary | None = None
        voi_stop_summary: VoiStopSummary | None = None
        try:
            if actual_pipeline_mode == "legacy":
                options, warnings, candidate_fetches, terrain_diag, candidate_diag = await asyncio.wait_for(
                    _collect_route_options_with_diagnostics(
                        osrm=osrm,
                        origin=req.origin,
                        destination=req.destination,
                        waypoints=req.waypoints,
                        max_alternatives=route_alternatives,
                        vehicle_type=req.vehicle_type,
                        scenario_mode=req.scenario_mode,
                        cost_toggles=req.cost_toggles,
                        terrain_profile=req.terrain_profile,
                        stochastic=req.stochastic,
                        emissions_context=req.emissions_context,
                        weather=req.weather,
                        incident_simulation=req.incident_simulation,
                        departure_time_utc=req.departure_time_utc,
                        pareto_method=req.pareto_method,
                        epsilon=req.epsilon,
                        optimization_mode=req.optimization_mode,
                        risk_aversion=req.risk_aversion,
                        utility_weights=(req.weights.time, req.weights.money, req.weights.co2),
                        option_prefix="route",
                    ),
                    timeout=timeout_s,
                )
            else:
                direct_result = await asyncio.wait_for(
                    _compute_direct_route_pipeline(
                        req=req,
                        osrm=osrm,
                        max_alternatives=route_alternatives,
                        pipeline_mode=actual_pipeline_mode,
                        run_seed=run_seed,
                    ),
                    timeout=timeout_s,
                )
                options = list(direct_result["candidates"])
                warnings = list(direct_result["warnings"])
                candidate_fetches = int(direct_result["candidate_fetches"])
                terrain_diag = direct_result["terrain_diag"]
                candidate_diag = direct_result["candidate_diag"]
                selected = direct_result["selected"]
                pareto_options = list(direct_result["candidates"])
                selected_certificate = direct_result.get("selected_certificate")
                voi_stop_summary = direct_result.get("voi_stop_summary")
                extra_json_artifacts = direct_result.get("extra_json_artifacts")
                extra_jsonl_artifacts = direct_result.get("extra_jsonl_artifacts")
                extra_csv_artifacts = direct_result.get("extra_csv_artifacts")
                extra_text_artifacts = direct_result.get("extra_text_artifacts")
        except asyncio.TimeoutError:
            _record_expected_calls_blocked(
                reason_code="route_compute_timeout",
                stage="collecting_candidates",
                detail="attempt_timeout_reached",
            )
            raise HTTPException(
                status_code=422,
                detail=_strict_error_detail(
                    reason_code="route_compute_timeout",
                    message="Route compute attempt exceeded timeout budget.",
                    warnings=[],
                    extra={
                        "stage": "collecting_candidates",
                        "stage_detail": "attempt_timeout_reached",
                        "timeout_s": timeout_s,
                    },
                ),
            ) from None

        if legacy_mode_warning is not None and legacy_mode_warning not in warnings:
            warnings.append(legacy_mode_warning)

        if not options:
            strict_detail = _strict_failure_detail_from_outcome(
                warnings=warnings,
                terrain_diag=terrain_diag,
                epsilon_requested=req.pareto_method == "epsilon_constraint" and req.epsilon is not None,
                terrain_message="Terrain DEM coverage is insufficient for strict routing policy.",
            )
            if strict_detail is not None:
                _record_expected_calls_blocked(
                    reason_code=str(strict_detail.get("reason_code", "strict_runtime_error")),
                    stage=str(strict_detail.get("stage", "finalizing_route")),
                    detail=str(strict_detail.get("message", "strict_runtime_error")),
                )
                raise HTTPException(
                    status_code=422,
                    detail=strict_detail,
                )
            raise HTTPException(
                status_code=422,
                detail=_strict_error_detail(
                    reason_code="no_route_candidates",
                    message="No route candidates could be computed.",
                    warnings=warnings,
                ),
            )

        if actual_pipeline_mode == "legacy":
            pareto_options = _finalize_pareto_options(
                options,
                max_alternatives=route_alternatives,
                pareto_method=req.pareto_method,
                epsilon=req.epsilon,
                optimization_mode=req.optimization_mode,
                risk_aversion=req.risk_aversion,
            )
            if not pareto_options:
                raise HTTPException(
                    status_code=422,
                    detail=_strict_error_detail(
                        reason_code="epsilon_infeasible",
                        message="No routes satisfy epsilon constraints for this request.",
                        warnings=warnings,
                    ),
                )

            selected = _pick_best_option(
                pareto_options,
                w_time=req.weights.time,
                w_money=req.weights.money,
                w_co2=req.weights.co2,
                optimization_mode=req.optimization_mode,
                risk_aversion=req.risk_aversion,
            )
            selected_certificate = selected.certification

        log_event(
            "route_request",
            request_id=request_id,
            pipeline_mode=actual_pipeline_mode,
            run_seed=int(run_seed),
            vehicle_type=req.vehicle_type,
            scenario_mode=req.scenario_mode.value,
            weights=req.weights.model_dump(),
            cost_toggles=req.cost_toggles.model_dump(),
            terrain_profile=req.terrain_profile,
            stochastic=req.stochastic.model_dump(mode="json"),
            weather=req.weather.model_dump(mode="json"),
            incident_simulation=req.incident_simulation.model_dump(mode="json"),
            optimization_mode=req.optimization_mode,
            risk_aversion=req.risk_aversion,
            emissions_context=req.emissions_context.model_dump(mode="json"),
            pareto_method=req.pareto_method,
            epsilon=req.epsilon.model_dump(mode="json") if req.epsilon is not None else None,
            departure_time_utc=(
                req.departure_time_utc.isoformat() if req.departure_time_utc is not None else None
            ),
            origin=req.origin.model_dump(),
            destination=req.destination.model_dump(),
            waypoint_count=len(req.waypoints),
            candidate_fetches=candidate_fetches,
            candidate_count=len(pareto_options),
            graph_explored_states=int(candidate_diag.graph_explored_states),
            graph_generated_paths=int(candidate_diag.graph_generated_paths),
            graph_emitted_paths=int(candidate_diag.graph_emitted_paths),
            graph_effective_max_hops=int(candidate_diag.graph_effective_max_hops),
            graph_effective_hops_floor=int(candidate_diag.graph_effective_hops_floor),
            graph_effective_state_budget_initial=int(candidate_diag.graph_effective_state_budget_initial),
            graph_effective_state_budget=int(candidate_diag.graph_effective_state_budget),
            graph_no_path_reason=str(candidate_diag.graph_no_path_reason or ""),
            precheck_reason_code=str(candidate_diag.precheck_reason_code or ""),
            precheck_elapsed_ms=float(candidate_diag.precheck_elapsed_ms),
            precheck_origin_node_id=str(candidate_diag.precheck_origin_node_id or ""),
            precheck_destination_node_id=str(candidate_diag.precheck_destination_node_id or ""),
            precheck_origin_nearest_m=float(candidate_diag.precheck_origin_nearest_m),
            precheck_destination_nearest_m=float(candidate_diag.precheck_destination_nearest_m),
            precheck_origin_selected_m=float(candidate_diag.precheck_origin_selected_m),
            precheck_destination_selected_m=float(candidate_diag.precheck_destination_selected_m),
            precheck_origin_candidate_count=int(candidate_diag.precheck_origin_candidate_count),
            precheck_destination_candidate_count=int(candidate_diag.precheck_destination_candidate_count),
            precheck_selected_component=int(candidate_diag.precheck_selected_component),
            precheck_selected_component_size=int(candidate_diag.precheck_selected_component_size),
            precheck_gate_action=str(candidate_diag.precheck_gate_action or ""),
            graph_retry_attempted=bool(candidate_diag.graph_retry_attempted),
            graph_retry_state_budget=int(candidate_diag.graph_retry_state_budget),
            graph_retry_outcome=str(candidate_diag.graph_retry_outcome or ""),
            graph_rescue_attempted=bool(candidate_diag.graph_rescue_attempted),
            graph_rescue_mode=str(candidate_diag.graph_rescue_mode or ""),
            graph_rescue_state_budget=int(candidate_diag.graph_rescue_state_budget),
            graph_rescue_outcome=str(candidate_diag.graph_rescue_outcome or ""),
            prefetch_total_sources=int(candidate_diag.prefetch_total_sources),
            prefetch_success_sources=int(candidate_diag.prefetch_success_sources),
            prefetch_failed_sources=int(candidate_diag.prefetch_failed_sources),
            prefetch_failed_keys=str(candidate_diag.prefetch_failed_keys or ""),
            prefetch_failed_details=str(candidate_diag.prefetch_failed_details or ""),
            prefetch_missing_expected_sources=str(candidate_diag.prefetch_missing_expected_sources or ""),
            prefetch_rows_json=str(candidate_diag.prefetch_rows_json or ""),
            scenario_gate_required_configured=int(candidate_diag.scenario_gate_required_configured),
            scenario_gate_required_effective=int(candidate_diag.scenario_gate_required_effective),
            scenario_gate_source_ok_count=int(candidate_diag.scenario_gate_source_ok_count),
            scenario_gate_waiver_applied=bool(candidate_diag.scenario_gate_waiver_applied),
            scenario_gate_waiver_reason=str(candidate_diag.scenario_gate_waiver_reason or ""),
            scenario_gate_road_hint=str(candidate_diag.scenario_gate_road_hint or ""),
            terrain_fail_closed_count=terrain_diag.fail_closed_count,
            terrain_unsupported_region_count=terrain_diag.unsupported_region_count,
            terrain_asset_unavailable_count=terrain_diag.asset_unavailable_count,
            terrain_dem_version=terrain_diag.dem_version,
            terrain_coverage_min_observed=round(terrain_diag.coverage_min_observed, 6),
            warning_count=len(warnings),
            selected_id=selected.id,
            selected_metrics=selected.metrics.model_dump(),
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
        route_run = _write_route_run_bundle(
            req=req,
            selected=selected,
            candidates=pareto_options,
            warnings=warnings,
            candidate_diag=candidate_diag,
            request_id=request_id,
            pipeline_mode=actual_pipeline_mode,
            run_seed=int(run_seed),
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
            selected_certificate=selected_certificate,
            voi_stop_summary=voi_stop_summary,
            extra_json_artifacts=extra_json_artifacts,
            extra_jsonl_artifacts=extra_jsonl_artifacts,
            extra_csv_artifacts=extra_csv_artifacts,
            extra_text_artifacts=extra_text_artifacts,
        )

        return RouteResponse(
            selected=selected,
            candidates=pareto_options,
            run_id=str(route_run["run_id"]),
            pipeline_mode=actual_pipeline_mode,  # type: ignore[arg-type]
            manifest_endpoint=str(route_run["manifest_endpoint"]),
            artifacts_endpoint=str(route_run["artifacts_endpoint"]),
            provenance_endpoint=str(route_run["provenance_endpoint"]),
            selected_certificate=selected_certificate,
            voi_stop_summary=voi_stop_summary,
        )
    except HTTPException as e:
        has_error = True
        detail_obj = e.detail if isinstance(e.detail, dict) else None
        reason_code = normalize_reason_code(
            str((detail_obj or {}).get("reason_code", "http_exception"))
        )
        trace_status = "error"
        trace_error = reason_code
        headers = dict(getattr(e, "headers", {}) or {})
        headers["x-route-request-id"] = request_id
        e.headers = headers
        detail_message = str((detail_obj or {}).get("message", str(e.detail))).strip()
        log_event(
            "route_request_failed",
            request_id=request_id,
            reason="http_exception",
            status_code=int(e.status_code),
            reason_code=reason_code,
            detail_message=detail_message,
            detail=e.detail,
            stage=(detail_obj or {}).get("stage"),
            stage_detail=(detail_obj or {}).get("stage_detail"),
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
        raise
    except Exception as exc:
        has_error = True
        trace_status = "error"
        trace_error = str(type(exc).__name__).strip() or "unexpected_exception"
        log_event(
            "route_request_failed",
            request_id=request_id,
            reason="unexpected_exception",
            error_type=type(exc).__name__,
            error_message=str(exc).strip() or type(exc).__name__,
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
        raise
    finally:
        finish_live_call_trace(
            request_id=request_id,
            endpoint="/route",
            status=trace_status,
            error_reason=trace_error,
        )
        reset_live_call_trace(trace_token)
        _record_endpoint_metric("route", t0, error=has_error)


@app.post("/route/baseline", response_model=RouteBaselineResponse)
async def compute_route_baseline(
    req: RouteRequest,
    response: Response,
    osrm: OSRMDep,
    _: UserAccessDep,
) -> RouteBaselineResponse:
    request_id = str(uuid.uuid4())
    t0 = time.perf_counter()
    response.headers["x-route-request-id"] = request_id
    try:
        via = [(float(waypoint.lat), float(waypoint.lon)) for waypoint in (req.waypoints or [])]
        routes = await osrm.fetch_routes(
            origin_lat=req.origin.lat,
            origin_lon=req.origin.lon,
            dest_lat=req.destination.lat,
            dest_lon=req.destination.lon,
            alternatives=False,
            via=via or None,
        )
        if not routes:
            raise OSRMError("OSRM returned no routes for baseline request.")
        vehicle = resolve_vehicle_profile(req.vehicle_type)
        baseline = _build_osrm_baseline_option(
            routes[0],
            option_id="baseline_osrm",
            vehicle=vehicle,
            scenario_mode=req.scenario_mode,
            cost_toggles=req.cost_toggles,
            emissions_context=req.emissions_context,
            weather=req.weather,
            departure_time_utc=req.departure_time_utc,
        )
        compute_ms = round((time.perf_counter() - t0) * 1000, 2)
        notes = [
            "OSRM quick baseline route; strict live-source enrichments are intentionally bypassed.",
            "Use this as a fast comparator against strict smart-route output.",
            (
                "Configured baseline realism multipliers applied: "
                f"duration x{float(settings.route_baseline_duration_multiplier):.2f}, "
                f"distance x{float(settings.route_baseline_distance_multiplier):.2f}."
            ),
        ]
        log_event(
            "route_baseline_request",
            request_id=request_id,
            vehicle_type=req.vehicle_type,
            waypoint_count=len(req.waypoints),
            distance_km=baseline.metrics.distance_km,
            duration_s=baseline.metrics.duration_s,
            monetary_cost=baseline.metrics.monetary_cost,
            emissions_kg=baseline.metrics.emissions_kg,
            compute_ms=compute_ms,
        )
        return RouteBaselineResponse(
            baseline=baseline,
            method="osrm_quick_baseline",
            compute_ms=compute_ms,
            notes=notes,
        )
    except OSRMError as exc:
        message = str(exc).strip() or "OSRM baseline route is unavailable."
        log_event(
            "route_baseline_request_failed",
            request_id=request_id,
            reason_code="baseline_route_unavailable",
            detail_message=message,
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
        raise HTTPException(
            status_code=502,
            detail=_strict_error_detail(
                reason_code="baseline_route_unavailable",
                message=f"OSRM baseline route is unavailable. (baseline_route_unavailable) cause={message}",
                warnings=[],
            ),
            headers={"x-route-request-id": request_id},
        ) from exc
    except HTTPException as e:
        headers = dict(getattr(e, "headers", {}) or {})
        headers["x-route-request-id"] = request_id
        e.headers = headers
        raise
    except Exception as exc:
        message = str(exc).strip() or "Unknown baseline route failure."
        log_event(
            "route_baseline_request_failed",
            request_id=request_id,
            reason_code="baseline_route_unavailable",
            detail_message=message,
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
        raise HTTPException(
            status_code=500,
            detail=_strict_error_detail(
                reason_code="baseline_route_unavailable",
                message=f"Baseline route compute failed unexpectedly: {message}",
                warnings=[],
            ),
            headers={"x-route-request-id": request_id},
        ) from exc


@app.post("/route/baseline/ors", response_model=RouteBaselineResponse)
async def compute_route_ors_baseline(
    req: RouteRequest,
    response: Response,
    osrm: OSRMDep,
    _: UserAccessDep,
) -> RouteBaselineResponse:
    request_id = str(uuid.uuid4())
    t0 = time.perf_counter()
    response.headers["x-route-request-id"] = request_id
    try:
        ors_duration_multiplier = max(1.0, float(settings.route_ors_baseline_duration_multiplier))
        ors_distance_multiplier = max(1.0, float(settings.route_ors_baseline_distance_multiplier))
        seed_route = await _fetch_ors_reference_route_seed(req=req)
        vehicle = resolve_vehicle_profile(req.vehicle_type)
        baseline = _build_osrm_baseline_option(
            seed_route,
            option_id="baseline_ors",
            vehicle=vehicle,
            scenario_mode=req.scenario_mode,
            cost_toggles=req.cost_toggles,
            emissions_context=req.emissions_context,
            weather=req.weather,
            departure_time_utc=req.departure_time_utc,
            baseline_duration_multiplier_override=ors_duration_multiplier,
            baseline_distance_multiplier_override=ors_distance_multiplier,
        )
        compute_ms = round((time.perf_counter() - t0) * 1000, 2)
        notes = [
            "OpenRouteService reference route via Directions API.",
            "This comparator bypasses strict live-source graph search and uses provider baseline geometry.",
            (
                "Configured baseline realism multipliers applied: "
                f"duration x{ors_duration_multiplier:.2f}, "
                f"distance x{ors_distance_multiplier:.2f}."
            ),
        ]
        log_event(
            "route_ors_baseline_request",
            request_id=request_id,
            vehicle_type=req.vehicle_type,
            waypoint_count=len(req.waypoints),
            distance_km=baseline.metrics.distance_km,
            duration_s=baseline.metrics.duration_s,
            monetary_cost=baseline.metrics.monetary_cost,
            emissions_kg=baseline.metrics.emissions_kg,
            compute_ms=compute_ms,
        )
        return RouteBaselineResponse(
            baseline=baseline,
            method="ors_reference",
            compute_ms=compute_ms,
            notes=notes,
        )
    except RuntimeError as exc:
        message = str(exc).strip() or "OpenRouteService baseline route is unavailable."
        reason_code = (
            "baseline_provider_unconfigured"
            if "ORS_DIRECTIONS_API_KEY" in message or "ORS_DIRECTIONS_URL_TEMPLATE" in message
            else "baseline_route_unavailable"
        )
        if (
            reason_code == "baseline_provider_unconfigured"
            and bool(settings.route_ors_baseline_allow_proxy_fallback)
        ):
            via = [(float(waypoint.lat), float(waypoint.lon)) for waypoint in (req.waypoints or [])]
            routes = await osrm.fetch_routes(
                origin_lat=req.origin.lat,
                origin_lon=req.origin.lon,
                dest_lat=req.destination.lat,
                dest_lon=req.destination.lon,
                alternatives=False,
                via=via or None,
            )
            if not routes:
                raise HTTPException(
                    status_code=502,
                        detail=_strict_error_detail(
                            reason_code="baseline_route_unavailable",
                            message=(
                                "OpenRouteService baseline provider is not configured and OSRM proxy fallback returned no route. "
                                "(baseline_route_unavailable)"
                            ),
                            warnings=[],
                    ),
                    headers={"x-route-request-id": request_id},
                )
            ors_duration_multiplier = max(1.0, float(settings.route_ors_baseline_duration_multiplier))
            ors_distance_multiplier = max(1.0, float(settings.route_ors_baseline_distance_multiplier))
            vehicle = resolve_vehicle_profile(req.vehicle_type)
            baseline = _build_osrm_baseline_option(
                routes[0],
                option_id="baseline_ors_proxy",
                vehicle=vehicle,
                scenario_mode=req.scenario_mode,
                cost_toggles=req.cost_toggles,
                emissions_context=req.emissions_context,
                weather=req.weather,
                departure_time_utc=req.departure_time_utc,
                baseline_duration_multiplier_override=ors_duration_multiplier,
                baseline_distance_multiplier_override=ors_distance_multiplier,
            )
            compute_ms = round((time.perf_counter() - t0) * 1000, 2)
            notes = [
                "OpenRouteService API key is not configured; using an OSRM proxy baseline.",
                "Proxy applies configured OpenRouteService realism multipliers for duration and distance.",
                (
                    "Proxy multipliers applied: "
                    f"duration x{ors_duration_multiplier:.2f}, distance x{ors_distance_multiplier:.2f}."
                ),
            ]
            log_event(
                "route_ors_baseline_request",
                request_id=request_id,
                vehicle_type=req.vehicle_type,
                waypoint_count=len(req.waypoints),
                distance_km=baseline.metrics.distance_km,
                duration_s=baseline.metrics.duration_s,
                monetary_cost=baseline.metrics.monetary_cost,
                emissions_kg=baseline.metrics.emissions_kg,
                compute_ms=compute_ms,
                provider_mode="osrm_proxy",
            )
            return RouteBaselineResponse(
                baseline=baseline,
                method="ors_proxy_baseline",
                compute_ms=compute_ms,
                notes=notes,
            )
        status_code = 503 if reason_code == "baseline_provider_unconfigured" else 502
        log_event(
            "route_ors_baseline_request_failed",
            request_id=request_id,
            reason_code=reason_code,
            detail_message=message,
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
        raise HTTPException(
            status_code=status_code,
            detail=_strict_error_detail(
                reason_code=reason_code,
                message=f"OpenRouteService baseline route is unavailable. ({reason_code}) cause={message}",
                warnings=[],
            ),
            headers={"x-route-request-id": request_id},
        ) from exc
    except HTTPException as e:
        headers = dict(getattr(e, "headers", {}) or {})
        headers["x-route-request-id"] = request_id
        e.headers = headers
        raise
    except Exception as exc:
        message = str(exc).strip() or "Unknown OpenRouteService baseline route failure."
        log_event(
            "route_ors_baseline_request_failed",
            request_id=request_id,
            reason_code="baseline_route_unavailable",
            detail_message=message,
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
        raise HTTPException(
            status_code=500,
            detail=_strict_error_detail(
                reason_code="baseline_route_unavailable",
                message=f"OpenRouteService baseline route compute failed unexpectedly: {message}",
                warnings=[],
            ),
            headers={"x-route-request-id": request_id},
        ) from exc


@app.post("/departure/optimize", response_model=DepartureOptimizeResponse)
async def optimize_departure_time(
    req: DepartureOptimizeRequest,
    osrm: OSRMDep,
    _: UserAccessDep,
) -> DepartureOptimizeResponse:
    t0 = time.perf_counter()
    has_error = False

    try:
        start_utc = req.window_start_utc
        end_utc = req.window_end_utc
        if start_utc.tzinfo is None:
            start_utc = start_utc.replace(tzinfo=UTC)
        else:
            start_utc = start_utc.astimezone(UTC)
        if end_utc.tzinfo is None:
            end_utc = end_utc.replace(tzinfo=UTC)
        else:
            end_utc = end_utc.astimezone(UTC)

        if end_utc <= start_utc:
            raise HTTPException(
                status_code=422,
                detail=_strict_error_detail(
                    reason_code="epsilon_infeasible",
                    message="window_end_utc must be after window_start_utc",
                    warnings=[],
                ),
            )

        earliest_arrival_utc: datetime | None = None
        latest_arrival_utc: datetime | None = None
        if req.time_window is not None:
            earliest_arrival_utc = req.time_window.earliest_arrival_utc
            latest_arrival_utc = req.time_window.latest_arrival_utc
            if earliest_arrival_utc is not None:
                if earliest_arrival_utc.tzinfo is None:
                    earliest_arrival_utc = earliest_arrival_utc.replace(tzinfo=UTC)
                else:
                    earliest_arrival_utc = earliest_arrival_utc.astimezone(UTC)
            if latest_arrival_utc is not None:
                if latest_arrival_utc.tzinfo is None:
                    latest_arrival_utc = latest_arrival_utc.replace(tzinfo=UTC)
                else:
                    latest_arrival_utc = latest_arrival_utc.astimezone(UTC)
            if (
                earliest_arrival_utc is not None
                and latest_arrival_utc is not None
                and latest_arrival_utc < earliest_arrival_utc
            ):
                raise HTTPException(
                    status_code=422,
                    detail="time_window.latest_arrival_utc must be >= earliest_arrival_utc",
                )

        departure_times: list[datetime] = []
        cursor = start_utc
        step = timedelta(minutes=req.step_minutes)
        while cursor <= end_utc:
            departure_times.append(cursor)
            cursor = cursor + step
        if departure_times[-1] != end_utc:
            departure_times.append(end_utc)

        selected_rows: list[dict[str, Any]] = []
        latest_warnings: list[str] = []
        strict_failure_detail: dict[str, Any] | None = None
        for departure_time in departure_times:
            options, warnings, _candidate_fetches, terrain_diag, _candidate_diag = await _collect_route_options_with_diagnostics(
                osrm=osrm,
                origin=req.origin,
                destination=req.destination,
                waypoints=req.waypoints,
                max_alternatives=req.max_alternatives,
                vehicle_type=req.vehicle_type,
                scenario_mode=req.scenario_mode,
                cost_toggles=req.cost_toggles,
                terrain_profile=req.terrain_profile,
                stochastic=req.stochastic,
                emissions_context=req.emissions_context,
                weather=req.weather,
                incident_simulation=req.incident_simulation,
                departure_time_utc=departure_time,
                pareto_method=req.pareto_method,
                epsilon=req.epsilon,
                optimization_mode=req.optimization_mode,
                risk_aversion=req.risk_aversion,
                utility_weights=(req.weights.time, req.weights.money, req.weights.co2),
                option_prefix=f"departure_{departure_time.strftime('%H%M')}",
            )
            latest_warnings = warnings
            if not options:
                detail = _strict_failure_detail_from_outcome(
                    warnings=warnings,
                    terrain_diag=terrain_diag,
                    epsilon_requested=req.pareto_method == "epsilon_constraint" and req.epsilon is not None,
                    terrain_message="All candidates were removed by terrain DEM strict coverage policy.",
                )
                if detail is not None and strict_failure_detail is None:
                    strict_failure_detail = detail
                continue

            pareto = _finalize_pareto_options(
                options,
                max_alternatives=req.max_alternatives,
                pareto_method=req.pareto_method,
                epsilon=req.epsilon,
                optimization_mode=req.optimization_mode,
                risk_aversion=req.risk_aversion,
            )
            if not pareto:
                if req.pareto_method == "epsilon_constraint" and req.epsilon is not None:
                    strict_failure_detail = _strict_error_detail(
                        reason_code="epsilon_infeasible",
                        message="No routes satisfy epsilon constraints for this request.",
                        warnings=warnings,
                    )
                continue
            selected = _pick_best_option(
                pareto,
                w_time=req.weights.time,
                w_money=req.weights.money,
                w_co2=req.weights.co2,
                optimization_mode=req.optimization_mode,
                risk_aversion=req.risk_aversion,
            )
            arrival_utc = departure_time + timedelta(seconds=selected.metrics.duration_s)
            if earliest_arrival_utc is not None and arrival_utc < earliest_arrival_utc:
                continue
            if latest_arrival_utc is not None and arrival_utc > latest_arrival_utc:
                continue
            selected_rows.append(
                {
                    "departure_time_utc": departure_time,
                    "selected": selected,
                    "arrival_time_utc": arrival_utc,
                    "warning_count": len(warnings),
                }
            )

        if not selected_rows:
            if strict_failure_detail is not None:
                raise HTTPException(status_code=422, detail=strict_failure_detail)
            if req.time_window is not None:
                raise HTTPException(
                    status_code=422,
                    detail=_strict_error_detail(
                        reason_code="epsilon_infeasible",
                        message="no feasible departures for provided time window",
                        warnings=latest_warnings,
                    ),
                )
            raise HTTPException(
                status_code=422,
                detail=_strict_error_detail(
                    reason_code="no_route_candidates",
                    message="No departure candidates could be computed.",
                    warnings=[],
                ),
            )

        candidates: list[DepartureOptimizeCandidate] = []
        if req.optimization_mode == "robust":
            for row in selected_rows:
                selected = row["selected"]
                score = _option_joint_robust_utility(selected, risk_aversion=req.risk_aversion)
                departure_time = row["departure_time_utc"]
                candidates.append(
                    DepartureOptimizeCandidate(
                        departure_time_utc=departure_time.isoformat().replace("+00:00", "Z"),
                        selected=selected,
                        score=round(score, 6),
                        warning_count=int(row["warning_count"]),
                    )
                )
        else:
            durations = [
                _option_objective_value(
                    row["selected"],
                    "duration",
                    optimization_mode=req.optimization_mode,
                    risk_aversion=req.risk_aversion,
                )
                for row in selected_rows
            ]
            costs = [
                _option_objective_value(
                    row["selected"],
                    "money",
                    optimization_mode=req.optimization_mode,
                    risk_aversion=req.risk_aversion,
                )
                for row in selected_rows
            ]
            emissions = [
                _option_objective_value(
                    row["selected"],
                    "co2",
                    optimization_mode=req.optimization_mode,
                    risk_aversion=req.risk_aversion,
                )
                for row in selected_rows
            ]
            d_min, d_max = min(durations), max(durations)
            c_min, c_max = min(costs), max(costs)
            e_min, e_max = min(emissions), max(emissions)
            wt, wm, we = normalise_weights(req.weights.time, req.weights.money, req.weights.co2)

            def _norm(v: float, mn: float, mx: float) -> float:
                return 0.0 if mx <= mn else (v - mn) / (mx - mn)

            for row in selected_rows:
                selected = row["selected"]
                score = (
                    wt
                    * _norm(
                        _option_objective_value(
                            selected,
                            "duration",
                            optimization_mode=req.optimization_mode,
                            risk_aversion=req.risk_aversion,
                        ),
                        d_min,
                        d_max,
                    )
                    + wm
                    * _norm(
                        _option_objective_value(
                            selected,
                            "money",
                            optimization_mode=req.optimization_mode,
                            risk_aversion=req.risk_aversion,
                        ),
                        c_min,
                        c_max,
                    )
                    + we
                    * _norm(
                        _option_objective_value(
                            selected,
                            "co2",
                            optimization_mode=req.optimization_mode,
                            risk_aversion=req.risk_aversion,
                        ),
                        e_min,
                        e_max,
                    )
                )
                departure_time = row["departure_time_utc"]
                candidates.append(
                    DepartureOptimizeCandidate(
                        departure_time_utc=departure_time.isoformat().replace("+00:00", "Z"),
                        selected=selected,
                        score=round(score, 6),
                        warning_count=int(row["warning_count"]),
                    )
                )

        tie_mode: OptimizationMode = "expected_value" if req.optimization_mode == "robust" else req.optimization_mode
        candidates.sort(
            key=lambda item: (
                item.score,
                _option_objective_value(
                    item.selected,
                    "duration",
                    optimization_mode=tie_mode,
                    risk_aversion=req.risk_aversion,
                ),
                item.selected.id,
                item.departure_time_utc,
            )
        )
        best = candidates[0] if candidates else None

        log_event(
            "departure_optimize_request",
            vehicle_type=req.vehicle_type,
            scenario_mode=req.scenario_mode.value,
            weather=req.weather.model_dump(mode="json"),
            incident_simulation=req.incident_simulation.model_dump(mode="json"),
            optimization_mode=req.optimization_mode,
            risk_aversion=req.risk_aversion,
            emissions_context=req.emissions_context.model_dump(mode="json"),
            pair={"origin": req.origin.model_dump(), "destination": req.destination.model_dump()},
            step_minutes=req.step_minutes,
            window_start_utc=start_utc.isoformat().replace("+00:00", "Z"),
            window_end_utc=end_utc.isoformat().replace("+00:00", "Z"),
            time_window=(
                {
                    "earliest_arrival_utc": (
                        earliest_arrival_utc.isoformat().replace("+00:00", "Z")
                        if earliest_arrival_utc is not None
                        else None
                    ),
                    "latest_arrival_utc": (
                        latest_arrival_utc.isoformat().replace("+00:00", "Z")
                        if latest_arrival_utc is not None
                        else None
                    ),
                }
                if req.time_window is not None
                else None
            ),
            evaluated_count=len(candidates),
            best_departure_time_utc=best.departure_time_utc if best else None,
            duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
        )

        return DepartureOptimizeResponse(best=best, candidates=candidates, evaluated_count=len(candidates))
    except HTTPException:
        has_error = True
        raise
    except Exception:
        has_error = True
        raise
    finally:
        _record_endpoint_metric("departure_optimize_post", t0, error=has_error)


def _to_latlng(stop: DutyChainStop) -> LatLng:
    return LatLng(lat=stop.lat, lon=stop.lon)


def _next_departure(
    departure_time_utc: datetime | None,
    selected: RouteOption | None,
) -> datetime | None:
    if departure_time_utc is None or selected is None:
        return departure_time_utc
    cursor = departure_time_utc
    if cursor.tzinfo is None:
        cursor = cursor.replace(tzinfo=UTC)
    else:
        cursor = cursor.astimezone(UTC)
    return cursor + timedelta(seconds=float(selected.metrics.duration_s))


@app.post("/duty/chain", response_model=DutyChainResponse)
async def run_duty_chain(
    req: DutyChainRequest,
    osrm: OSRMDep,
    _: UserAccessDep,
) -> DutyChainResponse:
    t0 = time.perf_counter()
    has_error = False

    try:
        legs: list[DutyChainLegResult] = []
        total_distance_km = 0.0
        total_duration_s = 0.0
        total_monetary_cost = 0.0
        total_emissions_kg = 0.0
        total_energy_kwh = 0.0
        total_weather_delay_s = 0.0
        total_incident_delay_s = 0.0
        has_energy = False

        departure_cursor = req.departure_time_utc
        for idx in range(len(req.stops) - 1):
            origin_stop = req.stops[idx]
            destination_stop = req.stops[idx + 1]
            origin = _to_latlng(origin_stop)
            destination = _to_latlng(destination_stop)

            options, warnings, _candidate_fetches, terrain_diag, _candidate_diag = await _collect_route_options_with_diagnostics(
                osrm=osrm,
                origin=origin,
                destination=destination,
                waypoints=[],
                max_alternatives=req.max_alternatives,
                vehicle_type=req.vehicle_type,
                scenario_mode=req.scenario_mode,
                cost_toggles=req.cost_toggles,
                terrain_profile=req.terrain_profile,
                stochastic=req.stochastic,
                emissions_context=req.emissions_context,
                weather=req.weather,
                incident_simulation=req.incident_simulation,
                departure_time_utc=departure_cursor,
                pareto_method=req.pareto_method,
                epsilon=req.epsilon,
                optimization_mode=req.optimization_mode,
                risk_aversion=req.risk_aversion,
                utility_weights=(req.weights.time, req.weights.money, req.weights.co2),
                option_prefix=f"duty_leg_{idx}",
            )

            if not options:
                strict_detail = _strict_failure_detail_from_outcome(
                    warnings=warnings,
                    terrain_diag=terrain_diag,
                    epsilon_requested=req.pareto_method == "epsilon_constraint" and req.epsilon is not None,
                    terrain_message="All candidates were removed by terrain DEM strict coverage policy.",
                )
                legs.append(
                    DutyChainLegResult(
                        leg_index=idx,
                        origin=origin_stop,
                        destination=destination_stop,
                        selected=None,
                        candidates=[],
                        warning_count=len(warnings),
                        error=(
                            _strict_error_text(strict_detail)
                            if strict_detail is not None
                            else "reason_code:no_route_candidates; message:No route candidates could be computed."
                        ),
                    )
                )
                continue

            pareto = _finalize_pareto_options(
                options,
                max_alternatives=req.max_alternatives,
                pareto_method=req.pareto_method,
                epsilon=req.epsilon,
                optimization_mode=req.optimization_mode,
                risk_aversion=req.risk_aversion,
            )
            if not pareto:
                strict_detail = _strict_error_detail(
                    reason_code="epsilon_infeasible",
                    message="No routes satisfy epsilon constraints for this request.",
                    warnings=warnings,
                )
                legs.append(
                    DutyChainLegResult(
                        leg_index=idx,
                        origin=origin_stop,
                        destination=destination_stop,
                        selected=None,
                        candidates=[],
                        warning_count=len(warnings),
                        error=_strict_error_text(strict_detail),
                    )
                )
                continue
            selected = _pick_best_option(
                pareto,
                w_time=req.weights.time,
                w_money=req.weights.money,
                w_co2=req.weights.co2,
                optimization_mode=req.optimization_mode,
                risk_aversion=req.risk_aversion,
            )
            legs.append(
                DutyChainLegResult(
                    leg_index=idx,
                    origin=origin_stop,
                    destination=destination_stop,
                    selected=selected,
                    candidates=pareto,
                    warning_count=len(warnings),
                )
            )

            total_distance_km += float(selected.metrics.distance_km)
            total_duration_s += float(selected.metrics.duration_s)
            total_monetary_cost += float(selected.metrics.monetary_cost)
            total_emissions_kg += float(selected.metrics.emissions_kg)
            total_weather_delay_s += float(selected.metrics.weather_delay_s)
            total_incident_delay_s += float(selected.metrics.incident_delay_s)
            if selected.metrics.energy_kwh is not None:
                total_energy_kwh += float(selected.metrics.energy_kwh)
                has_energy = True

            departure_cursor = _next_departure(departure_cursor, selected)

        successful_leg_count = sum(1 for leg in legs if leg.selected is not None)
        avg_speed_kmh = total_distance_km / (total_duration_s / 3600.0) if total_duration_s > 0 else 0.0
        total_metrics = RouteMetrics(
            distance_km=round(total_distance_km, 3),
            duration_s=round(total_duration_s, 2),
            monetary_cost=round(total_monetary_cost, 2),
            emissions_kg=round(total_emissions_kg, 3),
            avg_speed_kmh=round(avg_speed_kmh, 2),
            energy_kwh=round(total_energy_kwh, 3) if has_energy else None,
            weather_delay_s=round(total_weather_delay_s, 2),
            incident_delay_s=round(total_incident_delay_s, 2),
        )

        log_event(
            "duty_chain_request",
            stop_count=len(req.stops),
            leg_count=len(legs),
            successful_leg_count=successful_leg_count,
            vehicle_type=req.vehicle_type,
            scenario_mode=req.scenario_mode.value,
            weather=req.weather.model_dump(mode="json"),
            incident_simulation=req.incident_simulation.model_dump(mode="json"),
            emissions_context=req.emissions_context.model_dump(mode="json"),
            optimization_mode=req.optimization_mode,
            risk_aversion=req.risk_aversion,
            total_metrics=total_metrics.model_dump(mode="json"),
            duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
        )

        return DutyChainResponse(
            legs=legs,
            total_metrics=total_metrics,
            leg_count=len(legs),
            successful_leg_count=successful_leg_count,
        )
    except HTTPException:
        has_error = True
        raise
    except Exception:
        has_error = True
        raise
    finally:
        _record_endpoint_metric("duty_chain_post", t0, error=has_error)


def _scenario_metric_deltas(
    baseline: RouteOption | None,
    current: RouteOption | None,
    *,
    baseline_error: str | None = None,
    current_error: str | None = None,
    baseline_error_detail: dict[str, Any] | None = None,
    current_error_detail: dict[str, Any] | None = None,
) -> dict[str, float | str | None]:
    def _reason_and_source(
        detail: dict[str, Any] | None,
        *,
        label: str,
    ) -> tuple[str | None, str | None]:
        if isinstance(detail, dict):
            reason_raw = str(detail.get("reason_code", "")).strip()
            if reason_raw:
                # Keep historical compare provenance labels stable while using structured error payloads.
                return normalize_reason_code(reason_raw, default="metric_unavailable"), f"{label}_error"
        return None, None

    baseline_reason, baseline_reason_source = _reason_and_source(
        baseline_error_detail,
        label="baseline",
    )
    current_reason, current_reason_source = _reason_and_source(
        current_error_detail,
        label="current",
    )
    _ = baseline_error
    _ = current_error
    missing_reason = current_reason or baseline_reason or "metric_unavailable"
    if baseline is None or current is None:
        missing_source = "current" if current is None else "baseline"
        return {
            "duration_s_delta": None,
            "monetary_cost_delta": None,
            "emissions_kg_delta": None,
            "duration_s_status": "missing",
            "monetary_cost_status": "missing",
            "emissions_kg_status": "missing",
            "duration_s_reason_code": missing_reason,
            "monetary_cost_reason_code": missing_reason,
            "emissions_kg_reason_code": missing_reason,
            "duration_s_missing_source": missing_source,
            "monetary_cost_missing_source": missing_source,
            "emissions_kg_missing_source": missing_source,
            "duration_s_reason_source": (
                current_reason_source
                if current_reason
                else baseline_reason_source
                if baseline_reason
                else "structured_missing"
            ),
            "monetary_cost_reason_source": (
                current_reason_source
                if current_reason
                else baseline_reason_source
                if baseline_reason
                else "structured_missing"
            ),
            "emissions_kg_reason_source": (
                current_reason_source
                if current_reason
                else baseline_reason_source
                if baseline_reason
                else "structured_missing"
            ),
            "missing_source": missing_source,
        }

    def _metric_delta(metric_name: str) -> tuple[float | None, str, str | None, str | None, str | None]:
        baseline_value: float | None = None
        current_value: float | None = None
        baseline_ok = True
        current_ok = True
        try:
            baseline_value = float(getattr(baseline.metrics, metric_name))
        except Exception:
            baseline_ok = False
        try:
            current_value = float(getattr(current.metrics, metric_name))
        except Exception:
            current_ok = False
        baseline_missing = (not baseline_ok) or (baseline_value is None)
        current_missing = (not current_ok) or (current_value is None)
        if baseline_missing or current_missing:
            missing_source = "baseline" if baseline_missing else "current"
            reason_code = (baseline_reason if baseline_missing else current_reason) or "metric_unavailable"
            reason_source = (
                baseline_reason_source
                if (baseline_missing and baseline_reason)
                else current_reason_source
                if (current_missing and current_reason)
                else "structured_missing"
            )
            return None, "missing", reason_code, missing_source, reason_source
        if baseline_value is None or current_value is None:
            return None, "missing", "metric_unavailable", "unknown", "structured_missing"
        delta = round(current_value - baseline_value, 3)
        return delta, "ok", None, None, None

    duration_delta, duration_status, duration_reason, duration_missing_source, duration_reason_source = _metric_delta("duration_s")
    monetary_delta, monetary_status, monetary_reason, monetary_missing_source, monetary_reason_source = _metric_delta("monetary_cost")
    emissions_delta, emissions_status, emissions_reason, emissions_missing_source, emissions_reason_source = _metric_delta("emissions_kg")
    return {
        "duration_s_delta": duration_delta,
        "monetary_cost_delta": monetary_delta,
        "emissions_kg_delta": emissions_delta,
        "duration_s_status": duration_status,
        "monetary_cost_status": monetary_status,
        "emissions_kg_status": emissions_status,
        "duration_s_reason_code": duration_reason,
        "monetary_cost_reason_code": monetary_reason,
        "emissions_kg_reason_code": emissions_reason,
        "duration_s_missing_source": duration_missing_source,
        "monetary_cost_missing_source": monetary_missing_source,
        "emissions_kg_missing_source": emissions_missing_source,
        "duration_s_reason_source": duration_reason_source,
        "monetary_cost_reason_source": monetary_reason_source,
        "emissions_kg_reason_source": emissions_reason_source,
        "missing_source": None,
    }


async def _run_scenario_compare(
    req: ScenarioCompareRequest,
    osrm: OSRMClient,
) -> tuple[list[ScenarioCompareResult], dict[str, ScenarioCompareDelta]]:
    scenario_modes = [
        ScenarioMode.NO_SHARING,
        ScenarioMode.PARTIAL_SHARING,
        ScenarioMode.FULL_SHARING,
    ]
    results: list[ScenarioCompareResult] = []
    mode_error_details: dict[str, dict[str, Any] | None] = {}

    for scenario_mode in scenario_modes:
        options, warnings, _candidate_fetches, terrain_diag, _candidate_diag = await _collect_route_options_with_diagnostics(
            osrm=osrm,
            origin=req.origin,
            destination=req.destination,
            waypoints=req.waypoints,
            max_alternatives=req.max_alternatives,
            vehicle_type=req.vehicle_type,
            scenario_mode=scenario_mode,
            cost_toggles=req.cost_toggles,
            terrain_profile=req.terrain_profile,
            stochastic=req.stochastic,
            emissions_context=req.emissions_context,
            weather=req.weather,
            incident_simulation=req.incident_simulation,
            departure_time_utc=req.departure_time_utc,
            pareto_method=req.pareto_method,
            epsilon=req.epsilon,
            optimization_mode=req.optimization_mode,
            risk_aversion=req.risk_aversion,
            utility_weights=(req.weights.time, req.weights.money, req.weights.co2),
            option_prefix=f"scenario_{scenario_mode.value}",
        )

        if not options:
            strict_detail = _strict_failure_detail_from_outcome(
                warnings=warnings,
                terrain_diag=terrain_diag,
                epsilon_requested=req.pareto_method == "epsilon_constraint" and req.epsilon is not None,
                terrain_message="All candidates were removed by terrain DEM strict coverage policy.",
            )
            mode_error_details[scenario_mode.value] = (
                strict_detail
                if isinstance(strict_detail, dict)
                else {
                    "reason_code": "no_route_candidates",
                    "message": "No route candidates could be computed.",
                    "warnings": warnings,
                }
            )
            results.append(
                ScenarioCompareResult(
                    scenario_mode=scenario_mode,
                    selected=None,
                    candidates=[],
                    warnings=warnings,
                    error=(
                        _strict_error_text(strict_detail)
                        if strict_detail is not None
                        else "reason_code:no_route_candidates; message:No route candidates could be computed."
                    ),
                )
            )
            continue

        pareto_options = _finalize_pareto_options(
            options,
            max_alternatives=req.max_alternatives,
            pareto_method=req.pareto_method,
            epsilon=req.epsilon,
            optimization_mode=req.optimization_mode,
            risk_aversion=req.risk_aversion,
        )
        if not pareto_options:
            strict_detail = _strict_error_detail(
                reason_code="epsilon_infeasible",
                message="No routes satisfy epsilon constraints for this request.",
                warnings=warnings,
            )
            mode_error_details[scenario_mode.value] = strict_detail
            results.append(
                ScenarioCompareResult(
                    scenario_mode=scenario_mode,
                    selected=None,
                    candidates=[],
                    warnings=warnings,
                    error=_strict_error_text(strict_detail),
                )
            )
            continue
        selected = _pick_best_option(
            pareto_options,
            w_time=req.weights.time,
            w_money=req.weights.money,
            w_co2=req.weights.co2,
            optimization_mode=req.optimization_mode,
            risk_aversion=req.risk_aversion,
        )
        results.append(
            ScenarioCompareResult(
                scenario_mode=scenario_mode,
                selected=selected,
                candidates=pareto_options,
                warnings=warnings,
            )
        )
        mode_error_details[scenario_mode.value] = None

    baseline = next(
        (item.selected for item in results if item.scenario_mode == ScenarioMode.NO_SHARING),
        None,
    )
    baseline_error = next(
        (item.error for item in results if item.scenario_mode == ScenarioMode.NO_SHARING),
        None,
    )
    baseline_error_detail = mode_error_details.get(ScenarioMode.NO_SHARING.value)
    deltas: dict[str, ScenarioCompareDelta] = {}
    for item in results:
        deltas[item.scenario_mode.value] = ScenarioCompareDelta.model_validate(
            _scenario_metric_deltas(
                baseline,
                item.selected,
                baseline_error=baseline_error,
                current_error=item.error,
                baseline_error_detail=baseline_error_detail,
                current_error_detail=mode_error_details.get(item.scenario_mode.value),
            )
        )

    return results, deltas


@app.post("/scenario/compare", response_model=ScenarioCompareResponse)
async def compare_scenarios(
    req: ScenarioCompareRequest,
    osrm: OSRMDep,
    _: UserAccessDep,
) -> ScenarioCompareResponse:
    run_id = str(uuid.uuid4())
    t0 = time.perf_counter()
    has_error = False

    try:
        results, deltas = await _run_scenario_compare(req=req, osrm=osrm)
        deltas_payload = {mode: delta.model_dump(mode="json") for mode, delta in deltas.items()}

        manifest_payload = {
            "schema_version": "1.0.0",
            "type": "scenario_compare",
            "request": req.model_dump(mode="json"),
            "results": [item.model_dump(mode="json") for item in results],
            "deltas": deltas_payload,
            "execution": {
                "duration_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                "scenario_count": len(results),
                "terrain_profile": req.terrain_profile,
                "stochastic": req.stochastic.model_dump(mode="json"),
                "weather": req.weather.model_dump(mode="json"),
                "incident_simulation": req.incident_simulation.model_dump(mode="json"),
                "optimization_mode": req.optimization_mode,
                "risk_aversion": req.risk_aversion,
                "emissions_context": req.emissions_context.model_dump(mode="json"),
            },
        }
        scenario_manifest = write_scenario_manifest(run_id, manifest_payload)

        log_event(
            "scenario_compare_request",
            run_id=run_id,
            vehicle_type=req.vehicle_type,
            max_alternatives=req.max_alternatives,
            terrain_profile=req.terrain_profile,
            stochastic=req.stochastic.model_dump(mode="json"),
            weather=req.weather.model_dump(mode="json"),
            incident_simulation=req.incident_simulation.model_dump(mode="json"),
            optimization_mode=req.optimization_mode,
            risk_aversion=req.risk_aversion,
            emissions_context=req.emissions_context.model_dump(mode="json"),
            pareto_method=req.pareto_method,
            epsilon=req.epsilon.model_dump(mode="json") if req.epsilon is not None else None,
            scenario_manifest=str(scenario_manifest),
        )

        return ScenarioCompareResponse(
            run_id=run_id,
            results=results,
            deltas=deltas,
            baseline_mode=ScenarioMode.NO_SHARING,
            scenario_manifest_endpoint=f"/runs/{run_id}/scenario-manifest",
            scenario_signature_endpoint=f"/runs/{run_id}/scenario-signature",
        )
    except HTTPException:
        has_error = True
        raise
    except Exception:
        has_error = True
        raise
    finally:
        _record_endpoint_metric("scenario_compare_post", t0, error=has_error)


@app.get("/experiments", response_model=ExperimentListResponse)
async def get_experiments(
    q: Annotated[str | None, Query()] = None,
    vehicle_type: Annotated[str | None, Query()] = None,
    scenario_mode: Annotated[ScenarioMode | None, Query()] = None,
    sort: Annotated[str, Query()] = "updated_desc",
) -> ExperimentListResponse:
    t0 = time.perf_counter()
    has_error = False
    try:
        allowed_sorts = {"updated_desc", "updated_asc", "name_asc", "name_desc"}
        if sort not in allowed_sorts:
            raise HTTPException(
                status_code=400,
                detail=f"invalid sort value '{sort}', expected one of: {', '.join(sorted(allowed_sorts))}",
            )
        return ExperimentListResponse(
            experiments=list_experiments(
                q=q,
                vehicle_type=vehicle_type,
                scenario_mode=scenario_mode.value if scenario_mode is not None else None,
                sort=sort,
            )
        )
    except HTTPException:
        has_error = True
        raise
    except Exception:
        has_error = True
        raise
    finally:
        _record_endpoint_metric("experiments_get", t0, error=has_error)


@app.post("/experiments", response_model=ExperimentBundle)
async def post_experiment(payload: ExperimentBundleInput, _: AdminAccessDep) -> ExperimentBundle:
    t0 = time.perf_counter()
    has_error = False
    try:
        bundle, path = create_experiment(payload)
        log_event("experiment_create", experiment_id=bundle.id, path=str(path))
        return bundle
    except Exception:
        has_error = True
        raise
    finally:
        _record_endpoint_metric("experiments_post", t0, error=has_error)


@app.get("/experiments/{experiment_id}", response_model=ExperimentBundle)
async def get_experiment_by_id(experiment_id: str) -> ExperimentBundle:
    t0 = time.perf_counter()
    has_error = False
    try:
        return get_experiment(experiment_id)
    except KeyError as e:
        has_error = True
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        has_error = True
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception:
        has_error = True
        raise
    finally:
        _record_endpoint_metric("experiments_id_get", t0, error=has_error)


@app.put("/experiments/{experiment_id}", response_model=ExperimentBundle)
async def put_experiment(
    experiment_id: str,
    payload: ExperimentBundleInput,
    _: AdminAccessDep,
) -> ExperimentBundle:
    t0 = time.perf_counter()
    has_error = False
    try:
        bundle, path = update_experiment(experiment_id, payload)
        log_event("experiment_update", experiment_id=bundle.id, path=str(path))
        return bundle
    except KeyError as e:
        has_error = True
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        has_error = True
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception:
        has_error = True
        raise
    finally:
        _record_endpoint_metric("experiments_id_put", t0, error=has_error)


@app.delete("/experiments/{experiment_id}")
async def delete_experiment_by_id(experiment_id: str, _: AdminAccessDep) -> dict[str, object]:
    t0 = time.perf_counter()
    has_error = False
    try:
        deleted_id, index_path = delete_experiment(experiment_id)
        log_event("experiment_delete", experiment_id=deleted_id, index_path=str(index_path))
        return {"experiment_id": deleted_id, "deleted": True}
    except KeyError as e:
        has_error = True
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        has_error = True
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception:
        has_error = True
        raise
    finally:
        _record_endpoint_metric("experiments_id_delete", t0, error=has_error)


@app.post("/experiments/{experiment_id}/compare", response_model=ScenarioCompareResponse)
async def replay_experiment_compare(
    experiment_id: str,
    payload: ExperimentCompareRequest,
    osrm: OSRMDep,
    _: UserAccessDep,
) -> ScenarioCompareResponse:
    t0 = time.perf_counter()
    has_error = False

    try:
        bundle = get_experiment(experiment_id)
        request_data = bundle.request.model_dump(mode="json")
        if payload.overrides:
            request_data.update(payload.overrides)
        req = ScenarioCompareRequest.model_validate(request_data)

        run_id = str(uuid.uuid4())
        results, deltas = await _run_scenario_compare(req=req, osrm=osrm)
        deltas_payload = {mode: delta.model_dump(mode="json") for mode, delta in deltas.items()}

        manifest_payload = {
            "schema_version": "1.0.0",
            "type": "scenario_compare",
            "source": {"experiment_id": bundle.id, "experiment_name": bundle.name},
            "request": req.model_dump(mode="json"),
            "results": [item.model_dump(mode="json") for item in results],
            "deltas": deltas_payload,
            "execution": {
                "duration_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                "scenario_count": len(results),
                "terrain_profile": req.terrain_profile,
                "weather": req.weather.model_dump(mode="json"),
                "incident_simulation": req.incident_simulation.model_dump(mode="json"),
            },
        }
        scenario_manifest = write_scenario_manifest(run_id, manifest_payload)
        log_event(
            "experiment_compare_replay",
            experiment_id=bundle.id,
            run_id=run_id,
            scenario_manifest=str(scenario_manifest),
        )

        return ScenarioCompareResponse(
            run_id=run_id,
            results=results,
            deltas=deltas,
            baseline_mode=ScenarioMode.NO_SHARING,
            scenario_manifest_endpoint=f"/runs/{run_id}/scenario-manifest",
            scenario_signature_endpoint=f"/runs/{run_id}/scenario-signature",
        )
    except KeyError as e:
        has_error = True
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        has_error = True
        raise HTTPException(status_code=400, detail=str(e)) from e
    except HTTPException:
        has_error = True
        raise
    except Exception:
        has_error = True
        raise
    finally:
        _record_endpoint_metric("experiments_compare_post", t0, error=has_error)


def _route_results_csv_rows(
    req: RouteRequest,
    candidates: list[RouteOption],
    *,
    selected_id: str,
    error: str = "",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for route in candidates:
        rows.append(
            {
                "pair_index": 0,
                "origin_lat": req.origin.lat,
                "origin_lon": req.origin.lon,
                "destination_lat": req.destination.lat,
                "destination_lon": req.destination.lon,
                "error": error,
                "route_id": route.id,
                "distance_km": route.metrics.distance_km,
                "duration_s": route.metrics.duration_s,
                "monetary_cost": route.metrics.monetary_cost,
                "emissions_kg": route.metrics.emissions_kg,
                "avg_speed_kmh": route.metrics.avg_speed_kmh,
                "selected": route.id == selected_id,
            }
        )
    if rows:
        return rows
    return [
        {
            "pair_index": 0,
            "origin_lat": req.origin.lat,
            "origin_lon": req.origin.lon,
            "destination_lat": req.destination.lat,
            "destination_lon": req.destination.lon,
            "error": error,
            "route_id": "",
            "distance_km": "",
            "duration_s": "",
            "monetary_cost": "",
            "emissions_kg": "",
            "avg_speed_kmh": "",
            "selected": False,
        }
    ]


def _route_routes_geojson(
    req: RouteRequest,
    candidates: list[RouteOption],
    *,
    selected_id: str,
) -> dict[str, Any]:
    features: list[dict[str, Any]] = []
    for route in candidates:
        features.append(
            {
                "type": "Feature",
                "geometry": route.geometry.model_dump(mode="json"),
                "properties": {
                    "route_id": route.id,
                    "selected": route.id == selected_id,
                    "distance_km": route.metrics.distance_km,
                    "duration_s": route.metrics.duration_s,
                    "monetary_cost": route.metrics.monetary_cost,
                    "emissions_kg": route.metrics.emissions_kg,
                    "origin": req.origin.model_dump(),
                    "destination": req.destination.model_dump(),
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


def _write_route_run_bundle(
    *,
    req: RouteRequest,
    selected: RouteOption,
    candidates: list[RouteOption],
    warnings: list[str],
    candidate_diag: CandidateDiagnostics,
    request_id: str,
    pipeline_mode: str,
    run_seed: int,
    duration_ms: float,
    selected_certificate: RouteCertificationSummary | None = None,
    voi_stop_summary: VoiStopSummary | None = None,
    extra_json_artifacts: dict[str, dict[str, Any] | list[Any]] | None = None,
    extra_jsonl_artifacts: dict[str, list[dict[str, Any]]] | None = None,
    extra_csv_artifacts: dict[str, tuple[list[str], list[dict[str, Any]]]] | None = None,
    extra_text_artifacts: dict[str, str] | None = None,
) -> dict[str, Any]:
    run_id = str(uuid.uuid4())
    manifest_payload = {
        "schema_version": "1.0.0",
        "type": "route_compute",
        "request": req.model_dump(mode="json"),
        "pipeline": {
            "mode": pipeline_mode,
            "run_seed": int(run_seed),
        },
        "selected_route_id": selected.id,
        "selected_certificate": (
            selected_certificate.model_dump(mode="json") if selected_certificate is not None else None
        ),
        "voi_stop_summary": (
            voi_stop_summary.model_dump(mode="json") if voi_stop_summary is not None else None
        ),
        "warnings": list(warnings),
        "candidate_diagnostics": candidate_diag.__dict__,
        "execution": {
            "duration_ms": round(float(duration_ms), 3),
            "request_id": request_id,
            "candidate_count": len(candidates),
        },
    }
    manifest_path = write_manifest(run_id, manifest_payload)

    results_payload = {
        "run_id": run_id,
        "selected": selected.model_dump(mode="json"),
        "candidates": [route.model_dump(mode="json") for route in candidates],
        "warnings": list(warnings),
        "candidate_diagnostics": candidate_diag.__dict__,
    }
    write_json_artifact(run_id, "results.json", results_payload)
    results_rows = _route_results_csv_rows(req, candidates, selected_id=selected.id)
    write_csv_artifact(
        run_id,
        "results.csv",
        fieldnames=[
            "pair_index",
            "origin_lat",
            "origin_lon",
            "destination_lat",
            "destination_lon",
            "error",
            "route_id",
            "distance_km",
            "duration_s",
            "monetary_cost",
            "emissions_kg",
            "avg_speed_kmh",
            "selected",
        ],
        rows=results_rows,
    )
    write_json_artifact(
        run_id,
        "routes.geojson",
        _route_routes_geojson(req, candidates, selected_id=selected.id),
    )
    write_csv_artifact(
        run_id,
        "results_summary.csv",
        fieldnames=["route_id", "selected", "distance_km", "duration_s", "monetary_cost", "emissions_kg"],
        rows=[
            {
                "route_id": route.id,
                "selected": route.id == selected.id,
                "distance_km": route.metrics.distance_km,
                "duration_s": route.metrics.duration_s,
                "monetary_cost": route.metrics.monetary_cost,
                "emissions_kg": route.metrics.emissions_kg,
            }
            for route in candidates
        ],
    )

    for artifact_name, payload in (extra_json_artifacts or {}).items():
        write_json_artifact(run_id, artifact_name, payload)
    for artifact_name, rows in (extra_jsonl_artifacts or {}).items():
        write_jsonl_artifact(run_id, artifact_name, rows)
    for artifact_name, payload in (extra_csv_artifacts or {}).items():
        fieldnames, rows = payload
        write_csv_artifact(run_id, artifact_name, fieldnames=fieldnames, rows=rows)
    for artifact_name, text in (extra_text_artifacts or {}).items():
        write_text_artifact(run_id, artifact_name, text)

    artifact_names = sorted(list_artifact_paths_for_run(run_id))
    metadata_payload = {
        "run_id": run_id,
        "schema_version": "1.0.0",
        "type": "route_compute",
        "request_id": request_id,
        "pipeline_mode": pipeline_mode,
        "run_seed": int(run_seed),
        "manifest_endpoint": f"/runs/{run_id}/manifest",
        "artifacts_endpoint": f"/runs/{run_id}/artifacts",
        "provenance_endpoint": f"/runs/{run_id}/provenance",
        "provenance_file": str(provenance_path_for_run(run_id)),
        "artifact_names": artifact_names,
        "selected_route_id": selected.id,
        "candidate_count": len(candidates),
        "warning_count": len(warnings),
        "duration_ms": round(float(duration_ms), 3),
    }
    write_json_artifact(run_id, "metadata.json", metadata_payload)

    provenance_events = [
        provenance_event(
            run_id,
            "route_input_received",
            pipeline_mode=pipeline_mode,
            request=req.model_dump(mode="json"),
            request_id=request_id,
            run_seed=int(run_seed),
        ),
        provenance_event(
            run_id,
            "route_selected",
            selected_id=selected.id,
            candidate_count=len(candidates),
            warning_count=len(warnings),
            candidate_diagnostics=candidate_diag.__dict__,
        ),
        provenance_event(
            run_id,
            "artifacts_written",
            manifest=str(manifest_path),
            artifact_names=artifact_names,
        ),
    ]
    write_provenance(run_id, provenance_events)

    return {
        "run_id": run_id,
        "manifest_endpoint": f"/runs/{run_id}/manifest",
        "artifacts_endpoint": f"/runs/{run_id}/artifacts",
        "provenance_endpoint": f"/runs/{run_id}/provenance",
    }


def _batch_results_csv_rows(
    req: BatchParetoRequest,
    results: list[BatchParetoResult],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for idx, result in enumerate(results):
        base = {
            "pair_index": idx,
            "origin_lat": result.origin.lat,
            "origin_lon": result.origin.lon,
            "destination_lat": result.destination.lat,
            "destination_lon": result.destination.lon,
            "error": result.error or "",
        }

        if not result.routes:
            rows.append(
                {
                    **base,
                    "route_id": "",
                    "distance_km": "",
                    "duration_s": "",
                    "monetary_cost": "",
                    "emissions_kg": "",
                    "avg_speed_kmh": "",
                }
            )
            continue

        for route in result.routes:
            rows.append(
                {
                    **base,
                    "route_id": route.id,
                    "distance_km": route.metrics.distance_km,
                    "duration_s": route.metrics.duration_s,
                    "monetary_cost": route.metrics.monetary_cost,
                    "emissions_kg": route.metrics.emissions_kg,
                    "avg_speed_kmh": route.metrics.avg_speed_kmh,
                }
            )

    return rows


def _batch_routes_geojson(results: list[BatchParetoResult]) -> dict[str, Any]:
    features: list[dict[str, Any]] = []
    for pair_idx, result in enumerate(results):
        for route in result.routes:
            features.append(
                {
                    "type": "Feature",
                    "geometry": route.geometry.model_dump(mode="json"),
                    "properties": {
                        "pair_index": pair_idx,
                        "route_id": route.id,
                        "origin": result.origin.model_dump(),
                        "destination": result.destination.model_dump(),
                        "metrics": route.metrics.model_dump(mode="json"),
                    },
                }
            )

    return {
        "type": "FeatureCollection",
        "features": features,
    }


def _batch_summary_csv_rows(results: list[BatchParetoResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair_idx, result in enumerate(results):
        route_count = len(result.routes)
        durations = [r.metrics.duration_s for r in result.routes]
        moneys = [r.metrics.monetary_cost for r in result.routes]
        emissions = [r.metrics.emissions_kg for r in result.routes]
        rows.append(
            {
                "pair_index": pair_idx,
                "origin_lat": result.origin.lat,
                "origin_lon": result.origin.lon,
                "destination_lat": result.destination.lat,
                "destination_lon": result.destination.lon,
                "route_count": route_count,
                "error": result.error or "",
                "min_duration_s": min(durations) if durations else "",
                "min_monetary_cost": min(moneys) if moneys else "",
                "min_emissions_kg": min(emissions) if emissions else "",
            }
        )
    return rows


def _write_batch_additional_exports(run_id: str, results: list[BatchParetoResult]) -> list[str]:
    out_dir = artifact_dir_for_run(run_id)
    written: list[str] = []

    geojson_path = out_dir / "routes.geojson"
    geojson_path.write_text(
        json.dumps(_batch_routes_geojson(results), indent=2),
        encoding="utf-8",
    )
    written.append("routes.geojson")

    summary_rows = _batch_summary_csv_rows(results)
    summary_path = out_dir / "results_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "pair_index",
            "origin_lat",
            "origin_lon",
            "destination_lat",
            "destination_lon",
            "route_count",
            "error",
            "min_duration_s",
            "min_monetary_cost",
            "min_emissions_kg",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    written.append("results_summary.csv")

    return written


def _parse_pairs_from_csv_text(csv_text: str) -> list[ODPair]:
    reader = csv.DictReader(io.StringIO(csv_text))
    required = {"origin_lat", "origin_lon", "destination_lat", "destination_lon"}
    headers = set(reader.fieldnames or [])
    if not required.issubset(headers):
        missing = sorted(required - headers)
        raise HTTPException(
            status_code=422,
            detail=f"CSV missing required columns: {', '.join(missing)}",
        )

    pairs: list[ODPair] = []
    for idx, row in enumerate(reader, start=2):
        try:
            pairs.append(
                ODPair(
                    origin=LatLng(
                        lat=float(row["origin_lat"]),
                        lon=float(row["origin_lon"]),
                    ),
                    destination=LatLng(
                        lat=float(row["destination_lat"]),
                        lon=float(row["destination_lon"]),
                    ),
                )
            )
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=f"invalid CSV numeric value on row {idx}",
            ) from e

    if not pairs:
        raise HTTPException(
            status_code=422,
            detail=_strict_error_detail(
                reason_code="no_route_candidates",
                message="CSV contains no OD pairs",
                warnings=[],
            ),
        )

    return pairs


@app.post("/batch/import/csv", response_model=BatchParetoResponse)
async def batch_import_csv(
    req: BatchCSVImportRequest,
    osrm: OSRMDep,
    _: UserAccessDep,
) -> BatchParetoResponse:
    pairs = _parse_pairs_from_csv_text(req.csv_text)
    batch_req = BatchParetoRequest(
        pairs=pairs,
        waypoints=req.waypoints,
        vehicle_type=req.vehicle_type,
        scenario_mode=req.scenario_mode,
        max_alternatives=req.max_alternatives,
        weights=req.weights,
        cost_toggles=req.cost_toggles,
        terrain_profile=req.terrain_profile,
        stochastic=req.stochastic,
        optimization_mode=req.optimization_mode,
        risk_aversion=req.risk_aversion,
        emissions_context=req.emissions_context,
        weather=req.weather,
        incident_simulation=req.incident_simulation,
        departure_time_utc=req.departure_time_utc,
        pareto_method=req.pareto_method,
        epsilon=req.epsilon,
        seed=req.seed,
        toggles=req.toggles,
        model_version=req.model_version,
    )
    return await batch_pareto(batch_req, osrm, None)


@app.post("/batch/pareto", response_model=BatchParetoResponse)
async def batch_pareto(req: BatchParetoRequest, osrm: OSRMDep, _: UserAccessDep) -> BatchParetoResponse:
    run_id = str(uuid.uuid4())
    t0 = time.perf_counter()
    has_error = False

    try:
        sem = asyncio.Semaphore(settings.batch_concurrency)

        async def one(pair_idx: int) -> tuple[BatchParetoResult, dict[str, Any]]:
            async with sem:
                pair = req.pairs[pair_idx]
                pair_stats: dict[str, Any] = {
                    "pair_index": pair_idx,
                    "candidate_count": 0,
                    "option_count": 0,
                    "pareto_count": 0,
                    "error": None,
                }
                try:
                    options, warnings, candidate_fetches, terrain_diag, _candidate_diag = await _collect_route_options_with_diagnostics(
                        osrm=osrm,
                        origin=pair.origin,
                        destination=pair.destination,
                        waypoints=req.waypoints,
                        max_alternatives=req.max_alternatives,
                        vehicle_type=req.vehicle_type,
                        scenario_mode=req.scenario_mode,
                        cost_toggles=req.cost_toggles,
                        terrain_profile=req.terrain_profile,
                        stochastic=req.stochastic,
                        emissions_context=req.emissions_context,
                        weather=req.weather,
                        incident_simulation=req.incident_simulation,
                        departure_time_utc=req.departure_time_utc,
                        pareto_method=req.pareto_method,
                        epsilon=req.epsilon,
                        optimization_mode=req.optimization_mode,
                        risk_aversion=req.risk_aversion,
                        utility_weights=(req.weights.time, req.weights.money, req.weights.co2),
                        option_prefix=f"pair{pair_idx}_route",
                    )
                    pair_stats["candidate_count"] = candidate_fetches
                    pair_stats["option_count"] = len(options)
                    if not options:
                        strict_detail = _strict_failure_detail_from_outcome(
                            warnings=warnings,
                            terrain_diag=terrain_diag,
                            epsilon_requested=req.pareto_method == "epsilon_constraint" and req.epsilon is not None,
                            terrain_message="All candidates were removed by terrain DEM strict coverage policy.",
                        )
                        warning_msg = (
                            _strict_error_text(strict_detail)
                            if strict_detail is not None
                            else "reason_code:no_route_candidates; message:No route candidates could be computed."
                        )
                        pair_stats["error"] = warning_msg
                        return (
                            BatchParetoResult(
                                origin=pair.origin,
                                destination=pair.destination,
                                error=warning_msg,
                            ),
                            pair_stats,
                        )

                    pareto = _finalize_pareto_options(
                        options,
                        max_alternatives=req.max_alternatives,
                        pareto_method=req.pareto_method,
                        epsilon=req.epsilon,
                        optimization_mode=req.optimization_mode,
                        risk_aversion=req.risk_aversion,
                    )
                    pair_stats["pareto_count"] = len(pareto)
                    if (
                        not pareto
                        and req.pareto_method == "epsilon_constraint"
                        and req.epsilon is not None
                    ):
                        strict_detail = _strict_error_detail(
                            reason_code="epsilon_infeasible",
                            message="No routes satisfy epsilon constraints for this request.",
                            warnings=warnings,
                        )
                        error_text = _strict_error_text(strict_detail)
                        pair_stats["error"] = error_text
                        return (
                            BatchParetoResult(
                                origin=pair.origin,
                                destination=pair.destination,
                                error=error_text,
                            ),
                            pair_stats,
                        )

                    return (
                        BatchParetoResult(
                            origin=pair.origin,
                            destination=pair.destination,
                            routes=pareto,
                        ),
                        pair_stats,
                    )
                except ValueError as e:
                    msg = str(e).strip()
                    reason_code = "batch_pair_failed"
                    lowered = msg.lower()
                    if "tariff" in lowered or "toll" in lowered:
                        reason_code = "toll_tariff_unresolved"
                    elif "terrain_region_unsupported" in lowered or ("terrain" in lowered and "unsupported" in lowered):
                        reason_code = "terrain_region_unsupported"
                    elif "terrain_dem_asset_unavailable" in lowered or ("terrain" in lowered and "asset" in lowered):
                        reason_code = "terrain_dem_asset_unavailable"
                    elif "terrain" in lowered and ("coverage" in lowered or "dem" in lowered):
                        reason_code = "terrain_dem_coverage_insufficient"
                    elif "epsilon" in lowered and "infeasible" in lowered:
                        reason_code = "epsilon_infeasible"
                    elif "departure profile" in lowered or "departure_profile" in lowered:
                        reason_code = "departure_profile_unavailable"
                    elif "scenario_profile_unavailable" in lowered or (
                        "scenario profile" in lowered and "unavailable" in lowered
                    ):
                        reason_code = "scenario_profile_unavailable"
                    elif "scenario_profile_invalid" in lowered or (
                        "scenario profile" in lowered and "invalid" in lowered
                    ):
                        reason_code = "scenario_profile_invalid"
                    pair_stats["error"] = f"reason_code:{reason_code}; message:{msg or type(e).__name__}"
                    return (
                        BatchParetoResult(
                            origin=pair.origin,
                            destination=pair.destination,
                            error=pair_stats["error"],
                        ),
                        pair_stats,
                    )

        provenance_events: list[dict[str, Any]] = [
            provenance_event(
                run_id,
                "input_received",
                pair_count=len(req.pairs),
                vehicle_type=req.vehicle_type,
                scenario_mode=req.scenario_mode.value,
                max_alternatives=req.max_alternatives,
                cost_toggles=req.cost_toggles.model_dump(mode="json"),
                terrain_profile=req.terrain_profile,
                stochastic=req.stochastic.model_dump(mode="json"),
                weather=req.weather.model_dump(mode="json"),
                incident_simulation=req.incident_simulation.model_dump(mode="json"),
                optimization_mode=req.optimization_mode,
                risk_aversion=req.risk_aversion,
                emissions_context=req.emissions_context.model_dump(mode="json"),
                pareto_method=req.pareto_method,
                epsilon=req.epsilon.model_dump(mode="json") if req.epsilon is not None else None,
                departure_time_utc=(
                    req.departure_time_utc.isoformat() if req.departure_time_utc is not None else None
                ),
            )
        ]

        pair_outputs = await asyncio.gather(*[one(i) for i in range(len(req.pairs))])
        results = [result for result, _ in pair_outputs]
        pair_stats = [stats for _, stats in pair_outputs]

        duration_ms = round((time.perf_counter() - t0) * 1000, 2)
        error_count = sum(1 for r in results if r.error)
        candidate_count = sum(int(stats["candidate_count"]) for stats in pair_stats)
        option_count = sum(int(stats["option_count"]) for stats in pair_stats)
        pareto_count = sum(int(stats["pareto_count"]) for stats in pair_stats)

        provenance_events.append(
            provenance_event(
                run_id,
                "candidates_fetched",
                candidate_count=candidate_count,
                pairs=[
                    {
                        "pair_index": int(stats["pair_index"]),
                        "candidate_count": int(stats["candidate_count"]),
                        "error": stats["error"],
                    }
                    for stats in pair_stats
                ],
            )
        )
        provenance_events.append(
            provenance_event(
                run_id,
                "options_built",
                option_count=option_count,
                pairs=[
                    {
                        "pair_index": int(stats["pair_index"]),
                        "option_count": int(stats["option_count"]),
                    }
                    for stats in pair_stats
                ],
            )
        )
        provenance_events.append(
            provenance_event(
                run_id,
                "pareto_selected",
                pareto_count=pareto_count,
                error_count=error_count,
                pairs=[
                    {
                        "pair_index": int(stats["pair_index"]),
                        "pareto_count": int(stats["pareto_count"]),
                    }
                    for stats in pair_stats
                ],
            )
        )
        provenance_file = provenance_path_for_run(run_id)

        request_payload = req.model_dump(mode="json")
        manifest_payload: dict[str, Any] = {
            "schema_version": "1.0.0",
            "type": "batch_pareto",
            "request": request_payload,
            "reproducibility": {
                "seed": req.seed,
                "toggles": req.toggles,
            },
            "model_metadata": {
                "app_version": app.version,
                "model_version": req.model_version or app.version,
            },
            "execution": {
                "pair_count": len(req.pairs),
                "max_alternatives": req.max_alternatives,
                "batch_concurrency": settings.batch_concurrency,
                "cost_toggles": req.cost_toggles.model_dump(),
                "terrain_profile": req.terrain_profile,
                "stochastic": req.stochastic.model_dump(mode="json"),
                "weather": req.weather.model_dump(mode="json"),
                "incident_simulation": req.incident_simulation.model_dump(mode="json"),
                "optimization_mode": req.optimization_mode,
                "risk_aversion": req.risk_aversion,
                "emissions_context": req.emissions_context.model_dump(mode="json"),
                "pareto_method": req.pareto_method,
                "epsilon": req.epsilon.model_dump(mode="json") if req.epsilon is not None else None,
                "departure_time_utc": (
                    req.departure_time_utc.isoformat() if req.departure_time_utc is not None else None
                ),
                "duration_ms": duration_ms,
                "error_count": error_count,
            },
        }
        manifest_path = write_manifest(
            run_id,
            manifest_payload,
        )

        results_payload = {
            "run_id": run_id,
            "results": [r.model_dump(mode="json") for r in results],
        }
        metadata_payload = {
            "run_id": run_id,
            "schema_version": "1.0.0",
            "manifest_endpoint": f"/runs/{run_id}/manifest",
            "artifacts_endpoint": f"/runs/{run_id}/artifacts",
            "provenance_endpoint": f"/runs/{run_id}/provenance",
            "provenance_file": str(provenance_file),
            "artifact_names": sorted(artifact_paths_for_run(run_id)),
            "pair_count": len(req.pairs),
            "error_count": error_count,
            "duration_ms": duration_ms,
            "terrain_profile": req.terrain_profile,
            "weather": req.weather.model_dump(mode="json"),
            "incident_simulation": req.incident_simulation.model_dump(mode="json"),
        }

        artifact_paths = write_run_artifacts(
            run_id,
            results_payload=results_payload,
            metadata_payload=metadata_payload,
            csv_rows=_batch_results_csv_rows(req, results),
        )
        extra_artifacts = _write_batch_additional_exports(run_id, results)
        report_path = write_report_pdf(
            run_id,
            manifest=json.loads(manifest_path.read_text(encoding="utf-8")),
            metadata=metadata_payload,
            results=results_payload,
        )
        artifact_names = [name for name in sorted(list(artifact_paths) + extra_artifacts + [report_path.name])]
        provenance_events.append(
            provenance_event(
                run_id,
                "artifacts_written",
                manifest=str(manifest_path),
                artifact_names=artifact_names,
                provenance_file=str(provenance_file),
            )
        )
        written_provenance = write_provenance(run_id, provenance_events)

        log_event(
            "batch_pareto_request",
            run_id=run_id,
            pair_count=len(req.pairs),
            vehicle_type=req.vehicle_type,
            scenario_mode=req.scenario_mode.value,
            cost_toggles=req.cost_toggles.model_dump(),
            terrain_profile=req.terrain_profile,
            stochastic=req.stochastic.model_dump(mode="json"),
            weather=req.weather.model_dump(mode="json"),
            incident_simulation=req.incident_simulation.model_dump(mode="json"),
            optimization_mode=req.optimization_mode,
            risk_aversion=req.risk_aversion,
            emissions_context=req.emissions_context.model_dump(mode="json"),
            pareto_method=req.pareto_method,
            epsilon=req.epsilon.model_dump(mode="json") if req.epsilon is not None else None,
            departure_time_utc=(
                req.departure_time_utc.isoformat() if req.departure_time_utc is not None else None
            ),
            duration_ms=duration_ms,
            error_count=error_count,
            manifest=str(manifest_path),
            artifacts=artifact_names,
            provenance=str(written_provenance),
        )

        return BatchParetoResponse(run_id=run_id, results=results)
    except HTTPException:
        has_error = True
        raise
    except Exception:
        has_error = True
        raise
    finally:
        _record_endpoint_metric("batch_pareto", t0, error=has_error)


def _validated_run_id(run_id: str) -> str:
    try:
        return str(uuid.UUID(run_id))
    except ValueError as e:
        raise HTTPException(status_code=400, detail="invalid run_id") from e


def _manifest_path_for_id(run_id: str) -> Path:
    valid_run_id = _validated_run_id(run_id)
    p = Path(settings.out_dir) / "manifests" / f"{valid_run_id}.json"
    base = (Path(settings.out_dir) / "manifests").resolve()
    resolved = p.resolve()

    if not str(resolved).startswith(str(base)):
        raise HTTPException(status_code=400, detail="invalid run_id path")

    return resolved


def _scenario_manifest_path_for_id(run_id: str) -> Path:
    valid_run_id = _validated_run_id(run_id)
    p = Path(settings.out_dir) / "scenario_manifests" / f"{valid_run_id}.json"
    base = (Path(settings.out_dir) / "scenario_manifests").resolve()
    resolved = p.resolve()

    if not str(resolved).startswith(str(base)):
        raise HTTPException(status_code=400, detail="invalid run_id path")

    return resolved


def _provenance_path_for_id(run_id: str) -> Path:
    valid_run_id = _validated_run_id(run_id)
    p = Path(settings.out_dir) / "provenance" / f"{valid_run_id}.json"
    base = (Path(settings.out_dir) / "provenance").resolve()
    resolved = p.resolve()

    if not str(resolved).startswith(str(base)):
        raise HTTPException(status_code=400, detail="invalid run_id path")

    return resolved


@app.get("/runs/{run_id}/manifest")
async def get_manifest(run_id: str):
    t0 = time.perf_counter()
    has_error = False
    try:
        path = _manifest_path_for_id(run_id)
        if not path.exists():
            raise HTTPException(status_code=404, detail="manifest not found")
        return FileResponse(str(path), media_type="application/json")
    except HTTPException:
        has_error = True
        raise
    except Exception:
        has_error = True
        raise
    finally:
        _record_endpoint_metric("runs_manifest_get", t0, error=has_error)


@app.get("/runs/{run_id}/scenario-manifest")
async def get_scenario_manifest(run_id: str):
    t0 = time.perf_counter()
    has_error = False
    try:
        path = _scenario_manifest_path_for_id(run_id)
        if not path.exists():
            raise HTTPException(status_code=404, detail="scenario manifest not found")
        return FileResponse(str(path), media_type="application/json")
    except HTTPException:
        has_error = True
        raise
    except Exception:
        has_error = True
        raise
    finally:
        _record_endpoint_metric("runs_scenario_manifest_get", t0, error=has_error)


@app.get("/runs/{run_id}/provenance")
async def get_provenance(run_id: str):
    t0 = time.perf_counter()
    has_error = False
    try:
        path = _provenance_path_for_id(run_id)
        if not path.exists():
            raise HTTPException(status_code=404, detail="provenance not found")

        log_event("run_provenance_get", run_id=_validated_run_id(run_id))
        return FileResponse(str(path), media_type="application/json")
    except HTTPException:
        has_error = True
        raise
    except Exception:
        has_error = True
        raise
    finally:
        _record_endpoint_metric("runs_provenance_get", t0, error=has_error)


@app.get("/runs/{run_id}/signature")
async def get_manifest_signature(run_id: str) -> dict[str, object]:
    t0 = time.perf_counter()
    has_error = False
    try:
        path = _manifest_path_for_id(run_id)
        if not path.exists():
            raise HTTPException(status_code=404, detail="manifest not found")

        payload = json.loads(path.read_text(encoding="utf-8"))
        signature = payload.get("signature")
        if not isinstance(signature, dict):
            raise HTTPException(status_code=404, detail="manifest signature not found")

        log_event("manifest_signature_get", run_id=_validated_run_id(run_id))
        return {"run_id": _validated_run_id(run_id), "signature": signature}
    except HTTPException:
        has_error = True
        raise
    except Exception:
        has_error = True
        raise
    finally:
        _record_endpoint_metric("runs_signature_get", t0, error=has_error)


@app.get("/runs/{run_id}/scenario-signature")
async def get_scenario_signature(run_id: str) -> dict[str, object]:
    t0 = time.perf_counter()
    has_error = False
    try:
        path = _scenario_manifest_path_for_id(run_id)
        if not path.exists():
            raise HTTPException(status_code=404, detail="scenario manifest not found")

        payload = json.loads(path.read_text(encoding="utf-8"))
        signature = payload.get("signature")
        if not isinstance(signature, dict):
            raise HTTPException(status_code=404, detail="scenario signature not found")

        log_event("scenario_signature_get", run_id=_validated_run_id(run_id))
        return {"run_id": _validated_run_id(run_id), "signature": signature}
    except HTTPException:
        has_error = True
        raise
    except Exception:
        has_error = True
        raise
    finally:
        _record_endpoint_metric("runs_scenario_signature_get", t0, error=has_error)


@app.post("/verify/signature", response_model=SignatureVerificationResponse)
async def verify_signature(req: SignatureVerificationRequest) -> SignatureVerificationResponse:
    t0 = time.perf_counter()
    has_error = False
    try:
        valid, expected = verify_payload_signature(
            req.payload,
            req.signature,
            secret=req.secret,
        )
        log_event("signature_verify", valid=valid)
        return SignatureVerificationResponse(
            valid=valid,
            algorithm=SIGNATURE_ALGORITHM,
            signature=req.signature.strip().lower(),
            expected_signature=expected,
        )
    except Exception:
        has_error = True
        raise
    finally:
        _record_endpoint_metric("verify_signature_post", t0, error=has_error)


def _artifact_path_for_id(run_id: str, artifact_name: str) -> Path:
    valid_run_id = _validated_run_id(run_id)
    try:
        base = artifact_dir_for_run(valid_run_id).resolve()
        path = artifact_path_for_name(valid_run_id, artifact_name).resolve()
    except ValueError as e:
        raise HTTPException(status_code=400, detail="invalid artifact path") from e

    if not str(path).startswith(str(base)):
        raise HTTPException(status_code=400, detail="invalid artifact path")

    return path


@app.get("/runs/{run_id}/artifacts")
async def list_artifacts(run_id: str) -> dict[str, object]:
    t0 = time.perf_counter()
    has_error = False
    try:
        valid_run_id = _validated_run_id(run_id)
        paths = list_artifact_paths_for_run(valid_run_id)

        artifacts: list[dict[str, object]] = []
        for name in sorted(paths):
            path = paths[name]
            if not path.exists():
                continue
            artifacts.append(
                {
                    "name": name,
                    "endpoint": f"/runs/{valid_run_id}/artifacts/{name}",
                    "size_bytes": path.stat().st_size,
                }
            )

        if not artifacts:
            raise HTTPException(status_code=404, detail="artifacts not found")

        log_event(
            "run_artifacts_list",
            run_id=valid_run_id,
            artifact_count=len(artifacts),
        )
        return {
            "run_id": valid_run_id,
            "artifacts": artifacts,
            "provenance_endpoint": f"/runs/{valid_run_id}/provenance",
        }
    except HTTPException:
        has_error = True
        raise
    except Exception:
        has_error = True
        raise
    finally:
        _record_endpoint_metric("runs_artifacts_list_get", t0, error=has_error)


async def _get_artifact_file(run_id: str, artifact_name: str) -> FileResponse:
    t0 = time.perf_counter()
    has_error = False
    try:
        path = _artifact_path_for_id(run_id, artifact_name)
        if not path.exists():
            raise HTTPException(status_code=404, detail="artifact not found")

        media_type = "application/octet-stream"
        if artifact_name.endswith(".json"):
            media_type = "application/json"
        elif artifact_name.endswith(".jsonl"):
            media_type = "application/x-ndjson"
        elif artifact_name.endswith(".geojson"):
            media_type = "application/geo+json"
        elif artifact_name.endswith(".csv"):
            media_type = "text/csv"
        elif artifact_name.endswith(".md"):
            media_type = "text/markdown"
        elif artifact_name.endswith(".pdf"):
            media_type = "application/pdf"

        log_event(
            "run_artifact_get",
            run_id=_validated_run_id(run_id),
            artifact=artifact_name,
            size_bytes=path.stat().st_size,
        )
        return FileResponse(str(path), media_type=media_type, filename=artifact_name)
    except HTTPException:
        has_error = True
        raise
    except Exception:
        has_error = True
        raise
    finally:
        _record_endpoint_metric("runs_artifact_get", t0, error=has_error)


@app.get("/runs/{run_id}/artifacts/results.json")
async def get_artifact_results_json(run_id: str) -> FileResponse:
    return await _get_artifact_file(run_id, "results.json")


@app.get("/runs/{run_id}/artifacts/results.csv")
async def get_artifact_results_csv(run_id: str) -> FileResponse:
    return await _get_artifact_file(run_id, "results.csv")


@app.get("/runs/{run_id}/artifacts/metadata.json")
async def get_artifact_metadata_json(run_id: str) -> FileResponse:
    return await _get_artifact_file(run_id, "metadata.json")


@app.get("/runs/{run_id}/artifacts/routes.geojson")
async def get_artifact_routes_geojson(run_id: str) -> FileResponse:
    return await _get_artifact_file(run_id, "routes.geojson")


@app.get("/runs/{run_id}/artifacts/results_summary.csv")
async def get_artifact_results_summary_csv(run_id: str) -> FileResponse:
    return await _get_artifact_file(run_id, "results_summary.csv")


@app.get("/runs/{run_id}/artifacts/report.pdf")
async def get_artifact_report_pdf(run_id: str) -> FileResponse:
    return await _get_artifact_file(run_id, "report.pdf")


@app.get("/runs/{run_id}/artifacts/{artifact_name}")
async def get_artifact_generic(run_id: str, artifact_name: str) -> FileResponse:
    return await _get_artifact_file(run_id, artifact_name)


