from __future__ import annotations

import asyncio
import copy
import ctypes
import csv
import hashlib
import io
import json
import math
import os
import re
import time
import uuid
import httpx
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

try:
    import psutil
except Exception:  # pragma: no cover - optional runtime dependency
    psutil = None

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
from .certificate_witness import CertificateWitness
from .certification_cache import (
    certification_cache_stats,
    clear_certification_cache,
    get_cached_certification,
    set_cached_certification,
)
from .certification_models import CertificationState
from .k_raw_cache import clear_k_raw_cache, get_cached_k_raw, k_raw_cache_stats, set_cached_k_raw
from .departure_profile import time_of_day_multiplier_uk
from .decision_critical import (
    DCCSConfig,
    DCCSCandidateRecord,
    DCCSResult,
    build_candidate_ledger,
    record_refine_outcome,
    score_candidate,
    select_candidates,
    stable_candidate_id,
    summarize_refine_outcomes,
)
from .abstention import AbstentionRecord
from .evidence_certification import (
    CertificateResult,
    FragilityResult,
    _route_perturbed_objectives,
    active_evidence_families,
    annotate_world_manifest_cache_reuse,
    compute_certificate,
    compute_fragility_maps,
    evaluate_world_bundle,
    refc_requires_full_stress_worlds,
    sample_world_manifest,
    validate_route_evidence_provenance,
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
    CertifiedSetSummary,
    CostToggles,
    CustomVehicleListResponse,
    DecisionAbstentionSummary,
    DecisionControllerSummary,
    DecisionLaneManifest,
    DecisionPackage,
    DecisionPreferenceSummary,
    DecisionSupportSourceRecord,
    DecisionSupportSummary,
    DecisionTheoremHookRecord,
    DecisionTheoremHookSummary,
    DecisionWitnessSummary,
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
from .pareto_methods import annotate_knee_scores, filter_by_epsilon, select_pareto_routes
from .preference_state import build_preference_state
from .provenance_store import provenance_event, provenance_path_for_run, write_provenance
from .rbac import require_role
from .risk_model import robust_objective
from .route_cache import (
    checkpoint_route_cache,
    clear_route_cache,
    clear_route_cache_checkpoint,
    get_cached_routes,
    restore_checkpointed_route_cache,
    route_cache_checkpoint_stats,
    route_cache_stats,
    set_cached_routes,
)
from .route_option_cache import (
    CachedRouteOptionBuild,
    CachedRouteOptionCore,
    build_route_option_cache_key,
    build_route_option_core_cache_key,
    clear_route_option_cache,
    get_cached_route_option_core,
    get_cached_route_option_build,
    route_option_cache_stats,
    set_cached_route_option_core,
    set_cached_route_option_build,
)
from .route_state_cache import (
    CachedRouteState,
    clear_route_state_cache,
    get_cached_route_state,
    route_state_cache_stats,
    set_cached_route_state,
)
from .routing_graph import (
    GraphCandidateDiagnostics,
    begin_route_graph_warmup,
    route_graph_candidate_routes,
    route_graph_od_feasibility,
    route_graph_status,
    route_graph_via_paths,
    route_graph_warmup_status,
)
from .routing_ors import ORSClient, ORSError, ORSRoute, local_ors_runtime_manifest
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
from .voi_controller import (
    VOIConfig,
    VOIControllerState,
    build_action_menu,
    compute_search_completeness_metrics,
    enrich_controller_state_for_actioning,
    credible_evidence_uncertainty,
    credible_search_uncertainty,
    refresh_controller_state_after_action,
)
from .voi_dccs_cache import (
    clear_voi_dccs_cache,
    get_cached_voi_dccs,
    set_cached_voi_dccs,
    voi_dccs_cache_stats,
)
from .weather_adapter import weather_incident_multiplier, weather_speed_multiplier, weather_summary

try:
    UK_TZ = ZoneInfo("Europe/London")
except ZoneInfoNotFoundError:
    UK_TZ = UTC


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.osrm = OSRMClient(base_url=settings.osrm_base_url, profile=settings.osrm_profile)
    app.state.ors = ORSClient(base_url=settings.ors_base_url, timeout_ms=settings.ors_directions_timeout_ms)
    if bool(settings.route_graph_warmup_on_startup) and "PYTEST_CURRENT_TEST" not in os.environ:
        begin_route_graph_warmup()
    yield
    await app.state.osrm.aclose()
    await app.state.ors.aclose()


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


def ors_client(request: Request) -> ORSClient:
    ors: ORSClient | None = getattr(request.app.state, "ors", None)  # type: ignore[attr-defined]
    if ors is None:
        raise HTTPException(status_code=503, detail="ORS client not initialised")
    return ors


ORSDep = Annotated[ORSClient, Depends(ors_client)]


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


def _repo_local_live_source_policy_enabled() -> bool:
    policy = str(getattr(settings, "live_source_policy", "repo_local_fresh") or "repo_local_fresh").strip().lower()
    return policy == "repo_local_fresh"


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

    def _resolve_max_age_minutes(detail_data: dict[str, Any], fallback_minutes: int | None) -> int | None:
        raw_minutes = detail_data.get("max_age_minutes")
        if raw_minutes not in (None, ""):
            try:
                return int(raw_minutes)
            except (TypeError, ValueError):
                pass
        raw_days = detail_data.get("max_age_days")
        if raw_days not in (None, ""):
            try:
                return int(raw_days) * 1440
            except (TypeError, ValueError):
                pass
        return fallback_minutes

    dependency_summaries: list[dict[str, Any]] = []

    def _append_success(
        name: str,
        *,
        source: Any = None,
        as_of_text: Any = None,
        max_age_minutes_value: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        as_of_raw = str(as_of_text or "").strip()
        as_of_dt = _parse_iso_utc(as_of_raw)
        dependency_summaries.append(
            {
                "name": name,
                "ok": True,
                "reason_code": "ok",
                "status": "ok",
                "source": (str(source).strip() or None) if source not in (None, "") else None,
                "as_of_utc": as_of_dt.isoformat() if as_of_dt is not None else (as_of_raw or None),
                "age_minutes": _age_minutes(as_of_dt),
                "max_age_minutes": max_age_minutes_value,
                "details": details or {},
            }
        )

    def _failure_payload(
        name: str,
        exc: ModelDataError,
        *,
        default_reason_code: str,
        fallback_max_age_minutes: int | None,
    ) -> dict[str, Any]:
        reason_code = normalize_reason_code(str(exc.reason_code or "scenario_profile_unavailable"))
        detail_data = exc.details if isinstance(exc.details, dict) else {}
        as_of_text = str(
            detail_data.get("as_of_utc", detail_data.get("as_of", ""))
        ).strip()
        as_of_dt = _parse_iso_utc(as_of_text)
        effective_max_age = _resolve_max_age_minutes(detail_data, fallback_max_age_minutes)
        status = "stale" if as_of_dt is not None and isinstance(effective_max_age, int) and effective_max_age > 0 else "unavailable"
        dependency_summaries.append(
            {
                "name": name,
                "ok": False,
                "reason_code": reason_code or default_reason_code,
                "status": status,
                "source": detail_data.get("source"),
                "as_of_utc": as_of_dt.isoformat() if as_of_dt is not None else (as_of_text or None),
                "age_minutes": _age_minutes(as_of_dt),
                "max_age_minutes": effective_max_age,
                "details": detail_data,
            }
        )
        return {
            "ok": False,
            "status": status,
            "reason_code": reason_code or default_reason_code,
            "message": str(exc.message).strip() or reason_code,
            "as_of_utc": as_of_dt.isoformat() if as_of_dt is not None else (as_of_text or None),
            "age_minutes": _age_minutes(as_of_dt),
            "max_age_minutes": effective_max_age,
            "checked_at_utc": checked_at.isoformat(),
            "dependency": name,
            "dependencies": dependency_summaries,
        }

    try:
        try:
            profiles = _call_uncached(load_scenario_profiles)
            _append_success(
                "scenario_profiles",
                source=getattr(profiles, "source", None),
                as_of_text=(
                    getattr(profiles, "as_of_utc", None)
                    or getattr(profiles, "generated_at_utc", None)
                    or ""
                ),
                max_age_minutes_value=int(max_age_minutes),
                details={"contexts": len(getattr(profiles, "contexts", {}) or {})},
            )
        except ModelDataError as exc:
            return _failure_payload(
                "scenario_profiles",
                exc,
                default_reason_code="scenario_profile_unavailable",
                fallback_max_age_minutes=int(max_age_minutes),
            )

        fuel_max_age_minutes = int(settings.live_fuel_max_age_days) * 1440
        try:
            fuel_snapshot = _call_uncached(load_fuel_price_snapshot)
            _append_success(
                "fuel_snapshot",
                source=getattr(fuel_snapshot, "source", None),
                as_of_text=getattr(fuel_snapshot, "as_of", None),
                max_age_minutes_value=fuel_max_age_minutes,
                details={"signature_present": bool(getattr(fuel_snapshot, "signature", None))},
            )
        except ModelDataError as exc:
            return _failure_payload(
                "fuel_snapshot",
                exc,
                default_reason_code="fuel_price_source_unavailable",
                fallback_max_age_minutes=fuel_max_age_minutes,
            )

        toll_tariff_max_age_minutes = int(settings.live_toll_tariffs_max_age_days) * 1440
        try:
            toll_tariffs = _call_uncached(load_toll_tariffs)
            _append_success(
                "toll_tariffs",
                source=getattr(toll_tariffs, "source", None),
                max_age_minutes_value=toll_tariff_max_age_minutes,
                details={"rule_count": len(getattr(toll_tariffs, "rules", ()) or ())},
            )
        except ModelDataError as exc:
            return _failure_payload(
                "toll_tariffs",
                exc,
                default_reason_code="toll_tariff_unavailable",
                fallback_max_age_minutes=toll_tariff_max_age_minutes,
            )

        toll_topology_max_age_minutes = int(settings.live_toll_topology_max_age_days) * 1440
        try:
            toll_segments = _call_uncached(load_toll_segments_seed)
            _append_success(
                "toll_topology",
                max_age_minutes_value=toll_topology_max_age_minutes,
                details={"segment_count": len(toll_segments)},
            )
        except ModelDataError as exc:
            return _failure_payload(
                "toll_topology",
                exc,
                default_reason_code="toll_topology_unavailable",
                fallback_max_age_minutes=toll_topology_max_age_minutes,
            )

        stochastic_max_age_minutes = int(settings.live_stochastic_max_age_days) * 1440
        try:
            stochastic_regimes = _call_uncached(load_stochastic_regimes)
            _append_success(
                "stochastic_regimes",
                source=getattr(stochastic_regimes, "source", None),
                max_age_minutes_value=stochastic_max_age_minutes,
                details={"regime_count": len(getattr(stochastic_regimes, "regimes", {}) or {})},
            )
        except ModelDataError as exc:
            return _failure_payload(
                "stochastic_regimes",
                exc,
                default_reason_code="stochastic_regime_unavailable",
                fallback_max_age_minutes=stochastic_max_age_minutes,
            )

        departure_max_age_minutes = int(settings.live_departure_max_age_days) * 1440
        try:
            departure_profile = _call_uncached(load_departure_profile)
            contextual = getattr(departure_profile, "contextual", None)
            if isinstance(contextual, dict):
                region_count = len(contextual)
            else:
                profiles = getattr(departure_profile, "profiles", {})
                region_count = len(profiles) if isinstance(profiles, dict) else 0
            _append_success(
                "departure_profiles",
                source=getattr(departure_profile, "source", None),
                max_age_minutes_value=departure_max_age_minutes,
                details={"region_count": int(region_count)},
            )
        except ModelDataError as exc:
            return _failure_payload(
                "departure_profiles",
                exc,
                default_reason_code="departure_profile_unavailable",
                fallback_max_age_minutes=departure_max_age_minutes,
            )

        try:
            bank_holidays = _call_uncached(load_uk_bank_holidays)
            _append_success(
                "bank_holidays",
                max_age_minutes_value=departure_max_age_minutes,
                details={"count": len(bank_holidays)},
            )
        except ModelDataError as exc:
            return _failure_payload(
                "bank_holidays",
                exc,
                default_reason_code="bank_holidays_unavailable",
                fallback_max_age_minutes=departure_max_age_minutes,
            )

        freshest_candidates = [
            item
            for item in dependency_summaries
            if isinstance(item.get("age_minutes"), (int, float))
        ]
        least_fresh = max(freshest_candidates, key=lambda item: float(item["age_minutes"])) if freshest_candidates else None
        return {
            "ok": True,
            "status": "ok",
            "reason_code": "ok",
            "message": "Strict live runtime dependencies are ready.",
            "as_of_utc": least_fresh.get("as_of_utc") if isinstance(least_fresh, dict) else None,
            "age_minutes": least_fresh.get("age_minutes") if isinstance(least_fresh, dict) else None,
            "max_age_minutes": least_fresh.get("max_age_minutes") if isinstance(least_fresh, dict) else None,
            "checked_at_utc": checked_at.isoformat(),
            "dependency_count": len(dependency_summaries),
            "dependencies": dependency_summaries,
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
            "dependencies": dependency_summaries,
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
    payload["k_raw_cache"] = k_raw_cache_stats()
    payload["route_option_cache"] = route_option_cache_stats()
    payload["route_state_cache"] = route_state_cache_stats()
    payload["voi_dccs_cache"] = voi_dccs_cache_stats()
    return payload


@app.get("/cache/stats")
async def get_cache_stats() -> dict[str, dict[str, int]]:
    return {
        "route_cache": route_cache_stats(),
        "hot_rerun_route_cache_checkpoint": route_cache_checkpoint_stats(),
        "certification_cache": certification_cache_stats(),
        "k_raw_cache": k_raw_cache_stats(),
        "route_option_cache": route_option_cache_stats(),
        "route_state_cache": route_state_cache_stats(),
        "voi_dccs_cache": voi_dccs_cache_stats(),
    }


@app.delete("/cache")
async def delete_cache(
    scope: str = Query(default="full"),
    _: AdminAccessDep = None,
) -> dict[str, int]:
    t0 = time.perf_counter()
    has_error = False
    try:
        selected_scope = str(scope or "full").strip().lower()
        if selected_scope not in {"full", "thesis_cold", "hot_rerun_cold_source"}:
            raise HTTPException(status_code=400, detail="invalid_cache_scope")
        if selected_scope in {"full", "thesis_cold"}:
            clear_route_cache_checkpoint()
        elif selected_scope == "hot_rerun_cold_source":
            checkpoint_route_cache()
        cleared = clear_route_cache()
        cleared_certifications = 0
        if selected_scope != "hot_rerun_cold_source":
            cleared_certifications = clear_certification_cache()
        cleared_route_state = clear_route_state_cache()
        cleared_voi_dccs = clear_voi_dccs_cache()
        cleared_k_raw = 0
        cleared_route_option = 0
        if selected_scope == "full":
            cleared_k_raw = clear_k_raw_cache()
            cleared_route_option = clear_route_option_cache()
        log_event(
            "route_cache_clear",
            scope=selected_scope,
            cleared=cleared,
            cleared_k_raw=cleared_k_raw,
            cleared_route_option=cleared_route_option,
            cleared_certifications=cleared_certifications,
            cleared_route_state=cleared_route_state,
            cleared_voi_dccs=cleared_voi_dccs,
        )
        return {
            "cleared": cleared,
            "cleared_k_raw": cleared_k_raw,
            "cleared_route_option": cleared_route_option,
            "cleared_certifications": cleared_certifications,
            "cleared_route_state": cleared_route_state,
            "cleared_voi_dccs": cleared_voi_dccs,
        }
    except Exception:
        has_error = True
        raise
    finally:
        _record_endpoint_metric("cache_delete", t0, error=has_error)


@app.post("/cache/hot-rerun/restore")
async def restore_hot_rerun_cache(_: AdminAccessDep = None) -> dict[str, int]:
    t0 = time.perf_counter()
    has_error = False
    try:
        restored = restore_checkpointed_route_cache(clear_first=False)
        checkpoint_stats = route_cache_checkpoint_stats()
        log_event(
            "hot_rerun_route_cache_restore",
            restored=restored,
            checkpoint_size=int(checkpoint_stats.get("size", 0)),
        )
        return {
            "restored": restored,
            "checkpoint_size": int(checkpoint_stats.get("size", 0)),
        }
    except Exception:
        has_error = True
        raise
    finally:
        _record_endpoint_metric("cache_hot_rerun_restore_post", t0, error=has_error)


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
CANDIDATE_CACHE_SCHEMA_VERSION: int = 3
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
    elapsed_ms: float = 0.0


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
    graph_search_ms_supplemental: float = 0.0
    graph_k_raw_cache_hit: bool = False
    graph_low_ambiguity_fast_path: bool = False
    graph_supported_ambiguity_fast_fallback: bool = False
    graph_long_corridor_stress_probe: bool = False
    osrm_refine_ms: float = 0.0
    build_options_ms: float = 0.0
    refinement_policy: str = ""
    selected_candidate_count: int = 0
    selected_candidate_ids_json: str = "[]"
    diversity_collapse_detected: bool = False
    diversity_collapse_reason: str = ""
    raw_corridor_family_count: int = 0
    refined_corridor_family_count: int = 0
    supplemental_challenger_activated: bool = False
    supplemental_candidate_count: int = 0
    supplemental_selected_count: int = 0
    supplemental_budget_used: int = 0
    supplemental_sources_json: str = "[]"
    leftover_challenger_activated: bool = False
    leftover_challenger_candidate_count: int = 0
    leftover_challenger_selected_count: int = 0
    leftover_challenger_budget_used: int = 0
    preemptive_comparator_seed_activated: bool = False
    preemptive_comparator_candidate_count: int = 0
    preemptive_comparator_source_count: int = 0
    preemptive_comparator_sources_json: str = "[]"
    option_build_cache_hits: int = 0
    option_build_cache_misses: int = 0
    option_build_rebuild_count: int = 0
    option_build_reuse_rate: float = 0.0


def _is_toll_exclusion_label(label: str) -> bool:
    normalized = str(label or "").strip().lower()
    if not normalized or "exclude:" not in normalized:
        return False
    for match in re.finditer(r"(?:^|:)exclude:([^:]+)", normalized):
        excluded = [token.strip() for token in match.group(1).split(",") if token.strip()]
        if "toll" in excluded:
            return True
    return False


def _annotate_route_candidate_meta(
    route: dict[str, Any],
    *,
    source_labels: set[str],
    toll_exclusion_available: bool,
    observed_refine_cost_ms: float | None = None,
) -> None:
    def _coerce_cost_ms(value: Any) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return float("nan")
        return numeric if math.isfinite(numeric) else float("nan")

    existing_meta = route.get("_candidate_meta")
    existing_labels: set[str] = set()
    existing_cost_ms = float("nan")
    if isinstance(existing_meta, dict):
        raw_labels = existing_meta.get("source_labels")
        if isinstance(raw_labels, list):
            existing_labels = {str(label) for label in raw_labels if str(label).strip()}
        existing_cost_ms = _coerce_cost_ms(existing_meta.get("observed_refine_cost_ms"))
    combined_labels = existing_labels | {str(label) for label in source_labels if str(label).strip()}
    ordered_labels = sorted(combined_labels)
    seen_by_exclude_toll = any(_is_toll_exclusion_label(label) for label in ordered_labels)
    seen_by_non_exclude_toll = any(not _is_toll_exclusion_label(label) for label in ordered_labels)
    merged_cost_ms = _coerce_cost_ms(observed_refine_cost_ms)
    if math.isfinite(existing_cost_ms) and existing_cost_ms > 0.0:
        merged_cost_ms = (
            min(existing_cost_ms, merged_cost_ms)
            if math.isfinite(merged_cost_ms) and merged_cost_ms > 0.0
            else existing_cost_ms
        )
    if not (math.isfinite(merged_cost_ms) and merged_cost_ms > 0.0):
        merged_cost_ms = float("nan")

    candidate_meta = {
        "source_labels": ordered_labels,
        "seen_by_exclude_toll": seen_by_exclude_toll,
        "seen_by_non_exclude_toll": seen_by_non_exclude_toll,
        "toll_exclusion_available": bool(toll_exclusion_available),
    }
    if math.isfinite(merged_cost_ms):
        candidate_meta["observed_refine_cost_ms"] = round(float(merged_cost_ms), 6)
    route["_candidate_meta"] = candidate_meta


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


def _candidate_source_labels(route: Mapping[str, Any]) -> list[str]:
    meta = route.get("_candidate_meta")
    if not isinstance(meta, Mapping):
        return []
    raw_labels = meta.get("source_labels")
    if not isinstance(raw_labels, Sequence) or isinstance(raw_labels, (str, bytes)):
        return []
    labels = sorted({str(label).strip() for label in raw_labels if str(label).strip()})
    return labels


def _primary_candidate_source_label(source_labels: Sequence[str]) -> str | None:
    if not source_labels:
        return None

    def _priority(label: str) -> tuple[int, str]:
        normalized = str(label).strip().lower()
        if "direct_k_raw_fallback" in normalized:
            rank = 0
        elif "long_corridor_fallback" in normalized or "legacy_corridor_uniform_osrm_fallback" in normalized:
            rank = 1
        elif "supplemental_diversity_rescue" in normalized:
            rank = 2
        elif "preemptive_comparator_seed" in normalized:
            rank = 3
        else:
            rank = 4
        return rank, normalized

    ordered = sorted((str(label).strip() for label in source_labels if str(label).strip()), key=_priority)
    return ordered[0] if ordered else None


def _candidate_source_stage_from_labels(source_labels: Sequence[str]) -> str | None:
    normalized = [str(label).strip().lower() for label in source_labels if str(label).strip()]
    if not normalized:
        return None
    if any("preemptive_comparator_seed" in label for label in normalized):
        return "preemptive_comparator_seed"
    if any("supplemental_diversity_rescue" in label for label in normalized):
        return "supplemental_diversity_rescue"
    if any("direct_k_raw_fallback" in label for label in normalized):
        return "direct_k_raw_fallback"
    if any(
        "long_corridor_fallback" in label or "legacy_corridor_uniform_osrm_fallback" in label
        for label in normalized
    ):
        return "long_corridor_fallback"
    if any("osrm_refined" in label for label in normalized):
        return "osrm_refined"
    return None


def _candidate_source_engine_from_labels(source_labels: Sequence[str]) -> str | None:
    normalized = [str(label).strip().lower() for label in source_labels if str(label).strip()]
    if not normalized:
        return None
    if any("local_ors_seed" in label for label in normalized):
        return "ors_local_seed"
    if any("local_ors" in label for label in normalized):
        return "ors_local"
    if normalized:
        return "osrm"
    return None


def _selected_route_source_label(source_labels: Sequence[str]) -> str | None:
    if not source_labels:
        return None
    ordered = [str(label).strip() for label in source_labels if str(label).strip()]
    if not ordered:
        return None

    def _matches(fragment: str) -> str | None:
        for label in ordered:
            if fragment in label.lower():
                return label
        return None

    for fragment in (
        "preemptive_comparator_seed",
        "supplemental_diversity_rescue",
        "osrm_refined",
        "strict_fallback",
        "graph_path:",
        "direct_k_raw_fallback",
        "long_corridor_fallback",
        "legacy_corridor_uniform_osrm_fallback",
    ):
        matched = _matches(fragment)
        if matched is not None:
            return matched
    return ordered[0]


def _selected_route_source_stage_from_labels(source_labels: Sequence[str]) -> str | None:
    normalized = [str(label).strip().lower() for label in source_labels if str(label).strip()]
    if not normalized:
        return None
    if any("preemptive_comparator_seed" in label for label in normalized):
        return "preemptive_comparator_seed"
    if any("supplemental_diversity_rescue" in label for label in normalized):
        return "supplemental_diversity_rescue"
    if any("osrm_refined" in label for label in normalized):
        return "osrm_refined"
    if any("strict_fallback" in label or "graph_path:" in label for label in normalized):
        return "graph_native"
    if any("direct_k_raw_fallback" in label for label in normalized):
        return "direct_k_raw_fallback"
    if any(
        "long_corridor_fallback" in label or "legacy_corridor_uniform_osrm_fallback" in label
        for label in normalized
    ):
        return "long_corridor_fallback"
    return None


def _selected_route_source_engine_from_labels(source_labels: Sequence[str]) -> str | None:
    normalized = [str(label).strip().lower() for label in source_labels if str(label).strip()]
    if not normalized:
        return None
    if any("preemptive_comparator_seed" in label for label in normalized):
        if any("local_ors_seed" in label for label in normalized):
            return "ors_local_seed"
        if any("local_ors" in label for label in normalized):
            return "ors_local"
        return "osrm"
    if any("supplemental_diversity_rescue" in label for label in normalized):
        if any("local_ors_seed" in label for label in normalized):
            return "ors_local_seed"
        if any("local_ors" in label for label in normalized):
            return "ors_local"
        return "osrm"
    if any("osrm_refined" in label for label in normalized):
        return "internal"
    if any("strict_fallback" in label or "graph_path:" in label for label in normalized):
        return "internal"
    if any("local_ors_seed" in label for label in normalized):
        return "ors_local_seed"
    if any("local_ors" in label for label in normalized):
        return "ors_local"
    if normalized:
        return "osrm"
    return None


def _selected_route_source_payload(route: Mapping[str, Any]) -> dict[str, str | None]:
    labels = _candidate_source_labels(route)
    return {
        "source_label": _selected_route_source_label(labels),
        "source_stage": _selected_route_source_stage_from_labels(labels),
        "source_engine": _selected_route_source_engine_from_labels(labels),
    }


def _candidate_source_bucket(
    *,
    source_label: str | None,
    source_stage: str | None,
) -> str:
    normalized_label = str(source_label or "").strip().lower()
    normalized_stage = str(source_stage or "").strip().lower()
    if normalized_stage == "preemptive_comparator_seed":
        return "preemptive"
    if normalized_stage == "supplemental_diversity_rescue":
        return "supplemental"
    if "exclude:" in normalized_label:
        if "toll" in normalized_label:
            return "exclude_toll"
        if "motorway" in normalized_label:
            return "exclude_motorway"
        if "trunk" in normalized_label:
            return "exclude_trunk"
        return "exclude_other"
    if ":via:" in normalized_label or normalized_label.startswith("via:"):
        return "via"
    if "alternatives" in normalized_label:
        return "alternatives"
    if "local_ors" in normalized_label:
        return "ors_seed"
    if normalized_stage == "direct_k_raw_fallback":
        return "fallback_direct"
    if normalized_stage == "long_corridor_fallback":
        return "fallback_family"
    return "graph"


def _route_speed_proxy_mix(route: Mapping[str, Any]) -> dict[str, float] | None:
    try:
        seg_d_m, seg_t_s = extract_segment_annotations(dict(route))
    except OSRMError:
        return None
    totals = {
        "motorway_share": 0.0,
        "a_road_share": 0.0,
        "urban_share": 0.0,
        "other_share": 0.0,
    }
    observed_distance_m = 0.0
    for distance_m, duration_s in zip(seg_d_m, seg_t_s):
        if distance_m <= 1.0 or duration_s <= 0.1:
            continue
        observed_distance_m += float(distance_m)
        speed_kph = (float(distance_m) / float(duration_s)) * 3.6
        if speed_kph >= 82.0:
            bucket = "motorway_share"
        elif speed_kph >= 58.0:
            bucket = "a_road_share"
        elif speed_kph <= 32.0:
            bucket = "urban_share"
        else:
            bucket = "other_share"
        totals[bucket] += float(distance_m)
    if observed_distance_m <= 0.0:
        return None
    return {
        key: round(max(0.0, float(value)) / observed_distance_m, 6)
        for key, value in totals.items()
    }


def _route_speed_profile_features(route: Mapping[str, Any]) -> dict[str, float]:
    try:
        seg_d_m, seg_t_s = extract_segment_annotations(dict(route))
    except OSRMError:
        return {
            "speed_variability": 0.0,
            "slow_segment_share": 0.0,
        }

    weighted_speed_numerator = 0.0
    total_distance_m = 0.0
    weighted_speeds: list[tuple[float, float]] = []
    for distance_m, duration_s in zip(seg_d_m, seg_t_s):
        if distance_m <= 1.0 or duration_s <= 0.1:
            continue
        speed_mps = float(distance_m) / float(duration_s)
        weighted_speeds.append((float(distance_m), speed_mps))
        total_distance_m += float(distance_m)
        weighted_speed_numerator += float(distance_m) * speed_mps

    if total_distance_m <= 0.0 or not weighted_speeds:
        return {
            "speed_variability": 0.0,
            "slow_segment_share": 0.0,
        }

    mean_speed_mps = weighted_speed_numerator / total_distance_m
    weighted_variance = (
        sum(distance_m * ((speed_mps - mean_speed_mps) ** 2) for distance_m, speed_mps in weighted_speeds)
        / total_distance_m
    )
    coefficient_of_variation = math.sqrt(max(0.0, weighted_variance)) / max(1e-6, mean_speed_mps)
    slow_segment_share = (
        sum(distance_m for distance_m, speed_mps in weighted_speeds if speed_mps < (0.78 * mean_speed_mps))
        / total_distance_m
    )
    return {
        "speed_variability": round(max(0.0, min(1.0, coefficient_of_variation / 0.45)), 6),
        "slow_segment_share": round(max(0.0, min(1.0, slow_segment_share)), 6),
    }


def _segment_heading_deg(
    *,
    lon_a: float,
    lat_a: float,
    lon_b: float,
    lat_b: float,
) -> float:
    phi_a = math.radians(float(lat_a))
    phi_b = math.radians(float(lat_b))
    delta_lon = math.radians(float(lon_b) - float(lon_a))
    y = math.sin(delta_lon) * math.cos(phi_b)
    x = (
        math.cos(phi_a) * math.sin(phi_b)
        - math.sin(phi_a) * math.cos(phi_b) * math.cos(delta_lon)
    )
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def _heading_delta_deg(a: float, b: float) -> float:
    delta = abs(float(a) - float(b)) % 360.0
    return 360.0 - delta if delta > 180.0 else delta


def _route_shape_profile_features(
    route: Mapping[str, Any],
    *,
    route_distance_km: float,
    straight_line_km: float,
) -> dict[str, float]:
    detour_ratio = max(0.0, (float(route_distance_km) / max(0.001, float(straight_line_km))) - 1.0)
    try:
        coords = _validate_osrm_geometry(dict(route))
    except OSRMError:
        return {
            "shape_bend_density": 0.0,
            "shape_detour_factor": round(max(0.0, min(1.0, detour_ratio / 0.55)), 6),
        }

    sampled_coords: list[tuple[float, float]] = []
    sample_step = max(1, len(coords) // 48)
    for idx in range(0, len(coords), sample_step):
        sampled_coords.append(coords[idx])
    if sampled_coords[-1] != coords[-1]:
        sampled_coords.append(coords[-1])
    coords = sampled_coords

    total_heading_delta = 0.0
    for idx in range(2, len(coords)):
        lon_a, lat_a = coords[idx - 2]
        lon_b, lat_b = coords[idx - 1]
        lon_c, lat_c = coords[idx]
        seg_ab_m = _haversine_segment_m(lat_a=lat_a, lon_a=lon_a, lat_b=lat_b, lon_b=lon_b)
        seg_bc_m = _haversine_segment_m(lat_a=lat_b, lon_a=lon_b, lat_b=lat_c, lon_b=lon_c)
        if seg_ab_m <= 1.0 or seg_bc_m <= 1.0:
            continue
        heading_ab = _segment_heading_deg(lon_a=lon_a, lat_a=lat_a, lon_b=lon_b, lat_b=lat_b)
        heading_bc = _segment_heading_deg(lon_a=lon_b, lat_a=lat_b, lon_b=lon_c, lat_b=lat_c)
        total_heading_delta += _heading_delta_deg(heading_ab, heading_bc)

    bend_density = total_heading_delta / max(120.0, float(route_distance_km) * 6.0)
    return {
        "shape_bend_density": round(max(0.0, min(1.0, bend_density)), 6),
        "shape_detour_factor": round(max(0.0, min(1.0, detour_ratio / 0.55)), 6),
    }


def _fallback_source_feature_map(source_labels: Sequence[str]) -> dict[str, float]:
    normalized = [str(label).strip().lower() for label in source_labels if str(label).strip()]
    if not normalized:
        return {}
    return {
        "source_via_hint": 1.0 if any(":via:" in label or label.startswith("via:") for label in normalized) else 0.0,
        "source_alternatives_hint": 1.0 if any("alternatives" in label for label in normalized) else 0.0,
        "source_exclude_toll_hint": 1.0 if any(_is_toll_exclusion_label(label) for label in normalized) else 0.0,
        "source_exclude_motorway_hint": (
            1.0
            if any("exclude:" in label and ("motorway" in label or "trunk" in label) for label in normalized)
            else 0.0
        ),
        "source_local_ors_hint": 1.0 if any("local_ors" in label for label in normalized) else 0.0,
    }


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


def _coerce_utc_timestamp(value: Any) -> str | None:
    if isinstance(value, datetime):
        dt = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")
    text = str(value or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    else:
        dt = dt.astimezone(UTC)
    return dt.isoformat().replace("+00:00", "Z")


def _path_mtime_utc(path: Path) -> str | None:
    try:
        if not path.exists():
            return None
        return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat().replace("+00:00", "Z")
    except OSError:
        return None


def _repo_local_evidence_timestamp(*, family: str, source: str) -> str | None:
    backend_root = Path(__file__).resolve().parents[1]
    asset_root = backend_root / "assets" / "uk"
    model_root = backend_root / "out" / "model_assets"
    source_name = str(source or "").strip().lower()
    candidates: list[Path] = []
    if family == "scenario" or "scenario_profiles" in source_name:
        candidates.extend(
            [
                asset_root / "scenario_profiles_uk.json",
                model_root / "scenario_profiles_uk.json",
            ]
        )
    elif family == "toll" or "toll_tariffs" in source_name or "toll_topology" in source_name:
        candidates.extend(
            [
                asset_root / "toll_tariffs_uk.json",
                asset_root / "toll_topology_uk.json",
                model_root / "toll_tariffs_uk_compiled.json",
                model_root / "toll_segments_seed_compiled.json",
            ]
        )
    elif family == "terrain" or source_name in {"dem_real", "manifest_asset"}:
        candidates.extend(
            [
                model_root / "terrain" / "terrain_manifest.json",
                asset_root / "terrain" / "terrain_manifest.json",
            ]
        )
    elif family == "fuel" or "fuel_prices" in source_name:
        candidates.extend(
            [
                asset_root / "fuel_prices_uk.json",
                model_root / "fuel_prices_uk_compiled.json",
            ]
        )
    elif family == "carbon" or "carbon" in source_name:
        candidates.extend(
            [
                asset_root / "carbon_price_schedule_uk.json",
                model_root / "carbon_price_schedule_uk.json",
            ]
        )
    elif family == "stochastic" or "stochastic" in source_name:
        candidates.extend(
            [
                asset_root / "stochastic_regimes_uk.json",
                model_root / "stochastic_regimes_uk.json",
                model_root / "stochastic_regimes_uk_compiled.json",
            ]
        )
    for candidate in candidates:
        timestamp = _path_mtime_utc(candidate)
        if timestamp:
            return timestamp
    return None


def _resolve_evidence_timestamp(
    *,
    family: str,
    source: str,
    explicit: Any = None,
    departure_time_utc: datetime | None = None,
) -> str | None:
    explicit_ts = _coerce_utc_timestamp(explicit)
    if explicit_ts:
        return explicit_ts
    repo_local_ts = _repo_local_evidence_timestamp(family=family, source=source)
    if repo_local_ts:
        return repo_local_ts
    if family == "weather":
        return _coerce_utc_timestamp(departure_time_utc or datetime.now(UTC))
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return _coerce_utc_timestamp(departure_time_utc or datetime.now(UTC))
    return None


def _scenario_policy_cache_key(
    scenario_mode: ScenarioMode,
    scenario_context: ScenarioRouteContext,
) -> tuple[str, str]:
    return (scenario_mode.value, scenario_context.context_key)


def _resolve_route_scenario_policy(
    *,
    scenario_mode: ScenarioMode,
    scenario_context: ScenarioRouteContext,
    scenario_policy_cache: dict[tuple[str, str], Any] | None = None,
) -> Any:
    scenario_policy_cache_local: dict[tuple[str, str], Any] = (
        scenario_policy_cache if scenario_policy_cache is not None else {}
    )
    cache_key = _scenario_policy_cache_key(scenario_mode, scenario_context)
    scenario_policy = scenario_policy_cache_local.get(cache_key)
    if scenario_policy is None:
        scenario_policy = resolve_scenario_profile(
            scenario_mode,
            context=scenario_context,
        )
        scenario_policy_cache_local[cache_key] = scenario_policy
    return scenario_policy


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
    optimization_mode: OptimizationMode = "expected_value",
    pareto_method: ParetoMethod = "dominance",
    epsilon: EpsilonConstraints | None = None,
    max_alternatives: int | None = None,
    scenario_policy_cache: dict[tuple[str, str], Any] | None = None,
    reset_terrain_route_run: bool = True,
    lightweight: bool = False,
    force_uncertainty: bool = False,
) -> RouteOption:
    def _uncertainty_stochastic_config() -> tuple[StochasticConfig, bool, int]:
        cfg = stochastic or StochasticConfig()
        requested_samples = max(8, int(cfg.samples))
        if not build_lightweight or not bool(cfg.enabled):
            return cfg, False, requested_samples
        capped_samples = min(
            requested_samples,
            max(16, min(24, int(math.ceil(requested_samples * 0.5)))),
        )
        if capped_samples >= requested_samples:
            return cfg, False, requested_samples
        return cfg.model_copy(update={"samples": capped_samples}), True, requested_samples

    build_lightweight = bool(lightweight)
    uncertainty_required = bool(force_uncertainty) or not build_lightweight
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
    scenario_policy_cache_local: dict[tuple[str, str], Any] = (
        scenario_policy_cache if scenario_policy_cache is not None else {}
    )
    core_cache_key = build_route_option_core_cache_key(
        route,
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
        optimization_mode=optimization_mode,
        pareto_method=pareto_method,
        epsilon=epsilon,
        max_alternatives=None if max_alternatives is None else int(max_alternatives),
    )
    if core_cache_key is not None:
        cached_core = get_cached_route_option_core(core_cache_key)
        if cached_core is not None:
            cached_option = cached_core.option.model_copy(update={"id": option_id}, deep=True)
            if build_lightweight:
                return _route_option_lightweight_copy(cached_option)
            if _option_has_full_route_details(cached_option):
                return cached_option
    road_class_counts = _route_road_class_counts(route)
    scenario_context = build_scenario_route_context(
        route_points=route_points_lat_lon,
        road_class_counts=road_class_counts,
        vehicle_class=str(vehicle.vehicle_class),
        departure_time_utc=departure_time_utc,
        weather_bucket=(weather_cfg.profile if weather_cfg.enabled else "clear"),
    )
    scenario_policy = _resolve_route_scenario_policy(
        scenario_mode=scenario_mode,
        scenario_context=scenario_context,
        scenario_policy_cache=scenario_policy_cache_local,
    )
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
        probe_segment_boundaries=not build_lightweight,
    )
    max_segments = max(32, int(settings.route_option_segment_cap))
    long_threshold_km = max(20.0, float(settings.route_option_long_distance_threshold_km))
    if build_lightweight:
        # Lightweight builds only need route-ranking aggregates, so we can
        # coarsen the segment economics loop much more aggressively here
        # without affecting the fully hydrated frontier/display route objects.
        # Keep a higher floor on long-haul routes so lean-mode evaluation does
        # not collapse them to a trivially small bucket count.
        lightweight_cap = max(8, int(math.ceil(len(seg_t_s) / 8.0)))
        if distance_km >= long_threshold_km:
            lightweight_cap = max(
                lightweight_cap,
                max(16, int(math.ceil(float(settings.route_option_segment_cap_long) / 2.0))),
            )
        max_segments = min(max_segments, lightweight_cap)
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

    segment_breakdown_rows: list[dict[str, float | int]] = []
    segment_count = 0
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
        segment_count += 1
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
        if not build_lightweight:
            segment_breakdown_rows.append(
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
    segment_breakdown = (
        _compact_segment_breakdown(
            segment_count=segment_count,
            distance_km=total_distance_km,
            duration_s=total_duration_s,
            toll_cost=total_toll_cost,
            fuel_cost=total_fuel_cost,
            carbon_cost=total_carbon_cost,
        )
        if build_lightweight
        else segment_breakdown_rows
    )

    eta_explanations: list[str] = []
    eta_timeline: list[dict[str, float | str]] = []
    counterfactuals: list[dict[str, str | float | bool]] = []
    if not build_lightweight:
        eta_explanations = [f"Baseline ETA {base_duration_s / 60.0:.1f} min."]
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

        eta_timeline = [
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
        shifted_mode_policy = _resolve_route_scenario_policy(
            scenario_mode=shifted_mode,
            scenario_context=scenario_context,
            scenario_policy_cache=scenario_policy_cache_local,
        )
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

        counterfactuals = [
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
        option_weather_summary["fuel_cost_total_gbp"] = round(total_fuel_cost, 6)
        option_weather_summary["time_cost_total_gbp"] = round(total_time_cost, 6)
        option_weather_summary["toll_cost_total_gbp"] = round(total_toll_cost, 6)
        option_weather_summary["carbon_cost_total_gbp"] = round(total_carbon_cost, 6)
        option_weather_summary["fuel_liters_total"] = round(total_fuel_liters, 6)
        option_weather_summary["fuel_liters_p10"] = round(total_fuel_liters_p10, 6)
        option_weather_summary["fuel_liters_p50"] = round(total_fuel_liters_p50, 6)
        option_weather_summary["fuel_liters_p90"] = round(total_fuel_liters_p90, 6)
        option_weather_summary["fuel_cost_p10_gbp"] = round(total_fuel_cost_p10, 6)
        option_weather_summary["fuel_cost_p50_gbp"] = round(total_fuel_cost_p50, 6)
        option_weather_summary["fuel_cost_p90_gbp"] = round(total_fuel_cost_p90, 6)
        option_weather_summary["emissions_uncertainty_low_kg"] = round(total_emissions_uncertainty_low, 6)
        option_weather_summary["emissions_uncertainty_high_kg"] = round(total_emissions_uncertainty_high, 6)
        if not build_lightweight:
            fuel_cal = load_fuel_consumption_calibration()
            option_weather_summary["consumption_model_source"] = fuel_cal.source
            option_weather_summary["consumption_model_version"] = fuel_cal.version
            if fuel_cal.as_of_utc is not None:
                option_weather_summary["consumption_model_as_of_utc"] = fuel_cal.as_of_utc
            option_weather_summary["vehicle_profile_id"] = vehicle.id
            option_weather_summary["vehicle_profile_version"] = int(vehicle.schema_version)
            option_weather_summary["vehicle_profile_source"] = vehicle.profile_source

    terrain_diag_meta: dict[str, Any] | None = None
    if not build_lightweight:
        terrain_diag_meta = terrain_live_diagnostics()

    evidence_records: list[EvidenceSourceRecord] = []

    evidence_records.append(
        EvidenceSourceRecord(
            family="scenario",
            source=str(scenario_policy.source or "unknown"),
            active=True,
            freshness_timestamp_utc=_resolve_evidence_timestamp(
                family="scenario",
                source=str(scenario_policy.source or "unknown"),
                explicit=scenario_policy.live_as_of_utc or scenario_policy.as_of_utc,
                departure_time_utc=departure_time_utc,
            ),
            max_age_minutes=float(getattr(settings, "live_scenario_coefficient_max_age_minutes", 0)),
            signature=str(scenario_policy.version or ""),
            confidence=max(0.0, min(1.0, float(scenario_policy.live_coverage.get("overall", 1.0))))
            if scenario_policy.live_coverage
            else None,
            fallback_used=bool("fallback" in str(scenario_policy.source or "").lower()),
            details={
                "calibration_basis": str(scenario_policy.calibration_basis or ""),
                "mode_observation_source": str(scenario_policy.mode_observation_source or ""),
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
            freshness_timestamp_utc=_resolve_evidence_timestamp(
                family="toll",
                source=str(toll_result.source or "unknown"),
                explicit=toll_result.details.get("as_of_utc") if isinstance(toll_result.details, dict) else None,
                departure_time_utc=departure_time_utc,
            ),
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
                "tariff_rule_ids": str(toll_result.details.get("tariff_rule_ids", "") or ""),
                "matched_asset_ids": str(toll_result.details.get("matched_asset_ids", "") or ""),
            },
        )
    )
    evidence_records.append(
        EvidenceSourceRecord(
            family="terrain",
            source=str(terrain_summary.source if terrain_summary is not None else "missing"),
            active=terrain_summary is not None,
            freshness_timestamp_utc=_resolve_evidence_timestamp(
                family="terrain",
                source=str(terrain_summary.source if terrain_summary is not None else "missing"),
                explicit=terrain_diag_meta.get("as_of_utc") if terrain_diag_meta is not None else None,
                departure_time_utc=departure_time_utc,
            ),
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
            freshness_timestamp_utc=_resolve_evidence_timestamp(
                family="fuel",
                source=str(fuel_price_source or "unknown"),
                explicit=fuel_price_as_of,
                departure_time_utc=departure_time_utc,
            ),
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
            freshness_timestamp_utc=_resolve_evidence_timestamp(
                family="carbon",
                source=str(carbon_context.source or "unknown"),
                explicit=getattr(carbon_context, "as_of_utc", None),
                departure_time_utc=departure_time_utc,
            ),
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
            freshness_timestamp_utc=_resolve_evidence_timestamp(
                family="weather",
                source=str(option_weather_summary.get("profile", "clear") if option_weather_summary else "clear"),
                explicit=(str(option_weather_summary.get("observed_at_utc")) if option_weather_summary else None),
                departure_time_utc=departure_time_utc,
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

    option_uncertainty: dict[str, float] | None = None
    option_uncertainty_meta: dict[str, str | float | int | bool] | None = None
    if uncertainty_required:
        uncertainty_stochastic, lightweight_uncertainty_sample_cap_applied, lightweight_uncertainty_requested_samples = (
            _uncertainty_stochastic_config()
        )
        option_uncertainty, option_uncertainty_meta = _route_stochastic_uncertainty(
            option,
            stochastic=uncertainty_stochastic,
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
        option_uncertainty_meta["lightweight_uncertainty_sample_cap_applied"] = bool(
            build_lightweight and lightweight_uncertainty_sample_cap_applied
        )
        option_uncertainty_meta["lightweight_uncertainty_requested_samples"] = int(
            lightweight_uncertainty_requested_samples
        )

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
    if core_cache_key is not None and bool(settings.route_option_cache_enabled):
        set_cached_route_option_core(
            core_cache_key,
            CachedRouteOptionCore(
                option=option.model_copy(deep=True),
                estimated_build_ms=0.0,
            ),
        )
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
    apply_realism_multipliers: bool = True,
) -> RouteOption:
    coords_raw = _validate_osrm_geometry(route)
    coords = _downsample_coords(coords_raw)
    seg_d_m, seg_t_s = extract_segment_annotations(route)
    baseline_distance_multiplier = 1.0
    if apply_realism_multipliers:
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
    baseline_duration_multiplier = 1.0
    if apply_realism_multipliers:
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


async def _fetch_repo_local_ors_baseline_seed(
    *,
    req: RouteRequest,
    osrm: OSRMClient,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if req.waypoints:
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
            raise RuntimeError("Repo-local secondary baseline waypoint realization returned no route.")
        return routes[0], {
            "provider_mode": "repo_local",
            "baseline_policy": "waypoint_direct_realization",
            "selected_candidate_rank": 0,
            "selected_distinct_corridor": False,
            "graph_candidate_count": 0,
        }

    od_straight_line_km = _od_haversine_km(req.origin, req.destination)
    long_corridor_thresholds = [
        float(value)
        for value in (
            settings.route_graph_fast_startup_long_corridor_bypass_km,
            settings.route_graph_long_corridor_threshold_km,
        )
        if float(value) > 0.0
    ]
    long_corridor_threshold_km = min(long_corridor_thresholds) if long_corridor_thresholds else 150.0
    is_long_corridor = od_straight_line_km >= long_corridor_threshold_km

    if is_long_corridor:
        alternative_budget = 2 if od_straight_line_km >= (long_corridor_threshold_km * 1.5) else 3
        long_candidates = await osrm.fetch_routes(
            origin_lat=req.origin.lat,
            origin_lon=req.origin.lon,
            dest_lat=req.destination.lat,
            dest_lon=req.destination.lon,
            alternatives=alternative_budget,
        )
        direct_route = long_candidates[0] if long_candidates else None
        if direct_route is None:
            routes = await osrm.fetch_routes(
                origin_lat=req.origin.lat,
                origin_lon=req.origin.lon,
                dest_lat=req.destination.lat,
                dest_lon=req.destination.lon,
                alternatives=False,
            )
            if not routes:
                raise RuntimeError("Repo-local long-corridor secondary baseline returned no route.")
            direct_route = routes[0]
        selected_long_route, selected_long_meta = _select_secondary_baseline_candidate(
            direct_route=direct_route,
            candidates=[
                candidate
                for candidate in long_candidates[1:]
                if _route_signature(candidate) != _route_signature(direct_route)
            ],
        )
        if selected_long_route is not None:
            return selected_long_route, {
                "provider_mode": "repo_local",
                "baseline_policy": "long_corridor_osrm_alternative_min_overlap",
                "od_straight_line_km": round(od_straight_line_km, 3),
                "degraded_long_corridor": True,
                "alternative_budget": int(alternative_budget),
                **selected_long_meta,
            }
        return direct_route, {
            "provider_mode": "repo_local",
            "baseline_policy": "long_corridor_direct_fallback",
            "selected_candidate_rank": 0,
            "selected_distinct_corridor": False,
            "graph_candidate_count": 0,
            "direct_overlap_ratio": 1.0,
            "secondary_quality_score": 0.0,
            "od_straight_line_km": round(od_straight_line_km, 3),
            "degraded_long_corridor": True,
            "alternative_budget": int(alternative_budget),
        }

    direct_candidates = await osrm.fetch_routes(
        origin_lat=req.origin.lat,
        origin_lon=req.origin.lon,
        dest_lat=req.destination.lat,
        dest_lon=req.destination.lon,
        alternatives=False,
    )
    direct_route = direct_candidates[0] if direct_candidates else None
    if direct_route is None:
        routes = await osrm.fetch_routes(
            origin_lat=req.origin.lat,
            origin_lon=req.origin.lon,
            dest_lat=req.destination.lat,
            dest_lon=req.destination.lon,
            alternatives=False,
        )
        if not routes:
            raise RuntimeError("Repo-local secondary baseline corridor search returned no route.")
        direct_route = routes[0]

    graph_candidates: list[dict[str, Any]] = []
    try:
        graph_routes, _search_state = await _route_graph_candidate_routes_async(
            origin_lat=float(req.origin.lat),
            origin_lon=float(req.origin.lon),
            destination_lat=float(req.destination.lat),
            destination_lon=float(req.destination.lon),
            max_paths=max(4, min(10, int(req.max_alternatives) * 3)),
        )
        graph_candidates = _select_ranked_candidate_routes(
            graph_routes,
            max_routes=max(3, min(6, int(req.max_alternatives) + 2)),
        )
    except Exception:
        graph_candidates = []

    realized_graph_candidates: list[dict[str, Any]] = []
    direct_signature = _route_signature(direct_route)
    for graph_candidate in graph_candidates:
        via = _graph_family_via_points(
            graph_candidate,
            max_landmarks=max(2, int(settings.route_graph_via_landmarks_per_path)),
        )
        if not via:
            continue
        realized_routes = await osrm.fetch_routes(
            origin_lat=req.origin.lat,
            origin_lon=req.origin.lon,
            dest_lat=req.destination.lat,
            dest_lon=req.destination.lon,
            alternatives=False,
            via=via,
        )
        if not realized_routes:
            continue
        realized = realized_routes[0]
        if _route_signature(realized) == direct_signature:
            continue
        realized_graph_candidates.append(realized)

    selected_graph_route, selected_graph_meta = _select_secondary_baseline_candidate(
        direct_route=direct_route,
        candidates=realized_graph_candidates,
    )
    if selected_graph_route is not None:
        return selected_graph_route, {
            "provider_mode": "repo_local",
            "baseline_policy": "graph_realized_min_overlap",
            "od_straight_line_km": round(od_straight_line_km, 3),
            "degraded_long_corridor": False,
            **selected_graph_meta,
        }

    alternative_budget = max(2, min(6, int(req.max_alternatives) + 2))
    fallback_candidates = await osrm.fetch_routes(
        origin_lat=req.origin.lat,
        origin_lon=req.origin.lon,
        dest_lat=req.destination.lat,
        dest_lon=req.destination.lon,
        alternatives=alternative_budget,
    )
    ranked = _select_ranked_candidate_routes(
        fallback_candidates,
        max_routes=max(2, min(4, alternative_budget)),
    )
    selected_fallback_route, selected_fallback_meta = _select_secondary_baseline_candidate(
        direct_route=direct_route,
        candidates=[candidate for candidate in ranked if _route_signature(candidate) != direct_signature],
    )
    if selected_fallback_route is not None:
        return selected_fallback_route, {
            "provider_mode": "repo_local",
            "baseline_policy": "osrm_alternative_min_overlap",
            "od_straight_line_km": round(od_straight_line_km, 3),
            "degraded_long_corridor": False,
            **selected_fallback_meta,
        }

    via_candidates = _candidate_via_points(req.origin, req.destination)
    max_via_trials = min(len(via_candidates), 6)
    via_realizations: list[dict[str, Any]] = []
    for via_point in via_candidates[:max_via_trials]:
        realized_routes = await osrm.fetch_routes(
            origin_lat=req.origin.lat,
            origin_lon=req.origin.lon,
            dest_lat=req.destination.lat,
            dest_lon=req.destination.lon,
            alternatives=False,
            via=[via_point],
        )
        if not realized_routes:
            continue
        realized = realized_routes[0]
        if _route_signature(realized) == direct_signature:
            continue
        via_realizations.append(realized)

    selected_via_route, selected_via_meta = _select_secondary_baseline_candidate(
        direct_route=direct_route,
        candidates=via_realizations,
    )
    if selected_via_route is not None:
        return selected_via_route, {
            "provider_mode": "repo_local",
            "baseline_policy": "heuristic_via_min_overlap",
            "od_straight_line_km": round(od_straight_line_km, 3),
            "degraded_long_corridor": False,
            **selected_via_meta,
        }

    return direct_route, {
        "provider_mode": "repo_local",
        "baseline_policy": "direct_fallback",
        "selected_candidate_rank": 0,
        "selected_distinct_corridor": False,
        "graph_candidate_count": len(graph_candidates),
        "direct_overlap_ratio": 1.0,
        "secondary_quality_score": 0.0,
        "od_straight_line_km": round(od_straight_line_km, 3),
        "degraded_long_corridor": False,
    }


def _ors_profile_for_vehicle(vehicle_type: str) -> tuple[str, str | None]:
    normalized = str(vehicle_type or "").strip().lower()
    if normalized in {"rigid_hgv", "artic_hgv", "hgv", "truck"}:
        return str(settings.ors_directions_profile_hgv or "driving-hgv"), "hgv"
    return str(settings.ors_directions_profile_default or "driving-car"), None


async def _fetch_local_ors_baseline_seed(
    *,
    req: RouteRequest,
    ors: ORSClient,
) -> tuple[dict[str, Any], dict[str, Any]]:
    coordinates = [(float(req.origin.lon), float(req.origin.lat))]
    coordinates.extend((float(waypoint.lon), float(waypoint.lat)) for waypoint in (req.waypoints or []))
    coordinates.append((float(req.destination.lon), float(req.destination.lat)))
    profile, vehicle_type = _ors_profile_for_vehicle(str(req.vehicle_type or ""))
    ors_route = await ors.fetch_route(
        coordinates_lon_lat=coordinates,
        profile=profile,
        vehicle_type=vehicle_type,
    )
    route = _build_osrm_like_route_from_polyline(
        coordinates_lon_lat=ors_route.coordinates_lon_lat,
        distance_m=ors_route.distance_m,
        duration_s=ors_route.duration_s,
    )
    engine_manifest = local_ors_runtime_manifest(
        base_url=str(getattr(ors, "base_url", "") or ""),
        profile=profile,
        vehicle_type=vehicle_type,
    )
    return route, {
        "provider_mode": "local_service",
        "baseline_policy": "engine_shortest_path",
        "engine": "openrouteservice",
        "engine_service_base_url": str(getattr(ors, "base_url", "") or ""),
        "engine_profile": profile,
        "engine_vehicle_type": vehicle_type,
        "asset_manifest_hash": str(engine_manifest.get("manifest_hash") or ""),
        "asset_recorded_at": str(engine_manifest.get("recorded_at") or datetime.now(UTC).isoformat()),
        "asset_freshness_status": str(engine_manifest.get("identity_status") or "graph_identity_unknown"),
        "engine_manifest": engine_manifest,
        "selected_candidate_rank": 0,
        "selected_distinct_corridor": False,
        "graph_candidate_count": 0,
        "direct_overlap_ratio": 1.0,
        "secondary_quality_score": 1.0,
    }


def _route_signature(route: dict[str, Any]) -> str:
    coords = _validate_osrm_geometry(route)
    n = len(coords)
    step = max(1, n // 30)
    sample = coords[::step][:40]

    # round for stability; avoid huge hash variability
    parts = [f"{lon:.4f},{lat:.4f}" for lon, lat in sample]
    s = "|".join(parts)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _route_option_signature(option: RouteOption) -> str:
    coords = list(option.geometry.coordinates)
    if not coords:
        return hashlib.sha1(str(option.id).encode("utf-8")).hexdigest()
    step = max(1, len(coords) // 30)
    sample = coords[::step][:40]
    parts = [f"{float(lon):.4f},{float(lat):.4f}" for lon, lat in sample]
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()


def _pair_refined_routes_to_options_by_signature(
    refined_routes: Sequence[dict[str, Any]],
    current_options: Sequence[RouteOption],
) -> list[tuple[dict[str, Any], RouteOption]]:
    option_by_id = {
        str(option.id): option
        for option in current_options
        if str(option.id).strip()
    }
    paired_by_index: dict[int, tuple[dict[str, Any], RouteOption]] = {}
    matched_option_ids: set[str] = set()
    unmatched_indices: list[int] = []
    for index, route in enumerate(refined_routes):
        option_id = str(route.get("_built_option_id") or "").strip()
        option = option_by_id.get(option_id)
        if option is None or option_id in matched_option_ids:
            unmatched_indices.append(index)
            continue
        paired_by_index[index] = (route, option)
        matched_option_ids.add(option_id)

    signature_buckets: dict[str, list[RouteOption]] = {}
    for option in current_options:
        option_id = str(option.id).strip()
        if not option_id or option_id in matched_option_ids:
            continue
        signature_buckets.setdefault(_route_option_signature(option), []).append(option)

    for index in unmatched_indices:
        route = refined_routes[index]
        try:
            route_signature = _route_signature(route)
        except OSRMError:
            continue
        bucket = signature_buckets.get(route_signature)
        if not bucket:
            continue
        option = bucket.pop(0)
        if not bucket:
            signature_buckets.pop(route_signature, None)
        route["_built_option_id"] = option.id
        paired_by_index[index] = (route, option)
        matched_option_ids.add(str(option.id))

    return [paired_by_index[index] for index in sorted(paired_by_index)]


def _stable_route_signature_map_for_options(
    refined_routes: Sequence[dict[str, Any]],
    current_options: Sequence[RouteOption],
) -> dict[str, str]:
    signatures = {
        str(option.id): _route_option_signature(option)
        for option in current_options
        if str(option.id).strip()
    }
    for route, option in _pair_refined_routes_to_options_by_signature(refined_routes, current_options):
        option_id = str(option.id).strip()
        if not option_id:
            continue
        try:
            signatures[option_id] = _route_signature(route)
        except OSRMError:
            continue
    return signatures


def _certification_frontier_signature_map(
    current_options: Sequence[RouteOption],
) -> dict[str, str]:
    return {
        str(option.id): _route_option_signature(option)
        for option in current_options
        if str(option.id).strip()
    }


def _strict_frontier_rows_from_options(
    options: list[RouteOption],
    *,
    selected_id: str,
    evidence_snapshot_hash: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for option in options:
        certificate = option.certification
        active_families = list(certificate.active_families) if certificate is not None else []
        rows.append(
            {
                "route_id": option.id,
                "route_signature": _route_option_signature(option),
                "candidate_ids": [],
                "distance_km": float(option.metrics.distance_km),
                "duration_s": float(option.metrics.duration_s),
                "monetary_cost": float(option.metrics.monetary_cost),
                "emissions_kg": float(option.metrics.emissions_kg),
                "certificate": float(certificate.certificate) if certificate is not None else None,
                "certificate_threshold": float(certificate.threshold) if certificate is not None else None,
                "active_families": active_families,
                "evidence_snapshot_hash": str(evidence_snapshot_hash or ""),
                "top_fragility_families": list(certificate.top_fragility_families) if certificate is not None else [],
                "top_value_of_refresh_family": certificate.top_value_of_refresh_family if certificate is not None else None,
                "top_competitor_route_id": certificate.top_competitor_route_id if certificate is not None else None,
                "selected": option.id == selected_id,
            }
        )
    return rows


def _legacy_final_route_trace(
    *,
    selected: RouteOption,
    frontier_options: list[RouteOption],
    candidate_fetches: int,
    candidate_diag: CandidateDiagnostics,
    run_seed: int,
    stage_timings_ms: dict[str, float],
    route_cache_runtime: Mapping[str, Any] | None = None,
    route_option_cache_runtime: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    raw_candidate_count = int(candidate_diag.raw_count) if int(candidate_diag.raw_count) > 0 else int(candidate_fetches)
    try:
        selected_candidate_ids = json.loads(str(candidate_diag.selected_candidate_ids_json or "[]"))
    except (TypeError, ValueError, json.JSONDecodeError):
        selected_candidate_ids = []
    if not isinstance(selected_candidate_ids, list):
        selected_candidate_ids = []
    route_cache_runtime = dict(route_cache_runtime or {})
    route_option_cache_runtime = dict(route_option_cache_runtime or {})
    return {
        "pipeline_mode": "legacy",
        "refinement_policy": str(candidate_diag.refinement_policy or "full_legacy"),
        "run_seed": int(run_seed),
        "stage_events": [],
        "stage_timings_ms": {key: round(float(value), 2) for key, value in stage_timings_ms.items()},
        "resource_usage": _process_resource_snapshot(),
        "route_cache_runtime": {
            "cache_hits": int(route_cache_runtime.get("cache_hits", 0)),
            "cache_misses": int(route_cache_runtime.get("cache_misses", 0)),
            "reuse_rate": float(route_cache_runtime.get("reuse_rate", 0.0)),
            "last_cache_key": route_cache_runtime.get("last_cache_key"),
        },
        "route_cache_stats": route_cache_stats(),
        "k_raw_cache_stats": k_raw_cache_stats(),
        "route_option_cache_stats": route_option_cache_stats(),
        "route_state_cache_stats": route_state_cache_stats(),
        "voi_dccs_cache_stats": voi_dccs_cache_stats(),
        "route_option_cache_runtime": {
            "cache_hits": int(route_option_cache_runtime.get("cache_hits", 0)),
            "cache_hits_local": int(route_option_cache_runtime.get("cache_hits_local", 0)),
            "cache_hits_global": int(route_option_cache_runtime.get("cache_hits_global", 0)),
            "cache_misses": int(route_option_cache_runtime.get("cache_misses", 0)),
            "cache_key_missing": int(route_option_cache_runtime.get("cache_key_missing", 0)),
            "cache_disabled": int(route_option_cache_runtime.get("cache_disabled", 0)),
            "cache_set_failures": int(route_option_cache_runtime.get("cache_set_failures", 0)),
            "saved_ms_estimate": round(float(route_option_cache_runtime.get("saved_ms_estimate", 0.0)), 6),
            "reuse_rate": float(route_option_cache_runtime.get("reuse_rate", 0.0)),
            "last_cache_key": route_option_cache_runtime.get("last_cache_key"),
        },
        "counts": {
            "k_raw": max(0, raw_candidate_count),
            "refined": len(frontier_options),
            "strict_frontier": len(frontier_options),
        },
        "candidate_fetches": int(candidate_fetches),
        "candidate_diagnostics": candidate_diag.__dict__,
        "selected_route_id": selected.id,
        "selected_route_signature": _route_option_signature(selected),
        "selected_candidate_ids": selected_candidate_ids,
        "selected_certificate": (
            selected.certification.model_dump(mode="json") if selected.certification is not None else None
        ),
        "voi": {
            "stop_reason": "not_requested",
            "action_trace": [],
            "best_rejected_action": None,
            "controller_states_recorded": 0,
        },
        "artifact_pointers": {
            "strict_frontier": "strict_frontier.jsonl",
            "evidence_validation": "evidence_validation.json",
        },
    }


def _build_route_decision_package(
    *,
    req: RouteRequest,
    requested_pipeline_mode: str,
    actual_pipeline_mode: str,
    selected: RouteOption,
    candidates: Sequence[RouteOption],
    warnings: Sequence[str],
    selected_certificate: RouteCertificationSummary | None,
    voi_stop_summary: VoiStopSummary | None,
    evidence_validation: Mapping[str, Any] | None,
    extra_json_artifacts: Mapping[str, dict[str, Any] | list[Any]] | None = None,
    extra_jsonl_artifacts: Mapping[str, list[dict[str, Any]]] | None = None,
    extra_csv_artifacts: Mapping[str, tuple[list[str], list[dict[str, Any]]]] | None = None,
    extra_text_artifacts: Mapping[str, str] | None = None,
) -> DecisionPackage:
    def _coerce_float(value: Any) -> float | None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(numeric):
            return None
        return float(numeric)

    def _coerce_int(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _string_list(value: Any) -> list[str]:
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            if text.startswith("[") and text.endswith("]"):
                try:
                    decoded = json.loads(text)
                except (TypeError, ValueError, json.JSONDecodeError):
                    decoded = None
                if isinstance(decoded, list):
                    return [str(item).strip() for item in decoded if str(item).strip()]
            if "," in text:
                return [item.strip() for item in text.split(",") if item.strip()]
            return [text]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            values: list[str] = []
            for item in value:
                text = str(item or "").strip()
                if text:
                    values.append(text)
            return values
        return []

    requested_mode = str(requested_pipeline_mode or "legacy").strip().lower() or "legacy"
    actual_mode = str(actual_pipeline_mode or requested_mode).strip().lower() or "legacy"
    execution_mode = "dccs_refc" if actual_mode == "tri_source" else actual_mode
    warnings_list = [str(warning).strip() for warning in warnings if str(warning).strip()]
    ambiguity_context = _request_ambiguity_context(req)
    evidence_validation_payload = dict(evidence_validation or {})
    evidence_validation_status = str(
        evidence_validation_payload.get("status", "unknown")
    ).strip() or "unknown"

    source_mix = _string_list(ambiguity_context.get("od_ambiguity_source_mix"))
    observed_source_count = _coerce_int(ambiguity_context.get("od_ambiguity_source_count"))
    if observed_source_count is None:
        observed_source_count = len(source_mix)
    observed_source_count = max(observed_source_count, len(source_mix))
    support_notes = [
        f"evidence_validation_status={evidence_validation_status}",
        f"internal_execution_mode={execution_mode}",
    ]
    if requested_mode != actual_mode:
        support_notes.append(f"requested_mode={requested_mode}")
    if not source_mix and observed_source_count > 0:
        source_mix = [f"observed_source_{idx + 1}" for idx in range(observed_source_count)]
        support_notes.append("Observed ambiguity source labels were unavailable; generic source ids were used.")
    support_ratio = _coerce_float(ambiguity_context.get("od_ambiguity_support_ratio"))
    source_entropy = _coerce_float(ambiguity_context.get("od_ambiguity_source_entropy"))
    support_records = [
        DecisionSupportSourceRecord(
            source_id=source_id,
            role="ambiguity_source",
            required=index < 3,
            present=True,
            status="observed",
            provenance="request_ambiguity_context",
            details={
                **(
                    {"support_ratio": round(float(support_ratio), 6)}
                    if support_ratio is not None
                    else {}
                ),
                **(
                    {"selected_certificate": round(float(selected_certificate.certificate), 6)}
                    if selected_certificate is not None
                    else {}
                ),
                "evidence_validation_ok": evidence_validation_status == "ok",
            },
        )
        for index, source_id in enumerate(source_mix)
    ]
    missing_sources = [
        f"tri_source_slot_{slot_index}"
        for slot_index in range(len(support_records) + 1, 4)
    ]
    support_satisfied = (
        actual_mode != "legacy"
        and observed_source_count >= 3
        and evidence_validation_status == "ok"
    )
    if actual_mode == "legacy":
        support_notes.append("Legacy runtime path bypassed the tri_source support gate.")
    elif not support_satisfied:
        support_notes.append("tri_source support contract is not fully satisfied by the current runtime facts.")

    frontier_rows: list[dict[str, Any]] = []
    raw_frontier_rows = (extra_jsonl_artifacts or {}).get("strict_frontier.jsonl")
    if isinstance(raw_frontier_rows, list):
        for row in raw_frontier_rows:
            if isinstance(row, Mapping):
                frontier_rows.append(dict(row))
    if not frontier_rows:
        frontier_rows = _strict_frontier_rows_from_options(
            list(candidates),
            selected_id=selected.id,
            evidence_snapshot_hash="",
        )
    frontier_route_ids = [
        str(row.get("route_id", "")).strip()
        for row in frontier_rows
        if str(row.get("route_id", "")).strip()
    ]
    if not frontier_route_ids:
        frontier_route_ids = [option.id for option in candidates]
    certified_route_ids = [
        route_id
        for route_id, row in (
            (str(row.get("route_id", "")).strip(), row) for row in frontier_rows
        )
        if route_id
        and (
            bool(row.get("certified"))
            or (
                _coerce_float(row.get("certificate")) is not None
                and _coerce_float(row.get("certificate_threshold")) is not None
                and float(_coerce_float(row.get("certificate")) or 0.0)
                >= float(_coerce_float(row.get("certificate_threshold")) or 0.0)
            )
        )
    ]
    if not certified_route_ids:
        certified_route_ids = [
            option.id
            for option in candidates
            if option.certification is not None and bool(option.certification.certified)
        ]

    def _frontier_sort_key(row: Mapping[str, Any]) -> tuple[float, float, float, float, str]:
        cost = _coerce_float(row.get("monetary_cost"))
        certificate = _coerce_float(row.get("certificate"))
        duration = _coerce_float(row.get("duration_s"))
        emissions = _coerce_float(row.get("emissions_kg"))
        route_id = str(row.get("route_id", "")).strip()
        return (
            float(cost if cost is not None else float("inf")),
            float(-(certificate if certificate is not None else -1.0)),
            float(duration if duration is not None else float("inf")),
            float(emissions if emissions is not None else float("inf")),
            route_id,
        )

    minimum_cost_route_id = None
    if frontier_rows:
        minimum_cost_route_id = min(frontier_rows, key=_frontier_sort_key).get("route_id")
        minimum_cost_route_id = str(minimum_cost_route_id).strip() or None
    if minimum_cost_route_id is None and candidates:
        minimum_cost_route_id = min(
            candidates,
            key=lambda option: (
                float(option.metrics.monetary_cost),
                float(-(option.certification.certificate if option.certification is not None else -1.0)),
                float(option.metrics.duration_s),
                float(option.metrics.emissions_kg),
                option.id,
            ),
        ).id

    certificate_summary_payload = (extra_json_artifacts or {}).get("certificate_summary.json")
    certificate_basis = "pending_runtime_wiring"
    if isinstance(certificate_summary_payload, Mapping):
        certificate_basis = str(
            certificate_summary_payload.get(
                "selected_certificate_basis",
                certificate_summary_payload.get("status", certificate_basis),
            )
        ).strip() or certificate_basis
    elif selected_certificate is not None:
        certificate_basis = "route_certification_summary"
    elif actual_mode == "legacy":
        certificate_basis = "legacy_selected_route"

    selected_certificate_value = float(selected_certificate.certificate) if selected_certificate is not None else None
    selected_certificate_threshold = float(selected_certificate.threshold) if selected_certificate is not None else None
    selected_certified = bool(selected_certificate.certified) if selected_certificate is not None else False
    selected_validation = _selected_evidence_validation(
        evidence_validation_payload,
        route_id=selected.id,
    )
    certificate_map = _route_certificate_map_for_decision_package(
        candidates=candidates,
        selected=selected,
        selected_certificate=selected_certificate,
        certificate_summary_payload=certificate_summary_payload if isinstance(certificate_summary_payload, Mapping) else None,
    )
    sampled_world_manifest = (extra_json_artifacts or {}).get("sampled_world_manifest.json")
    route_fragility_map_payload = (extra_json_artifacts or {}).get("route_fragility_map.json")
    evidence_snapshot_manifest = (extra_json_artifacts or {}).get("evidence_snapshot_manifest.json")
    certification_state = None
    if (
        actual_mode in {"dccs_refc", "voi"}
        and certificate_map
        and isinstance(sampled_world_manifest, Mapping)
    ):
        try:
            certification_state = CertificationState.from_refc_outputs(
                certificate=certificate_map,
                threshold=(
                    float(selected_certificate.threshold)
                    if selected_certificate is not None
                    else float(
                        req.certificate_threshold
                        if req.certificate_threshold is not None
                        else settings.route_pipeline_certificate_threshold
                    )
                ),
                world_manifest=sampled_world_manifest,
                fragility=route_fragility_map_payload if isinstance(route_fragility_map_payload, Mapping) else None,
                evidence_snapshot_manifest=(
                    evidence_snapshot_manifest if isinstance(evidence_snapshot_manifest, Mapping) else None
                ),
                ambiguity_context=ambiguity_context,
                evidence_validation=selected_validation,
            )
        except Exception:
            certification_state = None
    support_summary, _support_provenance_payload, _support_trace_rows = _build_support_summary_for_decision_package(
        selected=selected,
        selected_validation=selected_validation,
        certification_state=certification_state,
    )
    support_satisfied = bool(support_summary.satisfied)
    source_mix = list(support_summary.source_mix)
    missing_sources = list(support_summary.missing_sources)
    if certification_state is not None:
        certified_route_ids = list(certification_state.certified_set.certified_route_ids)
        if not certified_route_ids and certification_state.certified:
            certified_route_ids = [certification_state.winner_id]
        certificate_basis = certification_state.certification_basis
        selected_certificate_value = certification_state.certificate_map.get(selected.id, selected_certificate_value)
        selected_certificate_threshold = float(certification_state.threshold)
        selected_certified = bool(
            certification_state.certified
            and selected.id in certification_state.certified_set.certified_route_ids
        )
    selective_gate_passed = bool(
        support_satisfied
        and selected_certified
        and minimum_cost_route_id is not None
        and selected.id == minimum_cost_route_id
    )

    preference_notes = [
        (
            "request_weights="
            f"time:{float(req.weights.time):.6f},"
            f"money:{float(req.weights.money):.6f},"
            f"co2:{float(req.weights.co2):.6f}"
        ),
        f"optimization_mode={req.optimization_mode}",
        f"risk_aversion={float(req.risk_aversion):.6f}",
    ]
    if requested_mode == "tri_source" and actual_mode != "legacy":
        preference_notes.append(f"tri_source executed through the {actual_mode} runtime path.")
    if requested_mode != actual_mode:
        preference_notes.append(f"requested_mode={requested_mode}")
    preference_state = build_preference_state(
        req.model_dump(mode="json"),
        candidates,
        selected_route_id=selected.id,
        selected_certificate=selected_certificate_value,
        stop_reason=(
            str(voi_stop_summary.stop_reason)
            if voi_stop_summary is not None
            else (
                certification_state.decision_region.status
                if certification_state is not None
                else None
            )
        ),
        certificate_map=certificate_map or None,
    )
    dominant_objective = str(preference_state.summary.get("dominant_objective", "")).strip()
    if dominant_objective:
        preference_notes.append(f"dominant_objective={dominant_objective}")
    compatible_route_ids = _string_list(preference_state.summary.get("compatible_route_ids"))
    if compatible_route_ids:
        preference_notes.append(f"compatible_routes={','.join(compatible_route_ids)}")
    stop_hint_codes = _string_list(preference_state.summary.get("stop_hint_codes"))
    if stop_hint_codes:
        preference_notes.append(f"stop_hints={','.join(stop_hint_codes)}")
    preference_summary = DecisionPreferenceSummary(notes=preference_notes)

    witness_world_count = None
    if certification_state is not None:
        witness_world_count = int(certification_state.world_bundle.effective_world_count)
    elif isinstance(sampled_world_manifest, Mapping):
        witness_world_count = _coerce_int(sampled_world_manifest.get("world_count"))
    challenger_route_ids = [route_id for route_id in frontier_route_ids if route_id != selected.id]
    top_competitor_route_id = (
        str(selected_certificate.top_competitor_route_id).strip()
        if selected_certificate is not None and selected_certificate.top_competitor_route_id
        else ""
    )
    if top_competitor_route_id and top_competitor_route_id not in challenger_route_ids:
        challenger_route_ids.insert(0, top_competitor_route_id)
    witness_notes: list[str] = []
    if actual_mode == "legacy":
        witness_notes.append("Legacy route responses do not carry sampled-world witnesses; frontier rows are used.")
    if witness_world_count is None:
        witness_notes.append("Witness world count is unavailable in the current runtime payloads.")
    witness_summary = DecisionWitnessSummary(
        primary_witness_route_id=selected.id,
        witness_route_ids=(certified_route_ids or [selected.id]),
        challenger_route_ids=challenger_route_ids,
        witness_world_count=witness_world_count,
        witness_source_ids=source_mix,
        notes=witness_notes,
    )
    if certification_state is not None:
        abstention_record = (
            AbstentionRecord.from_decision_region(certification_state.decision_region)
            if certification_state.decision_region.reason_codes
            else None
        )
        witness = CertificateWitness.from_state(
            certification_state,
            fragility=route_fragility_map_payload if isinstance(route_fragility_map_payload, Mapping) else None,
            abstention=abstention_record,
        )
        witness_summary = DecisionWitnessSummary(
            primary_witness_route_id=witness.winner_id,
            witness_route_ids=(list(witness.certified_route_ids) or [witness.winner_id]),
            challenger_route_ids=challenger_route_ids,
            witness_world_count=witness_world_count,
            witness_source_ids=list(certification_state.world_bundle.active_families),
            notes=list(witness.reasons),
        )

    voi_action_trace_payload = (extra_json_artifacts or {}).get("voi_action_trace.json")
    action_trace_rows: list[dict[str, Any]] = []
    if isinstance(voi_action_trace_payload, Mapping):
        raw_action_trace = voi_action_trace_payload.get("action_trace", voi_action_trace_payload.get("actions"))
        if isinstance(raw_action_trace, list):
            action_trace_rows = [dict(row) for row in raw_action_trace if isinstance(row, Mapping)]
    controller_notes: list[str] = []
    if requested_mode == "tri_source" and actual_mode != "legacy":
        controller_notes.append(f"tri_source executed with {actual_mode} internals in this runtime packet.")
    if requested_mode != actual_mode:
        controller_notes.append(f"request fell back from {requested_mode} to {actual_mode}")
    controller_summary = DecisionControllerSummary(
        controller_mode=(
            "tri_source_controller"
            if requested_mode == "tri_source" and actual_mode != "legacy"
            else ("voi" if actual_mode == "voi" else "single_pass")
        ),
        engaged=actual_mode == "voi",
        iteration_count=(
            int(voi_stop_summary.iteration_count)
            if voi_stop_summary is not None
            else len(action_trace_rows)
        ),
        action_count=len(action_trace_rows),
        stop_reason=(
            str(voi_stop_summary.stop_reason)
            if voi_stop_summary is not None
            else ("single_pass" if actual_mode != "legacy" else "legacy_single_pass")
        ),
        search_budget_used=(int(voi_stop_summary.search_budget_used) if voi_stop_summary is not None else 0),
        evidence_budget_used=(int(voi_stop_summary.evidence_budget_used) if voi_stop_summary is not None else 0),
        notes=controller_notes,
    )

    artifact_names = sorted(
        {
            "decision_package.json",
            "preference_summary.json",
            "support_summary.json",
            "support_provenance.json",
            "certified_set.json",
            "abstention_summary.json",
            "witness_summary.json",
            "controller_summary.json",
            "controller_trace.jsonl",
            "theorem_hook_map.json",
            "lane_manifest.json",
            *list((extra_json_artifacts or {}).keys()),
            *list((extra_jsonl_artifacts or {}).keys()),
            *list((extra_csv_artifacts or {}).keys()),
            *list((extra_text_artifacts or {}).keys()),
        }
    )
    hook_artifacts = (
        ("decision_package", "decision_package.json"),
        ("winner_summary", "winner_summary.json"),
        ("certificate_summary", "certificate_summary.json"),
        ("strict_frontier", "strict_frontier.jsonl"),
        ("sampled_world_manifest", "sampled_world_manifest.json"),
        ("evidence_validation", "evidence_validation.json"),
        ("voi_action_trace", "voi_action_trace.json"),
    )

    abstention_summary = None
    if certification_state is not None and certification_state.decision_region.reason_codes:
        abstention_record = AbstentionRecord.from_decision_region(certification_state.decision_region)
        blocking_sources = sorted(
            {
                *missing_sources,
                *([abstention_record.challenger_id] if abstention_record.challenger_id else []),
            }
        )
        abstention_summary = DecisionAbstentionSummary(
            abstained=bool(certification_state.decision_region.abstain),
            reason_code=abstention_record.reason_code,
            message=abstention_record.detail or certification_state.decision_region.status,
            blocking_sources=blocking_sources,
            retryable=not certification_state.certified,
        )
    elif actual_mode == "legacy":
        abstention_summary = DecisionAbstentionSummary(
            abstained=False,
            reason_code="legacy_runtime_selected",
            message="Legacy routing was used for this response; tri_source certification gating was not enforced.",
            blocking_sources=missing_sources,
            retryable=(requested_mode != "legacy"),
        )
    elif not support_satisfied:
        abstention_summary = DecisionAbstentionSummary(
            abstained=False,
            reason_code="tri_source_support_incomplete",
            message="decision_package is populated from current runtime facts, but the tri_source support contract is incomplete.",
            blocking_sources=missing_sources,
            retryable=True,
        )
    elif selected_certificate is not None and not selected_certificate.certified:
        abstention_summary = DecisionAbstentionSummary(
            abstained=False,
            reason_code="selected_route_not_certified",
            message="The selected route is returned, but its certificate remains below threshold.",
            blocking_sources=[],
            retryable=True,
        )
    certified_set_summary = CertifiedSetSummary(
        selected_route_id=selected.id,
        minimum_cost_route_id=minimum_cost_route_id,
        certified_route_ids=certified_route_ids,
        frontier_route_ids=frontier_route_ids,
        certificate_value=selected_certificate_value,
        certificate_threshold=selected_certificate_threshold,
        certificate_basis=certificate_basis,
        certified=selected_certified,
        selective_gate_passed=selective_gate_passed,
    )
    if certification_state is not None:
        certified_set_summary = CertifiedSetSummary(
            selected_route_id=selected.id,
            minimum_cost_route_id=minimum_cost_route_id,
            certified_route_ids=list(certification_state.certified_set.certified_route_ids),
            frontier_route_ids=frontier_route_ids,
            certificate_value=selected_certificate_value,
            certificate_threshold=selected_certificate_threshold,
            certificate_basis=certification_state.certification_basis,
            certified=certification_state.certified,
            selective_gate_passed=bool(certification_state.certified_set.safe),
        )
        selected_certified = bool(certification_state.certified)

    return DecisionPackage(
        pipeline_mode=("tri_source" if requested_mode == "tri_source" and actual_mode != "legacy" else actual_mode),  # type: ignore[arg-type]
        selected_route_id=selected.id,
        preference_summary=preference_summary,
        support_summary=support_summary,
        certified_set_summary=certified_set_summary,
        abstention_summary=abstention_summary,
        witness_summary=witness_summary,
        controller_summary=controller_summary,
        theorem_hook_summary=DecisionTheoremHookSummary(
            hooks=[
                DecisionTheoremHookRecord(
                    hook_id=hook_id,
                    artifact_name=artifact_name,
                    status=("present" if artifact_name in artifact_names else "planned"),
                    note=(
                        "Runtime placeholder hook mapped from existing artifact surfaces."
                        if artifact_name in artifact_names
                        else "Artifact not yet emitted on this route path."
                    ),
                )
                for hook_id, artifact_name in hook_artifacts
            ]
        ),
        lane_manifest=DecisionLaneManifest(
            lane_id=("tri_source" if requested_mode == "tri_source" and actual_mode != "legacy" else actual_mode),
            lane_name=(
                "Tri-Source Default Route"
                if requested_mode == "tri_source" and actual_mode != "legacy"
                else ("Legacy Route" if actual_mode == "legacy" else actual_mode.replace("_", " ").title())
            ),
            lane_version="0.1.0",
            artifact_names=artifact_names,
            notes=(
                ["decision_package assembled from current runtime facts."]
                + ([f"requested_mode={requested_mode}"] if requested_mode != actual_mode else [])
            ),
        ),
        provenance={
            "requested_pipeline_mode": requested_mode,
            "actual_pipeline_mode": (
                "tri_source" if requested_mode == "tri_source" and actual_mode != "legacy" else actual_mode
            ),
            "internal_execution_mode": actual_mode,
            "selected_certificate": selected_certificate_value,
            "selected_certified": selected_certified,
            "warning_count": len(warnings_list),
            "candidate_count": len(candidates),
            "evidence_validation_ok": evidence_validation_status == "ok",
            "weight_time": float(req.weights.time),
            "weight_money": float(req.weights.money),
            "weight_co2": float(req.weights.co2),
            "risk_aversion": float(req.risk_aversion),
            "ambiguity_source_count": int(observed_source_count),
            "ambiguity_source_entropy": source_entropy,
            "ambiguity_support_ratio": support_ratio,
        },
    )


def _route_signature_cells(
    route: dict[str, Any],
    *,
    precision: int = 2,
    max_points: int = 96,
) -> set[tuple[int, int]]:
    coords = _validate_osrm_geometry(route)
    if not coords:
        return set()
    step = max(1, len(coords) // max(1, int(max_points)))
    scale = 10 ** max(1, int(precision))
    cells: set[tuple[int, int]] = set()
    for lon, lat in coords[::step]:
        cells.add((int(round(lat * scale)), int(round(lon * scale))))
        if len(cells) >= max(1, int(max_points)):
            break
    return cells


def _route_overlap_ratio(route_a: dict[str, Any], route_b: dict[str, Any]) -> float:
    cells_a = _route_signature_cells(route_a)
    cells_b = _route_signature_cells(route_b)
    if not cells_a or not cells_b:
        return 1.0
    union = cells_a | cells_b
    if not union:
        return 1.0
    return float(len(cells_a & cells_b)) / float(len(union))


def _secondary_baseline_quality_score(
    *,
    candidate: dict[str, Any],
    direct_route: dict[str, Any],
) -> float:
    direct_distance_km = max(0.001, _route_distance_km(direct_route))
    direct_duration_s = max(1.0, _route_duration_s(direct_route))
    candidate_distance_km = max(0.001, _route_distance_km(candidate))
    candidate_duration_s = max(1.0, _route_duration_s(candidate))
    overlap = _route_overlap_ratio(candidate, direct_route)
    distance_ratio = candidate_distance_km / direct_distance_km
    duration_ratio = candidate_duration_s / direct_duration_s
    if distance_ratio > 1.60 or duration_ratio > 1.70:
        return float("-inf")
    # Prefer materially different geometry while bounding detour inflation.
    return float(
        (1.35 * (1.0 - overlap))
        - (0.45 * max(0.0, distance_ratio - 1.18))
        - (0.35 * max(0.0, duration_ratio - 1.18))
    )


def _select_secondary_baseline_candidate(
    *,
    direct_route: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if not candidates:
        return None, {
            "selected_candidate_rank": 0,
            "selected_distinct_corridor": False,
            "graph_candidate_count": 0,
            "direct_overlap_ratio": 1.0,
            "secondary_quality_score": float("-inf"),
        }
    direct_family = _graph_family_signature(direct_route) or _route_signature(direct_route)
    direct_corridor = _route_corridor_signature(direct_route)
    ranked: list[tuple[float, int, float, str, str, dict[str, Any]]] = []
    for index, candidate in enumerate(candidates):
        overlap = _route_overlap_ratio(candidate, direct_route)
        quality = _secondary_baseline_quality_score(candidate=candidate, direct_route=direct_route)
        family = _graph_family_signature(candidate) or _route_signature(candidate)
        corridor = _route_corridor_signature(candidate)
        ranked.append((quality, index, overlap, family, corridor, candidate))
    ranked.sort(
        key=lambda item: (
            item[0],
            item[2] <= 0.92,
            item[3] != direct_family,
            item[4] != direct_corridor,
            -_route_duration_s(item[5]),
            _route_signature(item[5]),
        ),
        reverse=True,
    )
    best_quality, best_index, overlap, family, corridor, best = ranked[0]
    if not math.isfinite(best_quality):
        return None, {
            "selected_candidate_rank": 0,
            "selected_distinct_corridor": False,
            "graph_candidate_count": len(candidates),
            "direct_overlap_ratio": round(overlap, 4),
            "secondary_quality_score": best_quality,
        }
    return best, {
        "selected_candidate_rank": int(best_index),
        "selected_distinct_corridor": bool(family != direct_family or corridor != direct_corridor or overlap < 0.92),
        "graph_candidate_count": len(candidates),
        "direct_overlap_ratio": round(overlap, 4),
        "secondary_quality_score": round(best_quality, 6),
    }


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
    legacy_search_context: Mapping[str, Any] | None = None,
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
        "legacy_search_context": _cache_key_component(legacy_search_context),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()


def _graph_refine_route_cache_key(
    *,
    origin: LatLng,
    destination: LatLng,
    alternatives: bool | int,
    exclude: str | None = None,
    via: Sequence[tuple[float, float]] | None = None,
    vehicle_type: str | None = None,
    scenario_mode: Any | None = None,
    cost_toggles: Any | None = None,
    terrain_profile: Any | None = None,
    departure_time_utc: datetime | None = None,
    scenario_cache_token: str | None = None,
) -> str:
    payload = {
        "schema_version": "graph_refine_v3",
        "origin": {"lat": round(float(origin.lat), 6), "lon": round(float(origin.lon), 6)},
        "destination": {"lat": round(float(destination.lat), 6), "lon": round(float(destination.lon), 6)},
        "alternatives": alternatives,
        "exclude": str(exclude).strip() or None,
        "via": [
            [round(float(via_lat), 6), round(float(via_lon), 6)]
            for via_lat, via_lon in (via or [])
        ],
        "vehicle_type": str(vehicle_type or "").strip() or None,
        "scenario_mode": (
            getattr(scenario_mode, "value", scenario_mode)
            if scenario_mode is not None
            else None
        ),
        "cost_toggles": _cache_key_component(cost_toggles),
        "terrain_profile": _cache_key_component(terrain_profile),
        "departure_time_utc": departure_time_utc.isoformat() if departure_time_utc else None,
        "scenario_cache_token": str(scenario_cache_token or "").strip() or None,
    }
    return hashlib.sha1(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _cache_key_component(value: Any) -> Any:
    if value is None:
        return None
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    raw_value = getattr(value, "value", None)
    if raw_value is not None and not callable(raw_value):
        return raw_value
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _cache_key_component(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_cache_key_component(item) for item in value]
    return str(value)


def _committed_refresh_route_state_cache_key(
    *,
    base_route_state_cache_key: str,
    active_families: Sequence[str],
    forced_refreshed_families: Sequence[str],
) -> str:
    payload = {
        "cache_kind": "voi_committed_refresh_v1",
        "base_route_state_cache_key": str(base_route_state_cache_key),
        "active_families": sorted(str(family) for family in active_families if str(family).strip()),
        "forced_refreshed_families": sorted(
            str(family) for family in forced_refreshed_families if str(family).strip()
        ),
    }
    return hashlib.sha1(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _legacy_candidate_cache_context(
    *,
    origin: LatLng,
    destination: LatLng,
    refinement_policy: str | None,
    search_budget: int | None,
    run_seed: int | None,
    od_ambiguity_index: float | None,
    od_engine_disagreement_prior: float | None,
    od_hard_case_prior: float | None,
    od_ambiguity_support_ratio: float | None,
    od_ambiguity_source_entropy: float | None,
    od_candidate_path_count: int | None,
    od_corridor_family_count: int | None,
    allow_supported_ambiguity_fast_fallback: bool,
) -> dict[str, Any] | None:
    normalized_policy = str(refinement_policy or "").strip().lower()
    if normalized_policy not in {"first_n", "random_n", "corridor_uniform"}:
        return None
    context: dict[str, Any] = {
        "refinement_policy": normalized_policy,
        "search_budget": int(search_budget or 0),
        "run_seed": int(run_seed or 0),
        "allow_supported_ambiguity_fast_fallback": bool(allow_supported_ambiguity_fast_fallback),
        "supported_ambiguity_fast_fallback": _supported_ambiguity_fast_fallback_active(
            origin=origin,
            destination=destination,
            od_ambiguity_index=od_ambiguity_index,
            od_engine_disagreement_prior=od_engine_disagreement_prior,
            od_hard_case_prior=od_hard_case_prior,
            od_ambiguity_support_ratio=od_ambiguity_support_ratio,
            od_ambiguity_source_entropy=od_ambiguity_source_entropy,
            allow_supported_ambiguity_fast_fallback=allow_supported_ambiguity_fast_fallback,
        ),
    }
    if normalized_policy == "corridor_uniform":
        context["support_rich_short_haul_fast_fallback"] = _support_rich_short_haul_fast_fallback_eligible(
            origin=origin,
            destination=destination,
            refinement_policy=normalized_policy,
            search_budget=search_budget,
            od_ambiguity_index=od_ambiguity_index,
            od_engine_disagreement_prior=od_engine_disagreement_prior,
            od_hard_case_prior=od_hard_case_prior,
            od_ambiguity_support_ratio=od_ambiguity_support_ratio,
            od_ambiguity_source_entropy=od_ambiguity_source_entropy,
            od_candidate_path_count=od_candidate_path_count,
            od_corridor_family_count=od_corridor_family_count,
            allow_supported_ambiguity_fast_fallback=allow_supported_ambiguity_fast_fallback,
        )
    return context


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


def _support_aware_fallback_specs(
    *,
    origin: LatLng,
    destination: LatLng,
    max_routes: int,
) -> list[CandidateFetchSpec]:
    """Build a bounded non-graph OSRM family set for support-aware recoveries."""
    alt_budget = max(2, min(int(settings.route_candidate_alternatives_max), max_routes))
    specs: list[CandidateFetchSpec] = [
        CandidateFetchSpec(label="support_fallback:alternatives", alternatives=alt_budget)
    ]
    via_candidates = _candidate_via_points(origin, destination)
    via_limit = max(1, min(len(via_candidates), max(2, min(max_routes, 3))))
    for idx, via in enumerate(via_candidates[:via_limit], start=1):
        specs.append(CandidateFetchSpec(label=f"support_fallback:via:{idx}", alternatives=False, via=[via]))
    for ex in ("toll", "ferry", "motorway", "trunk", "motorway,toll", "trunk,toll"):
        specs.append(
            CandidateFetchSpec(
                label=f"support_fallback:exclude:{ex}",
                alternatives=False,
                exclude=ex,
            )
        )
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
    queue_started = time.perf_counter()
    fetch_started: float | None = None
    try:
        async with sem:
            # Per-candidate refine-cost tracking should measure the fetch itself,
            # not time spent queued behind unrelated concurrent candidates.
            fetch_started = time.perf_counter()
            routes = await osrm.fetch_routes(
                origin_lat=origin.lat,
                origin_lon=origin.lon,
                dest_lat=destination.lat,
                dest_lon=destination.lon,
                alternatives=spec.alternatives,
                exclude=spec.exclude,
                via=spec.via,
            )
        elapsed_ms = max(0.001, (time.perf_counter() - fetch_started) * 1000.0)
        return CandidateFetchResult(spec=spec, routes=routes, elapsed_ms=elapsed_ms)
    except OSRMError as e:
        elapsed_ms = max(
            0.001,
            (time.perf_counter() - (fetch_started if fetch_started is not None else queue_started)) * 1000.0,
        )
        return CandidateFetchResult(spec=spec, routes=[], error=str(e), elapsed_ms=elapsed_ms)
    except Exception as e:  # pragma: no cover - defensive fallback for unexpected failures
        msg = str(e).strip()
        detail = f"{type(e).__name__}: {msg}" if msg else type(e).__name__
        elapsed_ms = max(
            0.001,
            (time.perf_counter() - (fetch_started if fetch_started is not None else queue_started)) * 1000.0,
        )
        return CandidateFetchResult(spec=spec, routes=[], error=detail, elapsed_ms=elapsed_ms)


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
    def _prefilter_source_profile(route: Mapping[str, Any]) -> tuple[str, str]:
        source_labels = _candidate_source_labels(route)
        source_stage = _selected_route_source_stage_from_labels(source_labels) or ""
        source_label = _selected_route_source_label(source_labels)
        source_bucket = _candidate_source_bucket(
            source_label=source_label,
            source_stage=source_stage,
        )
        return source_stage, source_bucket

    def _prefilter_observed_refine_cost_ms(route: Mapping[str, Any]) -> float:
        meta = route.get("_candidate_meta")
        if not isinstance(meta, Mapping):
            return float("nan")
        try:
            observed_ms = float(meta.get("observed_refine_cost_ms"))
        except (TypeError, ValueError):
            return float("nan")
        if not math.isfinite(observed_ms) or observed_ms <= 0.0:
            return float("nan")
        return observed_ms

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

    scored: list[tuple[float, str, str, str, str, str, float, dict[str, Any]]] = []
    for route in routes:
        sig = _route_signature(route)
        family_sig = _graph_family_signature(route) or sig
        corridor_sig = _route_corridor_signature(route)
        source_stage, source_bucket = _prefilter_source_profile(route)
        observed_refine_cost_ms = _prefilter_observed_refine_cost_ms(route)
        scored.append(
            (
                _route_duration_s(route),
                sig,
                family_sig,
                corridor_sig,
                source_stage,
                source_bucket,
                observed_refine_cost_ms,
                route,
            )
        )
    scored.sort(key=lambda item: (item[0], item[1]))

    if len(scored) <= prefilter_target:
        return [route for _, _, _, _, _, _, _, route in scored]

    selected: list[tuple[float, str, str, str, str, str, float, dict[str, Any]]] = []
    family_counts: dict[str, int] = {}
    corridor_counts: dict[str, int] = {}
    source_bucket_counts: dict[tuple[str, str], int] = {}
    remaining = list(scored)

    while remaining and len(selected) < prefilter_target:
        best_idx = 0
        best_score = float("inf")
        best_duration = float("inf")
        best_sig = ""
        for idx, (
            duration_s,
            sig,
            family_sig,
            corridor_sig,
            source_stage,
            source_bucket,
            observed_refine_cost_ms,
            _route,
        ) in enumerate(remaining):
            family_count = family_counts.get(family_sig, 0)
            corridor_count = corridor_counts.get(corridor_sig, 0)
            diversity_multiplier = 1.0 + (0.18 * family_count) + (0.12 * corridor_count)
            fallback_bucket_penalty = 0.0
            fallback_cost_penalty = 0.0
            if source_stage == "direct_k_raw_fallback":
                bucket_key = (source_stage, source_bucket)
                bucket_count = source_bucket_counts.get(bucket_key, 0)
                if source_bucket == "via":
                    fallback_bucket_penalty = 0.22 * bucket_count
                elif source_bucket in {
                    "alternatives",
                    "exclude_toll",
                    "exclude_motorway",
                    "exclude_trunk",
                    "exclude_other",
                    "ors_seed",
                }:
                    fallback_bucket_penalty = 0.10 * bucket_count
                if observed_refine_cost_ms > 60.0:
                    fallback_cost_penalty = min(
                        0.18,
                        max(0.0, observed_refine_cost_ms - 60.0) / 500.0,
                    )
            blended_score = duration_s * (
                diversity_multiplier + fallback_bucket_penalty + fallback_cost_penalty
            )
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
        (
            duration_s,
            sig,
            family_sig,
            corridor_sig,
            source_stage,
            source_bucket,
            observed_refine_cost_ms,
            route,
        ) = remaining.pop(best_idx)
        selected.append(
            (
                duration_s,
                sig,
                family_sig,
                corridor_sig,
                source_stage,
                source_bucket,
                observed_refine_cost_ms,
                route,
            )
        )
        family_counts[family_sig] = family_counts.get(family_sig, 0) + 1
        corridor_counts[corridor_sig] = corridor_counts.get(corridor_sig, 0) + 1
        source_bucket_key = (source_stage, source_bucket)
        source_bucket_counts[source_bucket_key] = source_bucket_counts.get(source_bucket_key, 0) + 1

    selected.sort(key=lambda item: (item[0], item[1]))
    return [route for _, _, _, _, _, _, _, route in selected]


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
    optimization_mode: OptimizationMode = "expected_value",
    pareto_method: ParetoMethod = "dominance",
    epsilon: EpsilonConstraints | None = None,
    max_alternatives: int | None = None,
    route_option_cache_runtime: dict[str, Any] | None = None,
    scenario_policy_cache: dict[tuple[str, str], Any] | None = None,
    lightweight: bool = False,
) -> tuple[list[RouteOption], list[str], TerrainDiagnostics]:
    options: list[RouteOption] = []
    warnings: list[str] = []
    fail_closed_count = 0
    unsupported_region_count = 0
    asset_unavailable_count = 0
    coverage_min_observed = 1.0
    dem_version = "unknown"
    scenario_policy_cache_local: dict[tuple[str, str], Any] = (
        scenario_policy_cache if scenario_policy_cache is not None else {}
    )
    require_uncertainty = str(getattr(optimization_mode, "value", optimization_mode)).strip().lower() != "expected_value"
    route_option_local_cache: dict[str, CachedRouteOptionBuild] = {}
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
            _resolve_route_scenario_policy(
                scenario_mode=scenario_mode,
                scenario_context=shared_context,
                scenario_policy_cache=scenario_policy_cache_local,
            )
            shifted_mode = _counterfactual_shift_scenario(scenario_mode)
            _resolve_route_scenario_policy(
                scenario_mode=shifted_mode,
                scenario_context=shared_context,
                scenario_policy_cache=scenario_policy_cache_local,
            )
        except Exception:
            # Fall back to per-route scenario context if shared bootstrap fails.
            pass

    def _route_option_cache_runtime_touch() -> None:
        if route_option_cache_runtime is None:
            return
        total_events = int(route_option_cache_runtime.get("cache_hits", 0)) + int(
            route_option_cache_runtime.get("cache_misses", 0)
        )
        route_option_cache_runtime["reuse_rate"] = round(
            int(route_option_cache_runtime.get("cache_hits", 0)) / float(max(1, total_events)),
            6,
        )

    def _route_option_cache_runtime_bump(field: str, amount: int = 1) -> None:
        if route_option_cache_runtime is None:
            return
        route_option_cache_runtime[field] = int(route_option_cache_runtime.get(field, 0)) + int(amount)
        _route_option_cache_runtime_touch()

    for route in routes:
        route.pop("_built_option_id", None)
        option_id = f"{option_prefix}_{len(options)}"
        try:
            cache_enabled = bool(settings.route_option_cache_enabled)
            cache_key = build_route_option_cache_key(
                route,
                vehicle_type=vehicle_type,
                detail_level="summary" if lightweight else "full",
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
                optimization_mode=optimization_mode,
                pareto_method=pareto_method,
                epsilon=epsilon,
                max_alternatives=max_alternatives,
            )
            if route_option_cache_runtime is not None:
                route_option_cache_runtime["last_cache_key"] = cache_key
            option: RouteOption | None = None
            cached_payload: CachedRouteOptionBuild | None = None
            if not cache_enabled:
                _route_option_cache_runtime_bump("cache_disabled")
            elif cache_key is None:
                _route_option_cache_runtime_bump("cache_key_missing")
            else:
                cached_payload = route_option_local_cache.get(cache_key)
                if cached_payload is not None:
                    _route_option_cache_runtime_bump("cache_hits")
                    _route_option_cache_runtime_bump("cache_hits_local")
                    if route_option_cache_runtime is not None:
                        route_option_cache_runtime["saved_ms_estimate"] = float(
                            route_option_cache_runtime.get("saved_ms_estimate", 0.0)
        ) + max(0.0, float(cached_payload.estimated_build_ms))
                    option = cached_payload.option.model_copy(update={"id": option_id}, deep=True)
                else:
                    cached_payload = get_cached_route_option_build(cache_key)
                    if cached_payload is not None:
                        route_option_local_cache[cache_key] = cached_payload
                        _route_option_cache_runtime_bump("cache_hits")
                        _route_option_cache_runtime_bump("cache_hits_global")
                        if route_option_cache_runtime is not None:
                            route_option_cache_runtime["saved_ms_estimate"] = float(
                                route_option_cache_runtime.get("saved_ms_estimate", 0.0)
                            ) + max(0.0, float(cached_payload.estimated_build_ms))
                        option = cached_payload.option.model_copy(update={"id": option_id}, deep=True)
            if option is None:
                build_started = time.perf_counter()
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
                    optimization_mode=optimization_mode,
                    pareto_method=pareto_method,
                    epsilon=epsilon,
                    max_alternatives=max_alternatives,
                    scenario_policy_cache=scenario_policy_cache_local,
                    reset_terrain_route_run=False,
                    lightweight=lightweight,
                    force_uncertainty=require_uncertainty,
                )
                build_elapsed_ms = round((time.perf_counter() - build_started) * 1000.0, 2)
                if cache_key is not None and cache_enabled:
                    cached_payload = CachedRouteOptionBuild(
                        option=option.model_copy(deep=True),
                        estimated_build_ms=build_elapsed_ms,
                    )
                    route_option_local_cache[cache_key] = cached_payload
                    if not set_cached_route_option_build(cache_key, cached_payload):
                        _route_option_cache_runtime_bump("cache_set_failures")
                    _route_option_cache_runtime_bump("cache_misses")
            options.append(option)
            route["_built_option_id"] = option.id
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


def _option_has_full_route_details(option: RouteOption | None) -> bool:
    if option is None:
        return False
    return (
        option.uncertainty is not None
        and option.uncertainty_samples_meta is not None
        and option.evidence_provenance is not None
        and (
            bool(option.segment_breakdown)
            or bool(option.eta_explanations)
            or bool(option.counterfactuals)
        )
    )


def _route_option_lightweight_copy(option: RouteOption) -> RouteOption:
    segment_rows = option.segment_breakdown or []
    weather_summary = option.weather_summary or {}
    segment_count = max(
        1,
        int(
            sum(
                max(1, int(float(row.get("segment_count", 1))))
                for row in segment_rows
                if isinstance(row, Mapping)
            )
        ),
    )
    distance_km = round(
        float(
            sum(
                max(0.0, float(row.get("distance_km", 0.0)))
                for row in segment_rows
                if isinstance(row, Mapping)
            )
            or float(option.metrics.distance_km)
        ),
        6,
    )
    duration_s = round(
        float(
            sum(
                max(0.0, float(row.get("duration_s", 0.0)))
                for row in segment_rows
                if isinstance(row, Mapping)
            )
            or float(option.metrics.duration_s)
        ),
        6,
    )
    toll_cost = round(
        float(
            sum(
                max(0.0, float(row.get("toll_cost", 0.0)))
                for row in segment_rows
                if isinstance(row, Mapping)
            )
            or float(weather_summary.get("toll_cost_total_gbp", 0.0))
        ),
        6,
    )
    fuel_cost = round(
        float(
            sum(
                max(0.0, float(row.get("fuel_cost", 0.0)))
                for row in segment_rows
                if isinstance(row, Mapping)
            )
            or float(weather_summary.get("fuel_cost_total_gbp", 0.0))
        ),
        6,
    )
    carbon_cost = round(
        float(
            sum(
                max(0.0, float(row.get("carbon_cost", 0.0)))
                for row in segment_rows
                if isinstance(row, Mapping)
            )
            or float(weather_summary.get("carbon_cost_total_gbp", 0.0))
        ),
        6,
    )
    return option.model_copy(
        update={
            "segment_breakdown": _compact_segment_breakdown(
                segment_count=segment_count,
                distance_km=distance_km,
                duration_s=duration_s,
                toll_cost=toll_cost,
                fuel_cost=fuel_cost,
                carbon_cost=carbon_cost,
            ),
            "eta_explanations": [],
            "eta_timeline": [],
            "counterfactuals": [],
        },
        deep=True,
    )


def _compact_segment_breakdown(
    *,
    segment_count: int,
    distance_km: float,
    duration_s: float,
    toll_cost: float,
    fuel_cost: float,
    carbon_cost: float,
) -> list[dict[str, float | int]]:
    return [
        {
            "segment_index": 0,
            "segment_count": max(1, int(segment_count)),
            "distance_km": round(float(distance_km), 6),
            "duration_s": round(float(duration_s), 6),
            "toll_cost": round(float(toll_cost), 6),
            "fuel_cost": round(float(fuel_cost), 6),
            "carbon_cost": round(float(carbon_cost), 6),
        }
    ]


def _route_option_detail_level(*, req: RouteRequest, pipeline_mode: str) -> str:
    if pipeline_mode == "dccs" or bool(getattr(req, "evaluation_lean_mode", False)):
        return "summary"
    return "full"


def _should_hydrate_priority_route_options(req: RouteRequest) -> bool:
    return not bool(getattr(req, "evaluation_lean_mode", False))


def _route_state_cache_profile(*, req: RouteRequest, pipeline_mode: str) -> dict[str, Any]:
    return {
        "route_option_detail_level": _route_option_detail_level(req=req, pipeline_mode=pipeline_mode),
        "priority_hydration_enabled": _should_hydrate_priority_route_options(req),
    }


def _hydrate_priority_route_options(
    *,
    routes: list[dict[str, Any]],
    options: list[RouteOption],
    strict_frontier: list[RouteOption],
    display_candidates: list[RouteOption],
    selected: RouteOption,
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
    optimization_mode: OptimizationMode = "expected_value",
    pareto_method: ParetoMethod = "dominance",
    epsilon: EpsilonConstraints | None = None,
    max_alternatives: int | None = None,
    scenario_policy_cache: dict[tuple[str, str], Any] | None = None,
) -> tuple[list[RouteOption], list[RouteOption], list[RouteOption], RouteOption, float]:
    priority_ids = (
        {selected.id}
        | {option.id for option in strict_frontier}
        | {option.id for option in display_candidates}
    )
    if not priority_ids:
        return options, strict_frontier, display_candidates, selected, 0.0

    route_by_option_id: dict[str, dict[str, Any]] = {}
    for route in routes:
        built_option_id = str(route.get("_built_option_id") or "").strip()
        if built_option_id:
            route_by_option_id[built_option_id] = route

    option_by_id: dict[str, RouteOption] = {option.id: option for option in options}
    hydration_started = time.perf_counter()
    hydrated_count = 0
    for option_id in priority_ids:
        route = route_by_option_id.get(option_id)
        if route is None:
            continue
        current_option = option_by_id.get(option_id)
        if _option_has_full_route_details(current_option):
            continue
        cache_key = build_route_option_cache_key(
            route,
            vehicle_type=vehicle_type,
            detail_level="full",
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
            optimization_mode=optimization_mode,
            pareto_method=pareto_method,
            epsilon=epsilon,
            max_alternatives=max_alternatives,
        )
        cached_payload = get_cached_route_option_build(cache_key) if cache_key is not None else None
        if cached_payload is not None and _option_has_full_route_details(cached_payload.option):
            option_by_id[option_id] = cached_payload.option.model_copy(update={"id": option_id}, deep=True)
            hydrated_count += 1
            continue
        build_started = time.perf_counter()
        hydrated_option = build_option(
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
            optimization_mode=optimization_mode,
            pareto_method=pareto_method,
            epsilon=epsilon,
            max_alternatives=max_alternatives,
            scenario_policy_cache=scenario_policy_cache,
            reset_terrain_route_run=False,
            lightweight=False,
        )
        option_by_id[option_id] = hydrated_option
        if cache_key is not None:
            set_cached_route_option_build(
                cache_key,
                CachedRouteOptionBuild(
                    option=hydrated_option.model_copy(deep=True),
                    estimated_build_ms=round((time.perf_counter() - build_started) * 1000.0, 2),
                ),
            )
        hydrated_count += 1

    if hydrated_count <= 0:
        return options, strict_frontier, display_candidates, selected, 0.0

    def _rewrite(sequence: list[RouteOption]) -> list[RouteOption]:
        return [option_by_id.get(option.id, option) for option in sequence]

    rebuilt_options = _rewrite(options)
    rebuilt_frontier = _rewrite(strict_frontier)
    rebuilt_display = _rewrite(display_candidates)
    rebuilt_selected = option_by_id.get(selected.id, selected)
    return (
        rebuilt_options,
        rebuilt_frontier,
        rebuilt_display,
        rebuilt_selected,
        round((time.perf_counter() - hydration_started) * 1000.0, 2),
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
        _option_joint_robust_utility(option, risk_aversion=risk_aversion),
        *vector,
        option.id,
    )


def _ordered_robust_selection_options(
    options: list[RouteOption],
    *,
    risk_aversion: float,
) -> list[RouteOption]:
    if not options:
        return []
    nondominated, dominance_matrix, robust_vectors = _robust_nondominated_indices(
        options,
        risk_aversion=risk_aversion,
    )
    index_by_id = {option.id: idx for idx, option in enumerate(options)}

    def _rank(option: RouteOption) -> tuple[float | str, ...]:
        return _robust_rank_key(
            idx=int(index_by_id.get(option.id, 0)),
            options=options,
            dominance_matrix=dominance_matrix,
            robust_vectors=robust_vectors,
            risk_aversion=risk_aversion,
        )

    preferred_pool = [options[idx] for idx in nondominated] if nondominated else list(options)
    ordered = sorted(preferred_pool, key=_rank)
    if len(ordered) == len(options):
        return ordered
    preferred_ids = {option.id for option in ordered}
    dominated_pool = sorted(
        [option for option in options if option.id not in preferred_ids],
        key=_rank,
    )
    return [*ordered, *dominated_pool]


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
        mean_value = float(option.uncertainty.get(mean_key, deterministic))
        if objective == "duration":
            cvar_key = "cvar95_duration_s"
        elif objective == "money":
            cvar_key = "cvar95_monetary_cost"
        else:
            cvar_key = "cvar95_emissions_kg"
        cvar_raw = option.uncertainty.get(cvar_key)
        cvar_value = float(cvar_raw) if cvar_raw is not None else mean_value
        cvar_value = max(mean_value, cvar_value)
        return robust_objective(
            mean_value=mean_value,
            cvar_value=cvar_value,
            risk_aversion=risk_aversion,
            risk_family=settings.risk_family,
            risk_theta=settings.risk_family_theta,
        )

    mean_value = float(option.uncertainty.get(mean_key, deterministic))
    return mean_value


def _time_preservation_tradeoff_penalties(
    times: Sequence[float],
    moneys: Sequence[float],
    emissions: Sequence[float],
    *,
    wt: float,
    wm: float,
    we: float,
) -> list[float]:
    if not times:
        return []
    fastest_idx = min(
        range(len(times)),
        key=lambda idx: (float(times[idx]), float(moneys[idx]), float(emissions[idx]), idx),
    )
    fastest_time = float(times[fastest_idx])
    fastest_money = float(moneys[fastest_idx])
    fastest_emissions = float(emissions[fastest_idx])
    time_reference = max(1.0, fastest_time)
    money_reference = max(1.0, abs(fastest_money))
    emissions_reference = max(1.0, abs(fastest_emissions))
    tradeoff_weight = 1.5
    tradeoff_slack = 0.01

    penalties: list[float] = []
    for time_v, money_v, co2_v in zip(times, moneys, emissions, strict=True):
        time_gap_ratio = max(0.0, (float(time_v) - fastest_time) / time_reference)
        money_gain_ratio = max(0.0, (fastest_money - float(money_v)) / money_reference)
        co2_gain_ratio = max(0.0, (fastest_emissions - float(co2_v)) / emissions_reference)
        compensated_gain = (wm * money_gain_ratio) + (we * co2_gain_ratio)
        tradeoff_gap = max(0.0, time_gap_ratio - compensated_gain - tradeoff_slack)
        penalties.append(tradeoff_weight * tradeoff_gap)
    return penalties


def _route_selection_scale_floor(objective: str, reference_value: float) -> float:
    reference = max(1.0, abs(float(reference_value)))
    if objective == "duration":
        return max(300.0, 0.05 * reference)
    if objective == "money":
        return max(5.0, 0.02 * reference)
    if objective == "co2":
        return max(20.0, 0.04 * reference)
    return 0.0


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

    time_scale = max(t_max - t_min, _route_selection_scale_floor("duration", t_min))
    money_scale = max(m_max - m_min, _route_selection_scale_floor("money", m_min))
    emissions_scale = max(e_max - e_min, _route_selection_scale_floor("co2", e_min))
    distance_scale = d_max - d_min

    def _norm(value: float, min_value: float, scale: float) -> float:
        return 0.0 if scale <= 0.0 else (value - min_value) / scale

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
            _norm(time_v, t_min, time_scale),
            _norm(money_v, m_min, money_scale),
            _norm(co2_v, e_min, emissions_scale),
            _norm(distance_v, d_min, distance_scale),
        )
        for time_v, money_v, co2_v, distance_v in zip(
            times,
            moneys,
            emissions,
            distances,
            strict=True,
        )
    ]
    time_preservation_penalties = _time_preservation_tradeoff_penalties(
        times,
        moneys,
        emissions,
        wt=wt,
        wm=wm,
        we=we,
    )

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
    for option, time_v, money_v, co2_v, distance_v, norms, weighted_sum, weighted_regret, time_preservation_penalty in zip(
        options,
        times,
        moneys,
        emissions,
        distances,
        norm_rows,
        weighted_sum_rows,
        weighted_regret_rows,
        time_preservation_penalties,
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
            score = (
                weighted_sum
                + (regret_weight * weighted_regret)
                + (balance_weight * balance_penalty)
                + time_preservation_penalty
            )
        elif math_profile == "modified_vikor_distance":
            score = (
                vikor_q
                + (balance_weight * balance_penalty)
                + (distance_weight * n_distance)
                + (eta_distance_weight * eta_distance_penalty)
                + (knee_weight * knee_penalty)
                - (entropy_weight * entropy_reward)
                + time_preservation_penalty
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
                + time_preservation_penalty
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
    def _annotate_final(options_to_annotate: list[RouteOption]) -> list[RouteOption]:
        return annotate_knee_scores(options_to_annotate, objective_key=objective_fn)

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
        return _annotate_final(sorted_pareto)
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
        return _annotate_final(sorted_pareto)
    backfill_pool = list(options)
    if pareto_method == "epsilon_constraint" and epsilon is not None:
        backfill_pool = filter_by_epsilon(backfill_pool, epsilon, objective_key=objective_fn)
    if not backfill_pool:
        return _annotate_final(sorted_pareto)
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
    return _annotate_final(sorted_pareto)


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
        "skipped_support_rich_short_haul_graph_search",
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


def _route_graph_search_support_confidence(
    *,
    od_ambiguity_support_ratio: float | None,
    od_ambiguity_source_entropy: float | None,
) -> float:
    support_ratio = max(0.0, min(1.0, float(od_ambiguity_support_ratio or 0.0)))
    source_entropy = max(0.0, min(1.0, float(od_ambiguity_source_entropy or 0.0)))
    # Down-weight unsupported hard-case priors so a single weak prior does not
    # force an expensive graph search on otherwise low-ambiguity corridors.
    return max(0.0, min(1.0, support_ratio * source_entropy))


def _route_graph_search_ambiguity_strength(
    *,
    od_ambiguity_index: float | None,
    od_engine_disagreement_prior: float | None,
    od_hard_case_prior: float | None,
    od_ambiguity_support_ratio: float | None,
    od_ambiguity_source_entropy: float | None,
) -> float:
    support_confidence = _route_graph_search_support_confidence(
        od_ambiguity_support_ratio=od_ambiguity_support_ratio,
        od_ambiguity_source_entropy=od_ambiguity_source_entropy,
    )
    raw_ambiguity = max(0.0, min(1.0, float(od_ambiguity_index or 0.0)))
    engine_disagreement = max(0.0, min(1.0, float(od_engine_disagreement_prior or 0.0)))
    hard_case_prior = max(0.0, min(1.0, float(od_hard_case_prior or 0.0)))
    return max(
        raw_ambiguity,
        engine_disagreement * support_confidence,
        hard_case_prior * support_confidence,
    )


def _support_rich_short_haul_fast_fallback_eligible(
    *,
    origin: LatLng,
    destination: LatLng,
    refinement_policy: str | None,
    search_budget: int | None,
    od_ambiguity_index: float | None,
    od_engine_disagreement_prior: float | None,
    od_hard_case_prior: float | None,
    od_ambiguity_support_ratio: float | None,
    od_ambiguity_source_entropy: float | None,
    od_candidate_path_count: int | None,
    od_corridor_family_count: int | None,
    allow_supported_ambiguity_fast_fallback: bool,
) -> bool:
    legacy_selection_policy = str(refinement_policy or "").strip().lower()
    legacy_budget = max(0, int(search_budget or 0))
    candidate_path_count = max(0, int(od_candidate_path_count or 0))
    corridor_family_count = max(0, int(od_corridor_family_count or 0))
    return bool(
        _supported_ambiguity_fast_fallback_active(
            origin=origin,
            destination=destination,
            od_ambiguity_index=od_ambiguity_index,
            od_engine_disagreement_prior=od_engine_disagreement_prior,
            od_hard_case_prior=od_hard_case_prior,
            od_ambiguity_support_ratio=od_ambiguity_support_ratio,
            od_ambiguity_source_entropy=od_ambiguity_source_entropy,
            allow_supported_ambiguity_fast_fallback=allow_supported_ambiguity_fast_fallback,
        )
        and legacy_selection_policy == "corridor_uniform"
        and legacy_budget > 0
        and candidate_path_count > 0
        and candidate_path_count <= 4
        and corridor_family_count > 0
        and corridor_family_count <= 2
    )


def _support_rich_short_haul_graph_probe_eligible(
    *,
    enabled: bool,
    od_engine_disagreement_prior: float | None,
    od_hard_case_prior: float | None,
) -> bool:
    """Restrict the expensive VOI-only graph probe to genuinely high-stress rows."""
    if not enabled:
        return False
    engine_disagreement = max(0.0, min(1.0, float(od_engine_disagreement_prior or 0.0)))
    hard_case_prior = max(0.0, min(1.0, float(od_hard_case_prior or 0.0)))
    return bool(max(engine_disagreement, hard_case_prior) >= 0.50)


def _support_rich_short_haul_probe_fail_open_needed(
    *,
    enabled: bool,
    routes: Sequence[dict[str, Any]],
    od_candidate_path_count: int | None,
    od_corridor_family_count: int | None,
) -> bool:
    if not enabled or not routes:
        return False

    candidate_path_count = max(0, int(od_candidate_path_count or 0))
    corridor_family_count = max(0, int(od_corridor_family_count or 0))
    route_count = len(routes)
    corridor_keys: set[str] = set()
    for route in routes:
        corridor_sig = _graph_family_signature(route) or _route_corridor_signature(route)
        if corridor_sig:
            corridor_keys.add(str(corridor_sig))
            continue
        try:
            corridor_keys.add(_route_signature(route))
        except OSRMError:
            continue

    observed_corridor_count = len(corridor_keys)
    expected_route_floor = max(3, candidate_path_count)
    expected_corridor_floor = max(1, min(2, corridor_family_count))
    return bool(
        route_count <= expected_route_floor
        or observed_corridor_count < expected_corridor_floor
    )


def _support_backed_single_corridor_probe_fail_open_needed(
    *,
    enabled: bool,
    routes: Sequence[dict[str, Any]],
    od_candidate_path_count: int | None,
) -> bool:
    if not enabled or not routes:
        return False

    candidate_path_count = max(0, int(od_candidate_path_count or 0))
    route_count = len(routes)
    expected_route_floor = max(2, min(3, max(1, candidate_path_count)))
    return bool(route_count < expected_route_floor)


def _long_corridor_stress_graph_probe_eligible(
    *,
    long_corridor: bool,
    ambiguity_strength: float,
    od_engine_disagreement_prior: float | None,
    od_hard_case_prior: float | None,
    od_ambiguity_support_ratio: float | None,
    od_ambiguity_source_entropy: float | None,
    od_candidate_path_count: int | None,
    od_corridor_family_count: int | None,
) -> bool:
    """Keep graph search enabled only on exceptionally well-supported long corridors.

    Empirical March 29, 2026 probes showed that prior pressure alone was too
    permissive here: low-observed-ambiguity long corridors could burn tens of
    seconds in graph search and still fall back to direct-k-raw refinement.
    The probe now requires strong observed ambiguity plus rich multi-family
    support before it bypasses the long-corridor fast-fallback.
    """
    if not long_corridor:
        return False

    support_ratio = max(0.0, min(1.0, float(od_ambiguity_support_ratio or 0.0)))
    source_entropy = max(0.0, min(1.0, float(od_ambiguity_source_entropy or 0.0)))
    candidate_path_count = max(0, int(od_candidate_path_count or 0))
    corridor_family_count = max(0, int(od_corridor_family_count or 0))

    minimum_supported_ambiguity = max(
        float(settings.route_dccs_preemptive_comparator_min_ambiguity),
        0.35,
    )
    minimum_family_rich_count = max(
        6,
        int(settings.route_dccs_preemptive_comparator_min_corridor_count) + 5,
    )
    minimum_path_rich_count = max(
        12,
        int(settings.route_dccs_preemptive_comparator_max_candidates) + 6,
    )
    return bool(
        ambiguity_strength >= minimum_supported_ambiguity
        and support_ratio >= 0.60
        and source_entropy >= 0.90
        and candidate_path_count >= minimum_path_rich_count
        and corridor_family_count >= minimum_family_rich_count
    )


def _supported_ambiguity_fast_fallback_active(
    *,
    origin: LatLng,
    destination: LatLng,
    od_ambiguity_index: float | None,
    od_engine_disagreement_prior: float | None,
    od_hard_case_prior: float | None,
    od_ambiguity_support_ratio: float | None,
    od_ambiguity_source_entropy: float | None,
    allow_supported_ambiguity_fast_fallback: bool,
) -> bool:
    ambiguity_strength = _route_graph_search_ambiguity_strength(
        od_ambiguity_index=od_ambiguity_index,
        od_engine_disagreement_prior=od_engine_disagreement_prior,
        od_hard_case_prior=od_hard_case_prior,
        od_ambiguity_support_ratio=od_ambiguity_support_ratio,
        od_ambiguity_source_entropy=od_ambiguity_source_entropy,
    )
    support_ratio = max(0.0, min(1.0, float(od_ambiguity_support_ratio or 0.0)))
    source_entropy = max(0.0, min(1.0, float(od_ambiguity_source_entropy or 0.0)))
    corridor_distance_km = _od_haversine_km(origin, destination)
    long_corridor_threshold_km = max(10.0, float(settings.route_graph_long_corridor_threshold_km))
    reliability_corridor_threshold_km = min(
        long_corridor_threshold_km,
        max(5.0, long_corridor_threshold_km * 0.6),
    )
    long_corridor = corridor_distance_km >= long_corridor_threshold_km
    reliability_corridor = corridor_distance_km >= reliability_corridor_threshold_km
    return bool(
        allow_supported_ambiguity_fast_fallback
        and not long_corridor
        and not reliability_corridor
        and support_ratio >= 0.65
        and source_entropy >= 0.50
        and (
            ambiguity_strength >= 0.40
            or float(od_engine_disagreement_prior or 0.0)
            >= float(settings.route_dccs_preemptive_comparator_min_engine_disagreement)
            or float(od_hard_case_prior or 0.0)
            >= float(settings.route_dccs_preemptive_comparator_min_hard_case)
        )
    )


def _process_resource_snapshot() -> dict[str, Any]:
    if psutil is None:
        return _process_resource_snapshot_windows()
    try:
        process = psutil.Process(os.getpid())
        memory = process.memory_info()
        payload = {
            "rss_bytes": int(getattr(memory, "rss", 0)),
            "vms_bytes": int(getattr(memory, "vms", 0)),
        }
        for attr in ("peak_wset", "peak_pagefile", "peak_rss"):
            if hasattr(memory, attr):
                payload[f"{attr}_bytes"] = int(getattr(memory, attr))
        return payload
    except Exception:
        return _process_resource_snapshot_windows()


def _process_resource_snapshot_windows() -> dict[str, Any]:
    if os.name != "nt":
        return {}
    try:
        # Windows memory accounting is available through GetProcessMemoryInfo:
        # https://learn.microsoft.com/windows/win32/api/psapi/nf-psapi-getprocessmemoryinfo
        class _ProcessMemoryCountersEx(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        counters = _ProcessMemoryCountersEx()
        counters.cb = ctypes.sizeof(_ProcessMemoryCountersEx)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_ProcessMemoryCountersEx),
            ctypes.c_ulong,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        process_handle = kernel32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(
            process_handle,
            ctypes.byref(counters),
            counters.cb,
        )
        if not ok:
            return {}
        return {
            "rss_bytes": int(counters.WorkingSetSize),
            "vms_bytes": int(counters.PrivateUsage or counters.PagefileUsage),
            "peak_wset_bytes": int(counters.PeakWorkingSetSize),
            "peak_pagefile_bytes": int(counters.PeakPagefileUsage),
            "private_bytes": int(counters.PrivateUsage),
            "page_fault_count": int(counters.PageFaultCount),
        }
    except Exception:
        return {}


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
    refinement_policy: str | None = None,
    search_budget: int | None = None,
    run_seed: int | None = None,
    od_ambiguity_index: float | None = None,
    od_engine_disagreement_prior: float | None = None,
    od_hard_case_prior: float | None = None,
    od_ambiguity_support_ratio: float | None = None,
    od_ambiguity_source_entropy: float | None = None,
    od_candidate_path_count: int | None = None,
    od_corridor_family_count: int | None = None,
    allow_supported_ambiguity_fast_fallback: bool = False,
    route_cache_runtime: dict[str, Any] | None = None,
    progress_cb: ProgressCallback | None = None,
) -> tuple[list[dict[str, Any]], list[str], int, CandidateDiagnostics]:
    warnings: list[str] = []
    legacy_selection_policy = str(refinement_policy or "").strip().lower()
    legacy_budget = max(0, int(search_budget or 0))
    selected_candidate_ids: list[str] = []
    ambiguity_context_available = any(
        value is not None
        for value in (
            od_ambiguity_index,
            od_engine_disagreement_prior,
            od_hard_case_prior,
            od_ambiguity_support_ratio,
            od_ambiguity_source_entropy,
        )
    )
    ambiguity_strength = _route_graph_search_ambiguity_strength(
        od_ambiguity_index=od_ambiguity_index,
        od_engine_disagreement_prior=od_engine_disagreement_prior,
        od_hard_case_prior=od_hard_case_prior,
        od_ambiguity_support_ratio=od_ambiguity_support_ratio,
        od_ambiguity_source_entropy=od_ambiguity_source_entropy,
    )
    support_ratio = max(0.0, min(1.0, float(od_ambiguity_support_ratio or 0.0)))
    source_entropy = max(0.0, min(1.0, float(od_ambiguity_source_entropy or 0.0)))
    candidate_path_count = max(0, int(od_candidate_path_count or 0))
    corridor_family_count = max(0, int(od_corridor_family_count or 0))
    support_aware_fast_path = False
    support_aware_aggressive_fast_path = False
    support_aware_fast_fallback_reasons = {
        "skipped_supported_ambiguity_fast_fallback",
        "skipped_support_aware_graph_search",
        "skipped_support_aware_long_corridor_graph_search",
        "skipped_support_rich_short_haul_graph_search",
        "skipped_support_backed_single_corridor_graph_search",
    }
    support_rich_short_haul_fast_path = False

    def _route_cache_runtime_touch(*, cache_key_value: str | None = None, hit: bool | None = None) -> None:
        if route_cache_runtime is None:
            return
        if cache_key_value is not None:
            route_cache_runtime["last_cache_key"] = cache_key_value
        if hit is True:
            route_cache_runtime["cache_hits"] = int(route_cache_runtime.get("cache_hits", 0)) + 1
        elif hit is False:
            route_cache_runtime["cache_misses"] = int(route_cache_runtime.get("cache_misses", 0)) + 1
        total_events = int(route_cache_runtime.get("cache_hits", 0)) + int(
            route_cache_runtime.get("cache_misses", 0)
        )
        route_cache_runtime["reuse_rate"] = round(
            int(route_cache_runtime.get("cache_hits", 0)) / float(max(1, total_events)),
            6,
        )

    def _ensure_candidate_ids(routes: Sequence[dict[str, Any]], *, prefix: str) -> None:
        seen_ids: set[str] = set()
        for index, route in enumerate(routes, start=1):
            candidate_id = str(route.get("candidate_id") or "").strip() or stable_candidate_id(route)
            if not candidate_id or candidate_id in seen_ids:
                candidate_id = f"{prefix}_{index:03d}"
            seen_ids.add(candidate_id)
            route["candidate_id"] = candidate_id

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
        _route_cache_runtime_touch(cache_key_value=cache_key)
        cached = get_cached_routes(cache_key)
        if cached is not None:
            _route_cache_runtime_touch(hit=True)
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
                refinement_policy=str(diag.get("refinement_policy", "")),
                selected_candidate_count=int(diag.get("selected_candidate_count", 0)),
                selected_candidate_ids_json=str(diag.get("selected_candidate_ids_json", "[]")),
            )
        _route_cache_runtime_touch(hit=False)

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
    reliability_corridor_threshold_km = min(
        long_corridor_threshold_km,
        max(60.0, long_corridor_threshold_km * 0.6),
    )
    reliability_corridor = corridor_distance_km >= reliability_corridor_threshold_km
    long_corridor = corridor_distance_km >= long_corridor_threshold_km
    long_corridor_stress_probe = bool(
        ambiguity_context_available
        and _long_corridor_stress_graph_probe_eligible(
            long_corridor=long_corridor,
            ambiguity_strength=ambiguity_strength,
            od_engine_disagreement_prior=od_engine_disagreement_prior,
            od_hard_case_prior=od_hard_case_prior,
            od_ambiguity_support_ratio=od_ambiguity_support_ratio,
            od_ambiguity_source_entropy=od_ambiguity_source_entropy,
            od_candidate_path_count=od_candidate_path_count,
            od_corridor_family_count=od_corridor_family_count,
        )
    )
    if long_corridor:
        long_corridor_path_cap = max(2, int(settings.route_graph_long_corridor_max_paths))
        max_paths = max(2, min(max_paths, max_routes, long_corridor_path_cap))
    reduced_initial_enabled = bool(settings.route_graph_reduced_initial_for_long_corridor)
    initial_use_transition_state = not (reduced_initial_enabled and reliability_corridor)
    long_corridor_fast_fallback_reasons = {
        "state_budget_exceeded",
        "path_search_exhausted",
        "no_path",
        "candidate_pool_exhausted",
        "skipped_long_corridor_graph_search",
    }
    skip_initial_graph_search = bool(
        long_corridor
        and settings.route_graph_skip_initial_search_long_corridor
        and not long_corridor_stress_probe
    )
    skip_retry_rescue_for_long_corridor = bool(
        long_corridor
        or (
            reliability_corridor
            and bool(settings.route_graph_skip_retry_rescue_reliability_corridor)
        )
    )
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
    support_rich_short_haul_fast_path = _support_rich_short_haul_fast_fallback_eligible(
        origin=origin,
        destination=destination,
        refinement_policy=legacy_selection_policy,
        search_budget=legacy_budget,
        od_ambiguity_index=od_ambiguity_index,
        od_engine_disagreement_prior=od_engine_disagreement_prior,
        od_hard_case_prior=od_hard_case_prior,
        od_ambiguity_support_ratio=od_ambiguity_support_ratio,
        od_ambiguity_source_entropy=od_ambiguity_source_entropy,
        od_candidate_path_count=od_candidate_path_count,
        od_corridor_family_count=od_corridor_family_count,
        allow_supported_ambiguity_fast_fallback=allow_supported_ambiguity_fast_fallback,
    )
    if skip_initial_graph_search:
        log_event(
            "route_graph_search_budget",
            request_id=request_id,
            pass_name="initial_skipped",
            max_paths=max_paths,
            configured_state_budget=configured_state_budget,
            corridor_distance_km=round(float(corridor_distance_km), 3),
            long_corridor_threshold_km=round(float(long_corridor_threshold_km), 3),
            long_corridor=bool(long_corridor),
            legacy_short_haul_corridor_uniform_fast_path=False,
            support_aware_fast_path=bool(support_aware_fast_path),
            reason=(
                "support_aware_long_corridor_fast_fallback"
                if support_aware_fast_path and long_corridor
                else (
                    "support_aware_fast_fallback"
                    if support_aware_fast_path
                    else "long_corridor_skip_enabled"
                )
            ),
        )
        support_aware_skip_reason = (
            "skipped_support_aware_long_corridor_graph_search"
            if support_aware_fast_path and long_corridor
            else (
                "skipped_supported_ambiguity_fast_fallback"
                if support_aware_fast_path
                else "skipped_long_corridor_graph_search"
            )
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
            no_path_reason=support_aware_skip_reason,
            no_path_detail="Skipped expensive graph search and used bounded fallback.",
        )
        graph_search_ms_initial = 0.0
    elif support_rich_short_haul_fast_path:
        skip_initial_graph_search = True
        log_event(
            "route_graph_search_budget",
            request_id=request_id,
            pass_name="initial_skipped",
            max_paths=max_paths,
            configured_state_budget=configured_state_budget,
            corridor_distance_km=round(float(corridor_distance_km), 3),
            long_corridor_threshold_km=round(float(long_corridor_threshold_km), 3),
            long_corridor=bool(long_corridor),
            legacy_short_haul_corridor_uniform_fast_path=True,
            support_aware_fast_path=bool(support_aware_fast_path),
            reason="support_rich_short_haul_fast_fallback",
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
            no_path_reason="skipped_support_rich_short_haul_graph_search",
            no_path_detail=(
                "Skipped expensive graph search because support-rich short-haul "
                "evidence made bounded fallback preferable."
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
            reliability_corridor_threshold_km=round(float(reliability_corridor_threshold_km), 3),
            long_corridor_threshold_km=round(float(long_corridor_threshold_km), 3),
            reliability_corridor=bool(reliability_corridor),
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
    legacy_short_haul_corridor_uniform_fast_path = bool(support_rich_short_haul_fast_path)
    short_haul_legacy_corridor_uniform_fail_open = bool(
        legacy_selection_policy == "corridor_uniform"
        and legacy_budget > 0
        and not long_corridor
        and not reliability_corridor
        and initial_no_path_reason in {"state_budget_exceeded", "path_search_exhausted", "candidate_pool_exhausted", "no_path"}
    )
    short_haul_legacy_corridor_uniform_fallback = bool(
        legacy_selection_policy == "corridor_uniform"
        and legacy_budget > 0
        and not long_corridor
        and not reliability_corridor
        and (
            legacy_short_haul_corridor_uniform_fast_path
            or short_haul_legacy_corridor_uniform_fail_open
            or initial_no_path_reason == "skipped_support_rich_short_haul_graph_search"
        )
    )
    supported_ambiguity_fast_fallback = _supported_ambiguity_fast_fallback_active(
        origin=origin,
        destination=destination,
        od_ambiguity_index=od_ambiguity_index,
        od_engine_disagreement_prior=od_engine_disagreement_prior,
        od_hard_case_prior=od_hard_case_prior,
        od_ambiguity_support_ratio=od_ambiguity_support_ratio,
        od_ambiguity_source_entropy=od_ambiguity_source_entropy,
        allow_supported_ambiguity_fast_fallback=allow_supported_ambiguity_fast_fallback,
    )
    should_retry = not (
        skip_retry_rescue_for_long_corridor and initial_no_path_reason in long_corridor_fast_fallback_reasons
    ) and not (
        supported_ambiguity_fast_fallback and initial_no_path_reason in {"state_budget_exceeded", "path_search_exhausted"}
    ) and not (
        support_aware_fast_path and initial_no_path_reason in support_aware_fast_fallback_reasons
    ) and not (
        short_haul_legacy_corridor_uniform_fail_open
        and initial_no_path_reason in {"state_budget_exceeded", "path_search_exhausted", "candidate_pool_exhausted", "no_path"}
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
    elif (
        not graph_routes
        and not should_retry
        and initial_no_path_reason in long_corridor_fast_fallback_reasons
        and not short_haul_legacy_corridor_uniform_fallback
    ):
        graph_retry_attempted = True
        graph_retry_state_budget = int(getattr(graph_diag, "effective_state_budget", 0) or configured_state_budget)
        graph_retry_outcome = (
            "skipped_support_rich_short_haul_fast_fallback"
            if support_rich_short_haul_fast_path and not long_corridor and not reliability_corridor
            else (
                "skipped_supported_ambiguity_fast_fallback"
                if supported_ambiguity_fast_fallback and not long_corridor and not reliability_corridor
                else (
                    "skipped_long_corridor_fast_fallback"
                    if long_corridor
                    else "skipped_reliability_corridor_fast_fallback"
                )
            )
        )
    elif (
        not graph_routes
        and not should_retry
        and initial_no_path_reason in support_aware_fast_fallback_reasons
        and not short_haul_legacy_corridor_uniform_fallback
    ):
        graph_retry_attempted = True
        graph_retry_state_budget = int(getattr(graph_diag, "effective_state_budget", 0) or configured_state_budget)
        graph_retry_outcome = initial_no_path_reason
    if not graph_routes:
        rescue_reason = str(graph_diag.no_path_reason or "").strip()
        rescue_mode_setting = str(settings.route_graph_state_space_rescue_mode or "reduced").strip().lower()
        rescue_mode = rescue_mode_setting if rescue_mode_setting in {"reduced", "full"} else "reduced"
        rescue_enabled = bool(settings.route_graph_state_space_rescue_enabled)
        rescue_candidate_reasons = {"state_budget_exceeded", "path_search_exhausted", "no_path"}
        short_haul_legacy_corridor_uniform_no_rescue = bool(
            legacy_selection_policy == "corridor_uniform"
            and legacy_budget > 0
            and not long_corridor
            and not reliability_corridor
            and (
                rescue_reason in {"state_budget_exceeded", "path_search_exhausted", "candidate_pool_exhausted", "no_path"}
                or rescue_reason == "skipped_support_rich_short_haul_graph_search"
            )
        )
        should_rescue = not (
            skip_retry_rescue_for_long_corridor and rescue_reason in long_corridor_fast_fallback_reasons
        ) and not (
            supported_ambiguity_fast_fallback and rescue_reason in {"state_budget_exceeded", "path_search_exhausted", "no_path"}
        ) and not (
            support_aware_fast_path and rescue_reason in support_aware_fast_fallback_reasons
        ) and not (
            short_haul_legacy_corridor_uniform_no_rescue
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
        elif (
            rescue_enabled
            and rescue_reason in rescue_candidate_reasons
            and not should_rescue
            and not short_haul_legacy_corridor_uniform_no_rescue
        ):
            graph_rescue_attempted = True
            graph_rescue_mode = (
                "supported_ambiguity_fast_fallback"
                if supported_ambiguity_fast_fallback and not long_corridor and not reliability_corridor
                else (
                    "long_corridor_fast_fallback"
                    if long_corridor
                    else "reliability_corridor_fast_fallback"
                )
            )
            graph_rescue_state_budget = int(getattr(graph_diag, "effective_state_budget", 0) or configured_state_budget)
            graph_rescue_outcome = (
                "skipped_supported_ambiguity_fast_fallback"
                if supported_ambiguity_fast_fallback and not long_corridor and not reliability_corridor
                else (
                    "skipped_long_corridor_fast_fallback"
                    if long_corridor
                    else "skipped_reliability_corridor_fast_fallback"
                )
            )
        elif rescue_enabled and short_haul_legacy_corridor_uniform_no_rescue and not should_rescue:
            graph_rescue_attempted = True
            graph_rescue_mode = "legacy_corridor_uniform_fast_fallback"
            graph_rescue_state_budget = int(getattr(graph_diag, "effective_state_budget", 0) or configured_state_budget)
            graph_rescue_outcome = rescue_reason
        elif (
            rescue_enabled
            and rescue_reason in support_aware_fast_fallback_reasons
            and not should_rescue
        ):
            graph_rescue_attempted = True
            graph_rescue_mode = "support_aware_fast_fallback"
            graph_rescue_state_budget = int(getattr(graph_diag, "effective_state_budget", 0) or configured_state_budget)
            graph_rescue_outcome = rescue_reason
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
    final_no_path_reason = str(graph_diag.no_path_reason or "").strip()
    recoverable_graph_search_fallback_reasons = {
        "state_budget_exceeded",
        "path_search_exhausted",
        "candidate_pool_exhausted",
        "no_path",
    }
    legacy_corridor_uniform_fast_fallback_reasons = {
        *recoverable_graph_search_fallback_reasons,
        "skipped_legacy_corridor_uniform_fast_fallback",
        "skipped_support_rich_short_haul_graph_search",
        "skipped_support_rich_short_haul_fast_fallback",
    }
    exhausted_reliability_corridor_search = bool(
        not graph_routes
        and reliability_corridor
        and not long_corridor
        and (graph_retry_attempted or graph_rescue_attempted)
        and (
            initial_no_path_reason in recoverable_graph_search_fallback_reasons
            or final_no_path_reason in recoverable_graph_search_fallback_reasons
        )
    )
    osrm_family_fallback_reasons = {
        "routing_graph_deferred_load",
    }
    short_haul_legacy_corridor_uniform_fallback = bool(
        not graph_routes
        and not long_corridor
        and not reliability_corridor
        and legacy_selection_policy == "corridor_uniform"
        and legacy_budget > 0
        and (
            legacy_short_haul_corridor_uniform_fast_path
            or initial_no_path_reason in legacy_corridor_uniform_fast_fallback_reasons
            or final_no_path_reason in legacy_corridor_uniform_fast_fallback_reasons
        )
    )
    should_run_osrm_family_fallback = bool(
        not graph_routes
        and (
            (
                long_corridor
                and initial_no_path_reason in long_corridor_fast_fallback_reasons
            )
            or exhausted_reliability_corridor_search
            or initial_no_path_reason in osrm_family_fallback_reasons
            or initial_no_path_reason in support_aware_fast_fallback_reasons
            or short_haul_legacy_corridor_uniform_fallback
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
        if legacy_selection_policy in {"first_n", "random_n"} and legacy_budget > 0:
            fallback_candidates = [
                {"candidate_id": str(spec.label), "graph_path": [str(spec.label)]}
                for spec in fallback_specs
            ]
            selected_fallback_ids = set(
                record.candidate_id
                for record in _baseline_policy_result(
                    candidates=fallback_candidates,
                    config=DCCSConfig(mode="challenger", search_budget=legacy_budget),
                    run_seed=int(run_seed or 0),
                    policy=legacy_selection_policy,
                ).selected
            )
            selected_candidate_ids = sorted(selected_fallback_ids)
            fallback_specs = [
                spec for spec in fallback_specs if str(spec.label) in selected_fallback_ids
            ]
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
                per_route_refine_cost_ms = max(
                    0.001,
                    float(result.elapsed_ms) / float(max(1, len(result.routes) or 1)),
                )
                for route in result.routes:
                    try:
                        sig = _route_signature(route)
                    except OSRMError:
                        continue
                    if sig not in fallback_routes_by_signature:
                        fallback_routes_by_signature[sig] = route
                    legacy_candidate_ids = fallback_routes_by_signature[sig].get("_legacy_candidate_ids")
                    if not isinstance(legacy_candidate_ids, list):
                        legacy_candidate_ids = []
                        fallback_routes_by_signature[sig]["_legacy_candidate_ids"] = legacy_candidate_ids
                    candidate_id = str(result.spec.label or "").strip()
                    if candidate_id and candidate_id not in legacy_candidate_ids:
                        legacy_candidate_ids.append(candidate_id)
                    _annotate_route_candidate_meta(
                        fallback_routes_by_signature[sig],
                        source_labels={
                            (
                                f"{result.spec.label}:legacy_corridor_uniform_osrm_fallback"
                                if short_haul_legacy_corridor_uniform_fallback
                                else f"{result.spec.label}:long_corridor_fallback"
                            )
                        },
                        toll_exclusion_available=False,
                        observed_refine_cost_ms=per_route_refine_cost_ms,
                    )
        osrm_refine_ms = round((time.perf_counter() - fallback_started) * 1000.0, 2)
        if fallback_routes_by_signature:
            if long_corridor and initial_no_path_reason in long_corridor_fast_fallback_reasons:
                warnings.append(
                    "route_graph: routing_graph_long_corridor_osrm_fallback "
                    f"(distance_km={corridor_distance_km:.1f}, engine_reason={initial_no_path_reason or 'unknown'})."
                )
                scenario_gate_action = "long_corridor_osrm_fallback"
            elif exhausted_reliability_corridor_search:
                warnings.append(
                    "route_graph: routing_graph_search_exhausted_osrm_fallback "
                    f"(distance_km={corridor_distance_km:.1f}, "
                    f"threshold_km={reliability_corridor_threshold_km:.1f}, "
                    f"initial_engine_reason={initial_no_path_reason or 'unknown'}, "
                    f"final_engine_reason={final_no_path_reason or 'unknown'})."
                )
                scenario_gate_action = "medium_long_osrm_fallback"
            elif short_haul_legacy_corridor_uniform_fallback:
                warnings.append(
                    "route_graph: routing_graph_legacy_corridor_uniform_osrm_fallback "
                    f"(initial_engine_reason={initial_no_path_reason or 'unknown'}, "
                    f"final_engine_reason={final_no_path_reason or 'unknown'})."
                )
                scenario_gate_action = "legacy_corridor_uniform_osrm_fallback"
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

    graph_candidate_count_raw = int(len(graph_routes))
    if graph_routes:
        _ensure_candidate_ids(graph_routes, prefix="legacy_graph")
        if legacy_selection_policy in {"first_n", "random_n", "corridor_uniform"} and legacy_budget > 0:
            selection_result = _baseline_policy_result(
                candidates=graph_routes,
                config=DCCSConfig(mode="challenger", search_budget=legacy_budget),
                run_seed=int(run_seed or 0),
                policy=legacy_selection_policy,
            )
            allowed_candidate_ids = {record.candidate_id for record in selection_result.selected}
            selected_candidate_ids = [record.candidate_id for record in selection_result.selected]
            graph_routes = [
                route
                for route in graph_routes
                if str(route.get("candidate_id") or "").strip() in allowed_candidate_ids
            ]
            warnings.append(
                "legacy_matched_budget_refinement: "
                f"policy={legacy_selection_policy} selected={len(graph_routes)}/{graph_candidate_count_raw} "
                f"under search_budget={legacy_budget}"
            )
        else:
            selected_candidate_ids = [
                str(route.get("candidate_id") or "").strip()
                for route in graph_routes
                if str(route.get("candidate_id") or "").strip()
            ]

    total_candidate_fetches = int(max(0, fallback_spec_count))
    family_specs: list[CandidateFetchSpec] = []
    family_candidate_ids: dict[str, str] = {}
    for idx, route in enumerate(graph_routes, start=1):
        via = _graph_family_via_points(
            route,
            max_landmarks=max(1, int(settings.route_graph_via_landmarks_per_path)),
        )
        candidate_id = str(route.get("candidate_id") or f"legacy_graph_{idx:03d}").strip()
        label = f"graph_family:{candidate_id}"
        family_specs.append(
            CandidateFetchSpec(
                label=label,
                alternatives=False,
                via=via if via else None,
            )
            )
        family_candidate_ids[label] = candidate_id
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
                legacy_candidate_ids = routes_by_signature[sig].get("_legacy_candidate_ids")
                if not isinstance(legacy_candidate_ids, list):
                    legacy_candidate_ids = []
                    routes_by_signature[sig]["_legacy_candidate_ids"] = legacy_candidate_ids
                candidate_id = family_candidate_ids.get(result.spec.label, "")
                if candidate_id and candidate_id not in legacy_candidate_ids:
                    legacy_candidate_ids.append(candidate_id)
                _annotate_route_candidate_meta(
                    routes_by_signature[sig],
                    source_labels={f"{result.spec.label}:osrm_refined"},
                    toll_exclusion_available=False,
                )
        osrm_refine_ms += round((time.perf_counter() - osrm_refine_started) * 1000.0, 2)

    if (
        not graph_routes
        and routes_by_signature
        and legacy_selection_policy == "corridor_uniform"
        and legacy_budget > 0
    ):
        realized_fallback_routes = list(routes_by_signature.values())
        _ensure_candidate_ids(realized_fallback_routes, prefix="legacy_fallback")
        selection_result = _baseline_policy_result(
            candidates=realized_fallback_routes,
            config=DCCSConfig(mode="challenger", search_budget=legacy_budget),
            run_seed=int(run_seed or 0),
            policy=legacy_selection_policy,
        )
        allowed_candidate_ids = {record.candidate_id for record in selection_result.selected}
        selected_candidate_ids = [record.candidate_id for record in selection_result.selected]
        routes_by_signature = {
            sig: route
            for sig, route in routes_by_signature.items()
            if str(route.get("candidate_id") or "").strip() in allowed_candidate_ids
        }
        warnings.append(
            "legacy_matched_budget_refinement: "
            f"policy={legacy_selection_policy} selected={len(routes_by_signature)}/{len(realized_fallback_routes)} "
            f"under search_budget={legacy_budget}"
        )

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
        raw_count=graph_candidate_count_raw,
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
        refinement_policy=legacy_selection_policy,
        selected_candidate_count=len(selected_candidate_ids),
        selected_candidate_ids_json=json.dumps(selected_candidate_ids, sort_keys=True),
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
                    "refinement_policy": diag.refinement_policy,
                    "selected_candidate_count": diag.selected_candidate_count,
                    "selected_candidate_ids_json": diag.selected_candidate_ids_json,
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
    route_cache_runtime_out: dict[str, Any] | None = None,
    route_option_cache_runtime_out: dict[str, Any] | None = None,
    refinement_policy: str | None = None,
    search_budget: int | None = None,
    run_seed: int | None = None,
    od_ambiguity_index: float | None = None,
    od_engine_disagreement_prior: float | None = None,
    od_hard_case_prior: float | None = None,
    od_ambiguity_support_ratio: float | None = None,
    od_ambiguity_source_entropy: float | None = None,
    od_candidate_path_count: int | None = None,
    od_corridor_family_count: int | None = None,
    allow_supported_ambiguity_fast_fallback: bool = False,
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
    default_route_option_cache_runtime = {
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
    route_option_cache_runtime = route_option_cache_runtime_out if route_option_cache_runtime_out is not None else {}
    route_option_cache_runtime.clear()
    route_option_cache_runtime.update(default_route_option_cache_runtime)
    default_route_cache_runtime = {
        "cache_hits": 0,
        "cache_misses": 0,
        "reuse_rate": 0.0,
        "last_cache_key": None,
    }
    route_cache_runtime = route_cache_runtime_out if route_cache_runtime_out is not None else {}
    route_cache_runtime.clear()
    route_cache_runtime.update(default_route_cache_runtime)
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
        should_prefetch = (not _repo_local_live_source_policy_enabled()) and (
            bool(settings.strict_live_data_required)
            or bool(settings.live_route_compute_require_all_expected)
            or refresh_mode in {
                "all_sources",
                "full",
            }
        )
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
    single_leg_legacy_search_context = _legacy_candidate_cache_context(
        origin=origin,
        destination=destination,
        refinement_policy=refinement_policy,
        search_budget=search_budget,
        run_seed=run_seed,
        od_ambiguity_index=od_ambiguity_index,
        od_engine_disagreement_prior=od_engine_disagreement_prior,
        od_hard_case_prior=od_hard_case_prior,
        od_ambiguity_support_ratio=od_ambiguity_support_ratio,
        od_ambiguity_source_entropy=od_ambiguity_source_entropy,
        od_candidate_path_count=od_candidate_path_count,
        od_corridor_family_count=od_corridor_family_count,
        allow_supported_ambiguity_fast_fallback=allow_supported_ambiguity_fast_fallback,
    )
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
            legacy_search_context=single_leg_legacy_search_context,
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
            refinement_policy=refinement_policy,
            search_budget=search_budget,
            run_seed=run_seed,
            od_ambiguity_index=od_ambiguity_index,
            od_engine_disagreement_prior=od_engine_disagreement_prior,
            od_hard_case_prior=od_hard_case_prior,
            od_ambiguity_support_ratio=od_ambiguity_support_ratio,
            od_ambiguity_source_entropy=od_ambiguity_source_entropy,
            od_candidate_path_count=od_candidate_path_count,
            od_corridor_family_count=od_corridor_family_count,
            allow_supported_ambiguity_fast_fallback=allow_supported_ambiguity_fast_fallback,
            route_cache_runtime=route_cache_runtime,
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
        scenario_policy_cache: dict[tuple[str, str], Any] = {}
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
            optimization_mode=optimization_mode,
            pareto_method=pareto_method,
            epsilon=epsilon,
            max_alternatives=max_alternatives,
            option_prefix=option_prefix,
            route_option_cache_runtime=route_option_cache_runtime,
            scenario_policy_cache=scenario_policy_cache,
            lightweight=True,
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
            legacy_search_context=_legacy_candidate_cache_context(
                origin=leg_origin,
                destination=leg_destination,
                refinement_policy=refinement_policy,
                search_budget=search_budget,
                run_seed=run_seed,
                od_ambiguity_index=None,
                od_engine_disagreement_prior=None,
                od_hard_case_prior=None,
                od_ambiguity_support_ratio=None,
                od_ambiguity_source_entropy=None,
                od_candidate_path_count=None,
                od_corridor_family_count=None,
                allow_supported_ambiguity_fast_fallback=False,
            ),
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
            refinement_policy=refinement_policy,
            search_budget=search_budget,
            run_seed=run_seed,
            od_ambiguity_index=None,
            od_engine_disagreement_prior=None,
            od_hard_case_prior=None,
            od_ambiguity_support_ratio=None,
            od_ambiguity_source_entropy=None,
            od_candidate_path_count=None,
            od_corridor_family_count=None,
            allow_supported_ambiguity_fast_fallback=False,
            route_cache_runtime=route_cache_runtime,
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
            optimization_mode=optimization_mode,
            pareto_method=pareto_method,
            epsilon=epsilon,
            max_alternatives=max_alternatives,
            option_prefix=f"{option_prefix}_leg{leg_index}",
            route_option_cache_runtime=route_option_cache_runtime,
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
                    od_ambiguity_index=req.od_ambiguity_index,
                    od_engine_disagreement_prior=req.od_engine_disagreement_prior,
                    od_hard_case_prior=req.od_hard_case_prior,
                    od_ambiguity_support_ratio=getattr(req, "od_ambiguity_support_ratio", None),
                    od_ambiguity_source_entropy=getattr(req, "od_ambiguity_source_entropy", None),
                    od_candidate_path_count=getattr(req, "od_candidate_path_count", None),
                    od_corridor_family_count=getattr(req, "od_corridor_family_count", None),
                    allow_supported_ambiguity_fast_fallback=True,
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
                    od_ambiguity_index=req.od_ambiguity_index,
                    od_engine_disagreement_prior=req.od_engine_disagreement_prior,
                    od_hard_case_prior=req.od_hard_case_prior,
                    od_ambiguity_support_ratio=getattr(req, "od_ambiguity_support_ratio", None),
                    od_ambiguity_source_entropy=getattr(req, "od_ambiguity_source_entropy", None),
                    od_candidate_path_count=getattr(req, "od_candidate_path_count", None),
                    od_corridor_family_count=getattr(req, "od_corridor_family_count", None),
                    allow_supported_ambiguity_fast_fallback=True,
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
        "tri_source": "tri_source",
        "thesis_voi": "voi",
        "voi_ad2r": "voi",
        "dccs+refc": "dccs_refc",
    }
    env_mode = aliases.get(
        str(settings.route_pipeline_default_mode or "tri_source").strip().lower(),
        "tri_source",
    )
    if env_mode not in {"legacy", "dccs", "dccs_refc", "voi", "tri_source"}:
        env_mode = "tri_source"
    if not bool(settings.route_pipeline_request_override_enabled):
        return env_mode
    request_mode = aliases.get(str(requested_mode or "").strip().lower(), "")
    if request_mode in {"legacy", "dccs", "dccs_refc", "voi", "tri_source"}:
        return request_mode
    return env_mode


def _selected_evidence_validation(
    evidence_validation: Mapping[str, Any] | None,
    *,
    route_id: str,
) -> dict[str, Any]:
    if not isinstance(evidence_validation, Mapping):
        return {}
    raw_validations = evidence_validation.get("validations", [])
    if not isinstance(raw_validations, Sequence) or isinstance(raw_validations, (str, bytes)):
        return {}
    for row in raw_validations:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("route_id", "")).strip() == route_id:
            return dict(row)
    return {}


def _route_certificate_map_for_decision_package(
    *,
    candidates: Sequence[RouteOption],
    selected: RouteOption,
    selected_certificate: RouteCertificationSummary | None,
    certificate_summary_payload: Mapping[str, Any] | None,
) -> dict[str, float]:
    certificate_map: dict[str, float] = {}
    payload = dict(certificate_summary_payload) if isinstance(certificate_summary_payload, Mapping) else {}
    raw_route_certificates = payload.get("route_certificates")
    if isinstance(raw_route_certificates, Mapping):
        for route_id, value in raw_route_certificates.items():
            route_text = str(route_id).strip()
            if not route_text:
                continue
            try:
                certificate_map[route_text] = round(float(value), 6)
            except (TypeError, ValueError):
                continue
    raw_rows = payload.get("route_certification_rows")
    if isinstance(raw_rows, Sequence) and not isinstance(raw_rows, (str, bytes)):
        for row in raw_rows:
            if not isinstance(row, Mapping):
                continue
            route_text = str(row.get("route_id", "")).strip()
            if not route_text:
                continue
            try:
                certificate_map[route_text] = round(float(row.get("certificate", 0.0)), 6)
            except (TypeError, ValueError):
                continue
    for option in candidates:
        if option.certification is None:
            continue
        certificate_map.setdefault(option.id, round(float(option.certification.certificate), 6))
    if selected_certificate is not None:
        certificate_map[selected.id] = round(float(selected_certificate.certificate), 6)
    return certificate_map


def _rewrite_public_pipeline_mode_in_artifacts(
    *,
    extra_json_artifacts: dict[str, dict[str, Any] | list[Any]],
    extra_jsonl_artifacts: dict[str, list[dict[str, Any]]],
    pipeline_mode: str,
) -> None:
    for payload in extra_json_artifacts.values():
        if isinstance(payload, dict) and "pipeline_mode" in payload:
            payload["pipeline_mode"] = pipeline_mode
    for rows in extra_jsonl_artifacts.values():
        if not isinstance(rows, Sequence):
            continue
        for row in rows:
            if isinstance(row, dict) and "pipeline_mode" in row:
                row["pipeline_mode"] = pipeline_mode


def _build_support_summary_for_decision_package(
    *,
    selected: RouteOption,
    selected_validation: Mapping[str, Any],
    certification_state: CertificationState | None,
) -> tuple[DecisionSupportSummary, dict[str, Any], list[dict[str, Any]]]:
    provenance = selected.evidence_provenance
    support_payload = certification_state.support_state.as_dict() if certification_state is not None else {}
    live_count = int(support_payload.get("live_family_count", selected_validation.get("live_family_count", 0)) or 0)
    snapshot_count = int(
        support_payload.get("snapshot_family_count", selected_validation.get("snapshot_family_count", 0)) or 0
    )
    model_count = int(support_payload.get("model_family_count", selected_validation.get("model_family_count", 0)) or 0)
    if provenance is not None and live_count == 0 and snapshot_count == 0 and model_count == 0:
        for record in provenance.families:
            marker = " ".join(
                str(value)
                for value in (record.family, record.source, record.signature, record.fallback_source)
                if value not in (None, "")
            ).lower()
            if record.family == "stochastic" or "model" in marker:
                model_count += 1
            elif "snapshot" in marker:
                snapshot_count += 1
            elif record.active:
                live_count += 1
    source_mix = list(provenance.active_families) if provenance is not None else []
    if not source_mix and certification_state is not None:
        source_mix = list(certification_state.world_bundle.active_families)
    if not source_mix and provenance is not None:
        source_mix = [record.family for record in provenance.families if str(record.family).strip()]
    source_mix = sorted({family for family in source_mix if str(family).strip()})
    recommended_fidelity = str(support_payload.get("recommended_fidelity", "")).strip()
    if not recommended_fidelity:
        if live_count > 0 and model_count > 0 and snapshot_count == 0:
            recommended_fidelity = "probabilistic"
        elif live_count > 0 and (snapshot_count > 0 or model_count > 0):
            recommended_fidelity = "probabilistic_audit"
        else:
            recommended_fidelity = "audit_first"
    if recommended_fidelity == "probabilistic":
        required_source_ids = ("live", "model")
    elif recommended_fidelity == "audit_first":
        required_source_ids = ("snapshot",) if snapshot_count > 0 else ("live",)
    else:
        required_source_ids = ("live", "snapshot", "model")
    latest_freshness_timestamp = None
    support_trace_rows: list[dict[str, Any]] = []
    if provenance is not None:
        for record in provenance.families:
            support_trace_rows.append(record.model_dump(mode="json"))
            if record.freshness_timestamp_utc:
                latest_freshness_timestamp = record.freshness_timestamp_utc
    support_strength = float(support_payload.get("support_strength", 1.0) or 1.0)
    source_entropy = support_payload.get("source_entropy")
    try:
        normalized_entropy = None if source_entropy is None else round(float(source_entropy), 6)
    except (TypeError, ValueError):
        normalized_entropy = None
    source_rows = [
        DecisionSupportSourceRecord(
            source_id="live",
            role="support_bucket",
            required="live" in required_source_ids,
            present=live_count > 0,
            status="ok" if live_count > 0 else "missing",
            freshness_timestamp_utc=latest_freshness_timestamp,
            provenance="live",
            details={"family_count": int(live_count)},
        ),
        DecisionSupportSourceRecord(
            source_id="snapshot",
            role="support_bucket",
            required="snapshot" in required_source_ids,
            present=snapshot_count > 0,
            status="ok" if snapshot_count > 0 else "missing",
            freshness_timestamp_utc=latest_freshness_timestamp,
            provenance="snapshot",
            details={"family_count": int(snapshot_count)},
        ),
        DecisionSupportSourceRecord(
            source_id="model",
            role="support_bucket",
            required="model" in required_source_ids,
            present=model_count > 0,
            status="ok" if model_count > 0 else "missing",
            freshness_timestamp_utc=latest_freshness_timestamp,
            provenance="model",
            details={"family_count": int(model_count)},
        ),
    ]
    missing_sources = [row.source_id for row in source_rows if row.required and not row.present]
    observed_source_count = len([row for row in source_rows if row.required and row.present])
    support_satisfied = bool(
        support_strength >= 0.5
        and not missing_sources
        and str(selected_validation.get("status", "ok")).strip().lower() != "rejected"
    )
    notes = [str(note) for note in support_payload.get("notes", ()) if str(note).strip()]
    if not notes and str(selected_validation.get("status", "ok")).strip().lower() != "ok":
        notes.append("validation_rejected")
    support_summary = DecisionSupportSummary(
        support_mode="tri_source_selective",
        required_source_count=len(required_source_ids),
        observed_source_count=observed_source_count,
        satisfied=support_satisfied,
        sources=source_rows,
        source_mix=source_mix,
        missing_sources=missing_sources,
        source_entropy=normalized_entropy,
        provenance_mode=recommended_fidelity,
        notes=notes,
    )
    support_provenance_payload = {
        "selected_route_id": selected.id,
        "active_families": list(provenance.active_families) if provenance is not None else [],
        "families": [record.model_dump(mode="json") for record in provenance.families] if provenance is not None else [],
        "validation": dict(selected_validation),
        "support_state": support_payload,
    }
    return support_summary, support_provenance_payload, support_trace_rows


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


def _refc_certification_frontier_options(
    *,
    pipeline_mode: str,
    strict_frontier: Sequence[RouteOption],
    options: Sequence[RouteOption],
    selected: RouteOption,
    ambiguity_context: Mapping[str, Any] | None,
    w_time: float,
    w_money: float,
    w_co2: float,
    optimization_mode: OptimizationMode,
    risk_aversion: float,
) -> tuple[list[RouteOption], dict[str, Any]]:
    strict_route_ids = [
        str(option.id).strip()
        for option in strict_frontier
        if str(option.id).strip()
    ]
    metadata = {
        "strict_frontier_route_ids": list(strict_route_ids),
        "strict_frontier_count": len(strict_route_ids),
        "certification_frontier_route_ids": list(strict_route_ids),
        "certification_frontier_count": len(strict_route_ids),
        "certification_frontier_rescue_applied": False,
        "certification_frontier_rescue_reason": "strict_frontier_reused",
        "certification_frontier_rescue_added_route_ids": [],
    }
    if pipeline_mode != "dccs_refc":
        metadata["certification_frontier_rescue_reason"] = "pipeline_not_refc"
        return list(strict_frontier), metadata
    if len(strict_route_ids) != 1:
        metadata["certification_frontier_rescue_reason"] = "strict_frontier_not_collapsed"
        return list(strict_frontier), metadata
    if not _refc_requires_full_stress_worlds(ambiguity_context):
        metadata["certification_frontier_rescue_reason"] = "ambiguity_support_insufficient"
        return list(strict_frontier), metadata
    if len(options) <= len(strict_frontier):
        metadata["certification_frontier_rescue_reason"] = "no_extra_refined_options"
        return list(strict_frontier), metadata

    selection_score_map = _route_selection_score_map(
        list(options),
        w_time=w_time,
        w_money=w_money,
        w_co2=w_co2,
        optimization_mode=optimization_mode,
        risk_aversion=risk_aversion,
    )
    ranked_options = sorted(
        list(options),
        key=lambda option: (
            float(selection_score_map.get(option.id, float("inf"))),
            str(option.id),
        ),
    )
    strict_ids = set(strict_route_ids)
    rescue_pool = [
        option
        for option in ranked_options
        if str(option.id).strip() and str(option.id).strip() not in strict_ids
    ]
    if not rescue_pool:
        metadata["certification_frontier_rescue_reason"] = "no_ranked_rescue_candidates"
        return list(strict_frontier), metadata

    frontier = list(strict_frontier)
    selected_id = str(selected.id).strip()
    if selected_id and all(str(option.id).strip() != selected_id for option in frontier):
        selected_option = next((option for option in ranked_options if str(option.id).strip() == selected_id), None)
        if selected_option is not None:
            frontier.insert(0, selected_option)

    target_count = min(3, len(strict_frontier) + len(rescue_pool))
    frontier_ids = {str(option.id).strip() for option in frontier if str(option.id).strip()}
    added_route_ids: list[str] = []
    for option in rescue_pool:
        option_id = str(option.id).strip()
        if not option_id or option_id in frontier_ids:
            continue
        frontier.append(option)
        frontier_ids.add(option_id)
        added_route_ids.append(option_id)
        if len(frontier_ids) >= target_count:
            break

    if not added_route_ids:
        metadata["certification_frontier_rescue_reason"] = "rescue_candidates_not_added"
        return list(strict_frontier), metadata

    certification_route_ids = [
        str(option.id).strip()
        for option in frontier
        if str(option.id).strip()
    ]
    metadata.update(
        {
            "certification_frontier_route_ids": certification_route_ids,
            "certification_frontier_count": len(certification_route_ids),
            "certification_frontier_rescue_applied": True,
            "certification_frontier_rescue_reason": "single_frontier_supported_ambiguity_rescue",
            "certification_frontier_rescue_added_route_ids": added_route_ids,
        }
    )
    return frontier, metadata


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

    time_scale = max(t_max - t_min, _route_selection_scale_floor("duration", t_min))
    money_scale = max(m_max - m_min, _route_selection_scale_floor("money", m_min))
    emissions_scale = max(e_max - e_min, _route_selection_scale_floor("co2", e_min))
    distance_scale = d_max - d_min

    def _norm(value: float, min_value: float, scale: float) -> float:
        return 0.0 if scale <= 0.0 else (value - min_value) / scale

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
            _norm(time_v, t_min, time_scale),
            _norm(money_v, m_min, money_scale),
            _norm(co2_v, e_min, emissions_scale),
            _norm(distance_v, d_min, distance_scale),
        )
        for time_v, money_v, co2_v, distance_v in zip(
            times,
            moneys,
            emissions,
            distances,
            strict=True,
        )
    ]
    time_preservation_penalties = _time_preservation_tradeoff_penalties(
        times,
        moneys,
        emissions,
        wt=wt,
        wm=wm,
        we=we,
    )
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
    for option, n_row, weighted_sum, weighted_regret, time_preservation_penalty in zip(
        options,
        norm_rows,
        weighted_sum_rows,
        weighted_regret_rows,
        time_preservation_penalties,
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
            score = (
                weighted_sum
                + (regret_weight * weighted_regret)
                + (balance_weight * balance_penalty)
                + time_preservation_penalty
            )
        elif math_profile == "modified_vikor_distance":
            score = (
                vikor_q
                + (balance_weight * balance_penalty)
                + (distance_weight * n_distance)
                + (eta_distance_weight * eta_distance_penalty)
                + (knee_weight * knee_penalty)
                - (entropy_weight * entropy_reward)
                + time_preservation_penalty
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
                + time_preservation_penalty
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
        # Preserve the original uncertainty shape under counterfactual objective
        # perturbations so robust selection sees refreshed/stressed tails instead
        # of the stale baseline summary.
        scaling_specs = (
            (
                float(option.metrics.duration_s),
                float(metrics.duration_s),
                (
                    "mean_duration_s",
                    "std_duration_s",
                    "q50_duration_s",
                    "q90_duration_s",
                    "q95_duration_s",
                    "p95_duration_s",
                    "cvar95_duration_s",
                ),
            ),
            (
                float(option.metrics.monetary_cost),
                float(metrics.monetary_cost),
                (
                    "mean_monetary_cost",
                    "std_monetary_cost",
                    "q50_monetary_cost",
                    "q90_monetary_cost",
                    "q95_monetary_cost",
                    "p95_monetary_cost",
                    "cvar95_monetary_cost",
                ),
            ),
            (
                float(option.metrics.emissions_kg),
                float(metrics.emissions_kg),
                (
                    "mean_emissions_kg",
                    "std_emissions_kg",
                    "q50_emissions_kg",
                    "q90_emissions_kg",
                    "q95_emissions_kg",
                    "p95_emissions_kg",
                    "cvar95_emissions_kg",
                ),
            ),
        )
        for base_value, updated_value, keys in scaling_specs:
            scale = 1.0
            if abs(base_value) > 1e-9:
                scale = max(0.0, float(updated_value)) / abs(base_value)
            else:
                raw_mean = uncertainty.get(keys[0])
                if raw_mean is not None:
                    try:
                        mean_value = abs(float(raw_mean))
                    except (TypeError, ValueError):
                        mean_value = 0.0
                    if mean_value > 1e-9:
                        scale = max(0.0, float(updated_value)) / mean_value
            for key in keys:
                raw_value = uncertainty.get(key)
                if raw_value is None:
                    continue
                try:
                    uncertainty[key] = max(0.0, float(raw_value) * scale)
                except (TypeError, ValueError):
                    continue
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
    segment_rows = option.segment_breakdown or []
    weather_summary = option.weather_summary or {}
    toll_cost = sum(
        max(0.0, float(row.get("toll_cost", 0.0)))
        for row in segment_rows
        if isinstance(row, dict)
    )
    if toll_cost <= 0.0:
        toll_cost = max(0.0, float(weather_summary.get("toll_cost_total_gbp", 0.0)))
    fuel_cost = sum(
        max(0.0, float(row.get("fuel_cost", 0.0)))
        for row in segment_rows
        if isinstance(row, dict)
    )
    if fuel_cost <= 0.0:
        fuel_cost = max(0.0, float(weather_summary.get("fuel_cost_total_gbp", 0.0)))
    carbon_cost = sum(
        max(0.0, float(row.get("carbon_cost", 0.0)))
        for row in segment_rows
        if isinstance(row, dict)
    )
    if carbon_cost <= 0.0:
        carbon_cost = max(0.0, float(weather_summary.get("carbon_cost_total_gbp", 0.0)))
    weather_delay_s = max(0.0, float(weather_summary.get("weather_delay_s", 0.0)))
    scenario_summary = option.scenario_summary
    terrain_summary = option.terrain_summary
    uncertainty = option.uncertainty or {}
    duration_s = max(1.0, float(option.metrics.duration_s))
    money_cost = max(1.0, float(option.metrics.monetary_cost))
    emissions_kg = max(1.0, float(option.metrics.emissions_kg))
    distance_km = max(1.0, float(option.metrics.distance_km))
    avg_speed_kmh = max(1.0, float(option.metrics.avg_speed_kmh or 0.0))
    terrain_ascent = float(terrain_summary.ascent_m if terrain_summary is not None else 0.0)
    terrain_descent = float(terrain_summary.descent_m if terrain_summary is not None else 0.0)
    terrain_grade_burden = min(1.0, (terrain_ascent + terrain_descent) / float(max(1.0, distance_km * 30.0)))
    speed_pressure = min(1.0, max(0.0, (avg_speed_kmh - 48.0) / 30.0))
    uncertainty_tail_time_ratio = min(
        1.0,
        max(
            0.0,
            (
                max(0.0, float(uncertainty.get("p95_duration_s", uncertainty.get("q95_duration_s", duration_s))) - duration_s)
                / duration_s
            )
            * 3.25,
        ),
    )
    emissions_intensity = min(1.0, max(0.0, emissions_kg / float(max(1.0, distance_km * 2.4))))
    stochastic_time_ratio = min(
        1.0,
        max(0.0, float(uncertainty.get("std_duration_s", 0.0)) / duration_s),
    )
    stochastic_money_ratio = min(
        1.0,
        max(0.0, float(uncertainty.get("std_monetary_cost", 0.0)) / money_cost),
    )
    stochastic_co2_ratio = min(
        1.0,
        max(0.0, float(uncertainty.get("std_emissions_kg", 0.0)) / emissions_kg),
    )
    scenario_time_delta = max(
        0.0,
        abs(float(scenario_summary.duration_multiplier) - 1.0) if scenario_summary is not None else 0.0,
    )
    scenario_money_delta = max(
        0.0,
        abs(float(scenario_summary.fuel_consumption_multiplier) - 1.0) if scenario_summary is not None else 0.0,
    )
    scenario_co2_delta = max(
        0.0,
        abs(float(scenario_summary.emissions_multiplier) - 1.0) if scenario_summary is not None else 0.0,
    )
    scenario_sigma_delta = max(
        0.0,
        abs(float(scenario_summary.stochastic_sigma_multiplier) - 1.0) if scenario_summary is not None else 0.0,
    )
    raw_components = {
        "time": {
            # Faster corridors are empirically less robust under scenario and
            # stochastic stress than slower but cleaner alternatives because
            # they rely more heavily on optimistic traffic/flow conditions.
            "scenario": max(0.02, scenario_time_delta + (0.35 * scenario_sigma_delta) + (0.28 * speed_pressure)),
            "toll": toll_cost / money_cost if toll_cost > 0.0 else 0.0,
            # Cap terrain's time contribution so steep routes remain distinguishable
            # without swamping stochastic and weather-driven ambiguity.
            "terrain": min(0.65, terrain_grade_burden * 0.75),
            "fuel": max(0.0, (fuel_cost / money_cost) * (0.15 + (0.05 * speed_pressure))),
            "carbon": 0.0,
            "weather": min(0.55, ((weather_delay_s / duration_s) * 2.0) + (0.08 * speed_pressure)),
            "stochastic": max(stochastic_time_ratio * 2.25, scenario_sigma_delta * 1.60)
            + (0.34 * speed_pressure)
            + (0.28 * uncertainty_tail_time_ratio),
        },
        "money": {
            "scenario": max(0.01, scenario_money_delta + (0.10 * uncertainty_tail_time_ratio)),
            "toll": toll_cost / money_cost,
            "terrain": terrain_grade_burden * 0.30,
            "fuel": (fuel_cost / money_cost) + (0.10 * speed_pressure),
            "carbon": carbon_cost / money_cost,
            "weather": (weather_delay_s / duration_s) * 0.25,
            "stochastic": max(stochastic_money_ratio, scenario_sigma_delta * 0.65) + (0.10 * uncertainty_tail_time_ratio),
        },
        "co2": {
            "scenario": max(0.01, scenario_co2_delta + (0.08 * uncertainty_tail_time_ratio)),
            "toll": 0.0,
            "terrain": terrain_grade_burden,
            "fuel": max(0.02, fuel_cost / money_cost) + (0.18 * emissions_intensity),
            "carbon": (max(0.0, carbon_cost / money_cost) * 0.15) + (0.08 * emissions_intensity),
            "weather": (weather_delay_s / duration_s) * 0.10,
            "stochastic": max(stochastic_co2_ratio, scenario_sigma_delta * 0.40) + (0.15 * uncertainty_tail_time_ratio),
        },
    }

    def _normalise_family_components(weights: Mapping[str, float]) -> dict[str, float]:
        clipped = {family: max(0.0, float(value)) for family, value in weights.items()}
        total = sum(clipped.values())
        if total <= 0.0:
            return {family: 0.0 for family in clipped}
        return {family: value / total for family, value in clipped.items()}

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
    families = sorted(out)
    for objective in ("time", "money", "co2"):
        objective_components = {
            family: raw_components[objective].get(family, 0.0) * max(0.05, out.get(family, {}).get(objective, 0.0))
            for family in families
        }
        normalized = _normalise_family_components(objective_components)
        for family in families:
            out.setdefault(family, {})[objective] = round(float(normalized.get(family, 0.0)), 6)
    return out


def _route_option_certification_payload(option: RouteOption) -> dict[str, Any]:
    dependency_weights = _route_option_dependency_weights(option)
    provenance_payload = option.evidence_provenance.model_dump(mode="json") if option.evidence_provenance else {}
    provenance_payload["dependency_weights"] = dependency_weights

    def _json_safe(value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return _json_safe(value.model_dump(mode="json"))
        if is_dataclass(value):
            return _json_safe(asdict(value))
        if isinstance(value, Mapping):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return [_json_safe(item) for item in value]
        if isinstance(value, list):
            return [_json_safe(item) for item in value]
        return value

    return {
        "route_id": option.id,
        "id": option.id,
        "objective_vector": (
            float(option.metrics.duration_s),
            float(option.metrics.monetary_cost),
            float(option.metrics.emissions_kg),
        ),
        "distance_km": float(option.metrics.distance_km),
        "metrics": _json_safe(option.metrics),
        "segment_breakdown": _json_safe(option.segment_breakdown),
        "weather_summary": _json_safe(option.weather_summary),
        "scenario_summary": _json_safe(option.scenario_summary),
        "terrain_summary": _json_safe(option.terrain_summary),
        "uncertainty": _json_safe(option.uncertainty),
        "evidence_tensor": dependency_weights,
        "evidence_provenance": provenance_payload,
    }


def _global_certification_cache_payload(
    *,
    certificate_result: CertificateResult,
    fragility_result: FragilityResult,
    world_manifest_payload: Mapping[str, Any] | None,
    active_families: Sequence[str],
) -> tuple[CertificateResult, FragilityResult, dict[str, Any], list[str]]:
    manifest_payload = (
        copy.deepcopy(dict(world_manifest_payload))
        if isinstance(world_manifest_payload, Mapping)
        else copy.deepcopy(dict(certificate_result.world_manifest))
    )
    compact_certificate = CertificateResult(
        winner_id=certificate_result.winner_id,
        certificate=copy.deepcopy(certificate_result.certificate),
        threshold=float(certificate_result.threshold),
        certified=bool(certificate_result.certified),
        selected_route_id=certificate_result.selected_route_id,
        route_scores={},
        world_manifest=copy.deepcopy(manifest_payload),
        selector_config=copy.deepcopy(certificate_result.selector_config),
    )
    compact_fragility = FragilityResult(
        route_fragility_map=copy.deepcopy(fragility_result.route_fragility_map),
        competitor_fragility_breakdown=copy.deepcopy(
            fragility_result.competitor_fragility_breakdown
        ),
        value_of_refresh=copy.deepcopy(fragility_result.value_of_refresh),
        route_fragility_details={},
        evidence_snapshot_manifest={},
    )
    return (
        compact_certificate,
        compact_fragility,
        manifest_payload,
        list(active_families),
    )


CERTIFICATION_CACHE_VERSION = "refc_margin_refresh_v4"


def _certification_cache_key(
    *,
    frontier_route_ids: Sequence[str],
    frontier_signatures: Mapping[str, str],
    route_payloads: Sequence[Mapping[str, Any]],
    evidence_snapshot_hash: str | None,
    selected_route_id: str,
    run_seed: int,
    world_count: int,
    threshold: float,
    weights: Sequence[float],
    optimization_mode: str,
    risk_aversion: float,
    forced_refreshed_families: Sequence[str],
    ambiguity_context: Mapping[str, Any] | None,
    force_single_frontier_full_stress_requested_worlds: bool = False,
) -> str:
    normalized_context = {
        key: _cache_key_component(value)
        for key, value in dict(ambiguity_context or {}).items()
    }
    cache_payload = {
        "cache_version": CERTIFICATION_CACHE_VERSION,
        "frontier_route_ids": [str(route_id) for route_id in frontier_route_ids],
        "frontier_signatures": {str(key): str(value) for key, value in frontier_signatures.items()},
        "route_payloads": list(route_payloads),
        "evidence_snapshot_hash": str(evidence_snapshot_hash or ""),
        "selected_route_id": str(selected_route_id),
        "run_seed": int(run_seed),
        "world_count": int(world_count),
        "threshold": float(threshold),
        "weights": tuple(float(weight) for weight in weights),
        "optimization_mode": str(optimization_mode),
        "risk_aversion": float(risk_aversion),
        "forced_refreshed_families": sorted(str(family) for family in forced_refreshed_families),
        "ambiguity_context": normalized_context,
        "force_single_frontier_full_stress_requested_worlds": bool(
            force_single_frontier_full_stress_requested_worlds
        ),
    }
    return hashlib.sha1(
        json.dumps(
            cache_payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _route_objective_vector(route: Mapping[str, Any]) -> tuple[float, float, float]:
    raw = route.get("objective_vector")
    if isinstance(raw, Mapping):
        return (
            float(raw.get("time", 0.0)),
            float(raw.get("money", 0.0)),
            float(raw.get("co2", 0.0)),
        )
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) >= 3:
        return (
            float(raw[0]),
            float(raw[1]),
            float(raw[2]),
        )
    metrics = route.get("metrics")
    if isinstance(metrics, Mapping):
        return (
            float(metrics.get("duration_s", 0.0)),
            float(metrics.get("monetary_cost", 0.0)),
            float(metrics.get("emissions_kg", 0.0)),
        )
    return (
        float(route.get("time", 0.0)),
        float(route.get("money", 0.0)),
        float(route.get("co2", 0.0)),
    )


def _validate_route_options_evidence(options: Sequence[RouteOption]) -> dict[str, Any]:
    validations: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for option in options:
        validation = validate_route_evidence_provenance(
            option.model_dump(mode="json"),
            allow_snapshot=True,
            require_freshness=True,
        ).as_dict()
        validation["route_id"] = option.id
        validations.append(validation)
        for issue in validation.get("issues", []):
            issue_payload = dict(issue)
            issue_payload["route_id"] = option.id
            issues.append(issue_payload)
    return {
        "status": "ok" if not issues else "rejected",
        "issues": issues,
        "validations": validations,
    }


def _apply_world_state_overrides(
    worlds: Sequence[Mapping[str, Any]],
    *,
    forced_refreshed_families: set[str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for world in worlds:
        world_payload = dict(world)
        states = dict(world.get("states", {}))
        for family in forced_refreshed_families:
            if family in states:
                states[family] = "refreshed"
        world_payload["states"] = states
        out.append(world_payload)
    return out


def _request_ambiguity_context(req: RouteRequest) -> dict[str, Any]:
    return {
        "od_ambiguity_index": req.od_ambiguity_index,
        "od_ambiguity_confidence": req.od_ambiguity_confidence,
        "od_engine_disagreement_prior": req.od_engine_disagreement_prior,
        "od_hard_case_prior": req.od_hard_case_prior,
        "od_ambiguity_source_count": req.od_ambiguity_source_count,
        "od_ambiguity_source_mix": req.od_ambiguity_source_mix,
        "od_ambiguity_support_ratio": getattr(req, "od_ambiguity_support_ratio", None),
        "od_ambiguity_source_mix_count": getattr(req, "od_ambiguity_source_mix_count", None),
        "od_ambiguity_source_entropy": getattr(req, "od_ambiguity_source_entropy", None),
        "od_ambiguity_prior_strength": getattr(req, "od_ambiguity_prior_strength", None),
        "od_ambiguity_family_density": getattr(req, "od_ambiguity_family_density", None),
        "od_ambiguity_margin_pressure": getattr(req, "od_ambiguity_margin_pressure", None),
        "od_ambiguity_spread_pressure": getattr(req, "od_ambiguity_spread_pressure", None),
        "od_ambiguity_toll_instability": getattr(req, "od_ambiguity_toll_instability", None),
        "od_candidate_path_count": req.od_candidate_path_count,
        "od_corridor_family_count": req.od_corridor_family_count,
        "od_objective_spread": req.od_objective_spread,
        "od_nominal_margin_proxy": req.od_nominal_margin_proxy,
        "od_toll_disagreement_rate": req.od_toll_disagreement_rate,
        "ambiguity_budget_prior": req.ambiguity_budget_prior,
        "ambiguity_budget_band": req.ambiguity_budget_band,
    }


def _ambiguity_strength_from_context(context: Mapping[str, Any] | None) -> float:
    if not isinstance(context, Mapping):
        return 0.0
    values = [
        float(context.get("od_ambiguity_index") or 0.0),
        float(context.get("od_engine_disagreement_prior") or 0.0),
        float(context.get("od_hard_case_prior") or 0.0),
    ]
    return max(0.0, min(1.0, max(values)))


def _single_frontier_shortcut_certificate_value(
    *,
    requested_world_count: int,
    threshold: float,
    ambiguity_context: Mapping[str, Any] | None,
) -> float:
    requested = max(1, int(requested_world_count))
    if requested <= 1:
        return 1.0
    context = ambiguity_context if isinstance(ambiguity_context, Mapping) else {}

    def _context_float(key: str) -> float:
        return max(0.0, min(1.0, float(context.get(key) or 0.0)))

    raw_source_mix = context.get("od_ambiguity_source_mix")
    explicit_source_mix_count = max(0, int(float(context.get("od_ambiguity_source_mix_count") or 0.0)))
    source_mix_count = explicit_source_mix_count
    if isinstance(raw_source_mix, str):
        source_mix_text = raw_source_mix.strip()
        parsed_source_mix = None
        if source_mix_text[:1] in {"{", "["}:
            try:
                parsed_source_mix = json.loads(source_mix_text)
            except json.JSONDecodeError:
                parsed_source_mix = None
        if isinstance(parsed_source_mix, Mapping):
            source_mix_count = max(
                source_mix_count,
                len([key for key in parsed_source_mix if str(key).strip()]),
            )
        elif isinstance(parsed_source_mix, Sequence) and not isinstance(parsed_source_mix, (str, bytes)):
            source_mix_count = max(
                source_mix_count,
                len([item for item in parsed_source_mix if str(item).strip()]),
            )
        elif source_mix_text:
            source_mix_count = max(
                source_mix_count,
                len([item for item in source_mix_text.replace("+", ",").split(",") if item.strip()]),
            )
    elif isinstance(raw_source_mix, Sequence) and not isinstance(raw_source_mix, (str, bytes)):
        source_mix_count = max(
            source_mix_count,
            len([item for item in raw_source_mix if str(item).strip()]),
        )

    ambiguity_confidence = _context_float("od_ambiguity_confidence")
    support_ratio = _context_float("od_ambiguity_support_ratio")
    source_entropy = _context_float("od_ambiguity_source_entropy")
    source_count = max(0, int(float(context.get("od_ambiguity_source_count") or 0.0)))
    source_count_strength = min(1.0, source_count / 4.0) if source_count > 0 else 0.0
    source_mix_strength = min(1.0, source_mix_count / 3.0) if source_mix_count > 0 else 0.0
    support_strength = max(
        0.0,
        min(
            1.0,
            (0.35 * ambiguity_confidence)
            + (0.18 * source_count_strength)
            + (0.12 * source_mix_strength)
            + (0.20 * support_ratio)
            + (0.15 * source_entropy),
        ),
    )
    path_count = max(0, int(float(context.get("od_candidate_path_count") or 0.0)))
    family_count = max(0, int(float(context.get("od_corridor_family_count") or 0.0)))
    structural_support = max(
        min(1.0, max(0, path_count - 1) / 3.0),
        min(1.0, max(0, family_count - 1) / 2.0),
        source_mix_strength,
        source_count_strength,
    )
    ambiguity_signal = max(
        _context_float("od_ambiguity_index"),
        _context_float("od_hard_case_prior"),
        _context_float("od_engine_disagreement_prior"),
        _context_float("ambiguity_budget_prior"),
    )
    requested_pressure = min(1.0, max(0, requested - 1) / 96.0)
    budget_band = str(context.get("ambiguity_budget_band") or "").strip().lower()
    budget_pressure = 1.0 if budget_band == "high" else (0.45 if budget_band == "medium" else 0.0)
    uncertainty_mass = max(
        0.0,
        min(
            0.33,
            0.04
            + (0.10 * ambiguity_signal)
            + (0.08 * support_strength)
            + (0.05 * structural_support)
            + (0.04 * requested_pressure)
            + (0.03 * budget_pressure),
        ),
    )
    conservative_value = max(
        min(0.98, 1.0 - uncertainty_mass),
        min(0.95, max(0.67, float(threshold) - 0.08)),
    )
    return round(conservative_value, 6)


def _single_frontier_observed_coverage_relief(
    *,
    requested_world_count: int,
    observed_world_count: int,
    world_manifest: Mapping[str, Any] | None,
    ambiguity_context: Mapping[str, Any] | None,
    full_stress_required: bool,
) -> tuple[float, float, float]:
    if not full_stress_required:
        return 0.0, 0.0, 0.0
    requested = max(1, int(requested_world_count))
    observed = max(0, int(observed_world_count))
    if observed <= 1:
        return 0.0, 0.0, 0.0
    context = ambiguity_context if isinstance(ambiguity_context, Mapping) else {}
    manifest = world_manifest if isinstance(world_manifest, Mapping) else {}

    def _context_float(key: str) -> float:
        return max(0.0, min(1.0, float(context.get(key) or 0.0)))

    stress_fraction_raw = (
        manifest.get("stress_world_fraction")
        if manifest.get("stress_world_fraction") is not None
        else (
            manifest.get("refc_stress_world_fraction")
            if manifest.get("refc_stress_world_fraction") is not None
            else manifest.get("hard_case_stress_world_fraction")
        )
    )
    try:
        stress_fraction = float(stress_fraction_raw or 0.0)
    except (TypeError, ValueError):
        stress_fraction = 0.0
    if not math.isfinite(stress_fraction):
        stress_fraction = 0.0
    stress_fraction = max(0.0, min(1.0, stress_fraction))
    coverage_ratio = max(0.0, min(1.0, float(observed) / float(requested)))
    depth_score = max(0.0, min(1.0, float(observed) / float(max(requested, 64))))
    stress_score = min(1.0, stress_fraction / 0.25) if stress_fraction > 0.0 else 0.0
    support_score = max(
        _context_float("od_ambiguity_support_ratio"),
        _context_float("od_hard_case_prior"),
        _context_float("od_engine_disagreement_prior"),
        _context_float("ambiguity_budget_prior"),
        _context_float("od_ambiguity_source_entropy"),
        _context_float("od_ambiguity_confidence"),
    )
    relief = min(
        0.07,
        0.07
        * max(coverage_ratio, depth_score)
        * max(0.35, support_score)
        * max(0.35, stress_score),
    )
    return round(relief, 6), round(coverage_ratio, 6), round(stress_fraction, 6)


def _apply_single_frontier_certificate_cap(
    *,
    certificate: CertificateResult,
    fragility: FragilityResult,
    world_manifest: Mapping[str, Any] | None,
    selected_route_id: str,
    requested_world_count: int,
    threshold: float,
    ambiguity_context: Mapping[str, Any] | None,
    full_stress_required: bool,
) -> tuple[CertificateResult, FragilityResult, dict[str, Any]]:
    manifest_payload = dict(world_manifest or {})
    selected_id = (
        str(selected_route_id).strip()
        or str(certificate.selected_route_id).strip()
        or str(certificate.winner_id).strip()
    )
    empirical_value = float(certificate.certificate.get(selected_id, 0.0))
    cap_value = _single_frontier_shortcut_certificate_value(
        requested_world_count=int(requested_world_count),
        threshold=float(threshold),
        ambiguity_context=ambiguity_context,
    )
    manifest_worlds = manifest_payload.get("worlds")
    manifest_world_count = int(manifest_payload.get("world_count") or 0)
    if isinstance(manifest_worlds, Sequence) and not isinstance(manifest_worlds, (str, bytes)):
        manifest_world_count = max(manifest_world_count, len(manifest_worlds))
    effective_world_count = int(
        manifest_payload.get(
            "effective_world_count",
            manifest_world_count,
        )
        or 0
    )
    if full_stress_required and manifest_world_count > 0:
        effective_world_count = manifest_world_count
        manifest_payload["world_count_policy"] = "single_frontier_full_stress"
    coverage_relief, observed_coverage_ratio, observed_stress_fraction = _single_frontier_observed_coverage_relief(
        requested_world_count=int(requested_world_count),
        observed_world_count=effective_world_count or manifest_world_count,
        world_manifest=manifest_payload,
        ambiguity_context=ambiguity_context,
        full_stress_required=full_stress_required,
    )
    observed_coverage_ceiling = min(
        empirical_value,
        max(
            cap_value,
            min(0.99, max(0.0, float(threshold) - 0.01)),
        ),
    )
    if coverage_relief > 0.0:
        cap_value = min(
            empirical_value,
            max(cap_value, min(observed_coverage_ceiling, cap_value + coverage_relief)),
        )
    adjusted_value = round(min(empirical_value, cap_value), 6)
    cap_applied = adjusted_value + 1e-9 < empirical_value
    manifest_payload["effective_world_count"] = effective_world_count
    manifest_payload["single_frontier_empirical_certificate"] = round(empirical_value, 6)
    manifest_payload["single_frontier_certificate_cap"] = round(cap_value, 6)
    manifest_payload["single_frontier_certificate_cap_applied"] = bool(cap_applied)
    manifest_payload["single_frontier_requires_full_stress"] = bool(full_stress_required)
    manifest_payload["single_frontier_observed_coverage_ratio"] = observed_coverage_ratio
    manifest_payload["single_frontier_observed_coverage_relief"] = round(coverage_relief, 6)
    manifest_payload["single_frontier_observed_coverage_ceiling"] = round(observed_coverage_ceiling, 6)
    manifest_payload["single_frontier_observed_stress_fraction"] = observed_stress_fraction
    manifest_payload["selected_certificate_basis"] = (
        "single_frontier_structural_cap" if cap_applied else "empirical"
    )

    selector_config = dict(certificate.selector_config)
    selector_config["single_frontier_empirical_certificate"] = round(empirical_value, 6)
    selector_config["single_frontier_certificate_cap"] = round(cap_value, 6)
    selector_config["single_frontier_certificate_cap_applied"] = bool(cap_applied)
    selector_config["single_frontier_requires_full_stress"] = bool(full_stress_required)
    selector_config["single_frontier_observed_coverage_ratio"] = observed_coverage_ratio
    selector_config["single_frontier_observed_coverage_relief"] = round(coverage_relief, 6)
    selector_config["single_frontier_observed_coverage_ceiling"] = round(observed_coverage_ceiling, 6)
    selector_config["single_frontier_observed_stress_fraction"] = observed_stress_fraction
    selector_config["selected_certificate_basis"] = manifest_payload["selected_certificate_basis"]

    adjusted_certificate_map = dict(certificate.certificate)
    adjusted_certificate_map[selected_id] = adjusted_value
    adjusted_certificate = CertificateResult(
        winner_id=certificate.winner_id,
        certificate=adjusted_certificate_map,
        threshold=float(certificate.threshold),
        certified=adjusted_value >= float(threshold),
        selected_route_id=certificate.selected_route_id,
        route_scores=copy.deepcopy(certificate.route_scores),
        world_manifest=dict(manifest_payload),
        selector_config=selector_config,
    )

    value_of_refresh = copy.deepcopy(fragility.value_of_refresh)
    if isinstance(value_of_refresh, Mapping):
        value_of_refresh = dict(value_of_refresh)
        value_of_refresh["empirical_baseline_certificate"] = float(
            value_of_refresh.get("baseline_certificate", empirical_value)
        )
        value_of_refresh["controller_baseline_certificate"] = adjusted_value
        value_of_refresh["single_frontier_certificate_cap"] = round(cap_value, 6)
        value_of_refresh["single_frontier_certificate_cap_applied"] = bool(cap_applied)
        value_of_refresh["single_frontier_requires_full_stress"] = bool(full_stress_required)
        value_of_refresh["single_frontier_observed_coverage_ratio"] = observed_coverage_ratio
        value_of_refresh["single_frontier_observed_coverage_relief"] = round(coverage_relief, 6)
        value_of_refresh["single_frontier_observed_coverage_ceiling"] = round(observed_coverage_ceiling, 6)
        value_of_refresh["single_frontier_observed_stress_fraction"] = observed_stress_fraction

    evidence_snapshot_manifest = copy.deepcopy(fragility.evidence_snapshot_manifest)
    if isinstance(evidence_snapshot_manifest, Mapping):
        evidence_snapshot_manifest = dict(evidence_snapshot_manifest)
        evidence_snapshot_manifest["single_frontier_certificate_cap"] = round(cap_value, 6)
        evidence_snapshot_manifest["single_frontier_certificate_cap_applied"] = bool(cap_applied)
        evidence_snapshot_manifest["single_frontier_requires_full_stress"] = bool(full_stress_required)
        evidence_snapshot_manifest["single_frontier_observed_coverage_ratio"] = observed_coverage_ratio
        evidence_snapshot_manifest["single_frontier_observed_coverage_relief"] = round(coverage_relief, 6)
        evidence_snapshot_manifest["single_frontier_observed_coverage_ceiling"] = round(observed_coverage_ceiling, 6)
        evidence_snapshot_manifest["single_frontier_observed_stress_fraction"] = observed_stress_fraction

    adjusted_fragility = FragilityResult(
        route_fragility_map=copy.deepcopy(fragility.route_fragility_map),
        competitor_fragility_breakdown=copy.deepcopy(fragility.competitor_fragility_breakdown),
        value_of_refresh=value_of_refresh,
        route_fragility_details=copy.deepcopy(fragility.route_fragility_details),
        evidence_snapshot_manifest=evidence_snapshot_manifest,
    )
    return adjusted_certificate, adjusted_fragility, manifest_payload


def _merge_controller_refresh_overlay(
    *,
    report_fragility: FragilityResult,
    controller_fragility: FragilityResult | None,
    controller_frontier_route_ids: Sequence[str],
    controller_frontier_mode: str,
) -> FragilityResult:
    value_of_refresh = copy.deepcopy(report_fragility.value_of_refresh)
    if isinstance(value_of_refresh, Mapping):
        value_of_refresh = dict(value_of_refresh)
    else:
        value_of_refresh = {}
    controller_route_ids = [
        str(route_id).strip()
        for route_id in controller_frontier_route_ids
        if str(route_id).strip()
    ]
    value_of_refresh["controller_refresh_frontier_mode"] = str(controller_frontier_mode or "").strip() or None
    value_of_refresh["controller_refresh_frontier_route_ids"] = list(controller_route_ids)
    value_of_refresh["controller_refresh_frontier_count"] = len(controller_route_ids)
    if controller_fragility is not None and isinstance(controller_fragility.value_of_refresh, Mapping):
        overlay = dict(controller_fragility.value_of_refresh)
        for key in (
            "controller_ranking_basis",
            "controller_ranking",
            "top_refresh_family_controller",
            "top_refresh_gain_controller",
            "empirical_baseline_certificate",
            "controller_baseline_certificate",
        ):
            if key in overlay:
                value_of_refresh[key] = copy.deepcopy(overlay[key])
    return FragilityResult(
        route_fragility_map=copy.deepcopy(report_fragility.route_fragility_map),
        competitor_fragility_breakdown=copy.deepcopy(report_fragility.competitor_fragility_breakdown),
        value_of_refresh=value_of_refresh,
        route_fragility_details=copy.deepcopy(report_fragility.route_fragility_details),
        evidence_snapshot_manifest=copy.deepcopy(report_fragility.evidence_snapshot_manifest),
    )


def _initial_controller_overconfidence_cap(
    *,
    controller_state: VOIControllerState,
    current_certificate: float,
    threshold: float,
) -> tuple[float, bool]:
    if not math.isfinite(current_certificate) or not math.isfinite(threshold):
        return current_certificate, False
    if controller_state.iteration_index != 0:
        return round(current_certificate, 6), False
    if current_certificate < threshold:
        return round(current_certificate, 6), False
    context = (
        controller_state.ambiguity_context
        if isinstance(controller_state.ambiguity_context, Mapping)
        else {}
    )
    if bool(context.get("single_frontier_certificate_cap_applied")):
        return round(current_certificate, 6), False

    completeness_gap = max(0.0, float(controller_state.search_completeness_gap or 0.0))
    frontier_recall = max(0.0, min(1.0, float(controller_state.frontier_recall_at_budget or 0.0)))
    recall_gap = max(0.0, 1.0 - frontier_recall)
    if completeness_gap < 0.18 and recall_gap < 0.45:
        return round(current_certificate, 6), False

    top_refresh_gain = max(0.0, min(1.0, float(controller_state.top_refresh_gain or 0.0)))
    top_fragility_mass = max(0.0, min(1.0, float(controller_state.top_fragility_mass or 0.0)))
    near_tie_mass = max(0.0, min(1.0, float(controller_state.near_tie_mass or 0.0)))
    signal_strength = max(top_refresh_gain, top_fragility_mass, near_tie_mass)
    if signal_strength < 0.18:
        return round(current_certificate, 6), False

    pending_mass = max(0.0, min(1.0, float(controller_state.pending_challenger_mass or 0.0)))
    pending_flip = max(0.0, min(1.0, float(controller_state.best_pending_flip_probability or 0.0)))
    if pending_mass < 0.20 and pending_flip < 0.35:
        return round(current_certificate, 6), False

    support_strength = max(
        0.0,
        min(
            1.0,
            max(
                float(controller_state.prior_support_strength or 0.0),
                float(context.get("od_ambiguity_support_ratio") or 0.0),
                float(context.get("ambiguity_budget_prior") or 0.0),
                float(context.get("od_hard_case_prior") or 0.0),
            ),
        ),
    )
    if support_strength < 0.35:
        return round(current_certificate, 6), False

    risk = min(
        1.0,
        (0.32 * completeness_gap)
        + (0.26 * recall_gap)
        + (0.22 * pending_mass)
        + (0.10 * pending_flip)
        + (0.10 * signal_strength),
    )
    cap_drop = min(0.35, 0.55 * risk)
    if cap_drop <= 1e-9:
        return round(current_certificate, 6), False
    capped_value = min(current_certificate, max(0.0, threshold - cap_drop))
    if capped_value + 1e-9 >= current_certificate:
        return round(current_certificate, 6), False
    return round(capped_value, 6), True


def _adaptive_refc_world_count(
    *,
    requested_world_count: int,
    frontier_size: int,
    ambiguity_context: Mapping[str, Any] | None,
) -> tuple[int, str]:
    requested = max(1, int(requested_world_count))
    if frontier_size <= 1:
        return 1, "single_frontier_shortcut"
    if not bool(settings.route_refc_adaptive_world_count_enabled):
        return requested, "configured"
    ambiguity_strength = _ambiguity_strength_from_context(ambiguity_context)
    if ambiguity_strength <= float(settings.route_graph_fast_path_max_ambiguity):
        cap = max(1, int(settings.route_refc_low_ambiguity_world_cap))
        return min(requested, cap), "adaptive_low_ambiguity"
    if ambiguity_strength >= max(
        float(settings.route_dccs_preemptive_comparator_min_ambiguity),
        float(settings.route_dccs_preemptive_comparator_min_hard_case),
    ) or frontier_size >= 3:
        floor = max(1, int(settings.route_refc_high_ambiguity_world_floor))
        return max(requested, floor), "adaptive_high_ambiguity"
    cap = max(1, int(settings.route_refc_medium_ambiguity_world_cap))
    return min(requested, cap), "adaptive_medium_ambiguity"


def _refc_requires_full_stress_worlds(ambiguity_context: Mapping[str, Any] | None) -> bool:
    return refc_requires_full_stress_worlds(ambiguity_context)


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
    ambiguity_context: Mapping[str, Any] | None = None,
    force_single_frontier_full_stress_requested_worlds: bool = False,
) -> tuple[Any, Any, dict[str, Any], list[str]]:
    route_payloads = [_route_option_certification_payload(option) for option in frontier_options]
    configured_families = _split_csv_config(
        settings.route_refc_evidence_families,
        fallback=("scenario", "toll", "terrain", "fuel", "carbon", "weather", "stochastic"),
    )
    active_families = active_evidence_families(route_payloads, configured_families=configured_families)
    forced_refresh = set(forced_refreshed_families or set())
    normalized_context = {
        key: value
        for key, value in dict(ambiguity_context or {}).items()
        if value not in (None, "")
    }
    single_frontier_requires_full_stress = len(route_payloads) == 1 and _refc_requires_full_stress_worlds(
        normalized_context
    )
    effective_world_count, world_count_policy = _adaptive_refc_world_count(
        requested_world_count=int(world_count),
        frontier_size=len(route_payloads),
        ambiguity_context=normalized_context,
    )
    if single_frontier_requires_full_stress and force_single_frontier_full_stress_requested_worlds:
        effective_world_count = max(int(world_count), int(effective_world_count))
        world_count_policy = "single_frontier_full_stress_requested"
    if len(route_payloads) == 1 and not single_frontier_requires_full_stress:
        selected_id = str(route_payloads[0].get("route_id") or selected_route_id).strip() or selected_route_id
        shortcut_certificate_value = _single_frontier_shortcut_certificate_value(
            requested_world_count=int(world_count),
            threshold=float(threshold),
            ambiguity_context=normalized_context,
        )
        worlds = [
            {
                "world_id": "shortcut:single_frontier",
                "states": {
                    family: ("refreshed" if family in forced_refresh else "nominal")
                    for family in active_families
                },
            }
        ]
        world_manifest = {
            "status": "single_frontier_shortcut",
            "seed": int(run_seed),
            "world_count": 1,
            "requested_world_count": 1,
            "effective_world_count": 1,
            "world_count_policy": world_count_policy,
            "unique_world_count": 1,
            "active_families": list(active_families),
            "worlds": worlds,
            "forced_refreshed_families": sorted(forced_refresh),
            "selected_route_id": selected_id,
            "shortcut_certificate_value": shortcut_certificate_value,
            "shortcut_frontier_uncertainty": round(max(0.0, 1.0 - shortcut_certificate_value), 6),
            "ambiguity_context": normalized_context,
            "hard_case_stress_pack_count": 0,
            "world_reuse_rate": 0.0,
        }
        world_manifest = annotate_world_manifest_cache_reuse(world_manifest, cache_reuse_origin="miss")
        selector_config = {
            "selector_weights": (float(w_time), float(w_money), float(w_co2)),
            "threshold": float(threshold),
            "optimization_mode": str(optimization_mode),
            "risk_aversion": float(risk_aversion),
            "world_count": 1,
            "shortcut": "single_frontier_certified",
        }
        certificate = CertificateResult(
            winner_id=selected_id,
            certificate={selected_id: shortcut_certificate_value},
            threshold=float(threshold),
            certified=shortcut_certificate_value >= float(threshold),
            selected_route_id=selected_id,
            route_scores={selected_id: [0.0]},
            world_manifest=dict(world_manifest),
            selector_config=selector_config,
        )
        fragility = FragilityResult(
            route_fragility_map={selected_id: {}},
            competitor_fragility_breakdown={selected_id: {}},
            value_of_refresh={"ranking": [], "top_refresh_family": None},
            route_fragility_details={selected_id: {}},
            evidence_snapshot_manifest={
                "status": "single_frontier_shortcut",
                "selected_route_id": selected_id,
                "active_families": list(active_families),
            },
        )
        return certificate, fragility, world_manifest, active_families
    world_manifest = sample_world_manifest(
        active_families=active_families,
        seed=int(run_seed),
        world_count=int(effective_world_count),
        state_catalog=tuple(
            _split_csv_config(
                settings.route_refc_state_catalog,
                fallback=("nominal", "mildly_stale", "severely_stale", "low_confidence", "proxy", "refreshed"),
            )
        ),
        routes=route_payloads,
        ambiguity_context=normalized_context,
        selector_weights=(float(w_time), float(w_money), float(w_co2)),
    )
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
    evaluated_bundle = None
    evaluated_bundle = evaluate_world_bundle(
        route_payloads,
        worlds,
        active_families=active_families,
        selector_weights=(w_time, w_money, w_co2),
        selector_score_map_fn=selector_score_map_fn,
    )
    certificate = compute_certificate(
        route_payloads,
        worlds=worlds,
        selector_weights=(w_time, w_money, w_co2),
        threshold=threshold,
        active_families=active_families,
        selector_score_map_fn=selector_score_map_fn,
        evaluated_bundle=evaluated_bundle,
        ambiguity_context=normalized_context,
    )
    fragility = compute_fragility_maps(
        route_payloads,
        worlds=worlds,
        selector_weights=(w_time, w_money, w_co2),
        active_families=active_families,
        selected_route_id=selected_route_id,
        selector_score_map_fn=selector_score_map_fn,
        evaluated_bundle=evaluated_bundle,
        baseline_certificate=certificate,
        ambiguity_context=normalized_context,
    )
    manifest_payload = dict(world_manifest)
    manifest_payload["sampler_requested_world_count"] = int(
        manifest_payload.get("requested_world_count", effective_world_count)
    )
    manifest_payload["worlds"] = worlds
    manifest_payload["forced_refreshed_families"] = sorted(forced_refresh)
    manifest_payload["selected_route_id"] = selected_route_id
    manifest_payload["ambiguity_context"] = normalized_context
    manifest_payload["requested_world_count"] = int(world_count)
    manifest_payload["effective_world_count"] = int(effective_world_count)
    manifest_payload["world_count_policy"] = str(world_count_policy)
    if len(route_payloads) == 1:
        certificate, fragility, manifest_payload = _apply_single_frontier_certificate_cap(
            certificate=certificate,
            fragility=fragility,
            world_manifest=manifest_payload,
            selected_route_id=selected_route_id,
            requested_world_count=int(world_count),
            threshold=float(threshold),
            ambiguity_context=normalized_context,
            full_stress_required=single_frontier_requires_full_stress,
        )
    manifest_payload = annotate_world_manifest_cache_reuse(manifest_payload, cache_reuse_origin="miss")
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


def _certificate_margin_from_certificate(
    certificate_map: Mapping[str, float],
    *,
    winner_id: str,
) -> float:
    winner_value = float(certificate_map.get(winner_id, 0.0))
    competitors = [
        float(value)
        for route_id, value in certificate_map.items()
        if route_id != winner_id
    ]
    runner_up = max(competitors) if competitors else 0.0
    return max(0.0, winner_value - runner_up)


def _runner_up_gap_from_score_map(
    score_map: Mapping[str, float],
    *,
    selected_route_id: str,
) -> float:
    selected_score = float(score_map.get(selected_route_id, 0.0))
    competitors = [
        float(value)
        for route_id, value in score_map.items()
        if route_id != selected_route_id
    ]
    runner_up = min(competitors) if competitors else selected_score
    return max(0.0, runner_up - selected_score)


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
    ambiguity_context: Mapping[str, Any] | None = None,
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
            ambiguity_context=(
                {
                    key: value
                    for key, value in dict(ambiguity_context or {}).items()
                    if value not in (None, "")
                }
                or None
            ),
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
    candidate_meta = route.get("_candidate_meta")
    candidate_meta_dict = candidate_meta if isinstance(candidate_meta, dict) else {}
    source_labels = _candidate_source_labels(route)
    primary_source_label = _primary_candidate_source_label(source_labels)
    candidate_source_stage = _candidate_source_stage_from_labels(source_labels)
    candidate_source_engine = _candidate_source_engine_from_labels(source_labels)
    seed_observed_refine_cost_ms = float("nan")
    try:
        seed_observed_refine_cost_ms = float(candidate_meta_dict.get("observed_refine_cost_ms"))
    except (TypeError, ValueError):
        seed_observed_refine_cost_ms = float("nan")
    if not math.isfinite(seed_observed_refine_cost_ms) or seed_observed_refine_cost_ms <= 0.0:
        seed_observed_refine_cost_ms = float("nan")
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
    if not graph_path:
        geometry = route.get("geometry")
        coords = geometry.get("coordinates") if isinstance(geometry, dict) else None
        if isinstance(coords, list) and coords:
            step = max(1, len(coords) // 10)
            for idx in range(0, len(coords), step):
                point = coords[idx]
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    continue
                try:
                    lon = float(point[0])
                    lat = float(point[1])
                except (TypeError, ValueError):
                    continue
                graph_path.append(f"{lon:.5f},{lat:.5f}")
                if len(graph_path) >= 12:
                    break
            if len(coords) > 1:
                tail = coords[-1]
                if isinstance(tail, (list, tuple)) and len(tail) >= 2:
                    try:
                        tail_token = f"{float(tail[0]):.5f},{float(tail[1]):.5f}"
                    except (TypeError, ValueError):
                        tail_token = ""
                    if tail_token and tail_token not in graph_path:
                        graph_path.append(tail_token)
    graph_length_km = float(max(0.0, _route_distance_km(route)))
    duration_s = float(max(0.0, _route_duration_s(route)))
    path_nodes = max(2, int(meta_dict.get("path_nodes", len(graph_path) or 2)))
    toll_share = max(0.0, float(meta_dict.get("toll_edges", 0))) / float(max(1, path_nodes - 1))
    turn_burden = max(0.0, float(meta_dict.get("turn_burden", 0.0)))
    terrain_burden = min(1.0, turn_burden / float(max(1, path_nodes - 2)))
    straight_line_km = max(0.001, float(_od_haversine_km(origin, destination)))
    has_graph_mechanism = isinstance(road_mix_counts, dict) and bool(road_mix_counts)
    if not has_graph_mechanism:
        explicit_nonlocal_share = max(0.0, motorway_share + a_road_share)
        speed_proxy_mix = _route_speed_proxy_mix(route)
        if speed_proxy_mix is not None and explicit_nonlocal_share <= 0.05:
            motorway_share = float(speed_proxy_mix.get("motorway_share", 0.0))
            a_road_share = float(speed_proxy_mix.get("a_road_share", 0.0))
            urban_share = float(speed_proxy_mix.get("urban_share", 0.0))
            other_share = float(
                max(
                    0.0,
                    1.0 - motorway_share - a_road_share - urban_share,
                )
            )
    proxy_money = (
        (float(vehicle.cost_per_km) * graph_length_km)
        + (float(vehicle.cost_per_hour) * (duration_s / 3600.0))
    )
    if bool(cost_toggles.use_tolls):
        proxy_money += float(cost_toggles.toll_cost_per_km) * graph_length_km * toll_share
    proxy_co2 = max(0.0, float(vehicle.emission_factor_kg_per_tkm) * float(vehicle.mass_tonnes) * graph_length_km)
    if float(cost_toggles.carbon_price_per_kg) > 0.0:
        proxy_money += proxy_co2 * float(cost_toggles.carbon_price_per_kg)
    mechanism_descriptor = {
        "motorway_share": round(motorway_share, 6),
        "a_road_share": round(a_road_share, 6),
        "urban_share": round(urban_share, 6),
        "toll_share": round(toll_share, 6),
        "terrain_burden": round(terrain_burden, 6),
    }
    if not has_graph_mechanism:
        mechanism_descriptor.update(_route_speed_profile_features(route))
        mechanism_descriptor.update(
            _route_shape_profile_features(
                route,
                route_distance_km=graph_length_km,
                straight_line_km=straight_line_km,
            )
        )
        mechanism_descriptor.update(_fallback_source_feature_map(source_labels))
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
        "mechanism_descriptor": mechanism_descriptor,
        "proxy_confidence": {
            "time": round(max(0.45, min(0.98, 0.82 + (0.08 * motorway_share))), 6),
            "money": round(max(0.40, min(0.95, 0.70 + (0.20 * (1.0 - toll_share)))), 6),
            "co2": round(max(0.40, min(0.95, 0.66 + (0.18 * (1.0 - urban_share)))), 6),
        },
    }
    if primary_source_label:
        candidate["candidate_source_label"] = primary_source_label
    if candidate_source_engine:
        candidate["candidate_source_engine"] = candidate_source_engine
    if candidate_source_stage:
        candidate["candidate_source_stage"] = candidate_source_stage
    if math.isfinite(seed_observed_refine_cost_ms):
        candidate["seed_observed_refine_cost_ms"] = round(float(seed_observed_refine_cost_ms), 6)
    candidate["candidate_id"] = stable_candidate_id(candidate)
    return candidate


def _dccs_runtime_config(
    *,
    mode: str,
    pipeline_variant: str,
    search_budget: int,
) -> DCCSConfig:
    return DCCSConfig(
        mode=mode,
        pipeline_variant=pipeline_variant,
        search_budget=max(0, int(search_budget)),
        bootstrap_seed_size=max(1, int(settings.route_dccs_bootstrap_count)),
        near_duplicate_threshold=float(settings.route_dccs_overlap_threshold),
        flip_bias=float(settings.route_dccs_pflip_bias),
        flip_objective_weight=float(settings.route_dccs_pflip_gap_weight),
        flip_mechanism_weight=float(settings.route_dccs_pflip_mechanism_weight),
        flip_overlap_weight=float(settings.route_dccs_pflip_overlap_weight),
        flip_stretch_weight=float(settings.route_dccs_pflip_detour_weight),
    )


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _effective_refinement_policy(req: RouteRequest, *, pipeline_mode: str) -> str:
    requested = str(req.refinement_policy or "").strip().lower()
    valid = {"dccs", "first_n", "random_n", "corridor_uniform"}
    if pipeline_mode != "dccs":
        return "dccs"
    if requested in valid:
        return requested
    return "dccs"


def _candidate_corridor_key(candidate: Mapping[str, Any]) -> str:
    road_mix = candidate.get("road_class_mix", {})
    mechanism = candidate.get("mechanism_descriptor", {})
    if not isinstance(road_mix, Mapping):
        road_mix = {}
    if not isinstance(mechanism, Mapping):
        mechanism = {}

    def _bucket(value: Any, *, step: float = 0.25) -> str:
        clipped = max(0.0, min(1.0, float(value or 0.0)))
        return f"{int(round(clipped / step))}"

    path = candidate.get("graph_path", [])
    path_tokens = list(path) if isinstance(path, list) else []
    start = str(path_tokens[0]) if path_tokens else "none"
    end = str(path_tokens[-1]) if path_tokens else "none"
    source_bucket = _candidate_source_bucket(
        source_label=str(candidate.get("candidate_source_label") or "").strip() or None,
        source_stage=str(candidate.get("candidate_source_stage") or "").strip() or None,
    )
    return "|".join(
        [
            f"mw={_bucket(road_mix.get('motorway_share'))}",
            f"ar={_bucket(road_mix.get('a_road_share'))}",
            f"ur={_bucket(road_mix.get('urban_share'))}",
            f"tl={_bucket(candidate.get('toll_share', mechanism.get('toll_share')))}",
            f"te={_bucket(candidate.get('terrain_burden', mechanism.get('terrain_burden')))}",
            f"sv={_bucket(mechanism.get('speed_variability'), step=0.2)}",
            f"sb={_bucket(mechanism.get('shape_bend_density'), step=0.2)}",
            f"sd={_bucket(mechanism.get('shape_detour_factor'), step=0.2)}",
            f"src={source_bucket}",
            f"s={start}",
            f"e={end}",
        ]
    )


def _baseline_policy_result(
    *,
    candidates: Sequence[Mapping[str, Any]],
    config: DCCSConfig,
    run_seed: int,
    policy: str,
    frontier: Sequence[Mapping[str, Any]] = (),
    refined: Sequence[Mapping[str, Any]] = (),
) -> DCCSResult:
    ledger = build_candidate_ledger(candidates, frontier=frontier, refined=refined, config=config)
    indexed = list(enumerate(ledger))
    budget = max(0, int(config.search_budget))
    policy_key = str(policy or "first_n").strip().lower()
    if policy_key not in {"first_n", "random_n", "corridor_uniform"}:
        policy_key = "first_n"

    if policy_key == "first_n":
        ordered = sorted(indexed, key=lambda item: (item[0], item[1].candidate_id))
    elif policy_key == "random_n":
        ordered = sorted(
            indexed,
            key=lambda item: (
                hashlib.sha1(
                    f"{int(run_seed)}|random_n|{item[1].candidate_id}|{item[0]}".encode("utf-8")
                ).hexdigest(),
                item[0],
                item[1].candidate_id,
            ),
        )
    else:
        corridor_groups: dict[str, list[tuple[int, DCCSCandidateRecord]]] = {}
        for item in indexed:
            corridor_groups.setdefault(
                _candidate_corridor_key(candidates[item[0]]),
                [],
            ).append(item)
        ordered = []
        group_keys = sorted(corridor_groups)
        for rows in corridor_groups.values():
            rows.sort(key=lambda item: (item[0], item[1].candidate_id))
        exhausted = False
        offset = 0
        while not exhausted:
            exhausted = True
            for key in group_keys:
                rows = corridor_groups[key]
                if offset < len(rows):
                    ordered.append(rows[offset])
                    exhausted = False
            offset += 1

    selected_ids = {record.candidate_id for _, record in ordered[:budget]}
    selected: list[DCCSCandidateRecord] = []
    skipped: list[DCCSCandidateRecord] = []
    for rank, (index, record) in enumerate(ordered):
        score = score_candidate(record, config=config)
        if record.candidate_id in selected_ids:
            selected.append(
                replace(
                    record,
                    final_score=float(score),
                    decision="refine",
                    decision_reason=f"selected_by_baseline_policy:{policy_key}",
                    mode=f"{config.mode}:{policy_key}",
                )
            )
        else:
            reason = "budget_exhausted" if rank >= budget else "not_selected"
            skipped.append(
                replace(
                    record,
                    final_score=float(score),
                    decision="skip",
                    decision_reason=reason,
                    mode=f"{config.mode}:{policy_key}",
                )
            )
    ordered_ledger = [record for _, record in ordered]
    summary = {
        "mode": config.mode,
        "transition_reason": f"baseline_policy:{policy_key}",
        "search_budget": budget,
        "candidate_count": len(ordered_ledger),
        "selected_count": len(selected),
        "skipped_count": len(skipped),
        "dc_yield": 0.0,
        "challenger_hit_rate": 0.0,
        "frontier_gain_per_refinement": 0.0,
        "decision_flips": 0,
        "frontier_additions": 0,
        "term_ablation_ready": True,
        "selection_policy": policy_key,
    }
    return DCCSResult(
        mode=f"{config.mode}:{policy_key}",
        search_budget=budget,
        transition_reason=f"baseline_policy:{policy_key}",
        selected=selected,
        skipped=skipped,
        candidate_ledger=ordered_ledger,
        summary=summary,
    )


def _select_candidate_records(
    *,
    candidates: Sequence[Mapping[str, Any]],
    config: DCCSConfig,
    run_seed: int,
    refinement_policy: str,
    frontier: Sequence[Mapping[str, Any]] = (),
    refined: Sequence[Mapping[str, Any]] = (),
) -> DCCSResult:
    if refinement_policy == "dccs":
        return select_candidates(candidates, frontier=frontier, refined=refined, config=config)
    return _baseline_policy_result(
        candidates=candidates,
        config=config,
        run_seed=run_seed,
        policy=refinement_policy,
        frontier=frontier,
        refined=refined,
    )


async def _refine_graph_candidate_batch(
    *,
    osrm: OSRMClient,
    origin: LatLng,
    destination: LatLng,
    selected_records: Sequence[DCCSCandidateRecord],
    raw_graph_routes_by_id: Mapping[str, dict[str, Any]],
    vehicle_type: str | None = None,
    scenario_mode: Any | None = None,
    cost_toggles: Any | None = None,
    terrain_profile: Any | None = None,
    departure_time_utc: datetime | None = None,
    scenario_cache_token: str | None = None,
    route_cache_runtime: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[str], dict[str, float], int, float]:
    warnings: list[str] = []
    routes_by_signature: dict[str, dict[str, Any]] = {}
    observed_costs: dict[str, float] = {}
    candidate_fetches = 0
    batch_started = time.perf_counter()
    sem = asyncio.Semaphore(
        max(
            1,
            min(
                int(settings.route_candidate_refine_max_concurrency),
                max(1, len(selected_records)),
            ),
        )
    )

    def _touch_route_cache_runtime(*, cache_key_value: str | None = None, hit: bool | None = None) -> None:
        if route_cache_runtime is None:
            return
        if cache_key_value is not None:
            route_cache_runtime["last_cache_key"] = cache_key_value
        if hit is True:
            route_cache_runtime["cache_hits"] = int(route_cache_runtime.get("cache_hits", 0)) + 1
        elif hit is False:
            route_cache_runtime["cache_misses"] = int(route_cache_runtime.get("cache_misses", 0)) + 1
        total_events = int(route_cache_runtime.get("cache_hits", 0)) + int(
            route_cache_runtime.get("cache_misses", 0)
        )
        route_cache_runtime["reuse_rate"] = round(
            int(route_cache_runtime.get("cache_hits", 0)) / float(max(1, total_events)),
            6,
        )

    def _refine_cache_key(spec: CandidateFetchSpec) -> str:
        return _graph_refine_route_cache_key(
            origin=origin,
            destination=destination,
            alternatives=spec.alternatives,
            exclude=spec.exclude,
            via=spec.via,
            vehicle_type=vehicle_type,
            scenario_mode=scenario_mode,
            cost_toggles=cost_toggles,
            terrain_profile=terrain_profile,
            departure_time_utc=departure_time_utc,
            scenario_cache_token=scenario_cache_token,
        )

    async def _refine_selected_record(
        record: DCCSCandidateRecord,
    ) -> tuple[DCCSCandidateRecord, list[str], str | None, bool, list[dict[str, Any]], float, int]:
        raw_route = raw_graph_routes_by_id.get(record.candidate_id)
        if raw_route is None:
            return (
                record,
                [f"graph_family:{record.candidate_id}: missing_raw_candidate"],
                None,
                False,
                [],
                0.0,
                0,
            )
        meta = raw_route.get("_candidate_meta")
        meta_dict = meta if isinstance(meta, dict) else {}
        source_labels = {
            str(label).strip().lower()
            for label in (meta_dict.get("source_labels") or [])
            if str(label).strip()
        }
        pre_realized_fallback = any(
            "direct_k_raw_fallback" in label
            or "long_corridor_fallback" in label
            or "legacy_corridor_uniform_osrm_fallback" in label
            for label in source_labels
        )
        stored_observed_cost_ms = float("nan")
        if pre_realized_fallback:
            try:
                stored_observed_cost_ms = float(meta_dict.get("observed_refine_cost_ms"))
            except (TypeError, ValueError):
                stored_observed_cost_ms = float("nan")
        via = _graph_family_via_points(
            raw_route,
            max_landmarks=max(1, int(settings.route_graph_via_landmarks_per_path)),
        )
        spec = CandidateFetchSpec(
            label=f"graph_family:{record.candidate_id}",
            alternatives=False,
            via=via if via else None,
        )
        cache_key = _refine_cache_key(spec)
        _touch_route_cache_runtime(cache_key_value=cache_key)
        cached = get_cached_routes(cache_key)
        if cached is not None:
            _touch_route_cache_runtime(hit=True)
            cached_routes = cached[0] if len(cached) >= 1 else []
            cached_warnings = cached[1] if len(cached) >= 2 else []
            return (
                record,
                list(cached_warnings) if isinstance(cached_warnings, list) else [],
                spec.label,
                False,
                copy.deepcopy(list(cached_routes) if isinstance(cached_routes, list) else []),
                0.0,
                0,
            )
        _touch_route_cache_runtime(hit=False)
        fetch_started = time.perf_counter()
        result = await _run_candidate_fetch(
            osrm=osrm,
            origin=origin,
            destination=destination,
            spec=spec,
            sem=sem,
        )
        elapsed_ms = max(0.001, (time.perf_counter() - fetch_started) * 1000.0)
        candidate_routes = result.routes or []
        local_warnings: list[str] = []
        if result.error:
            local_warnings.append(f"{result.spec.label}: {result.error}")
        if candidate_routes:
            # Only fresh successful re-realizations count as observed refine
            # cost samples. Seed fallback costs remain available separately via
            # `seed_observed_refine_cost_ms` for prediction re-anchoring.
            observed_refine_cost_value = float(elapsed_ms)
            set_cached_routes(
                cache_key,
                (
                    copy.deepcopy(candidate_routes),
                    [],
                    1,
                    {
                        "label": str(result.spec.label),
                        "elapsed_ms": round(float(elapsed_ms), 6),
                        "cache_kind": "graph_refine",
                    },
                ),
            )
            return (
                record,
                local_warnings,
                result.spec.label,
                False,
                candidate_routes,
                observed_refine_cost_value,
                1,
            )
        try:
            fallback_routes = [json.loads(json.dumps(raw_route, separators=(",", ":"), default=str))]
        except Exception:
            fallback_routes = [dict(raw_route)]
        return (
            record,
            local_warnings,
            None,
            bool(pre_realized_fallback),
            fallback_routes,
            float("nan"),
            1,
        )

    refine_results = await asyncio.gather(
        *[_refine_selected_record(record) for record in selected_records]
    )

    for record, record_warnings, result_label, pre_realized, candidate_routes, observed_cost_ms, fetch_count in refine_results:
        warnings.extend(record_warnings)
        if (not pre_realized) and int(fetch_count) > 0 and math.isfinite(float(observed_cost_ms)):
            observed_costs[record.candidate_id] = round(max(0.001, float(observed_cost_ms)), 6)
        candidate_fetches += int(fetch_count)
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
                source_labels=(
                    {"graph_family:pre_realized_fallback:osrm_refined"}
                    if pre_realized
                    else {
                        f"{str(result_label or f'graph_family:{record.candidate_id}').strip()}:osrm_refined"
                    }
                ),
                toll_exclusion_available=False,
                observed_refine_cost_ms=(
                    float(observed_cost_ms)
                    if (not pre_realized) and math.isfinite(float(observed_cost_ms)) and float(observed_cost_ms) > 0.0
                    else None
                ),
            )
    elapsed_total_ms = round((time.perf_counter() - batch_started) * 1000.0, 2)
    return list(routes_by_signature.values()), warnings, observed_costs, candidate_fetches, elapsed_total_ms


def _resolve_voi_refine_selected_records(
    *,
    action_metadata: Mapping[str, Any] | None,
    current_records: Sequence[DCCSCandidateRecord],
    fallback_records: Sequence[DCCSCandidateRecord],
) -> tuple[list[DCCSCandidateRecord], dict[str, Any]]:
    metadata = action_metadata if isinstance(action_metadata, Mapping) else {}
    requested_candidate_ids_raw = metadata.get("candidate_ids")
    requested_candidate_ids: list[str] = []
    if isinstance(requested_candidate_ids_raw, Sequence) and not isinstance(requested_candidate_ids_raw, (str, bytes)):
        seen_requested: set[str] = set()
        for raw_candidate_id in requested_candidate_ids_raw:
            candidate_id = str(raw_candidate_id or "").strip()
            if not candidate_id or candidate_id in seen_requested:
                continue
            seen_requested.add(candidate_id)
            requested_candidate_ids.append(candidate_id)
    try:
        top_k = max(1, int(metadata.get("top_k", 1)))
    except Exception:
        top_k = 1

    current_records_by_id: dict[str, DCCSCandidateRecord] = {}
    for record in current_records:
        current_records_by_id.setdefault(record.candidate_id, record)

    resolved_records: list[DCCSCandidateRecord] = []
    unresolved_candidate_ids: list[str] = []
    for candidate_id in requested_candidate_ids:
        record = current_records_by_id.get(candidate_id)
        if record is None:
            unresolved_candidate_ids.append(candidate_id)
            continue
        resolved_records.append(record)

    used_candidate_metadata = bool(requested_candidate_ids) and bool(resolved_records)
    if used_candidate_metadata:
        selected_records = resolved_records
        resolution_source = "action_metadata"
    else:
        selected_records = list(fallback_records[:top_k])
        resolution_source = "fallback_top_k"
    executed_candidate_ids = [record.candidate_id for record in selected_records]
    execution_matches_requested_candidate_ids: bool | None = None
    if requested_candidate_ids:
        execution_matches_requested_candidate_ids = (
            not unresolved_candidate_ids
            and executed_candidate_ids == requested_candidate_ids
        )
    return selected_records, {
        "requested_candidate_ids": requested_candidate_ids,
        "resolved_candidate_ids": [record.candidate_id for record in resolved_records],
        "unresolved_candidate_ids": unresolved_candidate_ids,
        "executed_candidate_ids": executed_candidate_ids,
        "execution_used_candidate_metadata": used_candidate_metadata,
        "execution_matches_requested_candidate_ids": execution_matches_requested_candidate_ids,
        "execution_candidate_resolution_source": resolution_source,
    }


async def _route_graph_k_raw_search(
    *,
    origin: LatLng,
    destination: LatLng,
    max_alternatives: int,
    scenario_edge_modifiers: dict[str, Any] | None = None,
    start_node_id: str | None = None,
    goal_node_id: str | None = None,
    od_ambiguity_index: float | None = None,
    od_engine_disagreement_prior: float | None = None,
    od_hard_case_prior: float | None = None,
    od_ambiguity_support_ratio: float | None = None,
    od_ambiguity_source_entropy: float | None = None,
    od_candidate_path_count: int | None = None,
    od_corridor_family_count: int | None = None,
    od_nominal_margin_proxy: float | None = None,
    od_objective_spread: float | None = None,
    allow_supported_ambiguity_fast_fallback: bool = False,
    enable_voi_support_rich_short_haul_graph_probe: bool = False,
) -> tuple[list[dict[str, Any]], GraphCandidateDiagnostics, dict[str, Any]]:
    request_id = current_live_trace_request_id()
    max_paths = max(4, int(max_alternatives) * 2)
    configured_state_budget = max(1000, int(settings.route_graph_max_state_budget))
    corridor_distance_km = _od_haversine_km(origin, destination)
    long_corridor_threshold_km = max(10.0, float(settings.route_graph_long_corridor_threshold_km))
    reliability_corridor_threshold_km = min(
        long_corridor_threshold_km,
        max(60.0, long_corridor_threshold_km * 0.6),
    )
    reliability_corridor = corridor_distance_km >= reliability_corridor_threshold_km
    long_corridor = corridor_distance_km >= long_corridor_threshold_km
    ambiguity_context_available = any(
        value is not None
        for value in (od_ambiguity_index, od_engine_disagreement_prior, od_hard_case_prior)
    )
    ambiguity_strength = _route_graph_search_ambiguity_strength(
        od_ambiguity_index=od_ambiguity_index,
        od_engine_disagreement_prior=od_engine_disagreement_prior,
        od_hard_case_prior=od_hard_case_prior,
        od_ambiguity_support_ratio=od_ambiguity_support_ratio,
        od_ambiguity_source_entropy=od_ambiguity_source_entropy,
    )
    support_ratio = max(0.0, min(1.0, float(od_ambiguity_support_ratio or 0.0)))
    source_entropy = max(0.0, min(1.0, float(od_ambiguity_source_entropy or 0.0)))
    nominal_margin_proxy = max(0.0, min(1.0, float(od_nominal_margin_proxy or 0.0)))
    objective_spread = max(0.0, float(od_objective_spread or 0.0))
    low_ambiguity_fast_path = bool(
        ambiguity_context_available
        and reliability_corridor
        and ambiguity_strength <= float(settings.route_graph_fast_path_max_ambiguity)
    )
    support_aware_fast_path = bool(
        ambiguity_context_available
        and ambiguity_strength <= 0.40
        and support_ratio <= 0.40
        and source_entropy <= 0.25
    )
    support_aware_aggressive_fast_path = bool(
        support_aware_fast_path
        and (support_ratio <= 0.25 or source_entropy <= 0.20 or ambiguity_strength <= 0.20)
    )
    if long_corridor:
        long_corridor_path_cap = max(2, int(settings.route_graph_long_corridor_max_paths))
        max_paths = max(2, min(max_paths, max_alternatives, long_corridor_path_cap))
    elif low_ambiguity_fast_path:
        max_paths = max(2, min(max_paths, max(2, min(max_alternatives, 3))))
        configured_state_budget = max(1000, int(math.ceil(configured_state_budget * 0.35)))
    elif support_aware_fast_path:
        max_paths = max(2, min(max_paths, max(2, min(max_alternatives, 4))))
        configured_state_budget = max(
            1000,
            int(math.ceil(configured_state_budget * (0.20 if support_aware_aggressive_fast_path else 0.25))),
        )
        if support_aware_aggressive_fast_path:
            max_paths = max(2, min(max_paths, max(2, min(max_alternatives, 3))))
    reduced_initial_enabled = bool(settings.route_graph_reduced_initial_for_long_corridor)
    # Alternative-route literature emphasizes that path diversity should be
    # preserved before downstream selection; see Abraham et al., "Alternative
    # Routes in Road Networks", https://doi.org/10.1007/978-3-540-68552-4_24 .
    # Mirror legacy medium/long-corridor search semantics here so the thesis
    # pipeline does not collapse K_raw relative to the production baseline.
    initial_use_transition_state = not (reduced_initial_enabled and reliability_corridor)
    long_corridor_fast_fallback_reasons = {
        "state_budget_exceeded",
        "path_search_exhausted",
        "no_path",
        "candidate_pool_exhausted",
        "skipped_long_corridor_graph_search",
    }
    support_aware_fast_fallback_reasons = {
        "skipped_supported_ambiguity_fast_fallback",
        "skipped_support_aware_graph_search",
        "skipped_support_aware_long_corridor_graph_search",
        "skipped_support_rich_short_haul_graph_search",
        "skipped_support_backed_single_corridor_graph_search",
    }
    candidate_path_count = max(0, int(od_candidate_path_count or 0))
    corridor_family_count = max(0, int(od_corridor_family_count or 0))
    support_rich_short_haul_fast_path = bool(
        ambiguity_context_available
        and not long_corridor
        and not reliability_corridor
        and support_ratio >= 0.75
        and source_entropy >= 0.60
        and (
            ambiguity_strength >= 0.60
            or float(od_engine_disagreement_prior or 0.0) >= float(settings.route_dccs_preemptive_comparator_min_engine_disagreement)
            or float(od_hard_case_prior or 0.0) >= float(settings.route_dccs_preemptive_comparator_min_hard_case)
        )
        and candidate_path_count > 0
        and candidate_path_count <= 4
        and corridor_family_count > 0
        and corridor_family_count <= 2
    )
    support_rich_short_haul_graph_probe = bool(
        support_rich_short_haul_fast_path
        and _support_rich_short_haul_graph_probe_eligible(
            enabled=enable_voi_support_rich_short_haul_graph_probe,
            od_engine_disagreement_prior=od_engine_disagreement_prior,
            od_hard_case_prior=od_hard_case_prior,
        )
    )
    support_backed_single_corridor_fast_path = bool(
        ambiguity_context_available
        and allow_supported_ambiguity_fast_fallback
        and support_ratio >= 0.60
        and source_entropy >= 0.70
        and candidate_path_count > 0
        and candidate_path_count <= 4
        and corridor_family_count == 1
        and nominal_margin_proxy >= 0.95
        and objective_spread <= 0.26
        and ambiguity_strength < float(settings.route_dccs_preemptive_comparator_min_ambiguity)
        and float(od_engine_disagreement_prior or 0.0)
        < float(settings.route_dccs_preemptive_comparator_min_engine_disagreement)
        and float(od_hard_case_prior or 0.0)
        < float(settings.route_dccs_preemptive_comparator_min_hard_case)
    )
    support_backed_single_corridor_diversity_probe = bool(
        ambiguity_context_available
        and allow_supported_ambiguity_fast_fallback
        and support_ratio >= 0.60
        and source_entropy >= 0.70
        and candidate_path_count > 0
        and candidate_path_count <= 4
        and corridor_family_count == 1
        and nominal_margin_proxy >= 0.95
        and objective_spread <= 0.26
        and not support_backed_single_corridor_fast_path
        and (
            ambiguity_strength >= float(settings.route_dccs_preemptive_comparator_min_ambiguity)
            or float(od_engine_disagreement_prior or 0.0)
            >= float(settings.route_dccs_preemptive_comparator_min_engine_disagreement)
            or float(od_hard_case_prior or 0.0)
            >= float(settings.route_dccs_preemptive_comparator_min_hard_case)
        )
    )
    long_corridor_stress_probe = bool(
        ambiguity_context_available
        and _long_corridor_stress_graph_probe_eligible(
            long_corridor=long_corridor,
            ambiguity_strength=ambiguity_strength,
            od_engine_disagreement_prior=od_engine_disagreement_prior,
            od_hard_case_prior=od_hard_case_prior,
            od_ambiguity_support_ratio=od_ambiguity_support_ratio,
            od_ambiguity_source_entropy=od_ambiguity_source_entropy,
            od_candidate_path_count=od_candidate_path_count,
            od_corridor_family_count=od_corridor_family_count,
        )
    )
    skip_initial_graph_search = bool(
        long_corridor
        and settings.route_graph_skip_initial_search_long_corridor
        and not long_corridor_stress_probe
    )
    if (
        not skip_initial_graph_search
        and low_ambiguity_fast_path
        and bool(settings.route_graph_skip_initial_search_reliability_low_ambiguity)
    ):
        skip_initial_graph_search = True
    if not skip_initial_graph_search and support_aware_fast_path and (
        long_corridor
        or support_ratio <= 0.35
        or source_entropy <= 0.20
        or ambiguity_strength <= 0.20
    ):
        skip_initial_graph_search = True
    if not skip_initial_graph_search and support_rich_short_haul_fast_path and not support_rich_short_haul_graph_probe:
        skip_initial_graph_search = True
    if not skip_initial_graph_search and support_backed_single_corridor_fast_path:
        skip_initial_graph_search = True
    skip_retry_rescue_for_long_corridor = bool(
        long_corridor
        or (
            reliability_corridor
            and bool(settings.route_graph_skip_retry_rescue_reliability_corridor)
        )
    )
    initial_max_hops_override: int | None = None
    if long_corridor:
        scaled_hops = int(math.ceil(max(1.0, corridor_distance_km) * 8.5))
        initial_max_hops_override = max(900, min(int(settings.route_graph_max_hops_cap), scaled_hops))
    initial_search_timeout_ms = max(0, int(settings.route_graph_search_initial_timeout_ms))
    retry_search_timeout_ms = max(0, int(settings.route_graph_search_retry_timeout_ms))
    rescue_search_timeout_ms = max(0, int(settings.route_graph_search_rescue_timeout_ms))
    use_explicit_state_budget = bool(long_corridor or low_ambiguity_fast_path or support_aware_fast_path)
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
    supported_ambiguity_fast_fallback = bool(
        allow_supported_ambiguity_fast_fallback
        and not long_corridor
        and not reliability_corridor
        and support_ratio >= 0.65
        and source_entropy >= 0.50
        and (
            ambiguity_strength >= float(settings.route_dccs_preemptive_comparator_min_ambiguity)
            or float(od_engine_disagreement_prior or 0.0)
            >= float(settings.route_dccs_preemptive_comparator_min_engine_disagreement)
            or float(od_hard_case_prior or 0.0)
            >= float(settings.route_dccs_preemptive_comparator_min_hard_case)
        )
    )
    cache_key = hashlib.sha1(
        json.dumps(
            {
                "origin": [round(float(origin.lat), 6), round(float(origin.lon), 6)],
                "destination": [round(float(destination.lat), 6), round(float(destination.lon), 6)],
                "max_alternatives": int(max_alternatives),
                "scenario_edge_modifiers": scenario_edge_modifiers or {},
                "start_node_id": str(start_node_id or ""),
                "goal_node_id": str(goal_node_id or ""),
                "ambiguity_context_available": bool(ambiguity_context_available),
                "ambiguity_strength": round(ambiguity_strength, 6),
                "od_ambiguity_support_ratio": round(support_ratio, 6),
                "od_ambiguity_source_entropy": round(source_entropy, 6),
                "od_nominal_margin_proxy": round(nominal_margin_proxy, 6),
                "od_objective_spread": round(objective_spread, 6),
                "low_ambiguity_fast_path": bool(low_ambiguity_fast_path),
                "long_corridor_stress_probe": bool(long_corridor_stress_probe),
                "skip_initial_graph_search": bool(skip_initial_graph_search),
                "allow_supported_ambiguity_fast_fallback": bool(allow_supported_ambiguity_fast_fallback),
                "od_candidate_path_count": int(candidate_path_count),
                "od_corridor_family_count": int(corridor_family_count),
                "support_rich_short_haul_fast_path": bool(support_rich_short_haul_fast_path),
                "support_rich_short_haul_graph_probe": bool(support_rich_short_haul_graph_probe),
                "search_shape": {
                    "max_paths": int(max_paths),
                    "configured_state_budget": int(configured_state_budget),
                    "initial_max_hops_override": initial_max_hops_override,
                    "reliability_corridor_threshold_km": round(float(reliability_corridor_threshold_km), 3),
                    "long_corridor_threshold_km": round(float(long_corridor_threshold_km), 3),
                    "reliability_corridor": bool(reliability_corridor),
                    "long_corridor": bool(long_corridor),
                    "reduced_initial_enabled": bool(reduced_initial_enabled),
                    "route_dccs_bootstrap_count": int(settings.route_dccs_bootstrap_count),
                    "route_graph_skip_initial_search_long_corridor": bool(
                        settings.route_graph_skip_initial_search_long_corridor
                    ),
                    "route_graph_skip_initial_search_reliability_low_ambiguity": bool(
                        settings.route_graph_skip_initial_search_reliability_low_ambiguity
                    ),
                    "route_graph_skip_retry_rescue_reliability_corridor": bool(
                        settings.route_graph_skip_retry_rescue_reliability_corridor
                    ),
                    "route_graph_skip_supplemental_probe_low_ambiguity": bool(
                        settings.route_graph_skip_supplemental_probe_low_ambiguity
                    ),
                    "route_graph_search_initial_timeout_ms": int(initial_search_timeout_ms),
                    "route_graph_search_retry_timeout_ms": int(retry_search_timeout_ms),
                    "route_graph_search_rescue_timeout_ms": int(rescue_search_timeout_ms),
                },
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    cached_k_raw = get_cached_k_raw(cache_key)
    if cached_k_raw is not None:
        cached_routes, cached_diag, cached_meta = cached_k_raw
        cached_meta = dict(cached_meta)
        cached_meta["graph_k_raw_cache_hit"] = True
        cached_meta["graph_low_ambiguity_fast_path"] = bool(low_ambiguity_fast_path)
        cached_meta["graph_supported_ambiguity_fast_fallback"] = bool(supported_ambiguity_fast_fallback)
        cached_meta["graph_long_corridor_stress_probe"] = bool(long_corridor_stress_probe)
        cached_meta["graph_support_rich_short_haul_fast_fallback"] = bool(support_rich_short_haul_fast_path)
        cached_meta["graph_support_rich_short_haul_probe"] = bool(support_rich_short_haul_graph_probe)
        cached_meta["graph_support_backed_single_corridor_diversity_probe"] = bool(
            support_backed_single_corridor_diversity_probe
        )
        cached_meta["graph_support_backed_single_corridor_fast_fallback"] = bool(
            support_backed_single_corridor_fast_path
        )
        return cached_routes, cached_diag, cached_meta
    if skip_initial_graph_search:
        log_event(
            "route_graph_search_budget",
            request_id=request_id,
            pass_name="initial_skipped",
            max_paths=max_paths,
            configured_state_budget=configured_state_budget,
            corridor_distance_km=round(float(corridor_distance_km), 3),
            long_corridor_threshold_km=round(float(long_corridor_threshold_km), 3),
            long_corridor=bool(long_corridor),
            long_corridor_stress_probe=bool(long_corridor_stress_probe),
            support_aware_fast_path=bool(support_aware_fast_path),
            support_aware_aggressive_fast_path=bool(support_aware_aggressive_fast_path),
            reason=(
                "support_rich_short_haul_fast_fallback"
                if support_rich_short_haul_fast_path
                else (
                    "support_backed_single_corridor_fast_fallback"
                    if support_backed_single_corridor_fast_path
                    else (
                        "support_aware_long_corridor_fast_fallback"
                        if support_aware_fast_path and long_corridor
                        else (
                            "support_aware_fast_fallback"
                            if support_aware_fast_path
                            else "long_corridor_skip_enabled"
                        )
                    )
                )
            ),
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
            no_path_reason=(
                "skipped_support_rich_short_haul_graph_search"
                if support_rich_short_haul_fast_path
                else (
                    "skipped_support_backed_single_corridor_graph_search"
                    if support_backed_single_corridor_fast_path
                    else (
                        "skipped_support_aware_long_corridor_graph_search"
                        if support_aware_fast_path and long_corridor
                        else (
                            "skipped_support_aware_graph_search"
                            if support_aware_fast_path
                            else "skipped_long_corridor_graph_search"
                        )
                    )
                )
            ),
            no_path_detail=(
                "Skipped expensive graph search because support-rich short-haul evidence made bounded direct fallback preferable."
                if support_rich_short_haul_fast_path
                else (
                    "Skipped expensive graph search because support-backed single-corridor evidence made bounded direct fallback preferable."
                    if support_backed_single_corridor_fast_path
                    else (
                        "Skipped expensive graph search because the ambiguity signal was weakly supported and used bounded fallback."
                        if support_aware_fast_path
                        else "Skipped expensive long-corridor graph search and used bounded fallback."
                    )
                )
            ),
        )
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
            reliability_corridor_threshold_km=round(float(reliability_corridor_threshold_km), 3),
            long_corridor_threshold_km=round(float(long_corridor_threshold_km), 3),
            reliability_corridor=bool(reliability_corridor),
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
            max_state_budget_override=(configured_state_budget if use_explicit_state_budget else None),
            use_transition_state=initial_use_transition_state,
            search_deadline_s=_search_deadline_s(initial_search_timeout_ms),
            start_node_id=start_node_id,
            goal_node_id=goal_node_id,
        )
        graph_search_ms_initial = round((time.perf_counter() - graph_initial_started) * 1000.0, 2)
    initial_no_path_reason = str(graph_diag.no_path_reason or "").strip()
    should_retry = not (
        skip_retry_rescue_for_long_corridor and initial_no_path_reason in long_corridor_fast_fallback_reasons
    ) and not (
        supported_ambiguity_fast_fallback
        and initial_no_path_reason in {"state_budget_exceeded", "path_search_exhausted"}
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
        graph_retry_outcome = (
            "skipped_supported_ambiguity_fast_fallback"
            if supported_ambiguity_fast_fallback and not long_corridor and not reliability_corridor
            else (
                "skipped_long_corridor_fast_fallback"
                if long_corridor
                else "skipped_reliability_corridor_fast_fallback"
            )
        )
    elif not graph_routes and initial_no_path_reason in support_aware_fast_fallback_reasons:
        graph_retry_attempted = True
        graph_retry_state_budget = int(getattr(graph_diag, "effective_state_budget", 0) or configured_state_budget)
        graph_retry_outcome = initial_no_path_reason
    if not graph_routes:
        rescue_reason = str(graph_diag.no_path_reason or "").strip()
        rescue_mode_setting = str(settings.route_graph_state_space_rescue_mode or "reduced").strip().lower()
        rescue_mode = rescue_mode_setting if rescue_mode_setting in {"reduced", "full"} else "reduced"
        rescue_enabled = bool(settings.route_graph_state_space_rescue_enabled)
        rescue_candidate_reasons = {"state_budget_exceeded", "path_search_exhausted", "no_path"}
        should_rescue = not (
            skip_retry_rescue_for_long_corridor and rescue_reason in long_corridor_fast_fallback_reasons
        ) and not (
            supported_ambiguity_fast_fallback
            and rescue_reason in {"state_budget_exceeded", "path_search_exhausted", "no_path"}
        ) and not (
            support_aware_fast_path and rescue_reason in support_aware_fast_fallback_reasons
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
        elif rescue_enabled and rescue_reason in rescue_candidate_reasons and not should_rescue:
            graph_rescue_attempted = True
            graph_rescue_mode = (
                "supported_ambiguity_fast_fallback"
                if supported_ambiguity_fast_fallback and not long_corridor and not reliability_corridor
                else (
                    "support_aware_fast_fallback"
                    if support_aware_fast_path
                    else (
                        "long_corridor_fast_fallback"
                        if long_corridor
                        else "reliability_corridor_fast_fallback"
                    )
                )
            )
            graph_rescue_state_budget = int(getattr(graph_diag, "effective_state_budget", 0) or configured_state_budget)
            graph_rescue_outcome = (
                "skipped_supported_ambiguity_fast_fallback"
                if supported_ambiguity_fast_fallback and not long_corridor and not reliability_corridor
                else (
                    rescue_reason
                    if support_aware_fast_path and rescue_reason in support_aware_fast_fallback_reasons
                    else (
                        "skipped_long_corridor_fast_fallback"
                        if long_corridor
                        else "skipped_reliability_corridor_fast_fallback"
                    )
                )
            )
        elif rescue_enabled and rescue_reason in support_aware_fast_fallback_reasons:
            graph_rescue_attempted = True
            graph_rescue_mode = "support_aware_fast_fallback"
            graph_rescue_state_budget = int(getattr(graph_diag, "effective_state_budget", 0) or configured_state_budget)
            graph_rescue_outcome = rescue_reason

    graph_search_ms_supplemental = 0.0
    supplemental_probe_attempted = False
    supplemental_target = min(
        max_paths,
        max(2, min(int(max_alternatives), int(settings.route_dccs_bootstrap_count) + 2)),
    )

    def _graph_route_merge_key(route: dict[str, Any]) -> str:
        family_sig = _graph_family_signature(route)
        if family_sig:
            return family_sig
        try:
            return _route_signature(route)
        except OSRMError:
            return hashlib.sha1(
                json.dumps(route, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
            ).hexdigest()

    if (
        graph_routes
        and reliability_corridor
        and not long_corridor
        and not support_aware_fast_path
        and not (
            low_ambiguity_fast_path
            and bool(settings.route_graph_skip_supplemental_probe_low_ambiguity)
        )
        and len(graph_routes) < supplemental_target
    ):
        supplemental_probe_attempted = True
        supplemental_use_transition_state = not initial_use_transition_state
        supplemental_search_timeout_ms = min(
            max(5_000, initial_search_timeout_ms),
            max(5_000, int(math.ceil(initial_search_timeout_ms * 0.5))),
        )
        log_event(
            "route_graph_search_budget",
            request_id=request_id,
            pass_name="supplemental_diversity_probe",
            max_paths=max_paths,
            configured_state_budget=configured_state_budget,
            corridor_distance_km=round(float(corridor_distance_km), 3),
            reliability_corridor_threshold_km=round(float(reliability_corridor_threshold_km), 3),
            target_candidates=int(supplemental_target),
            observed_candidates=int(len(graph_routes)),
            use_transition_state=bool(supplemental_use_transition_state),
            search_timeout_ms=supplemental_search_timeout_ms,
            start_node_id=start_node_id,
            goal_node_id=goal_node_id,
        )
        supplemental_started = time.perf_counter()
        supplemental_routes, _supplemental_diag = await _route_graph_candidate_routes_async(
            origin_lat=float(origin.lat),
            origin_lon=float(origin.lon),
            destination_lat=float(destination.lat),
            destination_lon=float(destination.lon),
            max_paths=max_paths,
            scenario_edge_modifiers=scenario_edge_modifiers,
            use_transition_state=supplemental_use_transition_state,
            search_deadline_s=_search_deadline_s(supplemental_search_timeout_ms),
            start_node_id=start_node_id,
            goal_node_id=goal_node_id,
        )
        graph_search_ms_supplemental = round((time.perf_counter() - supplemental_started) * 1000.0, 2)
        if supplemental_routes:
            merged_graph_routes: dict[str, dict[str, Any]] = {
                _graph_route_merge_key(route): route
                for route in graph_routes
            }
            for route in supplemental_routes:
                merged_graph_routes.setdefault(_graph_route_merge_key(route), route)
            graph_routes = list(merged_graph_routes.values())[:max_paths]
    graph_search_meta = {
        "graph_retry_attempted": graph_retry_attempted,
        "graph_retry_state_budget": graph_retry_state_budget,
        "graph_retry_outcome": graph_retry_outcome,
        "graph_rescue_attempted": graph_rescue_attempted,
        "graph_rescue_mode": graph_rescue_mode,
        "graph_rescue_state_budget": graph_rescue_state_budget,
        "graph_rescue_outcome": graph_rescue_outcome,
        "graph_search_ms_initial": graph_search_ms_initial,
        "graph_search_ms_retry": graph_search_ms_retry,
        "graph_search_ms_rescue": graph_search_ms_rescue,
        "graph_search_ms_supplemental": graph_search_ms_supplemental,
        "graph_supplemental_probe_attempted": supplemental_probe_attempted,
        "graph_supplemental_target": int(supplemental_target),
        "graph_k_raw_cache_hit": False,
        "graph_low_ambiguity_fast_path": bool(low_ambiguity_fast_path),
        "graph_supported_ambiguity_fast_fallback": bool(supported_ambiguity_fast_fallback),
        "graph_long_corridor_stress_probe": bool(long_corridor_stress_probe),
        "graph_support_rich_short_haul_fast_fallback": bool(support_rich_short_haul_fast_path),
        "graph_support_rich_short_haul_probe": bool(support_rich_short_haul_graph_probe),
        "graph_support_backed_single_corridor_diversity_probe": bool(
            support_backed_single_corridor_diversity_probe
        ),
        "graph_support_backed_single_corridor_fast_fallback": bool(
            support_backed_single_corridor_fast_path
        ),
    }
    set_cached_k_raw(cache_key, (graph_routes, graph_diag, graph_search_meta))
    return graph_routes, graph_diag, graph_search_meta


async def _compute_direct_route_pipeline(
    *,
    req: RouteRequest,
    osrm: OSRMClient,
    ors: ORSClient,
    max_alternatives: int,
    pipeline_mode: str,
    run_seed: int,
) -> dict[str, Any]:
    execution_pipeline_mode = "dccs_refc" if pipeline_mode == "tri_source" else pipeline_mode
    warnings: list[str] = []
    stage_timings: dict[str, float] = {}
    stage_events: list[dict[str, Any]] = []
    dccs_batches: list[dict[str, Any]] = []
    action_trace: list[dict[str, Any]] = []
    action_score_rows: list[dict[str, Any]] = []
    controller_state_rows: list[dict[str, Any]] = []
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
    refinement_policy = _effective_refinement_policy(req, pipeline_mode=execution_pipeline_mode)
    request_ambiguity_context = _request_ambiguity_context(req)
    ambiguity_strength = _ambiguity_strength_from_context(request_ambiguity_context)
    engine_disagreement_prior = max(0.0, float(req.od_engine_disagreement_prior or 0.0))
    hard_case_prior = max(0.0, float(req.od_hard_case_prior or 0.0))
    search_used = 0
    evidence_used = 0
    candidate_fetches = 0
    route_cache_runtime = {
        "cache_hits": 0,
        "cache_misses": 0,
        "reuse_rate": 0.0,
        "last_cache_key": None,
    }
    graph_search_meta: dict[str, Any] = {
        "graph_retry_attempted": False,
        "graph_retry_state_budget": 0,
        "graph_retry_outcome": "not_applicable",
        "graph_rescue_attempted": False,
        "graph_rescue_mode": "not_applicable",
        "graph_rescue_state_budget": 0,
        "graph_rescue_outcome": "not_applicable",
        "graph_search_ms_initial": 0.0,
        "graph_search_ms_retry": 0.0,
        "graph_search_ms_rescue": 0.0,
        "graph_search_ms_supplemental": 0.0,
        "graph_k_raw_cache_hit": False,
        "graph_low_ambiguity_fast_path": False,
        "graph_supported_ambiguity_fast_fallback": False,
    }
    preemptive_comparator_state: dict[str, Any] = {
        "activated": False,
        "candidate_count": 0,
        "sources": [],
        "trigger_reason": "",
        "strong_trigger_reason": "",
        "suppressed_reason": "",
    }

    def _stage_started(stage_name: str) -> float:
        stage_events.append({"stage": stage_name, "event": "enter", "timestamp_utc": _utc_now_iso()})
        return time.perf_counter()

    def _stage_finished(stage_name: str, started: float) -> None:
        stage_events.append({"stage": stage_name, "event": "exit", "timestamp_utc": _utc_now_iso()})
        stage_timings[f"{stage_name}_ms"] = stage_timings.get(f"{stage_name}_ms", 0.0) + round(
            (time.perf_counter() - started) * 1000.0,
            2,
        )

    refresh_live_runtime_route_caches(mode=str(settings.live_route_compute_refresh_mode or "route_compute").strip().lower())
    scenario_started = _stage_started("scenario_context")
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
    scenario_cache_token = hashlib.sha1(
        json.dumps(scenario_modifiers, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    _stage_finished("scenario_context", scenario_started)

    preflight_started = _stage_started("preflight")
    precheck = route_graph_od_feasibility(
        origin_lat=float(req.origin.lat),
        origin_lon=float(req.origin.lon),
        destination_lat=float(req.destination.lat),
        destination_lon=float(req.destination.lon),
    )
    _stage_finished("preflight", preflight_started)
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

    graph_started = _stage_started("k_raw")
    graph_routes, graph_diag, graph_search_meta = await _route_graph_k_raw_search(
        origin=req.origin,
        destination=req.destination,
        max_alternatives=max_alternatives,
        scenario_edge_modifiers=scenario_modifiers,
        start_node_id=start_node_id,
        goal_node_id=goal_node_id,
        od_ambiguity_index=req.od_ambiguity_index,
        od_engine_disagreement_prior=req.od_engine_disagreement_prior,
        od_hard_case_prior=req.od_hard_case_prior,
        od_ambiguity_support_ratio=getattr(req, "od_ambiguity_support_ratio", None),
        od_ambiguity_source_entropy=getattr(req, "od_ambiguity_source_entropy", None),
        od_candidate_path_count=getattr(req, "od_candidate_path_count", None),
        od_corridor_family_count=getattr(req, "od_corridor_family_count", None),
        od_nominal_margin_proxy=getattr(req, "od_nominal_margin_proxy", None),
        od_objective_spread=getattr(req, "od_objective_spread", None),
        allow_supported_ambiguity_fast_fallback=(
            execution_pipeline_mode in {"dccs", "dccs_refc", "voi"}
        ),
        enable_voi_support_rich_short_haul_graph_probe=(execution_pipeline_mode == "voi"),
    )
    _stage_finished("k_raw", graph_started)
    fallback_k_raw_reason = str(graph_diag.no_path_reason or "").strip()
    support_rich_short_haul_probe_fail_open = _support_rich_short_haul_probe_fail_open_needed(
        enabled=bool(graph_search_meta.get("graph_support_rich_short_haul_probe", False)),
        routes=graph_routes,
        od_candidate_path_count=getattr(req, "od_candidate_path_count", None),
        od_corridor_family_count=getattr(req, "od_corridor_family_count", None),
    )
    support_backed_single_corridor_probe_fail_open = _support_backed_single_corridor_probe_fail_open_needed(
        enabled=bool(graph_search_meta.get("graph_support_backed_single_corridor_diversity_probe", False)),
        routes=graph_routes,
        od_candidate_path_count=getattr(req, "od_candidate_path_count", None),
    )
    if support_rich_short_haul_probe_fail_open:
        fallback_k_raw_reason = "support_rich_short_haul_probe_underfilled"
        graph_search_meta["graph_support_rich_short_haul_probe_fail_open"] = True
    else:
        graph_search_meta["graph_support_rich_short_haul_probe_fail_open"] = False
    if support_backed_single_corridor_probe_fail_open:
        fallback_k_raw_reason = "support_backed_single_corridor_probe_underfilled"
        graph_search_meta["graph_support_backed_single_corridor_probe_fail_open"] = True
    else:
        graph_search_meta["graph_support_backed_single_corridor_probe_fail_open"] = False
    support_aware_fallback_reasons = {
        "skipped_supported_ambiguity_fast_fallback",
        "skipped_support_aware_graph_search",
        "skipped_support_aware_long_corridor_graph_search",
        "skipped_support_rich_short_haul_graph_search",
        "skipped_support_backed_single_corridor_graph_search",
        "support_rich_short_haul_probe_underfilled",
        "support_backed_single_corridor_probe_underfilled",
    }
    fallback_k_raw_specs = (
        _long_corridor_fallback_specs(
            origin=req.origin,
            destination=req.destination,
            max_routes=max_alternatives,
        )
        if fallback_k_raw_reason == "support_rich_short_haul_probe_underfilled"
        else (
            _support_aware_fallback_specs(
                origin=req.origin,
                destination=req.destination,
                max_routes=max_alternatives,
            )
            if fallback_k_raw_reason in support_aware_fallback_reasons
            else _long_corridor_fallback_specs(
                origin=req.origin,
                destination=req.destination,
                max_routes=max_alternatives,
            )
        )
    )
    fallback_k_raw_fetch_count = 0
    fallback_k_raw_routes_by_signature: dict[str, dict[str, Any]] = {}
    k_raw_from_refined_fallback = False
    fallback_k_raw_reasons = {
        "state_budget_exceeded",
        "path_search_exhausted",
        "no_path",
        "candidate_pool_exhausted",
        "skipped_long_corridor_graph_search",
        "skipped_supported_ambiguity_fast_fallback",
        "skipped_support_aware_graph_search",
        "skipped_support_aware_long_corridor_graph_search",
        "skipped_support_rich_short_haul_graph_search",
        "skipped_support_backed_single_corridor_graph_search",
        "routing_graph_deferred_load",
    }
    if (
        fallback_k_raw_specs
        and (
            (not graph_routes and fallback_k_raw_reason in fallback_k_raw_reasons)
            or support_rich_short_haul_probe_fail_open
            or support_backed_single_corridor_probe_fail_open
        )
    ):
        fallback_started = time.perf_counter()
        async for progress in _iter_candidate_fetches(
            osrm=osrm,
            origin=req.origin,
            destination=req.destination,
            specs=fallback_k_raw_specs,
        ):
            result = progress.result
            fallback_k_raw_fetch_count += 1
            if result.error:
                warnings.append(f"{result.spec.label}: {result.error}")
                continue
            per_route_refine_cost_ms = max(
                0.001,
                float(result.elapsed_ms) / float(max(1, len(result.routes) or 1)),
            )
            for route in result.routes:
                try:
                    signature = _route_signature(route)
                except OSRMError:
                    continue
                if signature not in fallback_k_raw_routes_by_signature:
                    fallback_k_raw_routes_by_signature[signature] = route
                _annotate_route_candidate_meta(
                    fallback_k_raw_routes_by_signature[signature],
                    source_labels={f"{result.spec.label}:direct_k_raw_fallback"},
                    toll_exclusion_available=False,
                    observed_refine_cost_ms=per_route_refine_cost_ms,
                )
        if bool(settings.route_graph_direct_k_raw_fallback_include_ors_seed):
            try:
                ors_started = time.perf_counter()
                ors_route, _ors_meta = await _fetch_local_ors_baseline_seed(req=req, ors=ors)
                ors_seed_elapsed_ms = max(0.001, (time.perf_counter() - ors_started) * 1000.0)
                try:
                    ors_signature = _route_signature(ors_route)
                except OSRMError:
                    ors_signature = ""
                if ors_signature and ors_signature not in fallback_k_raw_routes_by_signature:
                    fallback_k_raw_routes_by_signature[ors_signature] = dict(ors_route)
                    _annotate_route_candidate_meta(
                        fallback_k_raw_routes_by_signature[ors_signature],
                        source_labels={"local_ors:direct_k_raw_fallback"},
                        toll_exclusion_available=False,
                        observed_refine_cost_ms=ors_seed_elapsed_ms,
                    )
                ors_via = _graph_family_via_points(
                    ors_route,
                    max_landmarks=max(2, int(settings.route_graph_via_landmarks_per_path)),
                )
                if ors_via:
                    ors_realized_started = time.perf_counter()
                    ors_realized_routes = await osrm.fetch_routes(
                        origin_lat=req.origin.lat,
                        origin_lon=req.origin.lon,
                        dest_lat=req.destination.lat,
                        dest_lon=req.destination.lon,
                        alternatives=False,
                        via=ors_via,
                    )
                    if ors_realized_routes:
                        ors_realized_elapsed_ms = max(0.001, (time.perf_counter() - ors_realized_started) * 1000.0)
                        realized_route = ors_realized_routes[0]
                        try:
                            realized_signature = _route_signature(realized_route)
                        except OSRMError:
                            realized_signature = ""
                        if realized_signature and realized_signature not in fallback_k_raw_routes_by_signature:
                            fallback_k_raw_routes_by_signature[realized_signature] = realized_route
                            _annotate_route_candidate_meta(
                                fallback_k_raw_routes_by_signature[realized_signature],
                                source_labels={"local_ors_seed:direct_k_raw_fallback"},
                                toll_exclusion_available=False,
                                observed_refine_cost_ms=ors_realized_elapsed_ms,
                            )
            except (RuntimeError, ORSError) as exc:
                warnings.append(f"local_ors:direct_k_raw_fallback: {str(exc).strip() or 'unavailable'}")
        stage_timings["k_raw_osrm_fallback_ms"] = round(
            (time.perf_counter() - fallback_started) * 1000.0,
            2,
        )
        if fallback_k_raw_routes_by_signature:
            fallback_routes = list(fallback_k_raw_routes_by_signature.values())
            if (support_rich_short_haul_probe_fail_open or support_backed_single_corridor_probe_fail_open) and graph_routes:
                merged_graph_routes: dict[str, dict[str, Any]] = {}
                for route in [*graph_routes, *fallback_routes]:
                    try:
                        merged_graph_routes.setdefault(_route_signature(route), route)
                    except OSRMError:
                        continue
                fallback_routes = list(merged_graph_routes.values())
            ranked_fallback_routes = _select_ranked_candidate_routes(
                fallback_routes,
                max_routes=max_alternatives,
            )
            toll_excluded_routes = [
                route
                for route in ranked_fallback_routes
                if bool(
                    ((route.get("_candidate_meta") or {}) if isinstance(route.get("_candidate_meta"), dict) else {}).get(
                        "seen_by_exclude_toll",
                        False,
                    )
                )
            ]
            support_aware_direct_fallback = fallback_k_raw_reason in support_aware_fallback_reasons
            if toll_excluded_routes:
                toll_excluded_signatures = {
                    _route_signature(route)
                    for route in toll_excluded_routes
                }
                graph_routes = [
                    *toll_excluded_routes,
                    *[
                        route
                        for route in ranked_fallback_routes
                        if _route_signature(route) not in toll_excluded_signatures
                    ],
                ]
                prioritized_warning = (
                    "routing_graph_support_aware_toll_exclusion_prioritized"
                    if support_aware_direct_fallback
                    else "routing_graph_long_corridor_toll_exclusion_prioritized"
                )
                warnings.append(
                    f"route_graph: {prioritized_warning} "
                    f"(preferred={len(toll_excluded_routes)}, candidates={len(graph_routes)})."
                )
            else:
                graph_routes = ranked_fallback_routes
            if support_rich_short_haul_probe_fail_open:
                warnings.append(
                    "route_graph: routing_graph_support_rich_short_haul_probe_fail_open "
                    f"(graph_candidates={len(graph_routes)}, fallback_reason={fallback_k_raw_reason})."
                )
            if support_backed_single_corridor_probe_fail_open:
                warnings.append(
                    "route_graph: routing_graph_support_backed_single_corridor_probe_fail_open "
                    f"(graph_candidates={len(graph_routes)}, fallback_reason={fallback_k_raw_reason})."
                )
            fallback_warning = (
                "routing_graph_support_aware_k_raw_fallback"
                if support_aware_direct_fallback
                else "routing_graph_long_corridor_k_raw_fallback"
            )
            warnings.append(
                f"route_graph: {fallback_warning} "
                f"(engine_reason={fallback_k_raw_reason}, fallback_candidates={len(graph_routes)})."
            )
            k_raw_from_refined_fallback = True
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
    for candidate, route in zip(raw_candidate_payloads, graph_routes, strict=True):
        candidate_id = str(candidate["candidate_id"])
        route["_dccs_candidate_id"] = candidate_id
        existing_ids = route.get("_dccs_candidate_ids")
        route["_dccs_candidate_ids"] = list(existing_ids) if isinstance(existing_ids, list) else [candidate_id]
        if candidate_id not in route["_dccs_candidate_ids"]:
            route["_dccs_candidate_ids"].append(candidate_id)

    def _current_raw_corridor_count() -> int:
        return len(
            {
                _candidate_corridor_key(candidate)
                for candidate in raw_candidate_payloads
                if isinstance(candidate, Mapping)
            }
        )

    async def _apply_preemptive_comparator_seeds() -> None:
        nonlocal candidate_fetches
        if execution_pipeline_mode not in {"dccs", "dccs_refc", "voi"}:
            return
        if not bool(settings.route_dccs_preemptive_comparator_seed_enabled):
            return
        existing_corridor_count = _current_raw_corridor_count()
        support_aware_fallback_has_usable_raw_candidates = bool(
            existing_corridor_count > 0
            and (
                k_raw_from_refined_fallback
                or bool(graph_search_meta.get("graph_supported_ambiguity_fast_fallback", False))
                or bool(graph_search_meta.get("graph_support_rich_short_haul_fast_fallback", False))
            )
        )
        if support_aware_fallback_has_usable_raw_candidates:
            preemptive_comparator_state["activated"] = False
            preemptive_comparator_state["candidate_count"] = 0
            preemptive_comparator_state["sources"] = []
            preemptive_comparator_state["trigger_reason"] = ""
            preemptive_comparator_state["strong_trigger_reason"] = ""
            preemptive_comparator_state["suppressed_reason"] = "support_aware_fallback_usable_raw_candidates"
            return
        comparator_trigger_reasons: list[str] = []
        strong_trigger_reasons: list[str] = []
        support_ratio = max(
            0.0,
            min(1.0, _as_float_or_zero(getattr(req, "od_ambiguity_support_ratio", None))),
        )
        source_entropy = max(
            0.0,
            min(1.0, _as_float_or_zero(getattr(req, "od_ambiguity_source_entropy", None))),
        )
        corridor_coverage_empty = existing_corridor_count <= 0
        if ambiguity_strength >= float(settings.route_dccs_preemptive_comparator_min_ambiguity):
            comparator_trigger_reasons.append("ambiguity")
            if support_ratio >= 0.55:
                strong_trigger_reasons.append("ambiguity_supported")
        if engine_disagreement_prior >= float(settings.route_dccs_preemptive_comparator_min_engine_disagreement):
            comparator_trigger_reasons.append("engine_disagreement")
            strong_trigger_reasons.append("engine_disagreement")
        if hard_case_prior >= float(settings.route_dccs_preemptive_comparator_min_hard_case):
            comparator_trigger_reasons.append("hard_case")
            if support_ratio >= 0.45 or source_entropy >= 0.40:
                strong_trigger_reasons.append("hard_case_supported")
        if corridor_coverage_empty:
            comparator_trigger_reasons.append("corridor_coverage")
        if comparator_trigger_reasons and support_ratio >= 0.65 and source_entropy >= 0.50:
            strong_trigger_reasons.append("multi_source_support")
        if not comparator_trigger_reasons:
            preemptive_comparator_state["activated"] = False
            preemptive_comparator_state["candidate_count"] = 0
            preemptive_comparator_state["sources"] = []
            preemptive_comparator_state["trigger_reason"] = ",".join(sorted(comparator_trigger_reasons))
            preemptive_comparator_state["strong_trigger_reason"] = ""
            preemptive_comparator_state["suppressed_reason"] = "not_triggered"
            return
        allow_preemptive_seed = bool(
            (corridor_coverage_empty and len(set(strong_trigger_reasons)) >= 2)
            or (
                existing_corridor_count <= 0
                and ambiguity_strength >= 0.75
                and support_ratio >= 0.60
                and source_entropy >= 0.35
            )
        )
        if not allow_preemptive_seed:
            preemptive_comparator_state["activated"] = False
            preemptive_comparator_state["candidate_count"] = 0
            preemptive_comparator_state["sources"] = []
            preemptive_comparator_state["trigger_reason"] = ",".join(sorted(comparator_trigger_reasons))
            preemptive_comparator_state["strong_trigger_reason"] = ",".join(sorted(set(strong_trigger_reasons)))
            preemptive_comparator_state["suppressed_reason"] = "insufficient_upstream_support"
            return

        max_seed_candidates = max(
            1,
            min(
                int(settings.route_dccs_preemptive_comparator_max_candidates),
                max(2, int(max_alternatives)),
            ),
        )
        existing_signatures: set[str] = set()
        for route in graph_routes:
            try:
                existing_signatures.add(_route_signature(route))
            except OSRMError:
                continue
        preemptive_routes_by_signature: dict[str, tuple[dict[str, Any], str, str]] = {}

        def _register_preemptive_route(
            route: dict[str, Any],
            *,
            source_label: str,
            engine_name: str,
        ) -> None:
            try:
                signature = _route_signature(route)
            except OSRMError:
                return
            if signature in existing_signatures or signature in preemptive_routes_by_signature:
                return
            _annotate_route_candidate_meta(
                route,
                source_labels={source_label},
                toll_exclusion_available=False,
            )
            preemptive_routes_by_signature[signature] = (route, source_label, engine_name)

        comparator_started = time.perf_counter()
        osrm_specs = [
            CandidateFetchSpec(
                label="preemptive:osrm:alternatives",
                alternatives=max_seed_candidates,
            )
        ]
        async for progress in _iter_candidate_fetches(
            osrm=osrm,
            origin=req.origin,
            destination=req.destination,
            specs=osrm_specs,
        ):
            candidate_fetches += 1
            result = progress.result
            if result.error:
                warnings.append(f"{result.spec.label}: {result.error}")
                continue
            for route in result.routes:
                _register_preemptive_route(
                    route,
                    source_label=f"{result.spec.label}:preemptive_comparator_seed",
                    engine_name="osrm",
                )
        try:
            ors_route, _ors_meta = await _fetch_local_ors_baseline_seed(req=req, ors=ors)
            _register_preemptive_route(
                dict(ors_route),
                source_label="preemptive:local_ors:polyline_seed",
                engine_name="ors_local",
            )
            ors_via = _graph_family_via_points(
                ors_route,
                max_landmarks=max(2, int(settings.route_graph_via_landmarks_per_path)),
            )
            if ors_via:
                ors_realized_routes = await osrm.fetch_routes(
                    origin_lat=req.origin.lat,
                    origin_lon=req.origin.lon,
                    dest_lat=req.destination.lat,
                    dest_lon=req.destination.lon,
                    alternatives=False,
                    via=ors_via,
                )
                candidate_fetches += 1
                if ors_realized_routes:
                    _register_preemptive_route(
                        ors_realized_routes[0],
                        source_label="preemptive:local_ors:osrm_realized_seed",
                        engine_name="ors_local_seed",
                    )
        except (RuntimeError, ORSError) as exc:
            warnings.append(f"preemptive:local_ors: {str(exc).strip() or 'unavailable'}")

        if not preemptive_routes_by_signature:
            stage_timings["preemptive_comparator_seed_ms"] = round(
                stage_timings.get("preemptive_comparator_seed_ms", 0.0)
                + ((time.perf_counter() - comparator_started) * 1000.0),
                2,
            )
            return

        ranked_seed_candidates = _select_ranked_candidate_routes(
            [route for route, _source_label, _engine_name in preemptive_routes_by_signature.values()],
            max_routes=max_seed_candidates,
        )
        added_sources: set[str] = set()
        added_count = 0
        ranked_signatures: list[str] = []
        for route in ranked_seed_candidates:
            try:
                ranked_signatures.append(_route_signature(route))
            except OSRMError:
                continue
        ranked_signature_set = set(ranked_signatures)
        for signature, (route, source_label, engine_name) in preemptive_routes_by_signature.items():
            if signature not in ranked_signature_set or signature in existing_signatures:
                continue
            candidate_payload = _graph_route_candidate_payload(
                route,
                origin=req.origin,
                destination=req.destination,
                vehicle=vehicle,
                cost_toggles=req.cost_toggles,
            )
            candidate_id = f"preemptive:{engine_name}:{candidate_payload['candidate_id']}"
            candidate_payload["candidate_id"] = candidate_id
            candidate_payload["candidate_source_label"] = source_label
            candidate_payload["candidate_source_engine"] = engine_name
            candidate_payload["candidate_source_stage"] = "preemptive_comparator_seed"
            route["_dccs_candidate_id"] = candidate_id
            route["_dccs_candidate_ids"] = [candidate_id]
            raw_candidate_payloads.append(candidate_payload)
            raw_graph_routes_by_id[candidate_id] = route
            raw_candidate_by_id[candidate_id] = candidate_payload
            graph_routes.append(route)
            existing_signatures.add(signature)
            added_sources.add(engine_name)
            added_count += 1
        stage_timings["preemptive_comparator_seed_ms"] = round(
            stage_timings.get("preemptive_comparator_seed_ms", 0.0)
            + ((time.perf_counter() - comparator_started) * 1000.0),
            2,
        )
        if added_count > 0:
            preemptive_comparator_state["activated"] = True
            preemptive_comparator_state["candidate_count"] = int(added_count)
            preemptive_comparator_state["sources"] = sorted(added_sources)
            preemptive_comparator_state["trigger_reason"] = ",".join(sorted(comparator_trigger_reasons))
            preemptive_comparator_state["strong_trigger_reason"] = ",".join(sorted(set(strong_trigger_reasons)))
            preemptive_comparator_state["suppressed_reason"] = ""

    await _apply_preemptive_comparator_seeds()

    raw_candidate_corridor_count = _current_raw_corridor_count()
    raw_toll_values = {
        round(float(candidate.get("toll_share", 0.0) or 0.0), 3)
        for candidate in raw_candidate_payloads
        if isinstance(candidate, Mapping)
    }
    raw_proxy_durations = [
        float((candidate.get("proxy_objective") or (0.0, 0.0, 0.0))[0])
        for candidate in raw_candidate_payloads
        if isinstance(candidate, Mapping) and isinstance(candidate.get("proxy_objective"), (tuple, list))
    ]
    raw_duration_spread_ratio = (
        (max(raw_proxy_durations) - min(raw_proxy_durations)) / max(1.0, min(raw_proxy_durations))
        if len(raw_proxy_durations) >= 2
        else 0.0
    )
    reserve_diversity_rescue_slot = bool(
        execution_pipeline_mode in {"dccs", "voi"}
        and total_search_budget >= 2
        and (execution_pipeline_mode != "voi" or total_search_budget >= 3)
        and len(raw_candidate_payloads) >= 3
        and raw_candidate_corridor_count >= 2
        and (
            len(raw_candidate_payloads) >= 4
            or len(raw_toll_values) > 1
            or raw_duration_spread_ratio >= 0.08
        )
    )
    reserve_diversity_rescue_slots = 1 if reserve_diversity_rescue_slot else 0
    if (
        reserve_diversity_rescue_slot
        and total_search_budget >= 4
        and len(raw_candidate_payloads) >= 6
        and raw_candidate_corridor_count >= 4
        and raw_duration_spread_ratio >= 0.04
    ):
        reserve_diversity_rescue_slots = 2
    initial_budget = total_search_budget
    if execution_pipeline_mode == "voi":
        initial_budget = min(
            total_search_budget,
            max(1, int(settings.route_dccs_bootstrap_count)),
        )
        if reserve_diversity_rescue_slots > 0:
            initial_budget = min(
                initial_budget,
                max(1, total_search_budget - reserve_diversity_rescue_slots),
            )
    elif reserve_diversity_rescue_slots > 0:
        initial_budget = max(1, total_search_budget - reserve_diversity_rescue_slots)
    initial_dccs_started = _stage_started("dccs")
    initial_dccs = _select_candidate_records(
        candidates=raw_candidate_payloads,
        config=_dccs_runtime_config(
            mode="bootstrap",
            pipeline_variant=pipeline_mode,
            search_budget=initial_budget,
        ),
        run_seed=run_seed,
        refinement_policy=refinement_policy,
    )
    _stage_finished("dccs", initial_dccs_started)
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
        vehicle_type=req.vehicle_type,
        scenario_mode=req.scenario_mode,
        cost_toggles=req.cost_toggles,
        terrain_profile=req.terrain_profile,
        departure_time_utc=req.departure_time_utc,
        scenario_cache_token=scenario_cache_token,
        route_cache_runtime=route_cache_runtime,
    )
    warnings.extend(batch_warnings)
    _ingest_refined_routes(batch_routes)
    candidate_fetches += int(fallback_k_raw_fetch_count) + batch_fetches
    search_used += len(initial_dccs.selected)
    refined_candidate_ids.update(record.candidate_id for record in initial_dccs.selected)
    stage_timings["osrm_refine_ms"] = round(batch_refine_ms, 2)

    def _route_state_cache_key() -> str:
        route_state_cache_profile = _route_state_cache_profile(req=req, pipeline_mode=pipeline_mode)
        refined_route_evidence_fingerprint = [
            {
                "route_id": str(route.get("route_id") or route.get("id") or route.get("option_id") or ""),
                "evidence_provenance": _cache_key_component(route.get("evidence_provenance")),
                "evidence_tensor": _cache_key_component(route.get("evidence_tensor")),
            }
            for route in sorted(
                refined_routes,
                key=lambda item: str(item.get("route_id") or item.get("id") or item.get("option_id") or ""),
            )
        ]
        payload = {
            "refined_route_signatures": sorted(refined_route_signatures),
            "refined_route_evidence_fingerprint": refined_route_evidence_fingerprint,
            "ambiguity_context": _cache_key_component(request_ambiguity_context),
            "vehicle_type": str(req.vehicle_type),
            "scenario_mode": str(req.scenario_mode),
            "cost_toggles": _cache_key_component(req.cost_toggles),
            "terrain_profile": _cache_key_component(req.terrain_profile),
            "stochastic": _cache_key_component(req.stochastic),
            "emissions_context": _cache_key_component(req.emissions_context),
            "weather": _cache_key_component(req.weather),
            "incident_simulation": _cache_key_component(req.incident_simulation),
            "departure_time_utc": req.departure_time_utc.isoformat() if req.departure_time_utc else None,
            "weights": (
                float(req.weights.time),
                float(req.weights.money),
                float(req.weights.co2),
            ),
            "pipeline_mode": str(pipeline_mode),
            "route_state_cache_profile": route_state_cache_profile,
            "risk_aversion": float(req.risk_aversion),
            "optimization_mode": str(req.optimization_mode),
            "pareto_method": str(req.pareto_method),
            "epsilon": float(req.epsilon) if req.epsilon is not None else None,
            "max_alternatives": int(max_alternatives),
        }
        return hashlib.sha1(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest()

    def _update_option_build_reuse_rate() -> None:
        total_cache_events = int(option_build_runtime.get("cache_hits", 0)) + int(
            option_build_runtime.get("cache_misses", 0)
        )
        option_build_runtime["reuse_rate"] = round(
            int(option_build_runtime.get("cache_hits", 0)) / float(max(1, total_cache_events)),
            6,
        )

    def _restore_cached_route_state(
        cache_key: str,
        cached_entry: CachedRouteState,
        *,
        hit_scope: str,
    ) -> tuple[
        list[RouteOption],
        list[RouteOption],
        list[RouteOption],
        RouteOption,
        TerrainDiagnostics,
        dict[str, list[str]],
        dict[str, float],
    ]:
        option_build_runtime["last_cache_key"] = cache_key
        option_build_runtime["cache_hits"] = int(option_build_runtime.get("cache_hits", 0)) + 1
        if hit_scope == "local":
            option_build_runtime["cache_hits_local"] = int(option_build_runtime.get("cache_hits_local", 0)) + 1
        else:
            option_build_runtime["cache_hits_global"] = int(option_build_runtime.get("cache_hits_global", 0)) + 1
        option_build_runtime["saved_ms_estimate"] = float(option_build_runtime.get("saved_ms_estimate", 0.0)) + max(
            0.0,
            float(getattr(cached_entry, "estimated_option_build_ms", 0.0)),
        )
        option_build_runtime["saved_pareto_ms_estimate"] = float(
            option_build_runtime.get("saved_pareto_ms_estimate", 0.0)
        ) + max(
            0.0,
            float(getattr(cached_entry, "estimated_pareto_ms", 0.0)),
        )
        _update_option_build_reuse_rate()
        return cached_entry.state

    def _cache_route_state_payload(
        cache_key: str,
        *,
        state: tuple[
            list[RouteOption],
            list[RouteOption],
            list[RouteOption],
            RouteOption,
            TerrainDiagnostics,
            dict[str, list[str]],
            dict[str, float],
        ],
        estimated_option_build_ms: float,
        estimated_pareto_ms: float,
    ) -> None:
        cached_payload = CachedRouteState(
            state=state,
            estimated_option_build_ms=float(estimated_option_build_ms),
            estimated_pareto_ms=float(estimated_pareto_ms),
        )
        route_state_local_cache[cache_key] = cached_payload
        set_cached_route_state(cache_key, cached_payload)
        option_build_runtime["last_cache_key"] = cache_key
        _update_option_build_reuse_rate()

    def _rebuild_route_state() -> tuple[
        list[RouteOption],
        list[RouteOption],
        list[RouteOption],
        RouteOption,
        TerrainDiagnostics,
        dict[str, list[str]],
        dict[str, float],
    ]:
        cache_key = _route_state_cache_key()
        cached_entry = route_state_local_cache.get(cache_key)
        if cached_entry is not None:
            return _restore_cached_route_state(cache_key, cached_entry, hit_scope="local")
        cached_entry = get_cached_route_state(cache_key)
        if cached_entry is not None:
            route_state_local_cache[cache_key] = cached_entry
            return _restore_cached_route_state(cache_key, cached_entry, hit_scope="global")

        def _paired_refined_route_options(
            built_options: Sequence[RouteOption],
        ) -> list[tuple[dict[str, Any], RouteOption]]:
            option_by_id = {
                str(option.id): option
                for option in built_options
                if str(option.id).strip()
            }
            paired: list[tuple[dict[str, Any], RouteOption]] = []
            for route in refined_routes:
                option_id = str(route.get("_built_option_id") or "").strip()
                if not option_id:
                    continue
                option = option_by_id.get(option_id)
                if option is None:
                    continue
                paired.append((route, option))
            if len(paired) == len(option_by_id):
                return paired
            if not paired and len(built_options) == len(refined_routes):
                return list(zip(refined_routes, built_options))
            warning = (
                "option_build: route_option_pairing_mismatch "
                f"(refined={len(refined_routes)}, built={len(built_options)}, paired={len(paired)})."
            )
            if warning not in warnings:
                warnings.append(warning)
            return paired

        option_build_runtime["cache_misses"] = int(option_build_runtime.get("cache_misses", 0)) + 1
        option_build_runtime["rebuild_count"] = int(option_build_runtime.get("rebuild_count", 0)) + 1
        option_build_runtime["last_cache_key"] = cache_key
        build_started = _stage_started("option_build")
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
            optimization_mode=req.optimization_mode,
            pareto_method=req.pareto_method,
            epsilon=req.epsilon,
            max_alternatives=max_alternatives,
            option_prefix="route",
            route_option_cache_runtime=route_option_cache_runtime,
            scenario_policy_cache=scenario_policy_cache,
            lightweight=(_route_option_detail_level(req=req, pipeline_mode=pipeline_mode) == "summary"),
        )
        build_elapsed_ms = round((time.perf_counter() - build_started) * 1000.0, 2)
        _stage_finished("option_build", build_started)
        for warning in build_warnings:
            if warning not in warnings:
                warnings.append(warning)
        if len(options) != len(refined_routes):
            mismatch_warning = (
                "option_build: dropped_refined_routes "
                f"(refined={len(refined_routes)}, built={len(options)})."
            )
            if mismatch_warning not in warnings:
                warnings.append(mismatch_warning)
        pareto_started = _stage_started("pareto")
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
        pareto_elapsed_ms = round((time.perf_counter() - pareto_started) * 1000.0, 2)
        _stage_finished("pareto", pareto_started)
        enrichment_started = _stage_started("option_enrichment")
        option_enrichment_ms = 0.0
        if _should_hydrate_priority_route_options(req):
            options, strict_frontier, display_candidates, selected, option_enrichment_ms = _hydrate_priority_route_options(
                routes=refined_routes,
                options=options,
                strict_frontier=strict_frontier,
                display_candidates=display_candidates,
                selected=selected,
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
                optimization_mode=req.optimization_mode,
                pareto_method=req.pareto_method,
                epsilon=req.epsilon,
                max_alternatives=max_alternatives,
                scenario_policy_cache=scenario_policy_cache,
            )
        _stage_finished("option_enrichment", enrichment_started)
        if option_enrichment_ms > 0.0:
            build_elapsed_ms = round(build_elapsed_ms + option_enrichment_ms, 2)
        paired_refined_options = _paired_refined_route_options(options)
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
            for route, option in paired_refined_options
        }
        selection_score_map = _route_selection_score_map(
            display_candidates,
            w_time=req.weights.time,
            w_money=req.weights.money,
            w_co2=req.weights.co2,
            optimization_mode=req.optimization_mode,
            risk_aversion=req.risk_aversion,
        )
        rebuilt_state = (
            options,
            strict_frontier,
            display_candidates,
            selected,
            terrain_diag,
            option_candidate_ids,
            selection_score_map,
        )
        _cache_route_state_payload(
            cache_key,
            state=rebuilt_state,
            estimated_option_build_ms=float(build_elapsed_ms),
            estimated_pareto_ms=float(pareto_elapsed_ms),
        )
        return rebuilt_state

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
                observed_refine_cost=(
                    float(observed_refine_costs[record.candidate_id])
                    if record.candidate_id in observed_refine_costs
                    else None
                ),
                frontier_added=frontier_added,
                decision_flip=decision_flip,
                dominated_but_close=dominated_but_close,
                redundant=redundant,
            )

    certificate_result: Any | None = None
    fragility_result: Any | None = None
    world_manifest_payload: dict[str, Any] | None = None
    initial_certificate_artifacts: dict[str, Any] | None = None
    active_families: list[str] = []
    certification_frontier_route_ids: list[str] = []
    certification_frontier_meta: dict[str, Any] = {
        "strict_frontier_route_ids": [],
        "strict_frontier_count": 0,
        "certification_frontier_route_ids": [],
        "certification_frontier_count": 0,
        "certification_frontier_rescue_applied": False,
        "certification_frontier_rescue_reason": "not_requested",
        "certification_frontier_rescue_added_route_ids": [],
    }
    evidence_snapshot_manifest: dict[str, Any] = {}
    evidence_snapshot_hash = ""
    selected_certificate: RouteCertificationSummary | None = None
    stop_reason = "not_applicable"
    best_rejected_action_payload: dict[str, Any] | None = None
    search_completeness_score: float | None = None
    search_completeness_gap: float | None = None
    credible_search_uncertainty_flag: bool | None = None
    credible_evidence_uncertainty_flag: bool | None = None
    certification_cache: dict[str, tuple[Any, Any, dict[str, Any], list[str]]] = {}
    certification_runtime = {
        "cache_hits": 0,
        "cache_hits_local": 0,
        "cache_hits_global": 0,
        "cache_misses": 0,
        "shortcut_count": 0,
    }
    option_build_runtime = {
        "cache_hits": 0,
        "cache_hits_local": 0,
        "cache_hits_global": 0,
        "cache_misses": 0,
        "rebuild_count": 0,
        "refresh_rebuild_count": 0,
        "saved_ms_estimate": 0.0,
        "saved_pareto_ms_estimate": 0.0,
        "reuse_rate": 0.0,
        "last_cache_key": None,
    }
    scenario_policy_cache: dict[tuple[str, str], Any] = {}
    route_option_cache_runtime = {
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
    voi_dccs_runtime = {
        "cache_hits": 0,
        "cache_hits_local": 0,
        "cache_hits_global": 0,
        "cache_misses": 0,
        "reuse_rate": 0.0,
        "last_cache_key": None,
    }
    route_state_local_cache: dict[str, CachedRouteState] = {}
    voi_dccs_local_cache: dict[str, Any] = {}

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

    diversity_rescue_state: dict[str, Any] = {
        "reserved_slot": bool(reserve_diversity_rescue_slot),
        "collapse_detected": False,
        "collapse_reason": "",
        "raw_corridor_family_count": int(raw_candidate_corridor_count),
        "refined_corridor_family_count_before": 0,
        "refined_corridor_family_count_after": 0,
        "supplemental_challenger_activated": False,
        "supplemental_sources": [],
        "supplemental_candidate_count": 0,
        "supplemental_selected_count": 0,
        "supplemental_budget_used": 0,
    }
    leftover_challenger_state: dict[str, Any] = {
        "activated": False,
        "candidate_count": 0,
        "selected_count": 0,
        "budget_used": 0,
    }
    leftover_challenger_selected_ids: set[str] = set()

    def _realized_corridor_family_count(routes: Sequence[dict[str, Any]]) -> int:
        families: set[str] = set()
        for route in routes:
            family = _graph_family_signature(route) or _route_corridor_signature(route)
            if not family:
                try:
                    family = _route_signature(route)
                except OSRMError:
                    family = ""
            if family:
                families.add(str(family))
        return len(families)

    def _diversity_collapse_snapshot(current_frontier: Sequence[RouteOption]) -> dict[str, Any]:
        refined_corridor_count = _realized_corridor_family_count(refined_routes)
        collapse_reason = ""
        collapse_detected = False
        if (
            len(raw_candidate_payloads) > 1
            and raw_candidate_corridor_count > 1
            and len(current_frontier) <= 1
            and refined_corridor_count <= 1
        ):
            collapse_detected = True
            collapse_reason = "single_frontier_after_diverse_k_raw"
        elif (
            reserve_diversity_rescue_slots >= 2
            and len(current_frontier) <= 1
            and len(raw_candidate_payloads) >= 6
            and raw_candidate_corridor_count >= 4
            and len(refined_routes) >= 2
            and refined_corridor_count >= 2
        ):
            collapse_detected = True
            collapse_reason = "single_frontier_after_multi_family_refine"
        elif (
            len(current_frontier) <= 1
            and len(raw_candidate_payloads) >= 4
            and raw_candidate_corridor_count >= 3
            and len(refined_routes) >= 3
            and refined_corridor_count >= 2
        ):
            collapse_detected = True
            collapse_reason = "single_frontier_after_multi_family_refine"
        elif (
            len(raw_candidate_payloads) >= 4
            and raw_candidate_corridor_count >= 3
            and refined_corridor_count < min(2, raw_candidate_corridor_count)
        ):
            collapse_detected = True
            collapse_reason = "refined_family_collapse"
        return {
            "collapse_detected": collapse_detected,
            "collapse_reason": collapse_reason,
            "refined_corridor_family_count": int(refined_corridor_count),
        }

    async def _apply_leftover_budget_challenger_fill(
        *,
        current_options: Sequence[RouteOption],
        current_frontier: Sequence[RouteOption],
        current_selected: RouteOption,
        option_candidate_ids_map: Mapping[str, list[str]],
        selection_score_map: Mapping[str, float],
        allow_collapse: bool = False,
    ) -> tuple[
        list[RouteOption],
        list[RouteOption],
        list[RouteOption],
        RouteOption,
        TerrainDiagnostics,
        dict[str, list[str]],
        dict[str, float],
        bool,
    ]:
        nonlocal candidate_fetches
        nonlocal search_used
        if execution_pipeline_mode not in {"dccs", "dccs_refc"}:
            return (
                list(current_options),
                list(current_frontier),
                list(display_candidates),
                current_selected,
                terrain_diag,
                dict(option_candidate_ids_map),
                dict(selection_score_map),
                False,
            )
        remaining_search_budget = max(0, total_search_budget - search_used)
        if remaining_search_budget <= 0:
            return (
                list(current_options),
                list(current_frontier),
                list(display_candidates),
                current_selected,
                terrain_diag,
                dict(option_candidate_ids_map),
                dict(selection_score_map),
                False,
            )
        collapse = _diversity_collapse_snapshot(current_frontier)
        if collapse["collapse_detected"] and not allow_collapse:
            return (
                list(current_options),
                list(current_frontier),
                list(display_candidates),
                current_selected,
                terrain_diag,
                dict(option_candidate_ids_map),
                dict(selection_score_map),
                False,
            )
        if (
            not allow_collapse
            and reserve_diversity_rescue_slots >= 2
            and len(raw_candidate_payloads) >= 6
            and raw_candidate_corridor_count >= 4
            and len(refined_routes) < 3
        ):
            # Preserve the reserved two-slot family-rich budget until the
            # supplemental rescue lane has a chance to probe for collapse.
            return (
                list(current_options),
                list(current_frontier),
                list(display_candidates),
                current_selected,
                terrain_diag,
                dict(option_candidate_ids_map),
                dict(selection_score_map),
                False,
            )
        remaining_candidates = [
            raw_candidate_by_id[candidate_id]
            for candidate_id in sorted(raw_candidate_by_id)
            if candidate_id not in refined_candidate_ids
        ]
        if not remaining_candidates:
            return (
                list(current_options),
                list(current_frontier),
                list(display_candidates),
                current_selected,
                terrain_diag,
                dict(option_candidate_ids_map),
                dict(selection_score_map),
                False,
            )

        frontier_payloads = [
            _route_option_dccs_payload(option, option_candidate_ids_map=option_candidate_ids_map)
            for option in current_frontier
        ]
        refined_payloads = [
            _route_option_dccs_payload(option, option_candidate_ids_map=option_candidate_ids_map)
            for option in current_options
        ]
        challenger_started = _stage_started("dccs")
        challenger_dccs = select_candidates(
            remaining_candidates,
            frontier=frontier_payloads,
            refined=refined_payloads,
            config=_dccs_runtime_config(
                mode="challenger",
                pipeline_variant=pipeline_mode,
                search_budget=min(remaining_search_budget, len(remaining_candidates)),
            ),
        )
        _stage_finished("dccs", challenger_started)
        dccs_batches.append(
            {
                "iteration": len(dccs_batches),
                "leftover_budget_challenger": True,
                **challenger_dccs.summary,
            }
        )
        for record in [*challenger_dccs.selected, *challenger_dccs.skipped]:
            candidate_records_by_id[record.candidate_id] = record

        selected_records = challenger_dccs.selected[:remaining_search_budget]
        leftover_challenger_state["activated"] = True
        leftover_challenger_state["candidate_count"] = len(remaining_candidates)
        leftover_challenger_state["selected_count"] = len(selected_records)
        leftover_challenger_state["budget_used"] = len(selected_records)
        if not selected_records:
            return (
                list(current_options),
                list(current_frontier),
                list(display_candidates),
                current_selected,
                terrain_diag,
                dict(option_candidate_ids_map),
                dict(selection_score_map),
                False,
            )

        previous_frontier_ids = {option.id for option in current_frontier}
        previous_selected_id = current_selected.id
        batch_routes, batch_warnings, batch_observed_costs, batch_fetches, batch_refine_ms = await _refine_graph_candidate_batch(
            osrm=osrm,
            origin=req.origin,
            destination=req.destination,
            selected_records=selected_records,
            raw_graph_routes_by_id=raw_graph_routes_by_id,
            vehicle_type=req.vehicle_type,
            scenario_mode=req.scenario_mode,
            cost_toggles=req.cost_toggles,
            terrain_profile=req.terrain_profile,
            departure_time_utc=req.departure_time_utc,
            scenario_cache_token=scenario_cache_token,
            route_cache_runtime=route_cache_runtime,
        )
        warnings.extend(batch_warnings)
        _ingest_refined_routes(batch_routes)
        candidate_fetches += batch_fetches
        search_used += len(selected_records)
        refined_candidate_ids.update(record.candidate_id for record in selected_records)
        leftover_challenger_selected_ids.update(str(record.candidate_id) for record in selected_records)
        stage_timings["osrm_refine_ms"] = round(
            stage_timings.get("osrm_refine_ms", 0.0) + float(batch_refine_ms),
            2,
        )
        rebuilt = _rebuild_route_state()
        rebuilt_options, rebuilt_frontier, rebuilt_display, rebuilt_selected, rebuilt_terrain, rebuilt_option_ids, rebuilt_scores = rebuilt
        _record_candidate_batch_outcomes(
            selected_records=selected_records,
            previous_frontier_ids=previous_frontier_ids,
            previous_selected_id=previous_selected_id,
            option_candidate_ids=rebuilt_option_ids,
            current_frontier_ids={option.id for option in rebuilt_frontier},
            current_selected_id=rebuilt_selected.id,
            selection_score_map=rebuilt_scores,
            observed_refine_costs=batch_observed_costs,
        )
        return (
            list(rebuilt_options),
            list(rebuilt_frontier),
            list(rebuilt_display),
            rebuilt_selected,
            rebuilt_terrain,
            dict(rebuilt_option_ids),
            dict(rebuilt_scores),
            True,
        )

    async def _apply_supplemental_diversity_rescue(
        *,
        current_options: Sequence[RouteOption],
        current_frontier: Sequence[RouteOption],
        current_selected: RouteOption,
        option_candidate_ids_map: Mapping[str, list[str]],
        selection_score_map: Mapping[str, float],
    ) -> tuple[
        list[RouteOption],
        list[RouteOption],
        list[RouteOption],
        RouteOption,
        TerrainDiagnostics,
        dict[str, list[str]],
        dict[str, float],
        bool,
    ]:
        nonlocal candidate_fetches
        nonlocal search_used
        remaining_search_budget = max(0, total_search_budget - search_used)
        collapse = _diversity_collapse_snapshot(current_frontier)
        diversity_rescue_state["collapse_detected"] = bool(collapse["collapse_detected"])
        diversity_rescue_state["collapse_reason"] = str(collapse["collapse_reason"] or "")
        diversity_rescue_state["refined_corridor_family_count_before"] = int(
            collapse["refined_corridor_family_count"]
        )
        if not collapse["collapse_detected"] or remaining_search_budget <= 0:
            diversity_rescue_state["refined_corridor_family_count_after"] = int(
                collapse["refined_corridor_family_count"]
            )
            return (
                list(current_options),
                list(current_frontier),
                list(display_candidates),
                current_selected,
                terrain_diag,
                dict(option_candidate_ids_map),
                dict(selection_score_map),
                False,
            )

        rescue_sources: list[str] = []
        rescue_candidates: list[dict[str, Any]] = []
        rescue_observed_costs: dict[str, float] = {}
        rescue_routes_by_signature: dict[str, dict[str, Any]] = {}
        rescue_fetches = 0
        rescue_started = time.perf_counter()
        osrm_only_rescue = collapse["collapse_reason"] == "single_frontier_after_multi_family_refine"

        def _register_rescue_route(
            route: dict[str, Any],
            *,
            source_label: str,
            engine_name: str,
            observed_cost_ms: float,
        ) -> None:
            nonlocal rescue_candidates
            try:
                signature = _route_signature(route)
            except OSRMError:
                return
            if signature in refined_route_signatures or signature in rescue_routes_by_signature:
                return
            _annotate_route_candidate_meta(
                route,
                source_labels={source_label},
                toll_exclusion_available=False,
            )
            candidate_payload = _graph_route_candidate_payload(
                route,
                origin=req.origin,
                destination=req.destination,
                vehicle=vehicle,
                cost_toggles=req.cost_toggles,
            )
            candidate_id = f"supplemental:{engine_name}:{candidate_payload['candidate_id']}"
            candidate_payload["candidate_id"] = candidate_id
            candidate_payload["candidate_source_label"] = source_label
            candidate_payload["candidate_source_engine"] = engine_name
            candidate_payload["candidate_source_stage"] = "supplemental_diversity_rescue"
            route["_dccs_candidate_id"] = candidate_id
            route["_dccs_candidate_ids"] = [candidate_id]
            rescue_routes_by_signature[signature] = route
            raw_candidate_by_id[candidate_id] = candidate_payload
            raw_graph_routes_by_id[candidate_id] = route
            rescue_candidates.append(candidate_payload)
            rescue_observed_costs[candidate_id] = round(float(observed_cost_ms), 3)

        osrm_alt_budget = min(
            max(2, max_alternatives),
            int(settings.route_candidate_alternatives_max),
        )
        osrm_specs = [CandidateFetchSpec(label="supplemental:osrm:alternatives", alternatives=osrm_alt_budget)]
        async for progress in _iter_candidate_fetches(
            osrm=osrm,
            origin=req.origin,
            destination=req.destination,
            specs=osrm_specs,
        ):
            rescue_fetches += 1
            result = progress.result
            if result.error:
                warnings.append(f"{result.spec.label}: {result.error}")
                continue
            per_route_ms = 0.0
            if result.routes:
                per_route_ms = max(1.0, (time.perf_counter() - rescue_started) * 1000.0) / max(1, len(result.routes))
            for route in result.routes:
                _register_rescue_route(
                    route,
                    source_label=f"{result.spec.label}:supplemental_diversity_rescue",
                    engine_name="osrm",
                observed_cost_ms=per_route_ms,
            )
        if not osrm_only_rescue:
            try:
                ors_started = time.perf_counter()
                ors_route, _ors_meta = await _fetch_local_ors_baseline_seed(req=req, ors=ors)
                rescue_fetches += 1
                _register_rescue_route(
                    dict(ors_route),
                    source_label="supplemental:local_ors:polyline_seed",
                    engine_name="ors_local",
                    observed_cost_ms=(time.perf_counter() - ors_started) * 1000.0,
                )
                ors_via = _graph_family_via_points(
                    ors_route,
                    max_landmarks=max(2, int(settings.route_graph_via_landmarks_per_path)),
                )
                ors_realized_routes = []
                if ors_via:
                    ors_realized_routes = await osrm.fetch_routes(
                        origin_lat=req.origin.lat,
                        origin_lon=req.origin.lon,
                        dest_lat=req.destination.lat,
                        dest_lon=req.destination.lon,
                        alternatives=False,
                        via=ors_via,
                    )
                if ors_realized_routes:
                    _register_rescue_route(
                        ors_realized_routes[0],
                        source_label="supplemental:local_ors:osrm_realized_seed",
                        engine_name="ors_local_seed",
                        observed_cost_ms=(time.perf_counter() - ors_started) * 1000.0,
                    )
                else:
                    warnings.append("supplemental:local_ors: corridor seed could not be re-realized through OSRM.")
            except (RuntimeError, ORSError) as exc:
                warnings.append(f"supplemental:local_ors: {str(exc).strip() or 'unavailable'}")

        if not rescue_candidates:
            diversity_rescue_state["refined_corridor_family_count_after"] = int(
                collapse["refined_corridor_family_count"]
            )
            stage_timings["supplemental_rescue_ms"] = round(
                stage_timings.get("supplemental_rescue_ms", 0.0) + ((time.perf_counter() - rescue_started) * 1000.0),
                2,
            )
            return (
                list(current_options),
                list(current_frontier),
                list(display_candidates),
                current_selected,
                terrain_diag,
                dict(option_candidate_ids_map),
                dict(selection_score_map),
                False,
            )

        frontier_payloads = [
            _route_option_dccs_payload(option, option_candidate_ids_map=option_candidate_ids_map)
            for option in current_frontier
        ]
        refined_payloads = [
            _route_option_dccs_payload(option, option_candidate_ids_map=option_candidate_ids_map)
            for option in current_options
        ]
        rescue_dccs = select_candidates(
            rescue_candidates,
            frontier=frontier_payloads,
            refined=refined_payloads,
            config=_dccs_runtime_config(
                mode="challenger",
                pipeline_variant=pipeline_mode,
                search_budget=min(remaining_search_budget, len(rescue_candidates)),
            ),
        )
        dccs_batches.append(
            {
                "iteration": len(dccs_batches),
                "supplemental_diversity_rescue": True,
                **rescue_dccs.summary,
            }
        )
        for record in [*rescue_dccs.selected, *rescue_dccs.skipped]:
            candidate_records_by_id[record.candidate_id] = record

        selected_records = rescue_dccs.selected[:remaining_search_budget]
        if not selected_records:
            diversity_rescue_state["refined_corridor_family_count_after"] = int(
                collapse["refined_corridor_family_count"]
            )
            return (
                list(current_options),
                list(current_frontier),
                list(display_candidates),
                current_selected,
                terrain_diag,
                dict(option_candidate_ids_map),
                dict(selection_score_map),
                False,
            )

        previous_frontier_ids = {option.id for option in current_frontier}
        previous_selected_id = current_selected.id
        _ingest_refined_routes([raw_graph_routes_by_id[record.candidate_id] for record in selected_records])
        refined_candidate_ids.update(record.candidate_id for record in selected_records)
        search_used += len(selected_records)
        candidate_fetches += rescue_fetches
        rescue_sources = sorted(
            {
                str(raw_candidate_by_id.get(record.candidate_id, {}).get("candidate_source_engine") or "").strip()
                for record in selected_records
            }
        )
        stage_timings["supplemental_rescue_ms"] = round(
            stage_timings.get("supplemental_rescue_ms", 0.0) + ((time.perf_counter() - rescue_started) * 1000.0),
            2,
        )
        rebuilt = _rebuild_route_state()
        rebuilt_options, rebuilt_frontier, rebuilt_display, rebuilt_selected, rebuilt_terrain, rebuilt_option_ids, rebuilt_scores = rebuilt
        _record_candidate_batch_outcomes(
            selected_records=selected_records,
            previous_frontier_ids=previous_frontier_ids,
            previous_selected_id=previous_selected_id,
            option_candidate_ids=rebuilt_option_ids,
            current_frontier_ids={option.id for option in rebuilt_frontier},
            current_selected_id=rebuilt_selected.id,
            selection_score_map=rebuilt_scores,
            observed_refine_costs=rescue_observed_costs,
        )
        diversity_rescue_state["supplemental_challenger_activated"] = True
        diversity_rescue_state["supplemental_sources"] = rescue_sources
        diversity_rescue_state["supplemental_candidate_count"] = len(rescue_candidates)
        diversity_rescue_state["supplemental_selected_count"] = len(selected_records)
        diversity_rescue_state["supplemental_budget_used"] = len(selected_records)
        diversity_rescue_state["refined_corridor_family_count_after"] = _realized_corridor_family_count(refined_routes)
        return (
            list(rebuilt_options),
            list(rebuilt_frontier),
            list(rebuilt_display),
            rebuilt_selected,
            rebuilt_terrain,
            dict(rebuilt_option_ids),
            dict(rebuilt_scores),
            True,
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
    (
        options,
        strict_frontier,
        display_candidates,
        selected,
        terrain_diag,
        option_candidate_ids,
        selection_score_map,
        _leftover_budget_challenger_applied,
    ) = await _apply_leftover_budget_challenger_fill(
        current_options=options,
        current_frontier=strict_frontier,
        current_selected=selected,
        option_candidate_ids_map=option_candidate_ids,
        selection_score_map=selection_score_map,
    )
    (
        options,
        strict_frontier,
        display_candidates,
        selected,
        terrain_diag,
        option_candidate_ids,
        selection_score_map,
        _supplemental_rescue_applied,
    ) = await _apply_supplemental_diversity_rescue(
        current_options=options,
        current_frontier=strict_frontier,
        current_selected=selected,
        option_candidate_ids_map=option_candidate_ids,
        selection_score_map=selection_score_map,
    )
    if (
        not _supplemental_rescue_applied
        and str(diversity_rescue_state.get("collapse_reason") or "") == "single_frontier_after_multi_family_refine"
    ):
        (
            options,
            strict_frontier,
            display_candidates,
            selected,
            terrain_diag,
            option_candidate_ids,
            selection_score_map,
            _post_rescue_leftover_challenger_applied,
        ) = await _apply_leftover_budget_challenger_fill(
            current_options=options,
            current_frontier=strict_frontier,
            current_selected=selected,
            option_candidate_ids_map=option_candidate_ids,
            selection_score_map=selection_score_map,
            allow_collapse=True,
        )
    evidence_base_options = list(options)

    def _paired_refined_options_for_state(
        current_options: Sequence[RouteOption],
    ) -> list[tuple[dict[str, Any], RouteOption]]:
        return _pair_refined_routes_to_options_by_signature(refined_routes, current_options)

    def _route_signature_map(current_options: Sequence[RouteOption]) -> dict[str, str]:
        return _stable_route_signature_map_for_options(refined_routes, current_options)

    def _selected_final_route_source_payload(
        current_options: Sequence[RouteOption],
        *,
        selected_option_id: str,
    ) -> dict[str, str | None]:
        for route, option in _paired_refined_options_for_state(current_options):
            if str(option.id).strip() != str(selected_option_id).strip():
                continue
            return _selected_route_source_payload(route)
        return {
            "source_label": None,
            "source_stage": None,
            "source_engine": None,
        }

    def _evidence_snapshot_manifest_payload(current_options: Sequence[RouteOption]) -> dict[str, Any]:
        route_payloads: list[dict[str, Any]] = []
        for option in current_options:
            provenance = option.evidence_provenance
            families = []
            active_family_names: list[str] = []
            if provenance is not None:
                active_family_names = list(provenance.active_families)
                for family in provenance.families:
                    families.append(family.model_dump(mode="json"))
            route_payloads.append(
                {
                    "route_id": option.id,
                    "active_families": active_family_names,
                    "families": families,
                }
            )
        payload = {
            "schema_version": "1.0.0",
            "run_seed": int(run_seed),
            "pipeline_mode": pipeline_mode,
            "routes": route_payloads,
        }
        payload["snapshot_hash"] = hashlib.sha1(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest()
        return payload

    force_single_frontier_full_stress_requested_worlds = False

    def _refresh_certification_views() -> None:
        nonlocal certificate_result
        nonlocal fragility_result
        nonlocal world_manifest_payload
        nonlocal active_families
        nonlocal certification_frontier_route_ids
        nonlocal certification_frontier_meta
        nonlocal selected_certificate
        nonlocal strict_frontier
        nonlocal display_candidates
        nonlocal selected
        nonlocal force_single_frontier_full_stress_requested_worlds
        ambiguity_context = _request_ambiguity_context(req)
        if execution_pipeline_mode not in {"dccs_refc", "voi"}:
            return
        certification_frontier, certification_frontier_meta = _refc_certification_frontier_options(
            pipeline_mode=execution_pipeline_mode,
            strict_frontier=strict_frontier,
            options=options,
            selected=selected,
            ambiguity_context=ambiguity_context,
            w_time=req.weights.time,
            w_money=req.weights.money,
            w_co2=req.weights.co2,
            optimization_mode=req.optimization_mode,
            risk_aversion=req.risk_aversion,
        )
        certification_frontier_route_ids = list(
            certification_frontier_meta.get("certification_frontier_route_ids", [])
        )
        frontier_signature_map = _certification_frontier_signature_map(certification_frontier)
        certification_route_payloads = [
            _route_option_certification_payload(option)
            for option in certification_frontier
        ]
        evidence_snapshot_hash = str(_evidence_snapshot_manifest_payload(strict_frontier).get("snapshot_hash") or "")
        cache_key = _certification_cache_key(
            frontier_route_ids=[option.id for option in certification_frontier],
            frontier_signatures=frontier_signature_map,
            route_payloads=certification_route_payloads,
            evidence_snapshot_hash=evidence_snapshot_hash,
            selected_route_id=selected.id,
            run_seed=int(run_seed),
            world_count=int(current_world_count),
            threshold=float(certificate_threshold),
            weights=(
                float(req.weights.time),
                float(req.weights.money),
                float(req.weights.co2),
            ),
            optimization_mode=str(req.optimization_mode),
            risk_aversion=float(req.risk_aversion),
            forced_refreshed_families=sorted(forced_refreshed_families),
            ambiguity_context=ambiguity_context,
            force_single_frontier_full_stress_requested_worlds=(
                force_single_frontier_full_stress_requested_worlds
            ),
        )
        cached = certification_cache.get(cache_key)
        if cached is None:
            global_cached = get_cached_certification(cache_key)
            if global_cached is None:
                refc_started = _stage_started("refc")
                certificate_result, fragility_result, world_manifest_payload, active_families = _compute_frontier_certification(
                    frontier_options=certification_frontier,
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
                    ambiguity_context=ambiguity_context,
                    force_single_frontier_full_stress_requested_worlds=(
                        force_single_frontier_full_stress_requested_worlds
                    ),
                )
                _stage_finished("refc", refc_started)
                cached_payload = (
                    certificate_result,
                    fragility_result,
                    world_manifest_payload,
                    list(active_families),
                )
                if isinstance(world_manifest_payload, Mapping):
                    world_manifest_payload = annotate_world_manifest_cache_reuse(
                        world_manifest_payload,
                        cache_reuse_origin="miss",
                    )
                    cached_payload = (
                        certificate_result,
                        fragility_result,
                        world_manifest_payload,
                        list(active_families),
                    )
                global_cached_payload = _global_certification_cache_payload(
                    certificate_result=certificate_result,
                    fragility_result=fragility_result,
                    world_manifest_payload=world_manifest_payload,
                    active_families=active_families,
                )
                certification_cache[cache_key] = cached_payload
                certification_runtime["cache_store_attempts"] = int(
                    certification_runtime.get("cache_store_attempts", 0)
                ) + 1
                if set_cached_certification(cache_key, global_cached_payload):
                    certification_runtime["cache_store_successes"] = int(
                        certification_runtime.get("cache_store_successes", 0)
                    ) + 1
                else:
                    certification_runtime["cache_store_rejections"] = int(
                        certification_runtime.get("cache_store_rejections", 0)
                    ) + 1
                certification_runtime["cache_misses"] = int(certification_runtime.get("cache_misses", 0)) + 1
                if isinstance(world_manifest_payload, Mapping) and str(world_manifest_payload.get("status") or "").strip() == "single_frontier_shortcut":
                    certification_runtime["shortcut_count"] = int(certification_runtime.get("shortcut_count", 0)) + 1
            else:
                certificate_result, fragility_result, world_manifest_payload, cached_families = global_cached
                active_families = list(cached_families)
                if isinstance(world_manifest_payload, Mapping):
                    world_manifest_payload = annotate_world_manifest_cache_reuse(
                        world_manifest_payload,
                        cache_reuse_origin="global",
                    )
                    certification_cache[cache_key] = (
                        certificate_result,
                        fragility_result,
                        world_manifest_payload,
                        list(active_families),
                    )
                certification_runtime["cache_hits"] = int(certification_runtime.get("cache_hits", 0)) + 1
                certification_runtime["cache_hits_global"] = int(certification_runtime.get("cache_hits_global", 0)) + 1
                if isinstance(world_manifest_payload, Mapping) and str(world_manifest_payload.get("status") or "").strip() == "single_frontier_shortcut":
                    certification_runtime["shortcut_count"] = int(certification_runtime.get("shortcut_count", 0)) + 1
        else:
            certificate_result, fragility_result, world_manifest_payload, cached_families = cached
            active_families = list(cached_families)
            if isinstance(world_manifest_payload, Mapping):
                world_manifest_payload = annotate_world_manifest_cache_reuse(
                    world_manifest_payload,
                    cache_reuse_origin="local",
                )
                certification_cache[cache_key] = (
                    certificate_result,
                    fragility_result,
                    world_manifest_payload,
                    list(active_families),
                )
            certification_runtime["cache_hits"] = int(certification_runtime.get("cache_hits", 0)) + 1
            certification_runtime["cache_hits_local"] = int(certification_runtime.get("cache_hits_local", 0)) + 1
        controller_refresh_frontier_mode = "report_frontier"
        controller_refresh_frontier_route_ids = [option.id for option in certification_frontier]
        if (
            fragility_result is not None
            and bool(certification_frontier_meta.get("certification_frontier_rescue_applied", False))
            and len(certification_frontier) > len(strict_frontier)
            and len(strict_frontier) == 1
        ):
            strict_overlay_started = _stage_started("refc")
            (
                _strict_overlay_certificate,
                strict_controller_fragility,
                _strict_overlay_manifest,
                _strict_overlay_families,
            ) = _compute_frontier_certification(
                frontier_options=list(strict_frontier),
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
                ambiguity_context=ambiguity_context,
                force_single_frontier_full_stress_requested_worlds=(
                    force_single_frontier_full_stress_requested_worlds
                ),
            )
            _stage_finished("refc", strict_overlay_started)
            fragility_result = _merge_controller_refresh_overlay(
                report_fragility=fragility_result,
                controller_fragility=strict_controller_fragility,
                controller_frontier_route_ids=[option.id for option in strict_frontier],
                controller_frontier_mode="strict_frontier",
            )
            controller_refresh_frontier_mode = "strict_frontier"
            controller_refresh_frontier_route_ids = [option.id for option in strict_frontier]
        elif fragility_result is not None:
            fragility_result = _merge_controller_refresh_overlay(
                report_fragility=fragility_result,
                controller_fragility=None,
                controller_frontier_route_ids=controller_refresh_frontier_route_ids,
                controller_frontier_mode=controller_refresh_frontier_mode,
            )
        if isinstance(world_manifest_payload, Mapping):
            world_manifest_payload = {
                **dict(world_manifest_payload),
                **certification_frontier_meta,
                "controller_refresh_frontier_mode": controller_refresh_frontier_mode,
                "controller_refresh_frontier_route_ids": list(controller_refresh_frontier_route_ids),
                "controller_refresh_frontier_count": len(controller_refresh_frontier_route_ids),
            }
            ambiguity_context = {
                **ambiguity_context,
                "refc_world_count": int(world_manifest_payload.get("world_count") or 0),
                "refc_unique_world_count": int(world_manifest_payload.get("unique_world_count") or 0),
                "refc_effective_world_count": int(world_manifest_payload.get("effective_world_count") or 0),
                "refc_requested_world_count": int(current_world_count),
                "refc_sampler_requested_world_count": int(
                    world_manifest_payload.get("sampler_requested_world_count") or 0
                ),
                "refc_world_count_shortfall": max(
                    0,
                    int(current_world_count) - int(world_manifest_payload.get("world_count") or 0),
                ),
                "refc_world_reuse_rate": float(world_manifest_payload.get("world_reuse_rate") or 0.0),
                "refc_world_count_policy": str(world_manifest_payload.get("world_count_policy") or ""),
                "refc_hard_stress_pack_count": int(world_manifest_payload.get("hard_case_stress_pack_count") or 0),
                "refc_stress_world_fraction": float(world_manifest_payload.get("stress_world_fraction") or 0.0),
            }
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
            ambiguity_context=ambiguity_context,
        )
        display_candidates, display_summaries = _attach_route_certifications(
            display_candidates,
            certificate_map=certificate_result.certificate,
            threshold=certificate_threshold,
            active_families=active_families,
            fragility_map=fragility_result.route_fragility_map,
            competitor_breakdown=fragility_result.competitor_fragility_breakdown,
            top_refresh_family=top_refresh_family,
            ambiguity_context=ambiguity_context,
        )
        selected = next((option for option in display_candidates if option.id == selected.id), selected)
        selected_certificate = (
            display_summaries.get(selected.id)
            or strict_summaries.get(selected.id)
        )

    def _certification_artifact_payloads(
        *,
        selected_route_id: str,
        frontier_route_ids: Sequence[str],
        certification_frontier_route_ids: Sequence[str],
        certificate_snapshot: Any | None,
        fragility_snapshot: Any | None,
        world_manifest_snapshot: Mapping[str, Any] | None,
        active_family_names: Sequence[str],
        forced_refreshed_family_names: Sequence[str],
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        ambiguity_context = _request_ambiguity_context(req)
        if (
            certificate_snapshot is not None
            and fragility_snapshot is not None
            and world_manifest_snapshot is not None
        ):
            certificate_value = float(certificate_snapshot.certificate.get(selected_route_id, 0.0))
            world_manifest_dict = dict(world_manifest_snapshot)
            return (
                {
                    "pipeline_mode": pipeline_mode,
                    "winner_route_id": certificate_snapshot.winner_id,
                    "selected_route_id": selected_route_id,
                    "selected_certificate": certificate_value,
                    "empirical_selected_certificate": float(
                        world_manifest_dict.get("single_frontier_empirical_certificate", certificate_value)
                    ),
                    "selected_certificate_basis": str(
                        world_manifest_dict.get("selected_certificate_basis", "empirical")
                    ),
                    "single_frontier_certificate_cap": world_manifest_dict.get(
                        "single_frontier_certificate_cap"
                    ),
                    "single_frontier_certificate_cap_applied": bool(
                        world_manifest_dict.get("single_frontier_certificate_cap_applied", False)
                    ),
                    "single_frontier_requires_full_stress": bool(
                        world_manifest_dict.get("single_frontier_requires_full_stress", False)
                    ),
                    "certificate_threshold": float(certificate_threshold),
                    "certified": bool(certificate_value >= certificate_threshold),
                    "route_certificates": dict(certificate_snapshot.certificate),
                    "frontier_route_ids": list(frontier_route_ids),
                    "strict_frontier_route_ids": list(frontier_route_ids),
                    "strict_frontier_count": int(
                        world_manifest_dict.get("strict_frontier_count", len(frontier_route_ids))
                    ),
                    "certification_frontier_route_ids": list(certification_frontier_route_ids),
                    "certification_frontier_count": int(
                        world_manifest_dict.get(
                            "certification_frontier_count",
                            len(certification_frontier_route_ids),
                        )
                    ),
                    "certification_frontier_rescue_applied": bool(
                        world_manifest_dict.get("certification_frontier_rescue_applied", False)
                    ),
                    "certification_frontier_rescue_reason": str(
                        world_manifest_dict.get("certification_frontier_rescue_reason", "")
                    ),
                    "certification_frontier_rescue_added_route_ids": list(
                        world_manifest_dict.get("certification_frontier_rescue_added_route_ids", [])
                    ),
                    "world_count": int(world_manifest_dict.get("world_count", current_world_count)),
                    "requested_world_count": int(
                        world_manifest_dict.get("requested_world_count", current_world_count)
                    ),
                    "effective_world_count": int(
                        world_manifest_dict.get(
                            "effective_world_count",
                            world_manifest_dict.get("world_count", current_world_count),
                        )
                    ),
                    "world_count_policy": str(world_manifest_dict.get("world_count_policy", "configured")),
                    "active_families": list(active_family_names),
                    "selector_config": copy.deepcopy(certificate_snapshot.selector_config),
                    "forced_refreshed_families": sorted(forced_refreshed_family_names),
                    "ambiguity_context": ambiguity_context,
                    "unique_world_count": int(
                        world_manifest_dict.get(
                            "unique_world_count",
                            world_manifest_dict.get("world_count", current_world_count),
                        )
                    ),
                    "world_reuse_rate": float(world_manifest_dict.get("world_reuse_rate", 0.0)),
                    "hard_case_stress_pack_count": int(
                        world_manifest_dict.get("hard_case_stress_pack_count", 0)
                    ),
                },
                copy.deepcopy(fragility_snapshot.route_fragility_map),
                copy.deepcopy(fragility_snapshot.competitor_fragility_breakdown),
                copy.deepcopy(fragility_snapshot.value_of_refresh),
                copy.deepcopy(world_manifest_dict),
            )
        return (
            {
                "pipeline_mode": pipeline_mode,
                "status": "not_requested",
                "selected_route_id": selected_route_id,
            },
            {},
            {},
            {"ranking": [], "top_refresh_family": None},
            {
                "status": "not_requested",
                "world_count": 0,
                "active_families": [],
                "worlds": [],
                "ambiguity_context": ambiguity_context,
            },
        )

    def _apply_committed_refresh_objectives() -> None:
        nonlocal options
        nonlocal strict_frontier
        nonlocal display_candidates
        nonlocal selected
        nonlocal selection_score_map
        if execution_pipeline_mode != "voi":
            return
        if not forced_refreshed_families or not active_families or not evidence_base_options:
            return
        cache_key = _committed_refresh_route_state_cache_key(
            base_route_state_cache_key=_route_state_cache_key(),
            active_families=tuple(active_families),
            forced_refreshed_families=tuple(sorted(forced_refreshed_families)),
        )
        cached_entry = route_state_local_cache.get(cache_key)
        local_hit = cached_entry is not None
        if cached_entry is None:
            cached_entry = get_cached_route_state(cache_key)
            if cached_entry is not None:
                route_state_local_cache[cache_key] = cached_entry
        if cached_entry is not None:
            refreshed_state = _restore_cached_route_state(
                cache_key,
                cached_entry,
                hit_scope="local" if local_hit else "global",
            )
            (
                options,
                strict_frontier,
                display_candidates,
                selected,
                _cached_refresh_terrain_diag,
                _cached_refresh_candidate_ids,
                selection_score_map,
            ) = refreshed_state
            return
        option_build_runtime["cache_misses"] = int(option_build_runtime.get("cache_misses", 0)) + 1
        option_build_runtime["rebuild_count"] = int(option_build_runtime.get("rebuild_count", 0)) + 1
        option_build_runtime["refresh_rebuild_count"] = int(option_build_runtime.get("refresh_rebuild_count", 0)) + 1
        option_build_runtime["last_cache_key"] = cache_key
        refresh_world = {
            "world_id": "voi:committed_refresh",
            "states": {
                family: ("refreshed" if family in forced_refreshed_families else "nominal")
                for family in active_families
            },
        }
        refresh_build_started = _stage_started("option_build")
        refreshed_options: list[RouteOption] = []
        for option in evidence_base_options:
            payload = _route_option_certification_payload(option)
            refreshed_objective = _route_perturbed_objectives(
                payload,
                refresh_world,
                active_families=active_families,
            )
            refreshed_options.append(_clone_option_with_objectives(option, refreshed_objective))
        refresh_build_elapsed_ms = round((time.perf_counter() - refresh_build_started) * 1000.0, 2)
        _stage_finished("option_build", refresh_build_started)
        refresh_pareto_started = _stage_started("pareto")
        refreshed_frontier = _strict_frontier_options(
            refreshed_options,
            max_alternatives=max_alternatives,
            pareto_method=req.pareto_method,
            epsilon=req.epsilon,
            optimization_mode=req.optimization_mode,
            risk_aversion=req.risk_aversion,
        )
        if not refreshed_frontier:
            _stage_finished("pareto", refresh_pareto_started)
            return
        refreshed_display = _finalize_pareto_options(
            refreshed_options,
            max_alternatives=max_alternatives,
            pareto_method=req.pareto_method,
            epsilon=req.epsilon,
            optimization_mode=req.optimization_mode,
            risk_aversion=req.risk_aversion,
        )
        refreshed_selected = _pick_best_option(
            refreshed_frontier,
            w_time=req.weights.time,
            w_money=req.weights.money,
            w_co2=req.weights.co2,
            optimization_mode=req.optimization_mode,
            risk_aversion=req.risk_aversion,
        )
        refresh_pareto_elapsed_ms = round((time.perf_counter() - refresh_pareto_started) * 1000.0, 2)
        _stage_finished("pareto", refresh_pareto_started)
        if all(option.id != refreshed_selected.id for option in refreshed_display):
            refreshed_display = [
                refreshed_selected,
                *[option for option in refreshed_display if option.id != refreshed_selected.id],
            ][: max_alternatives]
        refreshed_state = (
            refreshed_options,
            refreshed_frontier,
            refreshed_display,
            refreshed_selected,
            terrain_diag,
            {},
            _route_selection_score_map(
                refreshed_display,
                w_time=req.weights.time,
                w_money=req.weights.money,
                w_co2=req.weights.co2,
                optimization_mode=req.optimization_mode,
                risk_aversion=req.risk_aversion,
            ),
        )
        _cache_route_state_payload(
            cache_key,
            state=refreshed_state,
            estimated_option_build_ms=float(refresh_build_elapsed_ms),
            estimated_pareto_ms=float(refresh_pareto_elapsed_ms),
        )
        options = refreshed_options
        strict_frontier = refreshed_frontier
        display_candidates = refreshed_display
        selected = refreshed_selected
        selection_score_map = dict(refreshed_state[-1])

    if execution_pipeline_mode in {"dccs_refc", "voi"}:
        _refresh_certification_views()
        (
            initial_certificate_summary,
            initial_route_fragility_map,
            initial_competitor_fragility_breakdown,
            initial_value_of_refresh,
            initial_sampled_world_manifest,
        ) = _certification_artifact_payloads(
            selected_route_id=selected.id,
            frontier_route_ids=[option.id for option in strict_frontier],
            certification_frontier_route_ids=(
                certification_frontier_route_ids or [option.id for option in strict_frontier]
            ),
            certificate_snapshot=certificate_result,
            fragility_snapshot=fragility_result,
            world_manifest_snapshot=world_manifest_payload,
            active_family_names=list(active_families),
            forced_refreshed_family_names=sorted(forced_refreshed_families),
        )
        initial_certificate_artifacts = {
            "certificate_summary": initial_certificate_summary,
            "route_fragility_map": initial_route_fragility_map,
            "competitor_fragility_breakdown": initial_competitor_fragility_breakdown,
            "value_of_refresh": initial_value_of_refresh,
            "sampled_world_manifest": initial_sampled_world_manifest,
        }

    if execution_pipeline_mode == "voi":
        voi_started = _stage_started("voi")
        voi_config = VOIConfig(
            certificate_threshold=certificate_threshold,
            stop_threshold=tau_stop,
            search_budget=total_search_budget,
            evidence_budget=total_evidence_budget,
            max_iterations=max(1, total_search_budget + total_evidence_budget + 2),
            top_k_refine=min(2, max(1, total_search_budget)),
            resample_increment=world_increment,
            search_completeness_threshold=float(settings.route_pipeline_search_completeness_threshold),
            search_completeness_action_bonus=float(settings.route_pipeline_search_completeness_action_bonus),
        )
        certified_no_gain_streak = 0
        stochastic_enabled = bool(getattr(req.stochastic, "enabled", False))

        def _controller_selected_source_context() -> dict[str, Any]:
            selected_candidate_ids = option_candidate_ids.get(selected.id, [])
            selected_primary_candidate = (
                raw_candidate_by_id.get(selected_candidate_ids[0], {})
                if selected_candidate_ids
                else {}
            )
            selected_candidate_source_engine = (
                str(selected_primary_candidate.get("candidate_source_engine") or "").strip() or None
            )
            selected_candidate_source_stage = (
                str(selected_primary_candidate.get("candidate_source_stage") or "").strip() or None
            )
            selected_final_source = _selected_final_route_source_payload(
                options,
                selected_option_id=selected.id,
            )
            selected_final_route_source_engine = (
                str(selected_final_source.get("source_engine") or "").strip() or None
            )
            selected_final_route_source_stage = (
                str(selected_final_source.get("source_stage") or "").strip() or None
            )
            return {
                "selected_candidate_source_engine": selected_candidate_source_engine,
                "selected_candidate_source_stage": selected_candidate_source_stage,
                "selected_final_route_source_engine": selected_final_route_source_engine,
                "selected_final_route_source_stage": selected_final_route_source_stage,
            }

        def _update_certified_no_gain_streak(
            *,
            previous_selected_id: str,
            previous_certificate_value: float,
            previous_frontier_ids: set[str],
        ) -> None:
            nonlocal certified_no_gain_streak
            if certificate_result is None:
                certified_no_gain_streak = 0
                return
            current_certificate_value = float(certificate_result.certificate.get(selected.id, 0.0))
            current_frontier_ids = {option.id for option in strict_frontier}
            made_progress = (
                selected.id != previous_selected_id
                or current_frontier_ids != previous_frontier_ids
                or current_certificate_value > (previous_certificate_value + 1e-6)
            )
            if previous_certificate_value >= certificate_threshold and not made_progress:
                certified_no_gain_streak += 1
            else:
                certified_no_gain_streak = 0

        def _post_action_controller_state() -> VOIControllerState:
            frontier_payloads_post_action = [
                _route_option_dccs_payload(option, option_candidate_ids_map=option_candidate_ids)
                for option in strict_frontier
            ]
            post_remaining_search_budget = max(0, total_search_budget - search_used)
            post_remaining_evidence_budget = max(0, total_evidence_budget - evidence_used)
            post_action_certificate = (
                dict(certificate_result.certificate)
                if certificate_result is not None
                else dict(controller_state.certificate)
            )
            controller_selected_source_context = _controller_selected_source_context()
            return refresh_controller_state_after_action(
                controller_state,
                dccs=challenger_dccs,
                fragility=fragility_result,
                config=voi_config,
                frontier=frontier_payloads_post_action,
                certificate=post_action_certificate,
                winner_id=selected.id,
                selected_route_id=selected.id,
                remaining_search_budget=post_remaining_search_budget,
                remaining_evidence_budget=post_remaining_evidence_budget,
                active_evidence_families=list(active_families),
                refreshed_evidence_families=sorted(forced_refreshed_families),
                stochastic_enabled=stochastic_enabled,
                ambiguity_context={
                    **_request_ambiguity_context(req),
                    **controller_selected_source_context,
                    "graph_k_raw_cache_hit": bool(graph_search_meta.get("graph_k_raw_cache_hit", False)),
                    "graph_low_ambiguity_fast_path": bool(
                        graph_search_meta.get("graph_low_ambiguity_fast_path", False)
                    ),
                    "graph_supported_ambiguity_fast_fallback": bool(
                        graph_search_meta.get("graph_supported_ambiguity_fast_fallback", False)
                    ),
                    "supplemental_challenger_activated": bool(
                        diversity_rescue_state.get("supplemental_challenger_activated", False)
                    ),
                    "refc_world_count": int((world_manifest_payload or {}).get("world_count") or 0),
                    "refc_unique_world_count": int((world_manifest_payload or {}).get("unique_world_count") or 0),
                    "refc_effective_world_count": int((world_manifest_payload or {}).get("effective_world_count") or 0),
                    "refc_requested_world_count": int(current_world_count),
                    "refc_sampler_requested_world_count": int(
                        (world_manifest_payload or {}).get("sampler_requested_world_count") or 0
                    ),
                    "refc_world_count_shortfall": max(
                        0,
                        int(current_world_count) - int((world_manifest_payload or {}).get("world_count") or 0),
                    ),
                    "refc_world_reuse_rate": float((world_manifest_payload or {}).get("world_reuse_rate") or 0.0),
                    "refc_world_count_policy": str((world_manifest_payload or {}).get("world_count_policy") or ""),
                    "refc_hard_stress_pack_count": int((world_manifest_payload or {}).get("hard_case_stress_pack_count") or 0),
                    "refc_stress_world_fraction": float((world_manifest_payload or {}).get("stress_world_fraction") or 0.0),
                },
                action_trace=list(action_trace),
            )

        def _challenger_dccs_for_iteration(
            *,
            remaining_candidates: Sequence[Mapping[str, Any]],
            frontier_payloads: Sequence[Mapping[str, Any]],
            refined_payloads: Sequence[Mapping[str, Any]],
            remaining_search_budget: int,
        ) -> Any:
            cache_key = hashlib.sha1(
                json.dumps(
                    {
                        "remaining_candidate_ids": sorted(
                            str(candidate.get("candidate_id") or "")
                            for candidate in remaining_candidates
                        ),
                        "frontier": [
                            {
                                "route_id": str(route.get("route_id") or route.get("id") or ""),
                                "objective_vector": [
                                    round(float(value), 6)
                                    for value in _route_objective_vector(route)
                                ],
                            }
                            for route in frontier_payloads
                        ],
                        "refined_route_ids": sorted(
                            str(route.get("route_id") or route.get("id") or "")
                            for route in refined_payloads
                        ),
                        "remaining_search_budget": int(remaining_search_budget),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
            ).hexdigest()
            voi_dccs_runtime["last_cache_key"] = cache_key
            cached = voi_dccs_local_cache.get(cache_key)
            if cached is not None:
                voi_dccs_runtime["cache_hits"] = int(voi_dccs_runtime.get("cache_hits", 0)) + 1
                voi_dccs_runtime["cache_hits_local"] = int(voi_dccs_runtime.get("cache_hits_local", 0)) + 1
            else:
                cached = get_cached_voi_dccs(cache_key)
                if cached is not None:
                    voi_dccs_local_cache[cache_key] = cached
                    voi_dccs_runtime["cache_hits"] = int(voi_dccs_runtime.get("cache_hits", 0)) + 1
                    voi_dccs_runtime["cache_hits_global"] = int(voi_dccs_runtime.get("cache_hits_global", 0)) + 1
                else:
                    challenger_started = _stage_started("dccs")
                    cached = select_candidates(
                        remaining_candidates,
                        frontier=frontier_payloads,
                        refined=refined_payloads,
                        config=_dccs_runtime_config(
                            mode="challenger",
                            pipeline_variant=pipeline_mode,
                            search_budget=remaining_search_budget,
                        ),
                    )
                    _stage_finished("dccs", challenger_started)
                    voi_dccs_local_cache[cache_key] = cached
                    set_cached_voi_dccs(cache_key, cached)
                    voi_dccs_runtime["cache_misses"] = int(voi_dccs_runtime.get("cache_misses", 0)) + 1
            total_events = int(voi_dccs_runtime.get("cache_hits", 0)) + int(voi_dccs_runtime.get("cache_misses", 0))
            voi_dccs_runtime["reuse_rate"] = round(
                int(voi_dccs_runtime.get("cache_hits", 0)) / float(max(1, total_events)),
                6,
            )
            return cached

        while True:
            remaining_search_budget = max(0, total_search_budget - search_used)
            remaining_evidence_budget = max(0, total_evidence_budget - evidence_used)
            selected_cert_value = (
                float(certificate_result.certificate.get(selected.id, 0.0))
                if certificate_result is not None
                else 0.0
            )
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
            challenger_dccs = _challenger_dccs_for_iteration(
                remaining_candidates=remaining_candidates,
                frontier_payloads=frontier_payloads,
                refined_payloads=refined_payloads,
                remaining_search_budget=remaining_search_budget,
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
            certificate_margin_value = _certificate_margin_from_certificate(
                certificate_result.certificate,
                winner_id=selected.id,
            )
            controller_selected_source_context = _controller_selected_source_context()
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
                refreshed_evidence_families=sorted(forced_refreshed_families),
                stochastic_enabled=stochastic_enabled,
                ambiguity_context={
                    **_request_ambiguity_context(req),
                    **controller_selected_source_context,
                    "refc_world_count": int((world_manifest_payload or {}).get("world_count") or 0),
                    "refc_unique_world_count": int((world_manifest_payload or {}).get("unique_world_count") or 0),
                    "refc_effective_world_count": int((world_manifest_payload or {}).get("effective_world_count") or 0),
                    "refc_requested_world_count": int(current_world_count),
                    "refc_sampler_requested_world_count": int(
                        (world_manifest_payload or {}).get("sampler_requested_world_count") or 0
                    ),
                    "refc_world_count_shortfall": max(
                        0,
                        int(current_world_count) - int((world_manifest_payload or {}).get("world_count") or 0),
                    ),
                    "refc_world_reuse_rate": float((world_manifest_payload or {}).get("world_reuse_rate") or 0.0),
                    "refc_world_count_policy": str((world_manifest_payload or {}).get("world_count_policy") or ""),
                    "refc_hard_stress_pack_count": int((world_manifest_payload or {}).get("hard_case_stress_pack_count") or 0),
                    "refc_stress_world_fraction": float((world_manifest_payload or {}).get("stress_world_fraction") or 0.0),
                    "single_frontier_certificate_cap": (
                        float((world_manifest_payload or {}).get("single_frontier_certificate_cap"))
                        if (world_manifest_payload or {}).get("single_frontier_certificate_cap") is not None
                        else None
                    ),
                    "single_frontier_certificate_cap_applied": bool(
                        (world_manifest_payload or {}).get("single_frontier_certificate_cap_applied", False)
                    ),
                    "single_frontier_requires_full_stress": bool(
                        (world_manifest_payload or {}).get("single_frontier_requires_full_stress", False)
                    ),
                    "empirical_baseline_certificate": (
                        float((fragility_result.value_of_refresh or {}).get("empirical_baseline_certificate"))
                        if isinstance(fragility_result.value_of_refresh, Mapping)
                        and (fragility_result.value_of_refresh or {}).get("empirical_baseline_certificate") is not None
                        else None
                    ),
                    "controller_baseline_certificate": (
                        float((fragility_result.value_of_refresh or {}).get("controller_baseline_certificate"))
                        if isinstance(fragility_result.value_of_refresh, Mapping)
                        and (fragility_result.value_of_refresh or {}).get("controller_baseline_certificate") is not None
                        else None
                    ),
                },
                near_tie_mass=near_tie_mass,
                certificate_margin=certificate_margin_value,
            )
            search_metrics = compute_search_completeness_metrics(
                controller_state,
                dccs=challenger_dccs,
                config=voi_config,
            )
            search_completeness_score = float(search_metrics.get("search_completeness_score", 1.0))
            search_completeness_gap = float(search_metrics.get("search_completeness_gap", 0.0))
            controller_state = replace(
                controller_state,
                search_completeness_score=search_completeness_score,
                search_completeness_gap=search_completeness_gap,
                prior_support_strength=float(search_metrics.get("prior_support_strength", 0.0)),
                pending_challenger_mass=float(search_metrics.get("pending_challenger_mass", 0.0)),
                best_pending_flip_probability=float(search_metrics.get("best_pending_flip_probability", 0.0)),
                corridor_family_recall=float(search_metrics.get("corridor_family_recall", 1.0)),
                frontier_recall_at_budget=float(search_metrics.get("frontier_recall_at_budget", 1.0)),
            )
            controller_state = enrich_controller_state_for_actioning(
                controller_state,
                dccs=challenger_dccs,
                fragility=fragility_result,
                config=voi_config,
            )
            capped_initial_certificate, initial_controller_cap_applied = _initial_controller_overconfidence_cap(
                controller_state=controller_state,
                current_certificate=float(selected_cert_value),
                threshold=float(certificate_threshold),
            )
            if initial_controller_cap_applied:
                controller_certificate = dict(controller_state.certificate)
                controller_certificate[selected.id] = float(capped_initial_certificate)
                controller_context = dict(
                    controller_state.ambiguity_context
                    if isinstance(controller_state.ambiguity_context, Mapping)
                    else {}
                )
                empirical_baseline_certificate = float(selected_cert_value)
                prior_controller_baseline = controller_context.get("controller_baseline_certificate")
                if prior_controller_baseline is None:
                    controller_baseline_certificate = float(capped_initial_certificate)
                else:
                    controller_baseline_certificate = min(
                        float(prior_controller_baseline),
                        float(capped_initial_certificate),
                    )
                controller_context["empirical_baseline_certificate"] = empirical_baseline_certificate
                controller_context["controller_baseline_certificate"] = round(
                    float(controller_baseline_certificate),
                    6,
                )
                controller_context["controller_search_certificate_cap"] = round(
                    float(capped_initial_certificate),
                    6,
                )
                controller_context["controller_search_certificate_cap_applied"] = True
                controller_state = replace(
                    controller_state,
                    certificate=controller_certificate,
                    ambiguity_context=controller_context,
                )
                selected_cert_value = float(capped_initial_certificate)
            credible_search_uncertainty_flag = credible_search_uncertainty(
                controller_state,
                config=voi_config,
                current_certificate=float(selected_cert_value),
            )
            credible_evidence_uncertainty_flag = credible_evidence_uncertainty(
                controller_state,
                fragility=fragility_result,
                config=voi_config,
                current_certificate=float(selected_cert_value),
            )
            controller_state = replace(
                controller_state,
                credible_search_uncertainty=bool(credible_search_uncertainty_flag),
                credible_evidence_uncertainty=bool(credible_evidence_uncertainty_flag),
            )
            if not controller_state_rows:
                controller_state_rows.append(
                    {
                        **controller_state.as_dict(),
                        "frontier_route_ids": [option.id for option in strict_frontier],
                        "world_count": int(current_world_count),
                        "forced_refreshed_families": sorted(forced_refreshed_families),
                        "certified_no_gain_streak": certified_no_gain_streak,
                        "credible_search_uncertainty": credible_search_uncertainty_flag,
                        "credible_evidence_uncertainty": credible_evidence_uncertainty_flag,
                        "feasible_actions": [],
                    }
                )
            if (
                selected_cert_value >= certificate_threshold
                and (
                    (
                        search_completeness_score >= float(voi_config.search_completeness_threshold)
                        or not credible_search_uncertainty_flag
                    )
                    and not credible_evidence_uncertainty_flag
                    or (
                        certified_no_gain_streak >= 1
                        and not credible_search_uncertainty_flag
                        and not credible_evidence_uncertainty_flag
                    )
                )
            ):
                stop_reason = "certified"
                break
            if remaining_search_budget <= 0 and remaining_evidence_budget <= 0:
                stop_reason = "budget_exhausted"
                break
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
            if action_trace:
                controller_state_rows.append(
                    {
                        **controller_state.as_dict(),
                        "frontier_route_ids": [option.id for option in strict_frontier],
                        "world_count": int(current_world_count),
                        "forced_refreshed_families": sorted(forced_refreshed_families),
                        "certified_no_gain_streak": certified_no_gain_streak,
                        "credible_search_uncertainty": credible_search_uncertainty_flag,
                        "credible_evidence_uncertainty": credible_evidence_uncertainty_flag,
                        "feasible_actions": [action.as_dict() for action in actions],
                    }
                )
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
                stop_reason = (
                    "search_incomplete_no_action_worth_it"
                    if search_completeness_score is not None
                    and search_completeness_score < float(voi_config.search_completeness_threshold)
                    else "no_action_worth_it"
                )
                break
            chosen_action = feasible_actions[0]
            best_rejected_action_payload = feasible_actions[1].as_dict() if len(feasible_actions) > 1 else None
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
                stop_reason = (
                    "search_incomplete_no_action_worth_it"
                    if search_completeness_score is not None
                    and search_completeness_score < float(voi_config.search_completeness_threshold)
                    else "no_action_worth_it"
                )
                break

            trace_entry["chosen_action"] = chosen_action.as_dict()
            action_trace.append(trace_entry)

            if chosen_action.kind in {"refine_top1_dccs", "refine_topk_dccs"}:
                selected_records, selected_record_diag = _resolve_voi_refine_selected_records(
                    action_metadata=chosen_action.metadata,
                    current_records=[*challenger_dccs.selected, *challenger_dccs.skipped],
                    fallback_records=challenger_dccs.selected,
                )
                action_trace[-1].update(selected_record_diag)
                if not selected_records:
                    stop_reason = "no_action_worth_it"
                    break
                previous_frontier_ids = {option.id for option in strict_frontier}
                previous_selected_id = selected.id
                previous_certificate_value = float(selected_cert_value)
                previous_selected_score = float(selection_score_map.get(selected.id, 0.0))
                previous_selection_score_map = dict(selection_score_map)
                previous_controller_state = controller_state
                batch_routes, batch_warnings, batch_observed_costs, batch_fetches, batch_refine_ms = await _refine_graph_candidate_batch(
                    osrm=osrm,
                    origin=req.origin,
                    destination=req.destination,
                    selected_records=selected_records,
                    raw_graph_routes_by_id=raw_graph_routes_by_id,
                    vehicle_type=req.vehicle_type,
                    scenario_mode=req.scenario_mode,
                    cost_toggles=req.cost_toggles,
                    terrain_profile=req.terrain_profile,
                    departure_time_utc=req.departure_time_utc,
                    scenario_cache_token=scenario_cache_token,
                    route_cache_runtime=route_cache_runtime,
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
                evidence_base_options = list(options)
                if forced_refreshed_families:
                    _apply_committed_refresh_objectives()
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
                controller_state = _post_action_controller_state()
                controller_state = replace(
                    controller_state,
                    ambiguity_context={
                        **(
                            controller_state.ambiguity_context
                            if isinstance(controller_state.ambiguity_context, Mapping)
                            else {}
                        ),
                        "refc_requested_world_count": int(current_world_count),
                        "refc_world_count_shortfall": max(
                            0,
                            int(current_world_count) - int((world_manifest_payload or {}).get("world_count") or 0),
                        ),
                    },
                )
                controller_state = replace(
                    controller_state,
                    ambiguity_context={
                        **(
                            controller_state.ambiguity_context
                            if isinstance(controller_state.ambiguity_context, Mapping)
                            else {}
                        ),
                        "refc_requested_world_count": int(current_world_count),
                        "refc_world_count_shortfall": max(
                            0,
                            int(current_world_count) - int((world_manifest_payload or {}).get("world_count") or 0),
                        ),
                    },
                )
                _update_certified_no_gain_streak(
                    previous_selected_id=previous_selected_id,
                    previous_certificate_value=previous_certificate_value,
                    previous_frontier_ids=previous_frontier_ids,
                )
                current_selected_cert_value = (
                    float(certificate_result.certificate.get(selected.id, 0.0))
                    if certificate_result is not None
                    else float(selected_cert_value)
                )
                action_trace[-1]["realized_certificate_before"] = round(previous_certificate_value, 6)
                action_trace[-1]["realized_certificate_after"] = round(current_selected_cert_value, 6)
                action_trace[-1]["realized_certificate_delta"] = round(current_selected_cert_value - previous_certificate_value, 6)
                action_trace[-1]["realized_frontier_gain"] = int(len(strict_frontier) - len(previous_frontier_ids))
                action_trace[-1]["realized_selected_route_changed"] = bool(selected.id != previous_selected_id)
                action_trace[-1]["realized_selected_score_delta"] = round(
                    float(selection_score_map.get(selected.id, 0.0)) - previous_selected_score,
                    6,
                )
                action_trace[-1]["realized_runner_up_gap_before"] = round(
                    _runner_up_gap_from_score_map(previous_selection_score_map, selected_route_id=previous_selected_id),
                    6,
                )
                action_trace[-1]["realized_runner_up_gap_after"] = round(
                    _runner_up_gap_from_score_map(selection_score_map, selected_route_id=selected.id),
                    6,
                )
                action_trace[-1]["realized_runner_up_gap_delta"] = round(
                    action_trace[-1]["realized_runner_up_gap_after"] - action_trace[-1]["realized_runner_up_gap_before"],
                    6,
                )
                previous_evidence_uncertainty = round(
                    max(
                        0.0,
                        _as_float_or_zero(previous_controller_state.top_refresh_gain)
                        + _as_float_or_zero(previous_controller_state.top_fragility_mass),
                    ),
                    6,
                )
                current_evidence_uncertainty = round(
                    max(
                        0.0,
                        _as_float_or_zero(controller_state.top_refresh_gain)
                        + _as_float_or_zero(controller_state.top_fragility_mass),
                    ),
                    6,
                )
                action_trace[-1]["realized_evidence_uncertainty_before"] = previous_evidence_uncertainty
                action_trace[-1]["realized_evidence_uncertainty_after"] = current_evidence_uncertainty
                action_trace[-1]["realized_evidence_uncertainty_delta"] = round(
                    current_evidence_uncertainty - previous_evidence_uncertainty,
                    6,
                )
                action_trace[-1]["realized_productive"] = bool(
                    (current_selected_cert_value - previous_certificate_value) > 1e-9
                    or (len(strict_frontier) - len(previous_frontier_ids)) > 0
                    or selected.id != previous_selected_id
                    or (float(selection_score_map.get(selected.id, 0.0)) - previous_selected_score) < -1e-9
                    or action_trace[-1]["realized_runner_up_gap_delta"] > 1e-9
                    or action_trace[-1]["realized_evidence_uncertainty_delta"] < -1e-9
                )
                if (
                    current_selected_cert_value >= certificate_threshold
                    and certified_no_gain_streak >= 1
                    and not credible_evidence_uncertainty(
                        controller_state,
                        fragility=fragility_result,
                        config=voi_config,
                        current_certificate=float(current_selected_cert_value),
                    )
                ):
                    stop_reason = "certified"
                    break
                continue

            if chosen_action.kind == "refresh_top1_vor":
                previous_frontier_ids = {option.id for option in strict_frontier}
                previous_selected_id = selected.id
                previous_certificate_value = float(selected_cert_value)
                previous_selected_score = float(selection_score_map.get(selected.id, 0.0))
                previous_selection_score_map = dict(selection_score_map)
                previous_controller_state = controller_state
                forced_refreshed_families.add(chosen_action.target)
                evidence_used += max(1, int(chosen_action.cost_evidence))
                _apply_committed_refresh_objectives()
                _refresh_certification_views()
                controller_state = _post_action_controller_state()
                _update_certified_no_gain_streak(
                    previous_selected_id=previous_selected_id,
                    previous_certificate_value=previous_certificate_value,
                    previous_frontier_ids=previous_frontier_ids,
                )
                current_selected_cert_value = (
                    float(certificate_result.certificate.get(selected.id, 0.0))
                    if certificate_result is not None
                    else float(selected_cert_value)
                )
                action_trace[-1]["realized_certificate_before"] = round(previous_certificate_value, 6)
                action_trace[-1]["realized_certificate_after"] = round(current_selected_cert_value, 6)
                action_trace[-1]["realized_certificate_delta"] = round(current_selected_cert_value - previous_certificate_value, 6)
                action_trace[-1]["realized_frontier_gain"] = int(len(strict_frontier) - len(previous_frontier_ids))
                action_trace[-1]["realized_selected_route_changed"] = bool(selected.id != previous_selected_id)
                action_trace[-1]["realized_selected_score_delta"] = round(
                    float(selection_score_map.get(selected.id, 0.0)) - previous_selected_score,
                    6,
                )
                action_trace[-1]["realized_runner_up_gap_before"] = round(
                    _runner_up_gap_from_score_map(previous_selection_score_map, selected_route_id=previous_selected_id),
                    6,
                )
                action_trace[-1]["realized_runner_up_gap_after"] = round(
                    _runner_up_gap_from_score_map(selection_score_map, selected_route_id=selected.id),
                    6,
                )
                action_trace[-1]["realized_runner_up_gap_delta"] = round(
                    action_trace[-1]["realized_runner_up_gap_after"] - action_trace[-1]["realized_runner_up_gap_before"],
                    6,
                )
                previous_evidence_uncertainty = round(
                    max(
                        0.0,
                        _as_float_or_zero(previous_controller_state.top_refresh_gain)
                        + _as_float_or_zero(previous_controller_state.top_fragility_mass),
                    ),
                    6,
                )
                current_evidence_uncertainty = round(
                    max(
                        0.0,
                        _as_float_or_zero(controller_state.top_refresh_gain)
                        + _as_float_or_zero(controller_state.top_fragility_mass),
                    ),
                    6,
                )
                action_trace[-1]["realized_evidence_uncertainty_before"] = previous_evidence_uncertainty
                action_trace[-1]["realized_evidence_uncertainty_after"] = current_evidence_uncertainty
                action_trace[-1]["realized_evidence_uncertainty_delta"] = round(
                    current_evidence_uncertainty - previous_evidence_uncertainty,
                    6,
                )
                action_trace[-1]["realized_productive"] = bool(
                    (current_selected_cert_value - previous_certificate_value) > 1e-9
                    or (len(strict_frontier) - len(previous_frontier_ids)) > 0
                    or selected.id != previous_selected_id
                    or (float(selection_score_map.get(selected.id, 0.0)) - previous_selected_score) < -1e-9
                    or action_trace[-1]["realized_runner_up_gap_delta"] > 1e-9
                    or action_trace[-1]["realized_evidence_uncertainty_delta"] < -1e-9
                )
                if (
                    current_selected_cert_value >= certificate_threshold
                    and certified_no_gain_streak >= 1
                    and not credible_evidence_uncertainty(
                        controller_state,
                        fragility=fragility_result,
                        config=voi_config,
                        current_certificate=float(current_selected_cert_value),
                    )
                ):
                    stop_reason = "certified"
                    break
                continue

            if chosen_action.kind == "increase_stochastic_samples":
                bridge_resample = bool(
                    isinstance(chosen_action.metadata, Mapping)
                    and bool(chosen_action.metadata.get("evidence_discovery_bridge"))
                )
                if not stochastic_enabled and not bridge_resample:
                    stop_reason = "no_action_worth_it"
                    break
                previous_frontier_ids = {option.id for option in strict_frontier}
                previous_selected_id = selected.id
                previous_certificate_value = float(selected_cert_value)
                previous_selected_score = float(selection_score_map.get(selected.id, 0.0))
                previous_selection_score_map = dict(selection_score_map)
                previous_controller_state = controller_state
                if (
                    bridge_resample
                    and len(strict_frontier) <= 1
                    and bool((world_manifest_payload or {}).get("single_frontier_requires_full_stress", False))
                ):
                    force_single_frontier_full_stress_requested_worlds = True
                current_world_count += world_increment
                evidence_used += max(1, int(chosen_action.cost_evidence))
                _refresh_certification_views()
                controller_state = _post_action_controller_state()
                _update_certified_no_gain_streak(
                    previous_selected_id=previous_selected_id,
                    previous_certificate_value=previous_certificate_value,
                    previous_frontier_ids=previous_frontier_ids,
                )
                current_selected_cert_value = (
                    float(certificate_result.certificate.get(selected.id, 0.0))
                    if certificate_result is not None
                    else float(selected_cert_value)
                )
                action_trace[-1]["realized_certificate_before"] = round(previous_certificate_value, 6)
                action_trace[-1]["realized_certificate_after"] = round(current_selected_cert_value, 6)
                action_trace[-1]["realized_certificate_delta"] = round(current_selected_cert_value - previous_certificate_value, 6)
                action_trace[-1]["realized_frontier_gain"] = int(len(strict_frontier) - len(previous_frontier_ids))
                action_trace[-1]["realized_selected_route_changed"] = bool(selected.id != previous_selected_id)
                action_trace[-1]["realized_selected_score_delta"] = round(
                    float(selection_score_map.get(selected.id, 0.0)) - previous_selected_score,
                    6,
                )
                action_trace[-1]["realized_runner_up_gap_before"] = round(
                    _runner_up_gap_from_score_map(previous_selection_score_map, selected_route_id=previous_selected_id),
                    6,
                )
                action_trace[-1]["realized_runner_up_gap_after"] = round(
                    _runner_up_gap_from_score_map(selection_score_map, selected_route_id=selected.id),
                    6,
                )
                action_trace[-1]["realized_runner_up_gap_delta"] = round(
                    action_trace[-1]["realized_runner_up_gap_after"] - action_trace[-1]["realized_runner_up_gap_before"],
                    6,
                )
                previous_evidence_uncertainty = round(
                    max(
                        0.0,
                        _as_float_or_zero(previous_controller_state.top_refresh_gain)
                        + _as_float_or_zero(previous_controller_state.top_fragility_mass),
                    ),
                    6,
                )
                current_evidence_uncertainty = round(
                    max(
                        0.0,
                        _as_float_or_zero(controller_state.top_refresh_gain)
                        + _as_float_or_zero(controller_state.top_fragility_mass),
                    ),
                    6,
                )
                action_trace[-1]["realized_evidence_uncertainty_before"] = previous_evidence_uncertainty
                action_trace[-1]["realized_evidence_uncertainty_after"] = current_evidence_uncertainty
                action_trace[-1]["realized_evidence_uncertainty_delta"] = round(
                    current_evidence_uncertainty - previous_evidence_uncertainty,
                    6,
                )
                action_trace[-1]["realized_productive"] = bool(
                    (current_selected_cert_value - previous_certificate_value) > 1e-9
                    or (len(strict_frontier) - len(previous_frontier_ids)) > 0
                    or selected.id != previous_selected_id
                    or (float(selection_score_map.get(selected.id, 0.0)) - previous_selected_score) < -1e-9
                    or action_trace[-1]["realized_runner_up_gap_delta"] > 1e-9
                    or action_trace[-1]["realized_evidence_uncertainty_delta"] < -1e-9
                )
                if (
                    current_selected_cert_value >= certificate_threshold
                    and certified_no_gain_streak >= 1
                    and not credible_evidence_uncertainty(
                        controller_state,
                        fragility=fragility_result,
                        config=voi_config,
                        current_certificate=float(current_selected_cert_value),
                    )
                ):
                    stop_reason = "certified"
                    break
                continue

            stop_reason = "no_action_worth_it"
            break

        _stage_finished("voi", voi_started)
    elif execution_pipeline_mode in {"dccs", "dccs_refc"}:
        stop_reason = "single_pass"

    route_signature_map = _route_signature_map(options)
    final_selected_certificate = (
        float(certificate_result.certificate.get(selected.id, 0.0))
        if certificate_result is not None
        else None
    )
    if execution_pipeline_mode == "voi":
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
            certified=bool(stop_reason == "certified"),
            iteration_count=len(action_trace),
            search_budget_used=int(search_used),
            evidence_budget_used=int(evidence_used),
            stop_reason=stop_reason,
            best_rejected_action=best_rejected_action,
            best_rejected_q=best_rejected_q,
            search_completeness_score=search_completeness_score,
            search_completeness_gap=search_completeness_gap,
            credible_search_uncertainty=credible_search_uncertainty_flag,
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
        graph_retry_attempted=bool(graph_search_meta.get("graph_retry_attempted", False)),
        graph_retry_state_budget=int(graph_search_meta.get("graph_retry_state_budget", 0)),
        graph_retry_outcome=str(graph_search_meta.get("graph_retry_outcome", "not_applicable")),
        graph_rescue_attempted=bool(graph_search_meta.get("graph_rescue_attempted", False)),
        graph_rescue_mode=str(graph_search_meta.get("graph_rescue_mode", "not_applicable")),
        graph_rescue_state_budget=int(graph_search_meta.get("graph_rescue_state_budget", 0)),
        graph_rescue_outcome=str(graph_search_meta.get("graph_rescue_outcome", "not_applicable")),
        scenario_context_ms=float(stage_timings.get("scenario_context_ms", 0.0)),
        graph_search_ms_initial=float(
            graph_search_meta.get("graph_search_ms_initial", stage_timings.get("k_raw_ms", 0.0))
        ),
        graph_search_ms_retry=float(graph_search_meta.get("graph_search_ms_retry", 0.0)),
        graph_search_ms_rescue=float(graph_search_meta.get("graph_search_ms_rescue", 0.0)),
        graph_search_ms_supplemental=float(graph_search_meta.get("graph_search_ms_supplemental", 0.0)),
        graph_k_raw_cache_hit=bool(graph_search_meta.get("graph_k_raw_cache_hit", False)),
        graph_low_ambiguity_fast_path=bool(graph_search_meta.get("graph_low_ambiguity_fast_path", False)),
        graph_supported_ambiguity_fast_fallback=bool(
            graph_search_meta.get("graph_supported_ambiguity_fast_fallback", False)
        ),
        graph_long_corridor_stress_probe=bool(
            graph_search_meta.get("graph_long_corridor_stress_probe", False)
        ),
        osrm_refine_ms=float(stage_timings.get("osrm_refine_ms", 0.0)),
        build_options_ms=float(stage_timings.get("option_build_ms", 0.0)),
        refinement_policy=refinement_policy,
        selected_candidate_count=int(search_used),
        selected_candidate_ids_json=json.dumps(
            sorted(
                {
                    str(candidate_id)
                    for candidate_ids in option_candidate_ids.values()
                    for candidate_id in candidate_ids
                    if str(candidate_id).strip()
                }
            ),
            sort_keys=True,
        ),
        diversity_collapse_detected=bool(diversity_rescue_state.get("collapse_detected", False)),
        diversity_collapse_reason=str(diversity_rescue_state.get("collapse_reason", "") or ""),
        raw_corridor_family_count=int(
            diversity_rescue_state.get("raw_corridor_family_count", raw_candidate_corridor_count)
        ),
        refined_corridor_family_count=int(
            diversity_rescue_state.get(
                "refined_corridor_family_count_after",
                diversity_rescue_state.get("refined_corridor_family_count_before", 0),
            )
        ),
        supplemental_challenger_activated=bool(
            diversity_rescue_state.get("supplemental_challenger_activated", False)
        ),
        supplemental_candidate_count=int(diversity_rescue_state.get("supplemental_candidate_count", 0)),
        supplemental_selected_count=int(diversity_rescue_state.get("supplemental_selected_count", 0)),
        supplemental_budget_used=int(diversity_rescue_state.get("supplemental_budget_used", 0)),
        supplemental_sources_json=json.dumps(
            list(diversity_rescue_state.get("supplemental_sources", [])),
            sort_keys=True,
        ),
        leftover_challenger_activated=bool(leftover_challenger_state.get("activated", False)),
        leftover_challenger_candidate_count=int(leftover_challenger_state.get("candidate_count", 0)),
        leftover_challenger_selected_count=int(leftover_challenger_state.get("selected_count", 0)),
        leftover_challenger_budget_used=int(leftover_challenger_state.get("budget_used", 0)),
        preemptive_comparator_seed_activated=bool(preemptive_comparator_state.get("activated", False)),
        preemptive_comparator_candidate_count=int(preemptive_comparator_state.get("candidate_count", 0)),
        preemptive_comparator_source_count=len(list(preemptive_comparator_state.get("sources", []))),
        preemptive_comparator_sources_json=json.dumps(
            list(preemptive_comparator_state.get("sources", [])),
            sort_keys=True,
        ),
        option_build_cache_hits=int(option_build_runtime.get("cache_hits", 0)),
        option_build_cache_misses=int(option_build_runtime.get("cache_misses", 0)),
        option_build_rebuild_count=int(option_build_runtime.get("rebuild_count", 0)),
        option_build_reuse_rate=float(option_build_runtime.get("reuse_rate", 0.0)),
        **precheck_kwargs,
    )

    dccs_rows = []
    for candidate_id in sorted(candidate_records_by_id):
        row = candidate_records_by_id[candidate_id].as_dict()
        raw_candidate = raw_candidate_by_id.get(candidate_id, {})
        row["candidate_source_label"] = str(raw_candidate.get("candidate_source_label") or "").strip() or None
        row["candidate_source_engine"] = str(raw_candidate.get("candidate_source_engine") or "").strip() or None
        row["candidate_source_stage"] = str(raw_candidate.get("candidate_source_stage") or "").strip() or None
        row["supplemental_diversity_rescue"] = bool(
            raw_candidate.get("candidate_source_stage") == "supplemental_diversity_rescue"
        )
        row["leftover_budget_challenger"] = bool(candidate_id in leftover_challenger_selected_ids)
        row["preemptive_comparator_seed"] = bool(
            raw_candidate.get("candidate_source_stage") == "preemptive_comparator_seed"
        )
        dccs_rows.append(row)
    dccs_outcome_summary = summarize_refine_outcomes(list(candidate_records_by_id.values()))
    observed_refinement_count = int(dccs_outcome_summary.get("observed_refinement_count") or 0)
    predicted_selected_count = sum(
        max(0, int(batch.get("selected_count") or 0))
        for batch in dccs_batches
        if isinstance(batch, Mapping)
    )
    predicted_frontier_additions = sum(
        max(0, int(batch.get("predicted_frontier_additions") or batch.get("frontier_additions") or 0))
        for batch in dccs_batches
        if isinstance(batch, Mapping)
    )
    predicted_decision_flips = sum(
        max(0, int(batch.get("predicted_decision_flips") or batch.get("decision_flips") or 0))
        for batch in dccs_batches
        if isinstance(batch, Mapping)
    )
    predicted_unique_critical = sum(
        max(
            0,
            int(
                batch.get("unique_critical_predictions")
                or round(float(batch.get("dc_yield") or 0.0) * float(max(1, int(batch.get("selected_count") or 0))))
            ),
        )
        for batch in dccs_batches
        if isinstance(batch, Mapping)
    )
    predicted_challenger_hits = sum(
        max(
            0,
            int(
                round(
                    float(batch.get("predicted_challenger_hit_rate") or batch.get("challenger_hit_rate") or 0.0)
                    * float(max(1, int(batch.get("selected_count") or 0)))
                )
            ),
        )
        for batch in dccs_batches
        if isinstance(batch, Mapping)
    )
    predicted_dc_yield = round(
        predicted_unique_critical / float(max(1, predicted_selected_count)),
        6,
    )
    predicted_challenger_hit_rate = round(
        predicted_challenger_hits / float(max(1, predicted_selected_count)),
        6,
    )
    predicted_frontier_gain = round(
        predicted_frontier_additions / float(max(1, predicted_selected_count)),
        6,
    )
    def _bounded_ratio(value: Any | None) -> float | None:
        try:
            ratio = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(ratio):
            return None
        return round(max(0.0, min(1.0, float(ratio))), 6)

    dc_yield = _bounded_ratio(predicted_dc_yield)
    challenger_hit_rate = _bounded_ratio(predicted_challenger_hit_rate)
    frontier_gain_per_refinement = _bounded_ratio(predicted_frontier_gain)
    observed_dc_yield = _bounded_ratio(dccs_outcome_summary.get("observed_dc_yield"))
    observed_challenger_hit_rate = _bounded_ratio(dccs_outcome_summary.get("observed_challenger_hit_rate"))
    observed_frontier_gain_per_refinement = _bounded_ratio(
        dccs_outcome_summary.get("observed_frontier_gain_per_refinement")
    )
    refined_count = int(predicted_selected_count)
    decision_flips = int(dccs_outcome_summary.get("observed_decision_flips") or predicted_decision_flips)
    frontier_additions = int(dccs_outcome_summary.get("observed_frontier_additions") or predicted_frontier_additions)
    challenger_hits = int(
        round((challenger_hit_rate or 0.0) * float(max(1, refined_count)))
    )
    selected_candidate_ids = option_candidate_ids.get(selected.id, [])
    selected_primary_candidate = (
        raw_candidate_by_id.get(selected_candidate_ids[0], {})
        if selected_candidate_ids
        else {}
    )
    selected_source_label = str(selected_primary_candidate.get("candidate_source_label") or "").strip() or None
    selected_source_engine = str(selected_primary_candidate.get("candidate_source_engine") or "").strip() or None
    selected_source_stage = str(selected_primary_candidate.get("candidate_source_stage") or "").strip() or None
    selected_final_source = _selected_final_route_source_payload(
        options,
        selected_option_id=selected.id,
    )
    selected_final_source_label = str(selected_final_source.get("source_label") or "").strip() or None
    selected_final_source_engine = str(selected_final_source.get("source_engine") or "").strip() or None
    selected_final_source_stage = str(selected_final_source.get("source_stage") or "").strip() or None
    selected_from_supplemental_rescue = selected_source_stage == "supplemental_diversity_rescue"
    selected_from_preemptive_comparator_seed = selected_source_stage == "preemptive_comparator_seed"

    dccs_summary = {
        "pipeline_mode": pipeline_mode,
        "refinement_policy": refinement_policy,
        "search_budget_total": int(total_search_budget),
        "search_budget_used": int(search_used),
        "candidate_count_raw": len(raw_candidate_payloads),
        "refined_count": refined_count,
        "observed_refined_count": observed_refinement_count,
        "frontier_count": len(strict_frontier),
        "selected_route_id": selected.id,
        "selected_candidate_ids": selected_candidate_ids,
        "selected_candidate_source_label": selected_source_label,
        "selected_candidate_source_engine": selected_source_engine,
        "selected_candidate_source_stage": selected_source_stage,
        "selected_final_route_source_label": selected_final_source_label,
        "selected_final_route_source_engine": selected_final_source_engine,
        "selected_final_route_source_stage": selected_final_source_stage,
        "selected_from_supplemental_rescue": bool(selected_from_supplemental_rescue),
        "selected_from_preemptive_comparator_seed": bool(selected_from_preemptive_comparator_seed),
        "preemptive_comparator_seed_activated": bool(preemptive_comparator_state.get("activated", False)),
        "preemptive_comparator_candidate_count": int(preemptive_comparator_state.get("candidate_count", 0)),
        "preemptive_comparator_sources": list(preemptive_comparator_state.get("sources", [])),
        "preemptive_comparator_trigger_reason": str(preemptive_comparator_state.get("trigger_reason", "") or ""),
        "dc_yield": round(float(dc_yield or 0.0), 6),
        "challenger_hit_rate": round(float(challenger_hit_rate or 0.0), 6),
        "frontier_gain_per_refinement": round(float(frontier_gain_per_refinement or 0.0), 6),
        "observed_dc_yield": round(float(observed_dc_yield or 0.0), 6),
        "observed_challenger_hit_rate": round(float(observed_challenger_hit_rate or 0.0), 6),
        "observed_frontier_gain_per_refinement": round(float(observed_frontier_gain_per_refinement or 0.0), 6),
        "decision_flips": decision_flips,
        "frontier_additions": frontier_additions,
        "observed_metrics_available": bool(dccs_outcome_summary.get("observed_metrics_available")),
        "metric_stage": str(dccs_outcome_summary.get("metric_stage") or "pre_refinement_prediction"),
        "observed_refinement_count": observed_refinement_count,
        "predicted_selected_refinement_count": predicted_selected_count,
        "predicted_dc_yield": predicted_dc_yield,
        "predicted_challenger_hit_rate": predicted_challenger_hit_rate,
        "predicted_frontier_gain_per_refinement": predicted_frontier_gain,
        "predicted_decision_flips": predicted_decision_flips,
        "predicted_frontier_additions": predicted_frontier_additions,
        "candidate_fetches": int(candidate_fetches),
        "overlap_threshold": float(settings.route_dccs_overlap_threshold),
        "baseline_policy": str(settings.route_dccs_default_baseline_policy),
        "diversity_collapse_detected": bool(diversity_rescue_state.get("collapse_detected", False)),
        "diversity_collapse_reason": str(diversity_rescue_state.get("collapse_reason", "") or ""),
        "raw_corridor_family_count": int(
            diversity_rescue_state.get("raw_corridor_family_count", raw_candidate_corridor_count)
        ),
        "refined_corridor_family_count_before": int(
            diversity_rescue_state.get("refined_corridor_family_count_before", 0)
        ),
        "refined_corridor_family_count_after": int(
            diversity_rescue_state.get("refined_corridor_family_count_after", 0)
        ),
        "supplemental_challenger_activated": bool(
            diversity_rescue_state.get("supplemental_challenger_activated", False)
        ),
        "supplemental_candidate_count": int(diversity_rescue_state.get("supplemental_candidate_count", 0)),
        "supplemental_selected_count": int(diversity_rescue_state.get("supplemental_selected_count", 0)),
        "supplemental_budget_used": int(diversity_rescue_state.get("supplemental_budget_used", 0)),
        "supplemental_sources": list(diversity_rescue_state.get("supplemental_sources", [])),
        "leftover_challenger_activated": bool(leftover_challenger_state.get("activated", False)),
        "leftover_challenger_candidate_count": int(leftover_challenger_state.get("candidate_count", 0)),
        "leftover_challenger_selected_count": int(leftover_challenger_state.get("selected_count", 0)),
        "leftover_challenger_budget_used": int(leftover_challenger_state.get("budget_used", 0)),
        "batches": dccs_batches,
    }
    evidence_snapshot_manifest = _evidence_snapshot_manifest_payload(strict_frontier)
    evidence_snapshot_hash = str(evidence_snapshot_manifest.get("snapshot_hash") or "")

    def _top_fragility_families_for_route(route_id: str) -> list[str]:
        if fragility_result is None:
            return []
        raw = fragility_result.route_fragility_map.get(route_id, {})
        if not isinstance(raw, Mapping):
            return []
        ranked = sorted(
            (
                (str(family), _as_float_or_zero(value))
                for family, value in raw.items()
                if _as_float_or_zero(value) > 0.0
            ),
            key=lambda item: (-item[1], item[0]),
        )
        return [family for family, _ in ranked[:3]]

    top_value_of_refresh_family = None
    if fragility_result is not None and isinstance(fragility_result.value_of_refresh, Mapping):
        family_raw = str(fragility_result.value_of_refresh.get("top_refresh_family", "")).strip()
        top_value_of_refresh_family = family_raw or None

    def _top_competitor_route_id_for_route(route_id: str) -> str | None:
        if fragility_result is None:
            return None
        raw = fragility_result.competitor_fragility_breakdown.get(route_id, {})
        if not isinstance(raw, Mapping):
            return None
        ranked: list[tuple[str, float]] = []
        for competitor_id, family_counts in raw.items():
            if not isinstance(family_counts, Mapping):
                continue
            total = sum(_as_float_or_zero(value) for value in family_counts.values())
            if total > 0.0:
                ranked.append((str(competitor_id), total))
        if not ranked:
            return None
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked[0][0]

    option_by_id = {str(option.id): option for option in options if str(option.id).strip()}
    refined_route_rows = []
    for route in refined_routes:
        option_id = str(route.get("_built_option_id") or "").strip()
        if not option_id:
            continue
        option = option_by_id.get(option_id)
        if option is None:
            continue
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
            "certificate_threshold": (
                float(certificate_result.threshold)
                if certificate_result is not None
                else None
            ),
            "active_families": list(active_families),
            "evidence_snapshot_hash": evidence_snapshot_hash,
            "top_fragility_families": _top_fragility_families_for_route(option.id),
            "top_value_of_refresh_family": top_value_of_refresh_family,
            "top_competitor_route_id": _top_competitor_route_id_for_route(option.id),
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
        "selected_candidate_source_stage": selected_source_stage,
        "selected_candidate_source_engine": selected_source_engine,
        "ambiguity_context": _request_ambiguity_context(req),
    }

    (
        certificate_summary,
        route_fragility_map,
        competitor_fragility_breakdown,
        value_of_refresh,
        sampled_world_manifest,
    ) = _certification_artifact_payloads(
        selected_route_id=selected.id,
        frontier_route_ids=[option.id for option in strict_frontier],
        certification_frontier_route_ids=(
            certification_frontier_route_ids or [option.id for option in strict_frontier]
        ),
        certificate_snapshot=certificate_result,
        fragility_snapshot=fragility_result,
        world_manifest_snapshot=world_manifest_payload,
        active_family_names=list(active_families),
        forced_refreshed_family_names=sorted(forced_refreshed_families),
    )
    if initial_certificate_artifacts is None:
        initial_certificate_artifacts = {
            "certificate_summary": copy.deepcopy(certificate_summary),
            "route_fragility_map": copy.deepcopy(route_fragility_map),
            "competitor_fragility_breakdown": copy.deepcopy(competitor_fragility_breakdown),
            "value_of_refresh": copy.deepcopy(value_of_refresh),
            "sampled_world_manifest": copy.deepcopy(sampled_world_manifest),
        }
    initial_certificate_summary = copy.deepcopy(initial_certificate_artifacts["certificate_summary"])
    initial_route_fragility_map = copy.deepcopy(initial_certificate_artifacts["route_fragility_map"])
    initial_competitor_fragility_breakdown = copy.deepcopy(
        initial_certificate_artifacts["competitor_fragility_breakdown"]
    )
    initial_value_of_refresh = copy.deepcopy(initial_certificate_artifacts["value_of_refresh"])
    initial_sampled_world_manifest = copy.deepcopy(
        initial_certificate_artifacts["sampled_world_manifest"]
    )

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
            "certified": bool(stop_reason == "certified"),
            "search_budget_used": int(search_used),
            "search_budget_remaining": int(max(0, total_search_budget - search_used)),
            "evidence_budget_used": int(evidence_used),
            "evidence_budget_remaining": int(max(0, total_evidence_budget - evidence_used)),
            "search_completeness_score": search_completeness_score,
            "search_completeness_gap": search_completeness_gap,
            "credible_search_uncertainty": credible_search_uncertainty_flag,
            "credible_evidence_uncertainty": credible_evidence_uncertainty_flag,
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
        if execution_pipeline_mode == "voi"
        else {
            "pipeline_mode": pipeline_mode,
            "status": "not_requested",
            "selected_route_id": selected.id,
        }
    )
    if "refinement_ms" not in stage_timings and "osrm_refine_ms" in stage_timings:
        stage_timings["refinement_ms"] = round(float(stage_timings.get("osrm_refine_ms", 0.0)), 2)
    final_route_trace = {
        "pipeline_mode": pipeline_mode,
        "refinement_policy": refinement_policy,
        "run_seed": int(run_seed),
        "stage_events": stage_events,
        "stage_timings_ms": {key: round(float(value), 2) for key, value in stage_timings.items()},
        "certification_runtime": dict(certification_runtime),
        "route_cache_runtime": {
            "cache_hits": int(route_cache_runtime.get("cache_hits", 0)),
            "cache_misses": int(route_cache_runtime.get("cache_misses", 0)),
            "reuse_rate": float(route_cache_runtime.get("reuse_rate", 0.0)),
            "last_cache_key": route_cache_runtime.get("last_cache_key"),
        },
        "option_build_runtime": {
            "cache_hits": int(option_build_runtime.get("cache_hits", 0)),
            "cache_hits_local": int(option_build_runtime.get("cache_hits_local", 0)),
            "cache_hits_global": int(option_build_runtime.get("cache_hits_global", 0)),
            "cache_misses": int(option_build_runtime.get("cache_misses", 0)),
            "rebuild_count": int(option_build_runtime.get("rebuild_count", 0)),
            "refresh_rebuild_count": int(option_build_runtime.get("refresh_rebuild_count", 0)),
            "saved_ms_estimate": round(float(option_build_runtime.get("saved_ms_estimate", 0.0)), 6),
            "saved_pareto_ms_estimate": round(float(option_build_runtime.get("saved_pareto_ms_estimate", 0.0)), 6),
            "reuse_rate": float(option_build_runtime.get("reuse_rate", 0.0)),
            "last_cache_key": option_build_runtime.get("last_cache_key"),
        },
        "route_option_cache_runtime": {
            "cache_hits": int(route_option_cache_runtime.get("cache_hits", 0)),
            "cache_hits_local": int(route_option_cache_runtime.get("cache_hits_local", 0)),
            "cache_hits_global": int(route_option_cache_runtime.get("cache_hits_global", 0)),
            "cache_misses": int(route_option_cache_runtime.get("cache_misses", 0)),
            "cache_key_missing": int(route_option_cache_runtime.get("cache_key_missing", 0)),
            "cache_disabled": int(route_option_cache_runtime.get("cache_disabled", 0)),
            "cache_set_failures": int(route_option_cache_runtime.get("cache_set_failures", 0)),
            "saved_ms_estimate": round(float(route_option_cache_runtime.get("saved_ms_estimate", 0.0)), 6),
            "reuse_rate": float(route_option_cache_runtime.get("reuse_rate", 0.0)),
            "last_cache_key": route_option_cache_runtime.get("last_cache_key"),
        },
        "voi_dccs_runtime": {
            "cache_hits": int(voi_dccs_runtime.get("cache_hits", 0)),
            "cache_hits_local": int(voi_dccs_runtime.get("cache_hits_local", 0)),
            "cache_hits_global": int(voi_dccs_runtime.get("cache_hits_global", 0)),
            "cache_misses": int(voi_dccs_runtime.get("cache_misses", 0)),
            "reuse_rate": float(voi_dccs_runtime.get("reuse_rate", 0.0)),
            "last_cache_key": voi_dccs_runtime.get("last_cache_key"),
        },
        "ambiguity_context": _request_ambiguity_context(req),
        "resource_usage": _process_resource_snapshot(),
        "route_cache_stats": route_cache_stats(),
        "k_raw_cache_stats": k_raw_cache_stats(),
        "route_option_cache_stats": route_option_cache_stats(),
        "route_state_cache_stats": route_state_cache_stats(),
        "voi_dccs_cache_stats": voi_dccs_cache_stats(),
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
        "candidate_diagnostics": candidate_diag.__dict__,
        "diversity_rescue": {
            "reserved_slot": bool(diversity_rescue_state.get("reserved_slot", False)),
            "collapse_detected": bool(diversity_rescue_state.get("collapse_detected", False)),
            "collapse_reason": str(diversity_rescue_state.get("collapse_reason", "") or ""),
            "raw_corridor_family_count": int(
                diversity_rescue_state.get("raw_corridor_family_count", raw_candidate_corridor_count)
            ),
            "refined_corridor_family_count_before": int(
                diversity_rescue_state.get("refined_corridor_family_count_before", 0)
            ),
            "refined_corridor_family_count_after": int(
                diversity_rescue_state.get("refined_corridor_family_count_after", 0)
            ),
            "supplemental_challenger_activated": bool(
                diversity_rescue_state.get("supplemental_challenger_activated", False)
            ),
            "supplemental_sources": list(diversity_rescue_state.get("supplemental_sources", [])),
            "supplemental_candidate_count": int(diversity_rescue_state.get("supplemental_candidate_count", 0)),
            "supplemental_selected_count": int(diversity_rescue_state.get("supplemental_selected_count", 0)),
            "supplemental_budget_used": int(diversity_rescue_state.get("supplemental_budget_used", 0)),
        },
        "selected_route_id": selected.id,
        "selected_route_signature": route_signature_map.get(selected.id, ""),
        "selected_candidate_ids": selected_candidate_ids,
        "selected_candidate_source_label": selected_source_label,
        "selected_candidate_source_engine": selected_source_engine,
        "selected_candidate_source_stage": selected_source_stage,
        "selected_final_route_source_label": selected_final_source_label,
        "selected_final_route_source_engine": selected_final_source_engine,
        "selected_final_route_source_stage": selected_final_source_stage,
        "selected_from_supplemental_rescue": bool(selected_from_supplemental_rescue),
        "selected_certificate": (
            selected_certificate.model_dump(mode="json") if selected_certificate is not None else None
        ),
        "voi": {
            "stop_reason": stop_reason,
            "action_trace": action_trace,
            "best_rejected_action": best_rejected_action_payload,
            "controller_states_recorded": len(controller_state_rows),
            "search_completeness_score": search_completeness_score,
            "search_completeness_gap": search_completeness_gap,
            "credible_search_uncertainty": credible_search_uncertainty_flag,
            "credible_evidence_uncertainty": credible_evidence_uncertainty_flag,
        },
        "artifact_pointers": {
            "dccs_candidates": "dccs_candidates.jsonl",
            "dccs_summary": "dccs_summary.json",
            "refined_routes": "refined_routes.jsonl",
            "strict_frontier": "strict_frontier.jsonl",
            "winner_summary": "winner_summary.json",
            "certificate_summary": "certificate_summary.json",
            "initial_certificate_summary": "initial_certificate_summary.json",
            "route_fragility_map": "route_fragility_map.json",
            "initial_route_fragility_map": "initial_route_fragility_map.json",
            "competitor_fragility_breakdown": "competitor_fragility_breakdown.json",
            "initial_competitor_fragility_breakdown": "initial_competitor_fragility_breakdown.json",
            "value_of_refresh": "value_of_refresh.json",
            "initial_value_of_refresh": "initial_value_of_refresh.json",
            "sampled_world_manifest": "sampled_world_manifest.json",
            "initial_sampled_world_manifest": "initial_sampled_world_manifest.json",
            "evidence_snapshot_manifest": "evidence_snapshot_manifest.json",
            "voi_action_trace": "voi_action_trace.json",
            "voi_controller_state": "voi_controller_state.jsonl",
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
            "initial_certificate_summary.json": initial_certificate_summary,
            "route_fragility_map.json": route_fragility_map,
            "initial_route_fragility_map.json": initial_route_fragility_map,
            "competitor_fragility_breakdown.json": competitor_fragility_breakdown,
            "initial_competitor_fragility_breakdown.json": initial_competitor_fragility_breakdown,
            "value_of_refresh.json": value_of_refresh,
            "initial_value_of_refresh.json": initial_value_of_refresh,
            "sampled_world_manifest.json": sampled_world_manifest,
            "initial_sampled_world_manifest.json": initial_sampled_world_manifest,
            "evidence_snapshot_manifest.json": evidence_snapshot_manifest,
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
            "voi_controller_state.jsonl": controller_state_rows,
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
    ors: ORSDep,
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
    response_pipeline_mode = effective_pipeline_mode
    actual_pipeline_mode = "voi" if effective_pipeline_mode == "tri_source" else effective_pipeline_mode
    legacy_mode_warning: str | None = None
    if req.waypoints and effective_pipeline_mode != "legacy":
        actual_pipeline_mode = "legacy"
        response_pipeline_mode = "legacy"
        legacy_mode_warning = (
            f"{effective_pipeline_mode} pipeline currently supports single-leg OD requests only; "
            "falling back to legacy routing for waypoint requests."
        )
    run_seed = _resolve_pipeline_seed(req)
    log_event(
        "route_request_started",
        request_id=request_id,
        requested_max_alternatives=requested_alternatives,
        effective_max_alternatives=route_alternatives,
        waypoint_count=len(req.waypoints or []),
        pipeline_mode=response_pipeline_mode,
        execution_pipeline_mode=actual_pipeline_mode,
        refinement_policy=str(req.refinement_policy or ""),
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
        decision_package: DecisionPackage | None = None
        legacy_strict_frontier: list[RouteOption] | None = None
        route_cache_runtime: dict[str, Any] = {}
        route_option_cache_runtime: dict[str, Any] = {}
        collect_candidates_elapsed_ms = 0.0
        pareto_selection_elapsed_ms = 0.0
        evidence_validation_elapsed_ms = 0.0
        collect_started = time.perf_counter()
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
                        route_cache_runtime_out=route_cache_runtime,
                        route_option_cache_runtime_out=route_option_cache_runtime,
                        refinement_policy=str(req.refinement_policy or "").strip().lower(),
                        search_budget=int(req.search_budget or 0),
                        run_seed=int(run_seed),
                        od_ambiguity_index=req.od_ambiguity_index,
                        od_engine_disagreement_prior=req.od_engine_disagreement_prior,
                        od_hard_case_prior=req.od_hard_case_prior,
                        od_ambiguity_support_ratio=getattr(req, "od_ambiguity_support_ratio", None),
                        od_ambiguity_source_entropy=getattr(req, "od_ambiguity_source_entropy", None),
                        od_candidate_path_count=getattr(req, "od_candidate_path_count", None),
                        od_corridor_family_count=getattr(req, "od_corridor_family_count", None),
                        allow_supported_ambiguity_fast_fallback=True,
                    ),
                    timeout=timeout_s,
                )
                collect_candidates_elapsed_ms = (time.perf_counter() - collect_started) * 1000.0
            else:
                direct_result = await asyncio.wait_for(
                    _compute_direct_route_pipeline(
                        req=req,
                        osrm=osrm,
                        ors=ors,
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
                collect_candidates_elapsed_ms = (time.perf_counter() - collect_started) * 1000.0
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
        except ModelDataError as exc:
            reason_code, detail_message = _prefetch_failure_text(exc)
            _record_expected_calls_blocked(
                reason_code=reason_code,
                stage="collecting_candidates",
                detail=detail_message,
            )
            extra_detail = _prefetch_failure_detail_data(exc) or {}
            extra_detail.setdefault("stage", "collecting_candidates")
            raise HTTPException(
                status_code=422,
                detail=_strict_error_detail(
                    reason_code=reason_code,
                    message=detail_message,
                    warnings=[],
                    extra=extra_detail,
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
            pareto_started = time.perf_counter()
            legacy_strict_frontier = _strict_frontier_options(
                options,
                max_alternatives=route_alternatives,
                pareto_method=req.pareto_method,
                epsilon=req.epsilon,
                optimization_mode=req.optimization_mode,
                risk_aversion=req.risk_aversion,
            )
            if not legacy_strict_frontier:
                raise HTTPException(
                    status_code=422,
                    detail=_strict_error_detail(
                        reason_code="epsilon_infeasible",
                        message="No routes satisfy epsilon constraints for this request.",
                        warnings=warnings,
                    ),
                )
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
            pareto_selection_elapsed_ms = (time.perf_counter() - pareto_started) * 1000.0

            selected = _pick_best_option(
                legacy_strict_frontier,
                w_time=req.weights.time,
                w_money=req.weights.money,
                w_co2=req.weights.co2,
                optimization_mode=req.optimization_mode,
                risk_aversion=req.risk_aversion,
            )
            if all(option.id != selected.id for option in pareto_options):
                pareto_options = [
                    selected,
                    *[option for option in pareto_options if option.id != selected.id],
                ][:route_alternatives]
            selected_certificate = selected.certification

        evidence_validation_started = time.perf_counter()
        evidence_validation = _validate_route_options_evidence(pareto_options)
        evidence_validation_elapsed_ms = (time.perf_counter() - evidence_validation_started) * 1000.0
        if evidence_validation["status"] != "ok":
            first_issue = dict(evidence_validation["issues"][0] or {})
            reason_code = normalize_reason_code(str(first_issue.get("reason_code", "evidence_provenance_rejected")))
            _record_expected_calls_blocked(
                reason_code=reason_code,
                stage="finalizing_route",
                detail=str(first_issue.get("message", "Strict evidence validation failed.")),
            )
            raise HTTPException(
                status_code=422,
                detail=_strict_error_detail(
                    reason_code=reason_code,
                    message="Strict evidence policy rejected the route because its provenance was not live/snapshot-clean.",
                    warnings=warnings,
                    extra={
                        "strict_evidence_policy": {
                            "mode": "no_synthetic_no_proxy_no_fallback",
                            "allow_snapshot": True,
                        },
                        "evidence_validation": evidence_validation,
                    },
                ),
            )
        extra_json_artifacts = extra_json_artifacts or {}
        extra_jsonl_artifacts = extra_jsonl_artifacts or {}
        extra_csv_artifacts = extra_csv_artifacts or {}
        extra_text_artifacts = extra_text_artifacts or {}
        extra_json_artifacts["evidence_validation.json"] = evidence_validation
        if actual_pipeline_mode == "legacy":
            extra_jsonl_artifacts["strict_frontier.jsonl"] = _strict_frontier_rows_from_options(
                list(legacy_strict_frontier or pareto_options),
                selected_id=selected.id,
                evidence_snapshot_hash="",
            )
            extra_json_artifacts["final_route_trace.json"] = _legacy_final_route_trace(
                selected=selected,
                frontier_options=list(legacy_strict_frontier or pareto_options),
                candidate_fetches=candidate_fetches,
                candidate_diag=candidate_diag,
                run_seed=int(run_seed),
                stage_timings_ms={
                    "collecting_candidates": collect_candidates_elapsed_ms,
                    "pareto_selection": pareto_selection_elapsed_ms,
                    "evidence_validation": evidence_validation_elapsed_ms,
                    "total": (time.perf_counter() - t0) * 1000.0,
                },
                route_cache_runtime=route_cache_runtime,
                route_option_cache_runtime=route_option_cache_runtime,
            )
        if response_pipeline_mode != actual_pipeline_mode:
            _rewrite_public_pipeline_mode_in_artifacts(
                extra_json_artifacts=extra_json_artifacts,
                extra_jsonl_artifacts=extra_jsonl_artifacts,
                pipeline_mode=response_pipeline_mode,
            )
        decision_package = _build_route_decision_package(
            req=req,
            requested_pipeline_mode=response_pipeline_mode,
            actual_pipeline_mode=actual_pipeline_mode,
            selected=selected,
            candidates=pareto_options,
            warnings=warnings,
            selected_certificate=selected_certificate,
            voi_stop_summary=voi_stop_summary,
            evidence_validation=evidence_validation,
            extra_json_artifacts=extra_json_artifacts,
            extra_jsonl_artifacts=extra_jsonl_artifacts,
            extra_csv_artifacts=extra_csv_artifacts,
            extra_text_artifacts=extra_text_artifacts,
        )
        extra_json_artifacts["decision_package.json"] = decision_package.model_dump(mode="json")
        extra_json_artifacts["preference_summary.json"] = decision_package.preference_summary.model_dump(mode="json")
        extra_json_artifacts["support_summary.json"] = decision_package.support_summary.model_dump(mode="json")
        extra_json_artifacts["support_provenance.json"] = (
            selected.evidence_provenance.model_dump(mode="json")
            if selected.evidence_provenance is not None
            else {"selected_route_id": selected.id, "active_families": [], "families": []}
        )
        extra_json_artifacts["certified_set.json"] = decision_package.certified_set_summary.model_dump(mode="json")
        if decision_package.abstention_summary is not None:
            extra_json_artifacts["abstention_summary.json"] = decision_package.abstention_summary.model_dump(mode="json")
        if decision_package.witness_summary is not None:
            extra_json_artifacts["witness_summary.json"] = decision_package.witness_summary.model_dump(mode="json")
        if decision_package.controller_summary is not None:
            extra_json_artifacts["controller_summary.json"] = decision_package.controller_summary.model_dump(mode="json")
        if decision_package.theorem_hook_summary is not None:
            extra_json_artifacts["theorem_hook_map.json"] = decision_package.theorem_hook_summary.model_dump(mode="json")
        if decision_package.lane_manifest is not None:
            extra_json_artifacts["lane_manifest.json"] = decision_package.lane_manifest.model_dump(mode="json")
        extra_jsonl_artifacts["support_trace.jsonl"] = (
            [record.model_dump(mode="json") for record in selected.evidence_provenance.families]
            if selected.evidence_provenance is not None
            else []
        )
        certified_route_ids = set(decision_package.certified_set_summary.certified_route_ids)
        frontier_route_ids = set(decision_package.certified_set_summary.frontier_route_ids)
        extra_jsonl_artifacts["certified_set_routes.jsonl"] = [
            {
                "route_id": option.id,
                "selected": option.id == selected.id,
                "frontier": option.id in frontier_route_ids,
                "certified": option.id in certified_route_ids,
                "minimum_cost_route_id": decision_package.certified_set_summary.minimum_cost_route_id,
                "certificate_value": (
                    float(option.certification.certificate)
                    if option.certification is not None
                    else None
                ),
            }
            for option in pareto_options
        ]
        if decision_package.witness_summary is not None:
            witness_route_ids = set(decision_package.witness_summary.witness_route_ids)
            challenger_route_ids = set(decision_package.witness_summary.challenger_route_ids)
            extra_jsonl_artifacts["witness_routes.jsonl"] = [
                {
                    "route_id": option.id,
                    "selected": option.id == selected.id,
                    "witness": option.id in witness_route_ids,
                    "challenger": option.id in challenger_route_ids,
                }
                for option in pareto_options
            ]
        if decision_package.controller_summary is not None:
            extra_jsonl_artifacts["controller_trace.jsonl"] = list(
                extra_jsonl_artifacts.get("voi_controller_state.jsonl", [])
            )
        final_route_trace_payload = extra_json_artifacts.get("final_route_trace.json")
        if isinstance(final_route_trace_payload, dict):
            artifact_pointers = dict(final_route_trace_payload.get("artifact_pointers", {}) or {})
            artifact_pointers.update(
                {
                    "decision_package": "decision_package.json",
                    "preference_summary": "preference_summary.json",
                    "support_summary": "support_summary.json",
                    "support_provenance": "support_provenance.json",
                    "support_trace": "support_trace.jsonl",
                    "certified_set": "certified_set.json",
                    "certified_set_routes": "certified_set_routes.jsonl",
                    "controller_summary": "controller_summary.json",
                    "controller_trace": "controller_trace.jsonl",
                    "theorem_hook_map": "theorem_hook_map.json",
                    "lane_manifest": "lane_manifest.json",
                }
            )
            if "abstention_summary.json" in extra_json_artifacts:
                artifact_pointers["abstention_summary"] = "abstention_summary.json"
            if "witness_summary.json" in extra_json_artifacts:
                artifact_pointers["witness_summary"] = "witness_summary.json"
                artifact_pointers["witness_routes"] = "witness_routes.jsonl"
            final_route_trace_payload["artifact_pointers"] = artifact_pointers

        log_event(
            "route_request",
            request_id=request_id,
            pipeline_mode=response_pipeline_mode,
            execution_pipeline_mode=actual_pipeline_mode,
            run_seed=int(run_seed),
            refinement_policy=str(candidate_diag.refinement_policy or req.refinement_policy or ""),
            selected_candidate_count=int(candidate_diag.selected_candidate_count),
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
            pipeline_mode=response_pipeline_mode,
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
            pipeline_mode=response_pipeline_mode,  # type: ignore[arg-type]
            manifest_endpoint=str(route_run["manifest_endpoint"]),
            artifacts_endpoint=str(route_run["artifacts_endpoint"]),
            provenance_endpoint=str(route_run["provenance_endpoint"]),
            decision_package=decision_package,
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
    realism: bool = Query(default=True),
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
            apply_realism_multipliers=bool(realism),
        )
        compute_ms = round((time.perf_counter() - t0) * 1000, 2)
        notes = [
            "OSRM quick baseline route; strict live-source enrichments are intentionally bypassed.",
            "Use this as a fast comparator against strict smart-route output.",
        ]
        if realism:
            notes.append(
                "Configured baseline realism multipliers applied: "
                f"duration x{float(settings.route_baseline_duration_multiplier):.2f}, "
                f"distance x{float(settings.route_baseline_distance_multiplier):.2f}."
            )
        else:
            notes.append("Returned as a raw OSRM engine baseline with no thesis-time realism multipliers.")
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
            method="osrm_quick_baseline" if realism else "osrm_engine_baseline",
            compute_ms=compute_ms,
            provider_mode="repo_local",
            baseline_policy="quick_shortest_path",
            asset_freshness_status=str(getattr(settings, "live_source_policy", "repo_local_fresh")),
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
    ors: ORSDep,
    osrm: OSRMDep,
    _: UserAccessDep,
    realism: bool = Query(default=True),
    provider_mode_override: str | None = Query(default=None, alias="policy"),
) -> RouteBaselineResponse:
    request_id = str(uuid.uuid4())
    t0 = time.perf_counter()
    response.headers["x-route-request-id"] = request_id
    try:
        ors_duration_multiplier = max(1.0, float(settings.route_ors_baseline_duration_multiplier))
        ors_distance_multiplier = max(1.0, float(settings.route_ors_baseline_distance_multiplier))
        requested_ors_mode = str(provider_mode_override or settings.route_ors_baseline_mode or "local_service").strip().lower()
        if requested_ors_mode not in {"local_service", "repo_local"}:
            raise RuntimeError(f"Unsupported ORS baseline mode: {requested_ors_mode}")
        ors_mode = requested_ors_mode
        if ors_mode == "local_service":
            seed_route, baseline_meta = await _fetch_local_ors_baseline_seed(req=req, ors=ors)
        elif ors_mode == "repo_local":
            seed_route, baseline_meta = await _fetch_repo_local_ors_baseline_seed(req=req, osrm=osrm)
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
            apply_realism_multipliers=bool(realism),
        )
        compute_ms = round((time.perf_counter() - t0) * 1000, 2)
        provider_mode = str(baseline_meta.get("provider_mode") or ors_mode or "local_service")
        baseline_policy = str(baseline_meta.get("baseline_policy") or "engine_shortest_path")
        if provider_mode != requested_ors_mode:
            raise RuntimeError(
                f"ORS baseline provider mismatch: requested={requested_ors_mode} resolved={provider_mode}"
            )
        asset_manifest_hash = str(baseline_meta.get("asset_manifest_hash") or "").strip() or None
        asset_recorded_at = str(baseline_meta.get("asset_recorded_at") or "").strip() or None
        asset_freshness_status = (
            str(baseline_meta.get("asset_freshness_status") or "").strip()
            or ("self_hosted_engine_runtime" if provider_mode == "local_service" else "repo_local_fresh")
        )
        notes: list[str]
        if provider_mode == "local_service":
            engine_profile = str(baseline_meta.get("engine_profile") or "")
            engine_manifest = baseline_meta.get("engine_manifest") if isinstance(baseline_meta.get("engine_manifest"), dict) else {}
            identity_status = str(engine_manifest.get("identity_status") or asset_freshness_status or "").strip()
            compose_image = str(engine_manifest.get("compose_image") or "").strip()
            notes = [
                "Self-hosted openrouteservice baseline realized from the local engine, not a paid API.",
                f"ORS profile: {engine_profile or 'driving-car'}.",
                f"Local ORS graph identity status: {identity_status or 'unknown'}.",
            ]
            if compose_image:
                notes.append(f"ORS image: {compose_image}.")
            if realism:
                notes.append(
                    "Configured baseline realism multipliers applied: "
                    f"duration x{ors_duration_multiplier:.2f}, "
                    f"distance x{ors_distance_multiplier:.2f}."
                )
            else:
                notes.append("Returned as a raw openrouteservice engine baseline with no thesis-time realism multipliers.")
        else:
            notes = [
                "Repo-local secondary baseline realized through route-graph candidate selection and OSRM geometry.",
                "This comparator uses an intentionally different corridor-selection policy from the quick OSRM baseline.",
            ]
            if realism:
                notes.append(
                    "Configured baseline realism multipliers applied: "
                    f"duration x{ors_duration_multiplier:.2f}, "
                    f"distance x{ors_distance_multiplier:.2f}."
                )
            else:
                notes.append("Returned without realism multipliers.")
        if baseline_meta.get("degraded_long_corridor"):
            notes.append(
                "Long-corridor request used the bounded repo-local degrade policy "
                "and skipped graph-seeded challenger expansion to stay transport-safe."
            )
        if baseline_meta.get("selected_distinct_corridor"):
            notes.append("Secondary baseline selected a distinct graph corridor from the primary-ranked family.")
        elif baseline_meta.get("graph_candidate_count", 0):
            notes.append("Secondary baseline used the next-best available graph family because no distinct corridor was available.")
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
            provider_mode=provider_mode,
            baseline_policy=baseline_policy,
        )
        return RouteBaselineResponse(
            baseline=baseline,
            method="ors_local_engine_baseline" if provider_mode == "local_service" else "ors_repo_local_baseline",
            compute_ms=compute_ms,
            provider_mode=provider_mode,
            baseline_policy=baseline_policy,
            asset_manifest_hash=asset_manifest_hash,
            asset_recorded_at=asset_recorded_at,
            asset_freshness_status=asset_freshness_status,
            engine_manifest=dict(baseline_meta.get("engine_manifest") or {})
            if isinstance(baseline_meta.get("engine_manifest"), dict)
            else None,
            notes=notes,
        )
    except (RuntimeError, ORSError) as exc:
        message = str(exc).strip() or "Local ORS secondary baseline route is unavailable."
        reason_code = "baseline_route_unavailable"
        log_event(
            "route_ors_baseline_request_failed",
            request_id=request_id,
            reason_code=reason_code,
            detail_message=message,
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
        raise HTTPException(
            status_code=502,
            detail=_strict_error_detail(
                reason_code=reason_code,
                message=f"Local ORS secondary baseline route is unavailable. ({reason_code}) cause={message}",
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
        message = str(exc).strip() or "Unknown local ORS secondary baseline route failure."
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
                message=f"Local ORS secondary baseline route compute failed unexpectedly: {message}",
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
                od_ambiguity_index=req.od_ambiguity_index,
                od_engine_disagreement_prior=req.od_engine_disagreement_prior,
                od_hard_case_prior=req.od_hard_case_prior,
                od_ambiguity_support_ratio=getattr(req, "od_ambiguity_support_ratio", None),
                od_ambiguity_source_entropy=getattr(req, "od_ambiguity_source_entropy", None),
                od_candidate_path_count=getattr(req, "od_candidate_path_count", None),
                od_corridor_family_count=getattr(req, "od_corridor_family_count", None),
                allow_supported_ambiguity_fast_fallback=True,
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
                od_ambiguity_index=req.od_ambiguity_index,
                od_engine_disagreement_prior=req.od_engine_disagreement_prior,
                od_hard_case_prior=req.od_hard_case_prior,
                od_ambiguity_support_ratio=getattr(req, "od_ambiguity_support_ratio", None),
                od_ambiguity_source_entropy=getattr(req, "od_ambiguity_source_entropy", None),
                od_candidate_path_count=getattr(req, "od_candidate_path_count", None),
                od_corridor_family_count=getattr(req, "od_corridor_family_count", None),
                allow_supported_ambiguity_fast_fallback=True,
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
            od_ambiguity_index=req.od_ambiguity_index,
            od_engine_disagreement_prior=req.od_engine_disagreement_prior,
            od_hard_case_prior=req.od_hard_case_prior,
            od_ambiguity_support_ratio=getattr(req, "od_ambiguity_support_ratio", None),
            od_ambiguity_source_entropy=getattr(req, "od_ambiguity_source_entropy", None),
            od_candidate_path_count=getattr(req, "od_candidate_path_count", None),
            od_corridor_family_count=getattr(req, "od_corridor_family_count", None),
            allow_supported_ambiguity_fast_fallback=True,
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
                        od_ambiguity_index=req.od_ambiguity_index,
                        od_engine_disagreement_prior=req.od_engine_disagreement_prior,
                        od_hard_case_prior=req.od_hard_case_prior,
                        od_ambiguity_support_ratio=getattr(req, "od_ambiguity_support_ratio", None),
                        od_ambiguity_source_entropy=getattr(req, "od_ambiguity_source_entropy", None),
                        od_candidate_path_count=getattr(req, "od_candidate_path_count", None),
                        od_corridor_family_count=getattr(req, "od_corridor_family_count", None),
                        allow_supported_ambiguity_fast_fallback=True,
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
        artifact_names = [name for name in sorted(list(artifact_paths) + extra_artifacts)]
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


@app.get("/runs/{run_id}/artifacts/{artifact_name}")
async def get_artifact_generic(run_id: str, artifact_name: str) -> FileResponse:
    return await _get_artifact_file(run_id, artifact_name)


