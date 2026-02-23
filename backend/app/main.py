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
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from .calibration_loader import (
    load_fuel_consumption_calibration,
    load_risk_normalization_reference,
    load_stochastic_residual_prior,
    refresh_live_runtime_route_caches,
)
from .carbon_model import apply_scope_emissions_adjustment, resolve_carbon_price
from .departure_profile import time_of_day_multiplier_uk
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
    VehicleDeleteResponse,
    VehicleListResponse,
    VehicleMutationResponse,
    Waypoint,
    WeatherImpactConfig,
)
from .multileg_engine import compose_multileg_route_options
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
from .routing_graph import route_graph_candidate_routes, route_graph_status, route_graph_via_paths
from .routing_osrm import OSRMClient, OSRMError, extract_segment_annotations
from .run_store import (
    ARTIFACT_FILES,
    artifact_dir_for_run,
    artifact_paths_for_run,
    write_manifest,
    write_run_artifacts,
    write_scenario_manifest,
)
from .scenario import (
    ScenarioMode,
    ScenarioRouteContext,
    apply_scenario_duration,
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
from .weather_adapter import weather_incident_multiplier, weather_speed_multiplier, weather_summary

try:
    UK_TZ = ZoneInfo("Europe/London")
except ZoneInfoNotFoundError:
    UK_TZ = UTC


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.osrm = OSRMClient(base_url=settings.osrm_base_url, profile=settings.osrm_profile)
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
    scenario_candidate_family_count: int = 0
    scenario_candidate_jaccard_vs_baseline: float = 1.0
    scenario_candidate_jaccard_threshold: float = 1.0
    scenario_candidate_stress_score: float = 0.0
    scenario_candidate_gate_action: str = "not_applicable"
    scenario_edge_scaling_version: str = "v3_live_transform"


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


def _scenario_context_from_od(
    *,
    origin: LatLng,
    destination: LatLng,
    vehicle_class: str,
    departure_time_utc: datetime | None,
    weather_bucket: str,
    road_class_counts: dict[str, int] | None = None,
) -> ScenarioRouteContext:
    od_road_counts = road_class_counts
    if not od_road_counts:
        try:
            graph_ok, _graph_status = route_graph_status()
            if graph_ok:
                graph_routes, _graph_diag = route_graph_candidate_routes(
                    origin_lat=float(origin.lat),
                    origin_lon=float(origin.lon),
                    destination_lat=float(destination.lat),
                    destination_lon=float(destination.lon),
                    max_paths=5,
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
                            aggregate_counts[key] = aggregate_counts.get(key, 0.0) + (max(0.0, float(v)) * rank_weight)
                    if aggregate_counts:
                        od_road_counts = {
                            key: int(round(value))
                            for key, value in aggregate_counts.items()
                        }
        except Exception:
            od_road_counts = road_class_counts
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
) -> RouteOption:
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
    seg_d_m, seg_t_s = extract_segment_annotations(route)
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
    scenario_policy = resolve_scenario_profile(
        scenario_mode,
        context=scenario_context,
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
    duration_after_scenario_s = apply_scenario_duration(
        duration_after_tod_s,
        mode=scenario_mode,
        context=scenario_context,
    )
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
        segment_distances_m=[float(seg) for seg in seg_d_m],
    )
    non_terrain_multiplier = tod_multiplier * scenario_multiplier * weather_speed
    terrain_vehicle_params = params_for_vehicle(vehicle)
    raw_terrain_multipliers: list[float] = []
    for idx, (d_m_raw, t_s_raw) in enumerate(zip(seg_d_m, seg_t_s, strict=True)):
        d_m = max(float(d_m_raw), 0.0)
        t_s = max(float(t_s_raw), 0.0)
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
            seg_departure_utc = departure_time_utc + timedelta(seconds=max(0.0, elapsed_route_s))
            seg_dep = time_of_day_multiplier_uk(
                seg_departure_utc,
                route_points=route_points_lat_lon,
                road_class_counts=road_class_counts,
            )
            seg_duration_s *= min(1.35, max(0.75, seg_dep.multiplier / max(tod_multiplier, 1e-6)))
        seg_grade = segment_grades[idx] if idx < len(segment_grades) else 0.0
        seg_incident_delay_s = incident_delay_by_segment.get(idx, 0.0)
        seg_duration_s += seg_incident_delay_s
        elapsed_route_s += seg_duration_s
        seg_energy_kwh = 0.0
        seg_fuel_liters = 0.0
        energy_result = segment_energy_and_emissions(
            vehicle=vehicle,
            emissions_context=ctx,
            distance_km=seg_distance_km,
            duration_s=seg_duration_s,
            grade=seg_grade,
            fuel_price_multiplier=fuel_multiplier,
            departure_time_utc=departure_time_utc,
            route_centroid_lat=route_centroid_lat,
            route_centroid_lon=route_centroid_lon,
        )
        seg_fuel_cost = energy_result.fuel_cost_gbp * scenario_fuel_multiplier
        seg_fuel_liters_p10 = float(energy_result.fuel_liters_p10) * scenario_fuel_multiplier
        seg_fuel_liters_p50 = float(energy_result.fuel_liters_p50) * scenario_fuel_multiplier
        seg_fuel_liters_p90 = float(energy_result.fuel_liters_p90) * scenario_fuel_multiplier
        seg_fuel_cost_p10 = float(energy_result.fuel_cost_p10_gbp) * scenario_fuel_multiplier
        seg_fuel_cost_p50 = float(energy_result.fuel_cost_p50_gbp) * scenario_fuel_multiplier
        seg_fuel_cost_p90 = float(energy_result.fuel_cost_p90_gbp) * scenario_fuel_multiplier
        seg_fuel_cost_low = float(energy_result.fuel_cost_uncertainty_low_gbp) * scenario_fuel_multiplier
        seg_fuel_cost_high = float(energy_result.fuel_cost_uncertainty_high_gbp) * scenario_fuel_multiplier
        seg_energy_kwh = energy_result.energy_kwh * scenario_fuel_multiplier
        seg_emissions = energy_result.emissions_kg * scenario_fuel_multiplier
        seg_emissions_low = float(energy_result.emissions_uncertainty_low_kg) * scenario_fuel_multiplier
        seg_emissions_high = float(energy_result.emissions_uncertainty_high_kg) * scenario_fuel_multiplier
        seg_fuel_liters = energy_result.fuel_liters * scenario_fuel_multiplier
        if fuel_price_source is None and energy_result.price_source:
            fuel_price_source = energy_result.price_source
        if fuel_price_as_of is None and energy_result.price_as_of:
            fuel_price_as_of = energy_result.price_as_of
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
        if energy_result.emissions_kg > 1e-9:
            scope_scale = seg_emissions / max(
                1e-9, float(energy_result.emissions_kg) * scenario_fuel_multiplier
            )
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
        shifted_after_scenario = apply_scenario_duration(
            shifted_after_tod,
            mode=scenario_mode,
            context=scenario_context,
        )
        shifted_after_weather = shifted_after_scenario * weather_speed
        shifted_departure_duration = shifted_after_weather * gradient_duration_multiplier

    shifted_mode = _counterfactual_shift_scenario(scenario_mode)
    shifted_mode_policy = resolve_scenario_profile(shifted_mode, context=scenario_context)
    shifted_mode_duration = (
        apply_scenario_duration(
            duration_after_tod_s,
            mode=shifted_mode,
            context=scenario_context,
        )
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

    scored: list[tuple[float, str, dict[str, Any]]] = []
    for route in routes:
        sig = _route_signature(route)
        scored.append((_route_duration_s(route), sig, route))

    scored.sort(key=lambda item: (item[0], item[1]))
    return [route for _, _, route in scored[:max_routes]]


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
    if _has_warning_code(warnings, "scenario_profile_invalid"):
        return _strict_error_detail(
            reason_code="scenario_profile_invalid",
            message="Scenario profile data is invalid for strict routing policy.",
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
    for key in ("terrain_coverage_min_observed", "terrain_coverage_required", "terrain_dem_version"):
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
    elif ("terrain" in lowered and "unsupported" in lowered) or "terrain_region_unsupported" in lowered:
        reason_code = "terrain_region_unsupported"
    elif ("terrain" in lowered and "asset" in lowered) or "terrain_dem_asset_unavailable" in lowered:
        reason_code = "terrain_dem_asset_unavailable"
    elif "terrain" in lowered and ("coverage" in lowered or "dem" in lowered):
        reason_code = "terrain_dem_coverage_insufficient"
    elif "toll_tariff_unavailable" in lowered:
        reason_code = "toll_tariff_unavailable"
    elif "routing_graph_unavailable" in lowered:
        reason_code = "routing_graph_unavailable"
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

    t_min, t_max = min(times), max(times)
    m_min, m_max = min(moneys), max(moneys)
    e_min, e_max = min(emissions), max(emissions)

    def _norm(value: float, min_value: float, max_value: float) -> float:
        return 0.0 if max_value <= min_value else (value - min_value) / (max_value - min_value)

    best = options[0]
    best_tuple = (float("inf"), float("inf"), "")
    for option, time_v, money_v, co2_v in zip(options, times, moneys, emissions, strict=True):
        score = (wt * _norm(time_v, t_min, t_max)) + (wm * _norm(money_v, m_min, m_max)) + (
            we * _norm(co2_v, e_min, e_max)
        )
        tie_break = (
            score,
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
        objective_key=lambda option: _pareto_objective_key(
            option,
            optimization_mode=optimization_mode,
            risk_aversion=risk_aversion,
        ),
    )
    return _sort_options_by_mode(
        pareto,
        optimization_mode=optimization_mode,
        risk_aversion=risk_aversion,
    )


async def _collect_candidate_routes(
    *,
    osrm: OSRMClient,
    origin: LatLng,
    destination: LatLng,
    max_routes: int,
    cache_key: str | None = None,
    scenario_edge_modifiers: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[str], int, CandidateDiagnostics]:
    warnings: list[str] = []
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
            return routes, warnings, spec_count, CandidateDiagnostics(
                raw_count=int(diag.get("raw_count", len(routes))),
                deduped_count=int(diag.get("deduped_count", len(routes))),
                graph_explored_states=int(diag.get("graph_explored_states", 0)),
                graph_generated_paths=int(diag.get("graph_generated_paths", 0)),
                graph_emitted_paths=int(diag.get("graph_emitted_paths", 0)),
                candidate_budget=int(diag.get("candidate_budget", spec_count)),
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
            )

    graph_ok, graph_status = route_graph_status()
    if not graph_ok:
        return (
            [],
            [f"route_graph: routing_graph_unavailable (Route graph unavailable: {graph_status})"],
            0,
            CandidateDiagnostics(),
        )
    graph_routes, graph_diag = route_graph_candidate_routes(
        origin_lat=float(origin.lat),
        origin_lon=float(origin.lon),
        destination_lat=float(destination.lat),
        destination_lon=float(destination.lon),
        max_paths=max(4, max_routes * 2),
        scenario_edge_modifiers=scenario_edge_modifiers,
    )
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
    if graph_routes and not _is_neutral_scenario_modifiers(scenario_edge_modifiers):
        baseline_routes, _baseline_diag = route_graph_candidate_routes(
            origin_lat=float(origin.lat),
            origin_lon=float(origin.lon),
            destination_lat=float(destination.lat),
            destination_lon=float(destination.lon),
            max_paths=max(4, max_routes * 2),
            scenario_edge_modifiers=_neutral_scenario_edge_modifiers(),
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
                if bool(settings.route_graph_scenario_separability_fail):
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
                            scenario_candidate_family_count=scenario_family_count,
                            scenario_candidate_jaccard_vs_baseline=round(float(scenario_jaccard), 6),
                            scenario_candidate_jaccard_threshold=round(float(scenario_jaccard_threshold), 6),
                            scenario_candidate_stress_score=round(float(scenario_stress_score), 6),
                            scenario_candidate_gate_action=scenario_gate_action,
                            scenario_edge_scaling_version=scenario_scaling_version,
                        ),
                    )
                scenario_gate_action = "warned"
                warnings.append(f"route_graph: scenario_profile_invalid ({message})")
        else:
            scenario_jaccard_threshold = max(0.0, min(1.0, float(settings.route_graph_scenario_jaccard_max)))
            scenario_stress_score = _scenario_stress_score(scenario_edge_modifiers)
            scenario_gate_action = "not_stressed"
    if not graph_routes:
        return (
            [],
            ["route_graph: routing_graph_unavailable (Route graph produced no candidate paths)."],
            0,
            CandidateDiagnostics(
                raw_count=0,
                deduped_count=0,
                graph_explored_states=graph_diag.explored_states,
                graph_generated_paths=graph_diag.generated_paths,
                graph_emitted_paths=graph_diag.emitted_paths,
                candidate_budget=graph_diag.candidate_budget,
                scenario_candidate_family_count=scenario_family_count,
                scenario_candidate_jaccard_vs_baseline=round(float(scenario_jaccard), 6),
                scenario_candidate_jaccard_threshold=round(float(scenario_jaccard_threshold), 6),
                scenario_candidate_stress_score=round(float(scenario_stress_score), 6),
                scenario_candidate_gate_action=scenario_gate_action,
                scenario_edge_scaling_version=scenario_scaling_version,
            ),
        )

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

    routes_by_signature: dict[str, dict[str, Any]] = {}
    if family_specs:
        async for progress in _iter_candidate_fetches(
            osrm=osrm,
            origin=origin,
            destination=destination,
            specs=family_specs,
        ):
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
        return (
            [],
            [
                "route_graph: no_route_candidates (Graph families produced no OSRM-refined routes).",
                *warnings[:6],
            ],
            len(family_specs),
            CandidateDiagnostics(
                raw_count=len(graph_routes),
                deduped_count=0,
                graph_explored_states=graph_diag.explored_states,
                graph_generated_paths=graph_diag.generated_paths,
                graph_emitted_paths=graph_diag.emitted_paths,
                candidate_budget=graph_diag.candidate_budget,
            ),
        )

    ranked_routes = _select_ranked_candidate_routes(
        list(routes_by_signature.values()),
        max_routes=max_routes,
    )
    diag = CandidateDiagnostics(
        raw_count=len(graph_routes),
        deduped_count=len(routes_by_signature),
        graph_explored_states=graph_diag.explored_states,
        graph_generated_paths=graph_diag.generated_paths,
        graph_emitted_paths=graph_diag.emitted_paths,
        candidate_budget=graph_diag.candidate_budget,
        scenario_candidate_family_count=scenario_family_count,
        scenario_candidate_jaccard_vs_baseline=round(float(scenario_jaccard), 6),
        scenario_candidate_jaccard_threshold=round(float(scenario_jaccard_threshold), 6),
        scenario_candidate_stress_score=round(float(scenario_stress_score), 6),
        scenario_candidate_gate_action=scenario_gate_action,
        scenario_edge_scaling_version=scenario_scaling_version,
    )
    if cache_key:
        set_cached_routes(
            cache_key,
            (
                ranked_routes,
                warnings,
                len(family_specs),
                {
                    "raw_count": diag.raw_count,
                    "deduped_count": diag.deduped_count,
                    "graph_explored_states": diag.graph_explored_states,
                    "graph_generated_paths": diag.graph_generated_paths,
                    "graph_emitted_paths": diag.graph_emitted_paths,
                    "candidate_budget": diag.candidate_budget,
                    "scenario_candidate_family_count": diag.scenario_candidate_family_count,
                    "scenario_candidate_jaccard_vs_baseline": diag.scenario_candidate_jaccard_vs_baseline,
                    "scenario_candidate_jaccard_threshold": diag.scenario_candidate_jaccard_threshold,
                    "scenario_candidate_stress_score": diag.scenario_candidate_stress_score,
                    "scenario_candidate_gate_action": diag.scenario_candidate_gate_action,
                    "scenario_edge_scaling_version": diag.scenario_edge_scaling_version,
                },
            ),
        )
    return ranked_routes, warnings, len(family_specs), diag


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
) -> tuple[list[RouteOption], list[str], int, TerrainDiagnostics, CandidateDiagnostics]:
    refresh_live_runtime_route_caches()
    try:
        vehicle = resolve_vehicle_profile(vehicle_type)
        weather_cfg = weather or WeatherImpactConfig()
        base_candidate_context = _scenario_context_from_od(
            origin=origin,
            destination=destination,
            vehicle_class=str(vehicle.vehicle_class),
            departure_time_utc=departure_time_utc,
            weather_bucket=(weather_cfg.profile if weather_cfg.enabled else "clear"),
        )
        base_scenario_modifiers = _scenario_candidate_modifiers(
            scenario_mode=scenario_mode,
            context=base_candidate_context,
        )
        scenario_cache_token = hashlib.sha1(
            json.dumps(base_scenario_modifiers, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    except ModelDataError as exc:
        return (
            [],
            [f"route_0: {normalize_reason_code(exc.reason_code)} ({exc.message})"],
            0,
            TerrainDiagnostics(),
            CandidateDiagnostics(),
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
        )
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
        warnings.extend(build_warnings)
        return options, warnings, candidate_fetches, terrain_diag, candidate_diag

    async def _solve_leg(
        leg_index: int,
        leg_origin: LatLng,
        leg_destination: LatLng,
    ) -> tuple[list[RouteOption], list[str], int, TerrainDiagnostics, CandidateDiagnostics]:
        refresh_live_runtime_route_caches()
        try:
            leg_context = _scenario_context_from_od(
                origin=leg_origin,
                destination=leg_destination,
                vehicle_class=str(vehicle.vehicle_class),
                departure_time_utc=departure_time_utc,
                weather_bucket=(weather_cfg.profile if weather_cfg.enabled else "clear"),
            )
            leg_modifiers = _scenario_candidate_modifiers(
                scenario_mode=scenario_mode,
                context=leg_context,
            )
            leg_scenario_cache_token = hashlib.sha1(
                json.dumps(leg_modifiers, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        except ModelDataError as exc:
            return (
                [],
                [f"leg_{leg_index}: {normalize_reason_code(exc.reason_code)} ({exc.message})"],
                0,
                TerrainDiagnostics(),
                CandidateDiagnostics(),
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
        )
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
        leg_warnings.extend(leg_build_warnings)
        leg_pareto = _finalize_pareto_options(
            leg_options,
            max_alternatives=max_alternatives,
            pareto_method="dominance",
            epsilon=None,
            optimization_mode=optimization_mode,
            risk_aversion=risk_aversion,
        )
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
    terrain_diag = TerrainDiagnostics(
        fail_closed_count=fail_closed_count,
        unsupported_region_count=unsupported_region_count,
        asset_unavailable_count=asset_unavailable_count,
        dem_version="|".join(sorted(dem_versions)) if dem_versions else "unknown",
        coverage_min_observed=min(coverage_values) if coverage_values else 1.0,
    )
    candidate_diag = CandidateDiagnostics(
        raw_count=len(composed.options),
        deduped_count=len(composed.options),
        graph_explored_states=0,
        graph_generated_paths=0,
        graph_emitted_paths=0,
        candidate_budget=max(0, int(composed.candidate_fetches)),
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


async def _collect_route_options_with_diagnostics(**kwargs: Any) -> tuple[
    list[RouteOption], list[str], int, TerrainDiagnostics, CandidateDiagnostics
]:
    result = await _collect_route_options(**kwargs)
    return _normalize_collect_route_options_result(result)


@app.post("/pareto", response_model=ParetoResponse)
async def compute_pareto(req: ParetoRequest, osrm: OSRMDep, _: UserAccessDep) -> ParetoResponse:
    request_id = str(uuid.uuid4())
    t0 = time.perf_counter()
    has_error = False
    try:
        options, warnings, candidate_fetches, terrain_diag, candidate_diag = await _collect_route_options_with_diagnostics(
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
        )
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
    except HTTPException:
        has_error = True
        raise
    except Exception:
        has_error = True
        raise
    finally:
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
    if req.waypoints:
        async def stream_multileg() -> AsyncIterator[bytes]:
            stream_has_error = False
            try:
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
                    departure_time_utc=req.departure_time_utc,
                    pareto_method=req.pareto_method,
                    epsilon=req.epsilon,
                    optimization_mode=req.optimization_mode,
                    risk_aversion=req.risk_aversion,
                    utility_weights=(req.weights.time, req.weights.money, req.weights.co2),
                    option_prefix="route",
                )
                if not options:
                    strict_detail = _strict_failure_detail_from_outcome(
                        warnings=warnings,
                        terrain_diag=terrain_diag,
                        epsilon_requested=req.pareto_method == "epsilon_constraint" and req.epsilon is not None,
                        terrain_message="Terrain DEM coverage is insufficient for strict routing policy.",
                    )
                    if strict_detail is not None:
                        yield _ndjson_line(_stream_fatal_event_from_detail(strict_detail))
                    else:
                        message = "No route candidates could be computed."
                        if warnings:
                            message = f"{message} {warnings[0]}"
                        yield _ndjson_line(
                            {
                                "type": "fatal",
                                "reason_code": "no_route_candidates",
                                "message": message,
                                "warnings": [],
                            }
                        )
                    return

                routes = _finalize_pareto_options(
                    options,
                    max_alternatives=req.max_alternatives,
                    pareto_method=req.pareto_method,
                    epsilon=req.epsilon,
                    optimization_mode=req.optimization_mode,
                    risk_aversion=req.risk_aversion,
                )
                if (
                    not routes
                    and req.pareto_method == "epsilon_constraint"
                    and req.epsilon is not None
                ):
                    yield _ndjson_line(
                        _stream_fatal_event_from_detail(
                            _strict_error_detail(
                                reason_code="epsilon_infeasible",
                                message="No routes satisfy epsilon constraints for this request.",
                                warnings=warnings,
                            )
                        )
                    )
                    return
                total = len(routes)
                yield _ndjson_line({"type": "meta", "total": total})
                for idx, route in enumerate(routes, start=1):
                    yield _ndjson_line({"type": "route", "done": idx, "total": total, "route": route.model_dump(mode="json")})
                yield _ndjson_line(
                    {
                        "type": "done",
                        "done": total,
                        "total": total,
                        "routes": [route.model_dump(mode="json") for route in routes],
                        "warning_count": len(warnings),
                        "warnings": warnings,
                    }
                )
            except HTTPException as e:
                stream_has_error = True
                yield _ndjson_line(_stream_fatal_event_from_exception(e))
            except Exception as e:
                stream_has_error = True
                yield _ndjson_line(_stream_fatal_event_from_exception(e))
            finally:
                _record_endpoint_metric("pareto_stream", t0, error=stream_has_error)

        return StreamingResponse(
            stream_multileg(),
            media_type="application/x-ndjson",
            headers={"cache-control": "no-store"},
        )

    async def stream() -> AsyncIterator[bytes]:
        stream_has_error = False

        try:
            options, warnings, candidate_fetches, terrain_diag, _candidate_diag = await _collect_route_options_with_diagnostics(
                osrm=osrm,
                origin=req.origin,
                destination=req.destination,
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
                departure_time_utc=req.departure_time_utc,
                pareto_method=req.pareto_method,
                epsilon=req.epsilon,
                optimization_mode=req.optimization_mode,
                risk_aversion=req.risk_aversion,
                utility_weights=(req.weights.time, req.weights.money, req.weights.co2),
                option_prefix="route",
            )
            yield _ndjson_line({"type": "meta", "total": int(max(1, candidate_fetches))})
            if not options:
                strict_detail = _strict_failure_detail_from_outcome(
                    warnings=warnings,
                    terrain_diag=terrain_diag,
                    epsilon_requested=req.pareto_method == "epsilon_constraint" and req.epsilon is not None,
                    terrain_message="All candidates were removed by terrain DEM strict coverage policy.",
                )
                if strict_detail is not None:
                    stream_has_error = True
                    yield _ndjson_line(_stream_fatal_event_from_detail(strict_detail))
                    return
                stream_has_error = True
                message = "No route candidates could be computed."
                if warnings:
                    message = f"{message} {warnings[0]}"
                yield _ndjson_line(
                    {
                        "type": "fatal",
                        "reason_code": "no_route_candidates",
                        "message": message,
                        "warnings": [],
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
            if (
                not pareto_sorted
                and req.pareto_method == "epsilon_constraint"
                and req.epsilon is not None
            ):
                stream_has_error = True
                yield _ndjson_line(
                    _stream_fatal_event_from_detail(
                        _strict_error_detail(
                            reason_code="epsilon_infeasible",
                            message="No routes satisfy epsilon constraints for this request.",
                            warnings=warnings,
                        )
                    )
                )
                return

            total_routes = len(pareto_sorted)
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
            )

            yield _ndjson_line(
                {
                    "type": "done",
                    "done": total_routes,
                    "total": total_routes,
                    "routes": [route.model_dump(mode="json") for route in pareto_sorted],
                    "warning_count": len(warnings),
                    "warnings": warnings,
                }
            )
        except asyncio.CancelledError:
            log_event(
                "pareto_stream_cancelled",
                request_id=request_id,
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
            )
            raise
        except Exception as e:
            stream_has_error = True
            msg = str(e).strip() or type(e).__name__
            log_event(
                "pareto_stream_fatal",
                request_id=request_id,
                message=msg,
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
            )
            yield _ndjson_line(_stream_fatal_event_from_exception(e))
        finally:
            _record_endpoint_metric("pareto_stream", t0, error=stream_has_error)

    return StreamingResponse(
        stream(),
        media_type="application/x-ndjson",
        headers={"cache-control": "no-store"},
    )


@app.post("/route", response_model=RouteResponse)
async def compute_route(req: RouteRequest, osrm: OSRMDep, _: UserAccessDep) -> RouteResponse:
    request_id = str(uuid.uuid4())
    t0 = time.perf_counter()
    has_error = False
    try:
        options, warnings, candidate_fetches, terrain_diag, _candidate_diag = await _collect_route_options_with_diagnostics(
            osrm=osrm,
            origin=req.origin,
            destination=req.destination,
            waypoints=req.waypoints,
            max_alternatives=max(1, int(settings.route_candidate_alternatives_max)),
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
        )

        if not options:
            strict_detail = _strict_failure_detail_from_outcome(
                warnings=warnings,
                terrain_diag=terrain_diag,
                epsilon_requested=req.pareto_method == "epsilon_constraint" and req.epsilon is not None,
                terrain_message="Terrain DEM coverage is insufficient for strict routing policy.",
            )
            if strict_detail is not None:
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

        pareto_options = _finalize_pareto_options(
            options,
            max_alternatives=max(1, int(settings.route_candidate_alternatives_max)),
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

        log_event(
            "route_request",
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
            waypoint_count=len(req.waypoints),
            candidate_fetches=candidate_fetches,
            candidate_count=len(pareto_options),
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

        return RouteResponse(selected=selected, candidates=pareto_options)
    except HTTPException:
        has_error = True
        raise
    except Exception:
        has_error = True
        raise
    finally:
        _record_endpoint_metric("route", t0, error=has_error)


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
    if artifact_name not in ARTIFACT_FILES:
        raise HTTPException(status_code=404, detail="artifact not found")

    base = (Path(settings.out_dir) / "artifacts" / valid_run_id).resolve()
    path = (base / artifact_name).resolve()

    if not str(path).startswith(str(base)):
        raise HTTPException(status_code=400, detail="invalid artifact path")

    return path


@app.get("/runs/{run_id}/artifacts")
async def list_artifacts(run_id: str) -> dict[str, object]:
    t0 = time.perf_counter()
    has_error = False
    try:
        valid_run_id = _validated_run_id(run_id)
        paths = artifact_paths_for_run(valid_run_id)

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
        elif artifact_name.endswith(".geojson"):
            media_type = "application/geo+json"
        elif artifact_name.endswith(".csv"):
            media_type = "text/csv"
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

