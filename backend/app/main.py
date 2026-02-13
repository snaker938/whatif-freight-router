from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import json
import math
import random
import statistics
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from .experiment_store import (
    create_experiment,
    delete_experiment,
    get_experiment,
    list_experiments,
    update_experiment,
)
from .fallback_store import load_route_snapshot, save_route_snapshot
from .gradient import gradient_multipliers
from .incident_simulator import simulate_incident_events
from .logging_utils import log_event
from .metrics_store import metrics_snapshot, record_request
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
    ScenarioCompareRequest,
    ScenarioCompareResponse,
    ScenarioCompareResult,
    SignatureVerificationRequest,
    SignatureVerificationResponse,
    StochasticConfig,
    TerrainProfile,
    VehicleDeleteResponse,
    VehicleListResponse,
    VehicleMutationResponse,
    WeatherImpactConfig,
)
from .objectives_emissions import route_emissions_kg, speed_factor
from .objectives_selection import normalise_weights
from .oracle_quality_store import (
    append_check_record,
    checks_path,
    compute_dashboard_payload,
    load_check_records,
    write_summary_artifacts,
)
from .pareto_methods import select_pareto_routes
from .provenance_store import provenance_event, provenance_path_for_run, write_provenance
from .rbac import require_role
from .reporting import write_report_pdf
from .route_cache import clear_route_cache, get_cached_routes, route_cache_stats, set_cached_routes
from .routing_osrm import OSRMClient, OSRMError, extract_segment_annotations
from .run_store import (
    ARTIFACT_FILES,
    artifact_dir_for_run,
    artifact_paths_for_run,
    write_scenario_manifest,
    write_manifest,
    write_run_artifacts,
)
from .scenario import ScenarioMode, apply_scenario_duration, scenario_duration_multiplier
from .settings import settings
from .signatures import SIGNATURE_ALGORITHM, verify_payload_signature
from .time_of_day import time_of_day_multiplier
from .weather_adapter import weather_incident_multiplier, weather_speed_multiplier, weather_summary
from .vehicles import (
    VehicleProfile,
    all_vehicles,
    create_custom_vehicle,
    delete_custom_vehicle,
    get_vehicle,
    list_custom_vehicles,
    update_custom_vehicle,
)


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
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
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
        ingested_at_utc=datetime.now(timezone.utc).isoformat(),
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
        _, csv_path = write_summary_artifacts(dashboard)
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


@dataclass(frozen=True)
class CandidateFetchSpec:
    label: str
    alternatives: bool | int
    exclude: str | None = None
    via: tuple[float, float] | None = None


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


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(math.ceil(q * len(ordered))) - 1))
    return ordered[idx]


def _route_stochastic_uncertainty(
    option: RouteOption,
    *,
    stochastic: StochasticConfig,
) -> dict[str, float] | None:
    if not stochastic.enabled:
        return None

    base_duration_s = max(float(option.metrics.duration_s), 1e-6)
    base_monetary_cost = max(float(option.metrics.monetary_cost), 0.0)
    base_emissions_kg = max(float(option.metrics.emissions_kg), 0.0)
    sigma = max(0.0, min(float(stochastic.sigma), 0.5))
    samples = max(5, min(int(stochastic.samples), 200))

    seed_base = stochastic.seed if stochastic.seed is not None else 0
    seed_material = (
        f"{option.id}|{base_duration_s:.6f}|{base_monetary_cost:.6f}|"
        f"{base_emissions_kg:.6f}|{seed_base}"
    )
    route_seed = int(hashlib.sha1(seed_material.encode("utf-8")).hexdigest()[:12], 16)
    rng = random.Random(route_seed)

    sampled_durations: list[float] = []
    sampled_monetary: list[float] = []
    sampled_emissions: list[float] = []

    for _ in range(samples):
        factor = max(0.5, min(1.8, rng.gauss(1.0, sigma)))
        sampled_duration = base_duration_s * factor
        ratio = sampled_duration / base_duration_s if base_duration_s > 0 else 1.0
        sampled_durations.append(sampled_duration)
        sampled_monetary.append(base_monetary_cost * ratio)
        sampled_emissions.append(base_emissions_kg * ratio)

    mean_duration = statistics.fmean(sampled_durations)
    std_duration = statistics.pstdev(sampled_durations) if len(sampled_durations) > 1 else 0.0
    mean_monetary = statistics.fmean(sampled_monetary)
    std_monetary = statistics.pstdev(sampled_monetary) if len(sampled_monetary) > 1 else 0.0
    mean_emissions = statistics.fmean(sampled_emissions)
    std_emissions = statistics.pstdev(sampled_emissions) if len(sampled_emissions) > 1 else 0.0

    return {
        "mean_duration_s": round(mean_duration, 6),
        "std_duration_s": round(std_duration, 6),
        "p95_duration_s": round(_quantile(sampled_durations, 0.95), 6),
        "mean_monetary_cost": round(mean_monetary, 6),
        "std_monetary_cost": round(std_monetary, 6),
        "mean_emissions_kg": round(mean_emissions, 6),
        "std_emissions_kg": round(std_emissions, 6),
        "robust_score": round(mean_duration + std_duration, 6),
    }


def _ice_fuel_multiplier(fuel_type: str) -> float:
    if fuel_type == "petrol":
        return 1.08
    if fuel_type == "lng":
        return 0.90
    return 1.0


def _ice_euro_multiplier(euro_class: str) -> float:
    if euro_class == "euro4":
        return 1.12
    if euro_class == "euro5":
        return 1.06
    return 1.0


def _ice_temperature_multiplier(ambient_temp_c: float) -> float:
    if ambient_temp_c < 15.0:
        return 1.0 + min(0.25, (15.0 - ambient_temp_c) * 0.008)
    return 1.0 + min(0.12, (ambient_temp_c - 15.0) * 0.003)


def _ev_energy_temperature_multiplier(ambient_temp_c: float) -> float:
    return 1.0 + min(0.35, abs(ambient_temp_c - 20.0) * 0.01)


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
) -> RouteOption:
    vehicle = get_vehicle(vehicle_type)
    ctx = emissions_context or EmissionsContext()
    weather_cfg = weather or WeatherImpactConfig()
    incident_cfg = incident_simulation or IncidentSimulatorConfig()
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
    ice_context_multiplier = (
        _ice_fuel_multiplier(ctx.fuel_type)
        * _ice_euro_multiplier(ctx.euro_class)
        * _ice_temperature_multiplier(ctx.ambient_temp_c)
    )
    ev_temp_energy_multiplier = _ev_energy_temperature_multiplier(ctx.ambient_temp_c)

    coords = _downsample_coords(_validate_osrm_geometry(route))
    seg_d_m, seg_t_s = extract_segment_annotations(route)

    distance_m = float(route.get("distance", 0.0))
    duration_s = float(route.get("duration", 0.0))

    if distance_m <= 0 or duration_s <= 0:
        # If OSRM omits these top-level fields, compute them from segments.
        distance_m = sum(seg_d_m)
        duration_s = sum(seg_t_s)

    distance_km = distance_m / 1000.0
    base_duration_s = duration_s
    tod_multiplier = time_of_day_multiplier(departure_time_utc)
    scenario_multiplier = scenario_duration_multiplier(scenario_mode)
    duration_after_tod_s = base_duration_s * tod_multiplier
    duration_after_scenario_s = apply_scenario_duration(duration_after_tod_s, mode=scenario_mode)
    duration_after_weather_s = duration_after_scenario_s * weather_speed
    gradient_duration_multiplier, gradient_emissions_multiplier = gradient_multipliers(terrain_profile)
    duration_after_gradient_s = duration_after_weather_s * gradient_duration_multiplier
    duration_s = duration_after_gradient_s
    total_duration_multiplier = duration_s / base_duration_s if base_duration_s > 0 else 1.0
    pre_incident_segment_durations_s = [max(float(seg), 0.0) * total_duration_multiplier for seg in seg_t_s]
    route_key = option_id
    if incident_cfg.enabled:
        try:
            route_key = _route_signature(route)
        except OSRMError:
            route_key = option_id
    incident_events = simulate_incident_events(
        config=incident_cfg,
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
    carbon_price = max(cost_toggles.carbon_price_per_kg, 0.0)

    contains_toll = bool(route.get("contains_toll", False))
    toll_component_total = (
        distance_km * cost_toggles.toll_cost_per_km if cost_toggles.use_tolls and contains_toll else 0.0
    )

    segment_breakdown: list[dict[str, float | int]] = []
    total_emissions = 0.0
    total_energy_kwh = 0.0
    total_monetary = 0.0
    total_distance_km = 0.0
    total_duration_s = 0.0
    total_fuel_cost = 0.0
    total_time_cost = 0.0
    total_toll_cost = 0.0
    total_carbon_cost = 0.0

    for idx, (d_m, t_s) in enumerate(zip(seg_d_m, seg_t_s, strict=True)):
        d_m = max(float(d_m), 0.0)
        t_s = max(float(t_s), 0.0)

        seg_distance_km = d_m / 1000.0
        total_distance_km += seg_distance_km

        base_speed_kmh = (d_m / t_s) * 3.6 if t_s > 0 else 0.0
        base_seg_fuel_cost = seg_distance_km * vehicle.cost_per_km * speed_factor(base_speed_kmh)
        seg_fuel_cost = base_seg_fuel_cost * fuel_multiplier

        seg_duration_s = t_s * total_duration_multiplier
        seg_incident_delay_s = incident_delay_by_segment.get(idx, 0.0)
        seg_duration_s += seg_incident_delay_s
        seg_extra_delay_s = max(seg_duration_s - t_s, 0.0)
        seg_energy_kwh = 0.0
        if is_ev_mode:
            seg_energy_kwh = (seg_distance_km * ev_kwh_per_km * ev_temp_energy_multiplier) + (
                (seg_extra_delay_s / 3600.0) * 0.4
            )
            seg_emissions = seg_energy_kwh * grid_co2_kg_per_kwh
        else:
            base_seg_emissions = route_emissions_kg(
                vehicle=vehicle,
                segment_distances_m=[d_m],
                segment_durations_s=[t_s],
            )
            seg_emissions = (base_seg_emissions * gradient_emissions_multiplier * ice_context_multiplier) + (
                (seg_extra_delay_s / 3600.0) * vehicle.idle_emissions_kg_per_hour
            )

        toll_share = (seg_distance_km / distance_km) if distance_km > 0 else 0.0
        seg_toll_cost = toll_component_total * toll_share
        seg_time_cost = (seg_duration_s / 3600.0) * vehicle.cost_per_hour * DRIVER_TIME_COST_WEIGHT
        seg_carbon_cost = seg_emissions * carbon_price
        seg_monetary = seg_fuel_cost + seg_time_cost + seg_toll_cost + seg_carbon_cost

        seg_speed_kmh = seg_distance_km / (seg_duration_s / 3600.0) if seg_duration_s > 0 else 0.0
        segment_breakdown.append(
            {
                "segment_index": idx,
                "distance_km": round(seg_distance_km, 6),
                "duration_s": round(seg_duration_s, 6),
                "incident_delay_s": round(seg_incident_delay_s, 6),
                "avg_speed_kmh": round(seg_speed_kmh, 6),
                "emissions_kg": round(seg_emissions, 6),
                "monetary_cost": round(seg_monetary, 6),
            }
        )

        total_duration_s += seg_duration_s
        total_emissions += seg_emissions
        total_energy_kwh += seg_energy_kwh
        total_monetary += seg_monetary
        total_fuel_cost += seg_fuel_cost
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
    if is_ev_mode:
        eta_explanations.append(
            f"EV energy model active ({ev_kwh_per_km:.2f} kWh/km, grid {grid_co2_kg_per_kwh:.2f} kg/kWh)."
        )
    elif ctx.fuel_type != "diesel" or ctx.euro_class != "euro6" or abs(ctx.ambient_temp_c - 15.0) > 1e-6:
        eta_explanations.append(
            "Emissions context adjusted for "
            f"fuel={ctx.fuel_type}, euro={ctx.euro_class}, temp={ctx.ambient_temp_c:.1f}C."
        )

    eta_timeline: list[dict[str, float | str]] = [
        {"stage": "baseline", "duration_s": round(base_duration_s, 2), "delta_s": 0.0},
        {
            "stage": "time_of_day",
            "duration_s": round(duration_after_tod_s, 2),
            "delta_s": round(time_of_day_delta_s, 2),
            "multiplier": round(tod_multiplier, 3),
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
    eta_timeline.append(
        {
            "stage": "gradient",
            "duration_s": round(duration_after_gradient_s, 2),
            "delta_s": round(gradient_delta_s, 2),
            "multiplier": round(gradient_duration_multiplier, 3),
            "profile": terrain_profile,
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
        shifted_tod_multiplier = time_of_day_multiplier(departure_time_utc + timedelta(hours=2))
        shifted_after_tod = base_duration_s * shifted_tod_multiplier
        shifted_after_scenario = apply_scenario_duration(shifted_after_tod, mode=scenario_mode)
        shifted_after_weather = shifted_after_scenario * weather_speed
        shifted_departure_duration = shifted_after_weather * gradient_duration_multiplier

    shifted_mode = _counterfactual_shift_scenario(scenario_mode)
    shifted_mode_duration = (
        apply_scenario_duration(duration_after_tod_s, mode=shifted_mode)
        * weather_speed
        * gradient_duration_multiplier
    )
    duration_ratio = shifted_mode_duration / total_duration_s if total_duration_s > 0 else 1.0
    shifted_mode_emissions = total_emissions * duration_ratio
    shifted_mode_time_cost = total_time_cost * duration_ratio
    shifted_mode_monetary = (
        total_fuel_cost + shifted_mode_time_cost + total_toll_cost + total_carbon_cost
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

    option_weather_summary = weather_summary(weather_cfg) if weather_cfg.enabled else None
    if option_weather_summary is not None:
        option_weather_summary["weather_delay_s"] = round(weather_delta_s, 6)
        option_weather_summary["incident_rate_multiplier"] = round(weather_incident, 6)

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
        weather_summary=option_weather_summary,
        incident_events=incident_events,
    )
    option.uncertainty = _route_stochastic_uncertainty(
        option,
        stochastic=stochastic or StochasticConfig(),
    )
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
    vehicle_type: str,
    scenario_mode: ScenarioMode,
    max_routes: int,
    cost_toggles: CostToggles,
    terrain_profile: TerrainProfile = "flat",
    departure_time_utc: datetime | None = None,
) -> str:
    payload = {
        "origin": {"lat": round(origin.lat, 6), "lon": round(origin.lon, 6)},
        "destination": {"lat": round(destination.lat, 6), "lon": round(destination.lon, 6)},
        "vehicle_type": vehicle_type,
        "scenario_mode": scenario_mode.value,
        "max_routes": max_routes,
        "cost_toggles": cost_toggles.model_dump(mode="json"),
        "terrain_profile": terrain_profile,
        "departure_time_utc": departure_time_utc.isoformat() if departure_time_utc else None,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()


def _candidate_via_points(origin: LatLng, destination: LatLng) -> list[tuple[float, float]]:
    """Generate a few via points around the midpoint to force detours.

    This is a heuristic. If via falls off the network, OSRM may fail that candidate.
    """
    # Midpoint in lat/lon
    mid_lat = (origin.lat + destination.lat) / 2.0
    mid_lon = (origin.lon + destination.lon) / 2.0

    # Choose a modest offset based on span
    span = max(abs(destination.lat - origin.lat), abs(destination.lon - origin.lon))
    offset = max(0.08, min(span * 0.18, 0.6))

    candidates = [
        (mid_lat + offset, mid_lon),
        (mid_lat - offset, mid_lon),
        (mid_lat, mid_lon + offset),
        (mid_lat, mid_lon - offset),
    ]

    return candidates


def _candidate_fetch_specs(
    *,
    origin: LatLng,
    destination: LatLng,
    max_routes: int,
) -> list[CandidateFetchSpec]:
    max_routes = max(1, min(max_routes, 5))

    specs: list[CandidateFetchSpec] = [
        CandidateFetchSpec(label="alternatives", alternatives=max_routes),
    ]

    for ex in ("motorway", "trunk", "toll", "ferry"):
        specs.append(CandidateFetchSpec(label=f"exclude:{ex}", alternatives=False, exclude=ex))

    for idx, via in enumerate(_candidate_via_points(origin, destination), start=1):
        specs.append(CandidateFetchSpec(label=f"via:{idx}", alternatives=False, via=via))

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
                via=[spec.via] if spec.via else None,
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


def _select_ranked_candidate_routes(
    routes: list[dict[str, Any]],
    *,
    max_routes: int,
) -> list[dict[str, Any]]:
    max_routes = max(1, min(max_routes, 5))

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
    option_prefix: str,
) -> tuple[list[RouteOption], list[str]]:
    options: list[RouteOption] = []
    warnings: list[str] = []

    for route in routes:
        option_id = f"{option_prefix}_{len(options)}"
        try:
            options.append(
                build_option(
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
                )
            )
        except (OSRMError, ValueError) as e:
            warnings.append(f"{option_id}: {e}")

    return options, warnings


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
        std_key = "std_duration_s"
    elif objective == "money":
        deterministic = float(option.metrics.monetary_cost)
        mean_key = "mean_monetary_cost"
        std_key = "std_monetary_cost"
    else:
        deterministic = float(option.metrics.emissions_kg)
        mean_key = "mean_emissions_kg"
        std_key = "std_emissions_kg"

    if not option.uncertainty:
        return deterministic

    mean_value = float(option.uncertainty.get(mean_key, deterministic))
    std_value = max(0.0, float(option.uncertainty.get(std_key, 0.0)))
    if optimization_mode == "robust":
        return mean_value + (max(0.0, risk_aversion) * std_value)
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
    return sorted(
        options,
        key=lambda option: (
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
            option.id,
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
    pareto = select_pareto_routes(
        options,
        max_alternatives=max_alternatives,
        pareto_method=pareto_method,
        epsilon=epsilon,
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
) -> tuple[list[dict[str, Any]], list[str], int, bool]:
    if cache_key:
        cached = get_cached_routes(cache_key)
        if cached is not None:
            routes, warnings, spec_count = cached
            return routes, warnings, spec_count, False

    specs = _candidate_fetch_specs(origin=origin, destination=destination, max_routes=max_routes)
    warnings: list[str] = []
    unique_raw_routes: list[dict[str, Any]] = []
    seen_signatures: set[str] = set()

    async for progress in _iter_candidate_fetches(
        osrm=osrm,
        origin=origin,
        destination=destination,
        specs=specs,
    ):
        result = progress.result
        if result.error:
            warnings.append(f"{result.spec.label}: {result.error}")
            continue

        for route in result.routes:
            try:
                sig = _route_signature(route)
            except OSRMError as e:
                warnings.append(f"{result.spec.label}: skipped invalid route ({e})")
                continue
            if sig in seen_signatures:
                continue
            seen_signatures.add(sig)
            unique_raw_routes.append(route)

    ranked_routes = _select_ranked_candidate_routes(unique_raw_routes, max_routes=max_routes)
    if ranked_routes and cache_key:
        set_cached_routes(cache_key, (ranked_routes, warnings, len(specs)))
        save_route_snapshot(cache_key, ranked_routes)
        return ranked_routes, warnings, len(specs), False

    if settings.offline_fallback_enabled and cache_key:
        snapshot = load_route_snapshot(cache_key)
        if snapshot is not None:
            routes, updated_at = snapshot
            warnings.append(f"offline fallback used from snapshot at {updated_at}")
            return routes, warnings, len(specs), True

    out = (ranked_routes, warnings, len(specs), False)
    if cache_key:
        set_cached_routes(cache_key, (ranked_routes, warnings, len(specs)))
    return out


@app.post("/pareto", response_model=ParetoResponse)
async def compute_pareto(req: ParetoRequest, osrm: OSRMDep, _: UserAccessDep) -> ParetoResponse:
    request_id = str(uuid.uuid4())
    t0 = time.perf_counter()
    has_error = False
    try:
        cache_key = _candidate_cache_key(
            origin=req.origin,
            destination=req.destination,
            vehicle_type=req.vehicle_type,
            scenario_mode=req.scenario_mode,
            max_routes=req.max_alternatives,
            cost_toggles=req.cost_toggles,
            terrain_profile=req.terrain_profile,
            departure_time_utc=req.departure_time_utc,
        )
        routes, warnings, candidate_fetches, fallback_used = await _collect_candidate_routes(
            osrm=osrm,
            origin=req.origin,
            destination=req.destination,
            max_routes=req.max_alternatives,
            cache_key=cache_key,
        )

        options, build_warnings = _build_options(
            routes,
            vehicle_type=req.vehicle_type,
            scenario_mode=req.scenario_mode,
            cost_toggles=req.cost_toggles,
            terrain_profile=req.terrain_profile,
            stochastic=req.stochastic,
            emissions_context=req.emissions_context,
            weather=req.weather,
            incident_simulation=req.incident_simulation,
            departure_time_utc=req.departure_time_utc,
            option_prefix="route",
        )
        warnings.extend(build_warnings)

        if not options:
            detail = "No route candidates could be computed."
            if warnings:
                detail = f"{detail} {warnings[0]}"
            raise HTTPException(status_code=502, detail=detail)

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
            fallback_used=fallback_used,
            warning_count=len(warnings),
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
        )

        return ParetoResponse(routes=pareto_sorted)
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
    specs = _candidate_fetch_specs(
        origin=req.origin,
        destination=req.destination,
        max_routes=req.max_alternatives,
    )

    async def stream() -> AsyncIterator[bytes]:
        warnings: list[str] = []
        unique_raw_routes: list[dict[str, Any]] = []
        seen_signatures: set[str] = set()
        streamed_option_count = 0
        stream_has_error = False

        try:
            yield _ndjson_line({"type": "meta", "total": len(specs)})

            async for progress in _iter_candidate_fetches(
                osrm=osrm,
                origin=req.origin,
                destination=req.destination,
                specs=specs,
            ):
                if await request.is_disconnected():
                    raise asyncio.CancelledError

                result = progress.result
                if result.error:
                    msg = f"{result.spec.label}: {result.error}"
                    warnings.append(msg)
                    yield _ndjson_line(
                        {
                            "type": "error",
                            "done": progress.done,
                            "total": progress.total,
                            "message": msg,
                        }
                    )
                    continue

                for route in result.routes:
                    try:
                        sig = _route_signature(route)
                    except OSRMError as e:
                        warnings.append(f"{result.spec.label}: skipped invalid route ({e})")
                        continue

                    if sig in seen_signatures:
                        continue

                    seen_signatures.add(sig)
                    unique_raw_routes.append(route)

                    option_id = f"route_{streamed_option_count}"
                    streamed_option_count += 1
                    try:
                        option = build_option(
                            route,
                            option_id=option_id,
                            vehicle_type=req.vehicle_type,
                            scenario_mode=req.scenario_mode,
                            cost_toggles=req.cost_toggles,
                            terrain_profile=req.terrain_profile,
                            stochastic=req.stochastic,
                            emissions_context=req.emissions_context,
                            weather=req.weather,
                            incident_simulation=req.incident_simulation,
                            departure_time_utc=req.departure_time_utc,
                        )
                    except (OSRMError, ValueError) as e:
                        warnings.append(f"{option_id}: {e}")
                        continue

                    yield _ndjson_line(
                        {
                            "type": "route",
                            "done": progress.done,
                            "total": progress.total,
                            "route": option.model_dump(mode="json"),
                        }
                    )

            ranked_routes = _select_ranked_candidate_routes(
                unique_raw_routes,
                max_routes=req.max_alternatives,
            )
            options, build_warnings = _build_options(
                ranked_routes,
                vehicle_type=req.vehicle_type,
                scenario_mode=req.scenario_mode,
                cost_toggles=req.cost_toggles,
                terrain_profile=req.terrain_profile,
                stochastic=req.stochastic,
                emissions_context=req.emissions_context,
                weather=req.weather,
                incident_simulation=req.incident_simulation,
                departure_time_utc=req.departure_time_utc,
                option_prefix="route",
            )
            warnings.extend(build_warnings)
            pareto_sorted = _finalize_pareto_options(
                options,
                max_alternatives=req.max_alternatives,
                pareto_method=req.pareto_method,
                epsilon=req.epsilon,
                optimization_mode=req.optimization_mode,
                risk_aversion=req.risk_aversion,
            )

            log_event(
                "pareto_stream_request",
                request_id=request_id,
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
                origin=req.origin.model_dump(),
                destination=req.destination.model_dump(),
                candidate_fetches=len(specs),
                candidate_count=len(options),
                pareto_count=len(pareto_sorted),
                warning_count=len(warnings),
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

            yield _ndjson_line(
                {
                    "type": "done",
                    "done": len(specs),
                    "total": len(specs),
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
            yield _ndjson_line({"type": "fatal", "message": msg})
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
        cache_key = _candidate_cache_key(
            origin=req.origin,
            destination=req.destination,
            vehicle_type=req.vehicle_type,
            scenario_mode=req.scenario_mode,
            max_routes=5,
            cost_toggles=req.cost_toggles,
            terrain_profile=req.terrain_profile,
            departure_time_utc=req.departure_time_utc,
        )
        routes, warnings, candidate_fetches, fallback_used = await _collect_candidate_routes(
            osrm=osrm,
            origin=req.origin,
            destination=req.destination,
            max_routes=5,
            cache_key=cache_key,
        )

        options, build_warnings = _build_options(
            routes,
            vehicle_type=req.vehicle_type,
            scenario_mode=req.scenario_mode,
            cost_toggles=req.cost_toggles,
            terrain_profile=req.terrain_profile,
            stochastic=req.stochastic,
            emissions_context=req.emissions_context,
            weather=req.weather,
            incident_simulation=req.incident_simulation,
            departure_time_utc=req.departure_time_utc,
            option_prefix="route",
        )
        warnings.extend(build_warnings)

        if not options:
            detail = "No route candidates could be computed."
            if warnings:
                detail = f"{detail} {warnings[0]}"
            raise HTTPException(status_code=502, detail=detail)

        pareto_options = _finalize_pareto_options(
            options,
            max_alternatives=5,
            pareto_method=req.pareto_method,
            epsilon=req.epsilon,
            optimization_mode=req.optimization_mode,
            risk_aversion=req.risk_aversion,
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
            candidate_fetches=candidate_fetches,
            candidate_count=len(pareto_options),
            fallback_used=fallback_used,
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
            start_utc = start_utc.replace(tzinfo=timezone.utc)
        else:
            start_utc = start_utc.astimezone(timezone.utc)
        if end_utc.tzinfo is None:
            end_utc = end_utc.replace(tzinfo=timezone.utc)
        else:
            end_utc = end_utc.astimezone(timezone.utc)

        if end_utc <= start_utc:
            raise HTTPException(status_code=422, detail="window_end_utc must be after window_start_utc")

        earliest_arrival_utc: datetime | None = None
        latest_arrival_utc: datetime | None = None
        if req.time_window is not None:
            earliest_arrival_utc = req.time_window.earliest_arrival_utc
            latest_arrival_utc = req.time_window.latest_arrival_utc
            if earliest_arrival_utc is not None:
                if earliest_arrival_utc.tzinfo is None:
                    earliest_arrival_utc = earliest_arrival_utc.replace(tzinfo=timezone.utc)
                else:
                    earliest_arrival_utc = earliest_arrival_utc.astimezone(timezone.utc)
            if latest_arrival_utc is not None:
                if latest_arrival_utc.tzinfo is None:
                    latest_arrival_utc = latest_arrival_utc.replace(tzinfo=timezone.utc)
                else:
                    latest_arrival_utc = latest_arrival_utc.astimezone(timezone.utc)
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
        for departure_time in departure_times:
            cache_key = _candidate_cache_key(
                origin=req.origin,
                destination=req.destination,
                vehicle_type=req.vehicle_type,
                scenario_mode=req.scenario_mode,
                max_routes=req.max_alternatives,
                cost_toggles=req.cost_toggles,
                terrain_profile=req.terrain_profile,
                departure_time_utc=departure_time,
            )
            routes, warnings, _candidate_fetches, fallback_used = await _collect_candidate_routes(
                osrm=osrm,
                origin=req.origin,
                destination=req.destination,
                max_routes=req.max_alternatives,
                cache_key=cache_key,
            )
            options, build_warnings = _build_options(
                routes,
                vehicle_type=req.vehicle_type,
                scenario_mode=req.scenario_mode,
                cost_toggles=req.cost_toggles,
                terrain_profile=req.terrain_profile,
                stochastic=req.stochastic,
                emissions_context=req.emissions_context,
                weather=req.weather,
                incident_simulation=req.incident_simulation,
                departure_time_utc=departure_time,
                option_prefix=f"departure_{departure_time.strftime('%H%M')}",
            )
            warnings.extend(build_warnings)
            if not options:
                continue

            pareto = _finalize_pareto_options(
                options,
                max_alternatives=req.max_alternatives,
                pareto_method=req.pareto_method,
                epsilon=req.epsilon,
                optimization_mode=req.optimization_mode,
                risk_aversion=req.risk_aversion,
            )
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
                    "fallback_used": fallback_used,
                }
            )

        if not selected_rows:
            if req.time_window is not None:
                raise HTTPException(status_code=422, detail="no feasible departures for provided time window")
            raise HTTPException(status_code=502, detail="No departure candidates could be computed.")

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

        candidates: list[DepartureOptimizeCandidate] = []
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
                    fallback_used=bool(row["fallback_used"]),
                )
            )

        candidates.sort(
            key=lambda item: (
                item.score,
                _option_objective_value(
                    item.selected,
                    "duration",
                    optimization_mode=req.optimization_mode,
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
        cursor = cursor.replace(tzinfo=timezone.utc)
    else:
        cursor = cursor.astimezone(timezone.utc)
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

            cache_key = _candidate_cache_key(
                origin=origin,
                destination=destination,
                vehicle_type=req.vehicle_type,
                scenario_mode=req.scenario_mode,
                max_routes=req.max_alternatives,
                cost_toggles=req.cost_toggles,
                terrain_profile=req.terrain_profile,
                departure_time_utc=departure_cursor,
            )
            routes, warnings, _candidate_fetches, fallback_used = await _collect_candidate_routes(
                osrm=osrm,
                origin=origin,
                destination=destination,
                max_routes=req.max_alternatives,
                cache_key=cache_key,
            )
            options, build_warnings = _build_options(
                routes,
                vehicle_type=req.vehicle_type,
                scenario_mode=req.scenario_mode,
                cost_toggles=req.cost_toggles,
                terrain_profile=req.terrain_profile,
                stochastic=req.stochastic,
                emissions_context=req.emissions_context,
                weather=req.weather,
                incident_simulation=req.incident_simulation,
                departure_time_utc=departure_cursor,
                option_prefix=f"duty_leg_{idx}",
            )
            warnings.extend(build_warnings)

            if not options:
                legs.append(
                    DutyChainLegResult(
                        leg_index=idx,
                        origin=origin_stop,
                        destination=destination_stop,
                        selected=None,
                        candidates=[],
                        warning_count=len(warnings),
                        fallback_used=fallback_used,
                        error="No route candidates could be computed.",
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
                    fallback_used=fallback_used,
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
) -> dict[str, float]:
    if baseline is None or current is None:
        return {
            "duration_s_delta": 0.0,
            "monetary_cost_delta": 0.0,
            "emissions_kg_delta": 0.0,
        }
    return {
        "duration_s_delta": round(current.metrics.duration_s - baseline.metrics.duration_s, 3),
        "monetary_cost_delta": round(current.metrics.monetary_cost - baseline.metrics.monetary_cost, 3),
        "emissions_kg_delta": round(current.metrics.emissions_kg - baseline.metrics.emissions_kg, 3),
    }


async def _run_scenario_compare(
    req: ScenarioCompareRequest,
    osrm: OSRMClient,
) -> tuple[list[ScenarioCompareResult], dict[str, dict[str, float]]]:
    scenario_modes = [
        ScenarioMode.NO_SHARING,
        ScenarioMode.PARTIAL_SHARING,
        ScenarioMode.FULL_SHARING,
    ]
    results: list[ScenarioCompareResult] = []

    for scenario_mode in scenario_modes:
        cache_key = _candidate_cache_key(
            origin=req.origin,
            destination=req.destination,
            vehicle_type=req.vehicle_type,
            scenario_mode=scenario_mode,
            max_routes=req.max_alternatives,
            cost_toggles=req.cost_toggles,
            terrain_profile=req.terrain_profile,
            departure_time_utc=req.departure_time_utc,
        )
        routes, warnings, _candidate_fetches, fallback_used = await _collect_candidate_routes(
            osrm=osrm,
            origin=req.origin,
            destination=req.destination,
            max_routes=req.max_alternatives,
            cache_key=cache_key,
        )
        options, build_warnings = _build_options(
            routes,
            vehicle_type=req.vehicle_type,
            scenario_mode=scenario_mode,
            cost_toggles=req.cost_toggles,
            terrain_profile=req.terrain_profile,
            stochastic=req.stochastic,
            emissions_context=req.emissions_context,
            weather=req.weather,
            incident_simulation=req.incident_simulation,
            departure_time_utc=req.departure_time_utc,
            option_prefix=f"scenario_{scenario_mode.value}",
        )
        warnings.extend(build_warnings)

        if not options:
            results.append(
                ScenarioCompareResult(
                    scenario_mode=scenario_mode,
                    selected=None,
                    candidates=[],
                    warnings=warnings,
                    fallback_used=fallback_used,
                    error="No route candidates could be computed.",
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
                fallback_used=fallback_used,
            )
        )

    baseline = next(
        (item.selected for item in results if item.scenario_mode == ScenarioMode.NO_SHARING),
        None,
    )
    deltas: dict[str, dict[str, float]] = {}
    for item in results:
        deltas[item.scenario_mode.value] = _scenario_metric_deltas(baseline, item.selected)

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

        manifest_payload = {
            "schema_version": "1.0.0",
            "type": "scenario_compare",
            "request": req.model_dump(mode="json"),
            "results": [item.model_dump(mode="json") for item in results],
            "deltas": deltas,
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
    q: str | None = Query(default=None),
    vehicle_type: str | None = Query(default=None),
    scenario_mode: ScenarioMode | None = Query(default=None),
    sort: str = Query(default="updated_desc"),
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

        manifest_payload = {
            "schema_version": "1.0.0",
            "type": "scenario_compare",
            "source": {"experiment_id": bundle.id, "experiment_name": bundle.name},
            "request": req.model_dump(mode="json"),
            "results": [item.model_dump(mode="json") for item in results],
            "deltas": deltas,
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
                "fallback_used": result.fallback_used,
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
            "fallback_used",
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


def _parse_pairs_from_csv_text(csv_text: str) -> list[dict[str, Any]]:
    reader = csv.DictReader(io.StringIO(csv_text))
    required = {"origin_lat", "origin_lon", "destination_lat", "destination_lon"}
    headers = set(reader.fieldnames or [])
    if not required.issubset(headers):
        missing = sorted(required - headers)
        raise HTTPException(
            status_code=422,
            detail=f"CSV missing required columns: {', '.join(missing)}",
        )

    pairs: list[dict[str, Any]] = []
    for idx, row in enumerate(reader, start=2):
        try:
            pairs.append(
                {
                    "origin": {
                        "lat": float(row["origin_lat"]),
                        "lon": float(row["origin_lon"]),
                    },
                    "destination": {
                        "lat": float(row["destination_lat"]),
                        "lon": float(row["destination_lon"]),
                    },
                }
            )
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=f"invalid CSV numeric value on row {idx}",
            ) from e

    if not pairs:
        raise HTTPException(status_code=422, detail="CSV contains no OD pairs")

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
        vehicle_type=req.vehicle_type,
        scenario_mode=req.scenario_mode,
        max_alternatives=req.max_alternatives,
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
                cache_key = _candidate_cache_key(
                    origin=pair.origin,
                    destination=pair.destination,
                    vehicle_type=req.vehicle_type,
                    scenario_mode=req.scenario_mode,
                    max_routes=req.max_alternatives,
                    cost_toggles=req.cost_toggles,
                    terrain_profile=req.terrain_profile,
                    departure_time_utc=req.departure_time_utc,
                )
                pair_stats: dict[str, Any] = {
                    "pair_index": pair_idx,
                    "candidate_count": 0,
                    "option_count": 0,
                    "pareto_count": 0,
                    "error": None,
                    "fallback_used": False,
                }
                try:
                    routes = await osrm.fetch_routes(
                        origin_lat=pair.origin.lat,
                        origin_lon=pair.origin.lon,
                        dest_lat=pair.destination.lat,
                        dest_lon=pair.destination.lon,
                        alternatives=req.max_alternatives,
                    )
                    routes = routes[: req.max_alternatives]
                    pair_stats["candidate_count"] = len(routes)
                    save_route_snapshot(cache_key, routes)
                    options: list[RouteOption] = [
                        build_option(
                            r,
                            option_id=f"pair{pair_idx}_route{i}",
                            vehicle_type=req.vehicle_type,
                            scenario_mode=req.scenario_mode,
                            cost_toggles=req.cost_toggles,
                            terrain_profile=req.terrain_profile,
                            stochastic=req.stochastic,
                            emissions_context=req.emissions_context,
                            weather=req.weather,
                            incident_simulation=req.incident_simulation,
                            departure_time_utc=req.departure_time_utc,
                        )
                        for i, r in enumerate(routes)
                    ]
                    pair_stats["option_count"] = len(options)
                    pareto = _finalize_pareto_options(
                        options,
                        max_alternatives=req.max_alternatives,
                        pareto_method=req.pareto_method,
                        epsilon=req.epsilon,
                        optimization_mode=req.optimization_mode,
                        risk_aversion=req.risk_aversion,
                    )
                    pair_stats["pareto_count"] = len(pareto)

                    return (
                        BatchParetoResult(
                            origin=pair.origin,
                            destination=pair.destination,
                            routes=pareto,
                            fallback_used=bool(pair_stats["fallback_used"]),
                        ),
                        pair_stats,
                    )
                except OSRMError as e:
                    if settings.offline_fallback_enabled:
                        snapshot = load_route_snapshot(cache_key)
                        if snapshot is not None:
                            routes, _updated_at = snapshot
                            routes = routes[: req.max_alternatives]
                            pair_stats["candidate_count"] = len(routes)
                            pair_stats["fallback_used"] = True
                            options = [
                                build_option(
                                    r,
                                    option_id=f"pair{pair_idx}_route{i}",
                                    vehicle_type=req.vehicle_type,
                                    scenario_mode=req.scenario_mode,
                                    cost_toggles=req.cost_toggles,
                                    terrain_profile=req.terrain_profile,
                                    stochastic=req.stochastic,
                                    emissions_context=req.emissions_context,
                                    weather=req.weather,
                                    incident_simulation=req.incident_simulation,
                                    departure_time_utc=req.departure_time_utc,
                                )
                                for i, r in enumerate(routes)
                            ]
                            pair_stats["option_count"] = len(options)
                            pareto = _finalize_pareto_options(
                                options,
                                max_alternatives=req.max_alternatives,
                                pareto_method=req.pareto_method,
                                epsilon=req.epsilon,
                                optimization_mode=req.optimization_mode,
                                risk_aversion=req.risk_aversion,
                            )
                            pair_stats["pareto_count"] = len(pareto)
                            return (
                                BatchParetoResult(
                                    origin=pair.origin,
                                    destination=pair.destination,
                                    routes=pareto,
                                    fallback_used=True,
                                ),
                                pair_stats,
                            )

                    pair_stats["error"] = str(e)
                    return (
                        BatchParetoResult(
                            origin=pair.origin,
                            destination=pair.destination,
                            error=str(e),
                            fallback_used=False,
                        ),
                        pair_stats,
                    )
                except ValueError as e:
                    pair_stats["error"] = str(e)
                    return (
                        BatchParetoResult(
                            origin=pair.origin,
                            destination=pair.destination,
                            error=str(e),
                            fallback_used=bool(pair_stats["fallback_used"]),
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
        fallback_count = sum(1 for stats in pair_stats if bool(stats["fallback_used"]))

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
                        "fallback_used": bool(stats["fallback_used"]),
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
                fallback_count=fallback_count,
                pairs=[
                    {
                        "pair_index": int(stats["pair_index"]),
                        "pareto_count": int(stats["pareto_count"]),
                        "fallback_used": bool(stats["fallback_used"]),
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
                "fallback_used": fallback_count > 0,
                "fallback_count": fallback_count,
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
            "fallback_used": fallback_count > 0,
            "fallback_count": fallback_count,
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
                fallback_count=fallback_count,
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
            fallback_used=fallback_count > 0,
            fallback_count=fallback_count,
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
