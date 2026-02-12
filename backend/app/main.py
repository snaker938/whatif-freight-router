from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from .logging_utils import log_event
from .metrics_store import metrics_snapshot, record_request
from .models import (
    BatchParetoRequest,
    BatchParetoResponse,
    BatchParetoResult,
    CostToggles,
    CustomVehicleListResponse,
    GeoJSONLineString,
    LatLng,
    ParetoRequest,
    ParetoResponse,
    RouteMetrics,
    RouteOption,
    RouteRequest,
    RouteResponse,
    VehicleDeleteResponse,
    VehicleListResponse,
    VehicleMutationResponse,
)
from .objectives_emissions import route_emissions_kg, speed_factor
from .objectives_selection import pick_best_by_weighted_sum
from .pareto import pareto_filter
from .route_cache import clear_route_cache, get_cached_routes, route_cache_stats, set_cached_routes
from .routing_osrm import OSRMClient, OSRMError, extract_segment_annotations
from .run_store import ARTIFACT_FILES, artifact_paths_for_run, write_manifest, write_run_artifacts
from .scenario import ScenarioMode, apply_scenario_duration
from .settings import settings
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
async def create_vehicle_profile(vehicle: VehicleProfile) -> VehicleMutationResponse:
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
async def update_vehicle_profile(vehicle_id: str, vehicle: VehicleProfile) -> VehicleMutationResponse:
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
async def delete_vehicle_profile(vehicle_id: str) -> VehicleDeleteResponse:
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
async def delete_cache() -> dict[str, int]:
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


def build_option(
    route: dict[str, Any],
    *,
    option_id: str,
    vehicle_type: str,
    scenario_mode: ScenarioMode,
    cost_toggles: CostToggles,
) -> RouteOption:
    vehicle = get_vehicle(vehicle_type)

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
    base_duration_h = base_duration_s / 3600.0

    base_emissions = route_emissions_kg(
        vehicle=vehicle,
        segment_distances_m=seg_d_m,
        segment_durations_s=seg_t_s,
    )

    fuel_cost = 0.0
    for d_m, t_s in zip(seg_d_m, seg_t_s, strict=True):
        d_m = max(float(d_m), 0.0)
        t_s = max(float(t_s), 0.0)

        d_km = d_m / 1000.0
        speed_kmh = (d_m / t_s) * 3.6 if t_s > 0 else 0.0

        # Approximate fuel / wear cost as distance * speed_factor(speed).
        # This helps create time-vs-cost trade-offs so the UI can show multiple routes.
        fuel_cost += d_km * vehicle.cost_per_km * speed_factor(speed_kmh)

    fuel_cost *= max(cost_toggles.fuel_price_multiplier, 0.0)

    contains_toll = bool(route.get("contains_toll", False))
    toll_component = (
        distance_km * cost_toggles.toll_cost_per_km if cost_toggles.use_tolls and contains_toll else 0.0
    )

    base_monetary = fuel_cost + (base_duration_h * vehicle.cost_per_hour * DRIVER_TIME_COST_WEIGHT)
    base_monetary += toll_component

    duration_s = apply_scenario_duration(base_duration_s, mode=scenario_mode)
    extra_time_s = max(duration_s - base_duration_s, 0.0)

    # Scenario also adds idle emissions and driver time cost for extra delay.
    monetary = (
        base_monetary + (extra_time_s / 3600.0) * vehicle.cost_per_hour * DRIVER_TIME_COST_WEIGHT
    )
    emissions = base_emissions + (extra_time_s / 3600.0) * vehicle.idle_emissions_kg_per_hour
    monetary += emissions * max(cost_toggles.carbon_price_per_kg, 0.0)

    avg_speed_kmh = distance_km / (duration_s / 3600.0) if duration_s > 0 else 0.0

    return RouteOption(
        id=option_id,
        geometry=GeoJSONLineString(type="LineString", coordinates=coords),
        metrics=RouteMetrics(
            distance_km=round(distance_km, 3),
            duration_s=round(duration_s, 2),
            monetary_cost=round(monetary, 2),
            emissions_kg=round(emissions, 3),
            avg_speed_kmh=round(avg_speed_kmh, 2),
        ),
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
    vehicle_type: str,
    scenario_mode: ScenarioMode,
    max_routes: int,
    cost_toggles: CostToggles,
) -> str:
    payload = {
        "origin": {"lat": round(origin.lat, 6), "lon": round(origin.lon, 6)},
        "destination": {"lat": round(destination.lat, 6), "lon": round(destination.lon, 6)},
        "vehicle_type": vehicle_type,
        "scenario_mode": scenario_mode.value,
        "max_routes": max_routes,
        "cost_toggles": cost_toggles.model_dump(mode="json"),
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
                )
            )
        except (OSRMError, ValueError) as e:
            warnings.append(f"{option_id}: {e}")

    return options, warnings


def _finalize_pareto_options(
    options: list[RouteOption],
    *,
    max_alternatives: int,
) -> list[RouteOption]:
    pareto = pareto_filter(
        options,
        key=lambda o: (o.metrics.duration_s, o.metrics.monetary_cost, o.metrics.emissions_kg),
    )
    pareto_sorted = sorted(pareto, key=lambda o: (o.metrics.duration_s, o.id))

    if len(pareto_sorted) < 2 and len(options) > 1:
        # In practice, cost and emissions can be correlated with time, so one candidate
        # can dominate and the Pareto set collapses to a single route. That makes the UI
        # feel "broken" (sliders can't change anything), so return a small diverse
        # subset as a fallback.
        desired = min(len(options), max(2, min(max_alternatives, 4)))

        ranked_time = sorted(options, key=lambda o: o.metrics.duration_s)
        ranked_cost = sorted(options, key=lambda o: o.metrics.monetary_cost)
        ranked_co2 = sorted(options, key=lambda o: o.metrics.emissions_kg)

        chosen: dict[str, RouteOption] = {}

        def pick_first(ranked: list[RouteOption]) -> None:
            for route in ranked:
                if route.id not in chosen:
                    chosen[route.id] = route
                    return

        # Always include extremes where possible.
        for ranked in (ranked_time, ranked_cost, ranked_co2):
            pick_first(ranked)

        # Fill remaining slots with next-fastest distinct routes.
        for route in ranked_time:
            if len(chosen) >= desired:
                break
            if route.id not in chosen:
                chosen[route.id] = route

        pareto_sorted = sorted(
            chosen.values(),
            key=lambda route: (route.metrics.duration_s, route.id),
        )

    return pareto_sorted


async def _collect_candidate_routes(
    *,
    osrm: OSRMClient,
    origin: LatLng,
    destination: LatLng,
    max_routes: int,
    cache_key: str | None = None,
) -> tuple[list[dict[str, Any]], list[str], int]:
    if cache_key:
        cached = get_cached_routes(cache_key)
        if cached is not None:
            return cached

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
    out = (ranked_routes, warnings, len(specs))
    if cache_key:
        set_cached_routes(cache_key, out)
    return out


@app.post("/pareto", response_model=ParetoResponse)
async def compute_pareto(req: ParetoRequest, osrm: OSRMDep) -> ParetoResponse:
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
        )
        routes, warnings, candidate_fetches = await _collect_candidate_routes(
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
            option_prefix="route",
        )
        warnings.extend(build_warnings)

        if not options:
            detail = "No route candidates could be computed."
            if warnings:
                detail = f"{detail} {warnings[0]}"
            raise HTTPException(status_code=502, detail=detail)

        pareto_sorted = _finalize_pareto_options(options, max_alternatives=req.max_alternatives)

        log_event(
            "pareto_request",
            request_id=request_id,
            vehicle_type=req.vehicle_type,
            scenario_mode=req.scenario_mode.value,
            cost_toggles=req.cost_toggles.model_dump(),
            origin=req.origin.model_dump(),
            destination=req.destination.model_dump(),
            candidate_fetches=candidate_fetches,
            candidate_count=len(options),
            pareto_count=len(pareto_sorted),
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
                option_prefix="route",
            )
            warnings.extend(build_warnings)
            pareto_sorted = _finalize_pareto_options(options, max_alternatives=req.max_alternatives)

            log_event(
                "pareto_stream_request",
                request_id=request_id,
                vehicle_type=req.vehicle_type,
                scenario_mode=req.scenario_mode.value,
                cost_toggles=req.cost_toggles.model_dump(),
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
async def compute_route(req: RouteRequest, osrm: OSRMDep) -> RouteResponse:
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
        )
        routes, warnings, candidate_fetches = await _collect_candidate_routes(
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
            option_prefix="route",
        )
        warnings.extend(build_warnings)

        if not options:
            detail = "No route candidates could be computed."
            if warnings:
                detail = f"{detail} {warnings[0]}"
            raise HTTPException(status_code=502, detail=detail)

        selected = pick_best_by_weighted_sum(
            options,
            w_time=req.weights.time,
            w_money=req.weights.money,
            w_co2=req.weights.co2,
        )

        log_event(
            "route_request",
            request_id=request_id,
            vehicle_type=req.vehicle_type,
            scenario_mode=req.scenario_mode.value,
            weights=req.weights.model_dump(),
            cost_toggles=req.cost_toggles.model_dump(),
            origin=req.origin.model_dump(),
            destination=req.destination.model_dump(),
            candidate_fetches=candidate_fetches,
            candidate_count=len(options),
            warning_count=len(warnings),
            selected_id=selected.id,
            selected_metrics=selected.metrics.model_dump(),
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
        )

        return RouteResponse(selected=selected, candidates=options)
    except HTTPException:
        has_error = True
        raise
    except Exception:
        has_error = True
        raise
    finally:
        _record_endpoint_metric("route", t0, error=has_error)


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


@app.post("/batch/pareto", response_model=BatchParetoResponse)
async def batch_pareto(req: BatchParetoRequest, osrm: OSRMDep) -> BatchParetoResponse:
    run_id = str(uuid.uuid4())
    t0 = time.perf_counter()
    has_error = False

    try:
        sem = asyncio.Semaphore(settings.batch_concurrency)

        async def one(pair_idx: int) -> BatchParetoResult:
            async with sem:
                try:
                    routes = await osrm.fetch_routes(
                        origin_lat=req.pairs[pair_idx].origin.lat,
                        origin_lon=req.pairs[pair_idx].origin.lon,
                        dest_lat=req.pairs[pair_idx].destination.lat,
                        dest_lon=req.pairs[pair_idx].destination.lon,
                        alternatives=req.max_alternatives,
                    )
                    routes = routes[: req.max_alternatives]
                    options: list[RouteOption] = [
                        build_option(
                            r,
                            option_id=f"pair{pair_idx}_route{i}",
                            vehicle_type=req.vehicle_type,
                            scenario_mode=req.scenario_mode,
                            cost_toggles=req.cost_toggles,
                        )
                        for i, r in enumerate(routes)
                    ]
                    pareto = pareto_filter(
                        options,
                        key=lambda o: (
                            o.metrics.duration_s,
                            o.metrics.monetary_cost,
                            o.metrics.emissions_kg,
                        ),
                    )
                    pareto_sorted = sorted(pareto, key=lambda o: o.metrics.duration_s)

                    return BatchParetoResult(
                        origin=req.pairs[pair_idx].origin,
                        destination=req.pairs[pair_idx].destination,
                        routes=pareto_sorted,
                    )
                except (OSRMError, ValueError) as e:
                    return BatchParetoResult(
                        origin=req.pairs[pair_idx].origin,
                        destination=req.pairs[pair_idx].destination,
                        error=str(e),
                    )

        results = await asyncio.gather(*[one(i) for i in range(len(req.pairs))])

        duration_ms = round((time.perf_counter() - t0) * 1000, 2)
        error_count = sum(1 for r in results if r.error)

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
                "duration_ms": duration_ms,
                "error_count": error_count,
            },
        }
        manifest_path = write_manifest(
            run_id,
            manifest_payload,
        )

        artifact_paths = write_run_artifacts(
            run_id,
            results_payload={
                "run_id": run_id,
                "results": [r.model_dump(mode="json") for r in results],
            },
            metadata_payload={
                "run_id": run_id,
                "schema_version": "1.0.0",
                "manifest_endpoint": f"/runs/{run_id}/manifest",
                "artifacts_endpoint": f"/runs/{run_id}/artifacts",
                "artifact_names": sorted(artifact_paths_for_run(run_id)),
                "pair_count": len(req.pairs),
                "error_count": error_count,
                "duration_ms": duration_ms,
            },
            csv_rows=_batch_results_csv_rows(req, results),
        )

        log_event(
            "batch_pareto_request",
            run_id=run_id,
            pair_count=len(req.pairs),
            vehicle_type=req.vehicle_type,
            scenario_mode=req.scenario_mode.value,
            cost_toggles=req.cost_toggles.model_dump(),
            duration_ms=duration_ms,
            error_count=error_count,
            manifest=str(manifest_path),
            artifacts=[name for name in sorted(artifact_paths)],
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
        return {"run_id": valid_run_id, "artifacts": artifacts}
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
        elif artifact_name.endswith(".csv"):
            media_type = "text/csv"

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
