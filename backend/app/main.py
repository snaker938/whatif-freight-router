from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from collections.abc import Awaitable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .logging_utils import log_event
from .models import (
    BatchParetoRequest,
    BatchParetoResponse,
    BatchParetoResult,
    GeoJSONLineString,
    LatLng,
    ParetoRequest,
    ParetoResponse,
    RouteMetrics,
    RouteOption,
    RouteRequest,
    RouteResponse,
    VehicleListResponse,
)
from .objectives_emissions import route_emissions_kg, speed_factor
from .objectives_selection import pick_best_by_weighted_sum
from .pareto import pareto_filter
from .routing_osrm import OSRMClient, OSRMError, extract_segment_annotations
from .run_store import write_manifest
from .scenario import ScenarioMode, apply_scenario_duration
from .settings import settings
from .vehicles import VEHICLES, get_vehicle


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
    return VehicleListResponse(vehicles=list(VEHICLES.values()))


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


def build_option(
    route: dict[str, Any],
    *,
    option_id: str,
    vehicle_type: str,
    scenario_mode: ScenarioMode,
) -> RouteOption:
    vehicle = get_vehicle(vehicle_type)

    coords = _validate_osrm_geometry(route)
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

    base_monetary = fuel_cost + (base_duration_h * vehicle.cost_per_hour * DRIVER_TIME_COST_WEIGHT)

    duration_s = apply_scenario_duration(base_duration_s, mode=scenario_mode)
    extra_time_s = max(duration_s - base_duration_s, 0.0)

    # Scenario also adds idle emissions and driver time cost for extra delay.
    monetary = (
        base_monetary + (extra_time_s / 3600.0) * vehicle.cost_per_hour * DRIVER_TIME_COST_WEIGHT
    )
    emissions = base_emissions + (extra_time_s / 3600.0) * vehicle.idle_emissions_kg_per_hour

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


def _candidate_via_points(
    origin: LatLng, destination: LatLng, *, base_route: dict[str, Any] | None
) -> list[tuple[float, float]]:
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


async def _collect_candidate_routes(
    *,
    osrm: OSRMClient,
    origin: LatLng,
    destination: LatLng,
    max_routes: int,
) -> list[dict[str, Any]]:
    max_routes = max(1, min(max_routes, 5))

    # Start by asking OSRM for alternatives.
    raw = await osrm.fetch_routes(
        origin_lat=origin.lat,
        origin_lon=origin.lon,
        dest_lat=destination.lat,
        dest_lon=destination.lon,
        alternatives=max_routes,
    )

    # If OSRM doesn't give many alternatives, try a few forced variants:
    # - exclude certain classes (if supported)
    # - via points around midpoint
    tasks: list[Awaitable[list[dict[str, Any]]]] = []
    if len(raw) < max_routes:
        for ex in ["motorway", "trunk", "toll", "ferry"]:
            tasks.append(
                osrm.fetch_routes(
                    origin_lat=origin.lat,
                    origin_lon=origin.lon,
                    dest_lat=destination.lat,
                    dest_lon=destination.lon,
                    alternatives=False,
                    exclude=ex,
                )
            )

        for via in _candidate_via_points(origin, destination, base_route=raw[0] if raw else None):
            tasks.append(
                osrm.fetch_routes(
                    origin_lat=origin.lat,
                    origin_lon=origin.lon,
                    dest_lat=destination.lat,
                    dest_lon=destination.lon,
                    alternatives=False,
                    via=[via],
                )
            )

    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            # ✅ fixes your "raw.extend(r)" type error: gather can return BaseException
            if isinstance(r, BaseException):
                continue
            raw.extend(r)

    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for r in raw:
        sig = _route_signature(r)
        if sig in seen:
            continue
        seen.add(sig)
        unique.append(r)
        if len(unique) >= max_routes:
            break

    return unique


@app.post("/pareto", response_model=ParetoResponse)
async def compute_pareto(req: ParetoRequest, osrm: OSRMDep) -> ParetoResponse:
    request_id = str(uuid.uuid4())
    t0 = time.perf_counter()

    try:
        routes = await _collect_candidate_routes(
            osrm=osrm,
            origin=req.origin,
            destination=req.destination,
            max_routes=req.max_alternatives,
        )
    except OSRMError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    options: list[RouteOption] = [
        build_option(
            r,
            option_id=f"route_{i}",
            vehicle_type=req.vehicle_type,
            scenario_mode=req.scenario_mode,
        )
        for i, r in enumerate(routes)
    ]

    pareto = pareto_filter(
        options,
        key=lambda o: (o.metrics.duration_s, o.metrics.monetary_cost, o.metrics.emissions_kg),
    )
    pareto_sorted = sorted(pareto, key=lambda o: o.metrics.duration_s)

    if len(pareto_sorted) < 2 and len(options) > 1:
        # In practice, cost and emissions can be correlated with time, so one candidate
        # can dominate and the Pareto set collapses to a single route. That makes the UI
        # feel "broken" (sliders can't change anything), so return a small diverse
        # subset as a fallback.
        desired = min(len(options), max(2, min(req.max_alternatives, 4)))

        ranked_time = sorted(options, key=lambda o: o.metrics.duration_s)
        ranked_cost = sorted(options, key=lambda o: o.metrics.monetary_cost)
        ranked_co2 = sorted(options, key=lambda o: o.metrics.emissions_kg)

        chosen: dict[str, RouteOption] = {}

        def pick_first(ranked: list[RouteOption]) -> None:
            for r in ranked:
                if r.id not in chosen:
                    chosen[r.id] = r
                    return

        # Always include extremes where possible.
        for ranked in (ranked_time, ranked_cost, ranked_co2):
            pick_first(ranked)

        # Fill remaining slots with next-fastest distinct routes.
        for r in ranked_time:
            if len(chosen) >= desired:
                break
            if r.id not in chosen:
                chosen[r.id] = r

        pareto_sorted = sorted(chosen.values(), key=lambda o: o.metrics.duration_s)

    log_event(
        "pareto_request",
        request_id=request_id,
        vehicle_type=req.vehicle_type,
        scenario_mode=req.scenario_mode.value,
        origin=req.origin.model_dump(),
        destination=req.destination.model_dump(),
        candidate_count=len(options),
        pareto_count=len(pareto_sorted),
        duration_ms=round((time.perf_counter() - t0) * 1000, 2),
    )

    return ParetoResponse(routes=pareto_sorted)


@app.post("/route", response_model=RouteResponse)
async def compute_route(req: RouteRequest, osrm: OSRMDep) -> RouteResponse:
    request_id = str(uuid.uuid4())
    t0 = time.perf_counter()

    try:
        routes = await _collect_candidate_routes(
            osrm=osrm,
            origin=req.origin,
            destination=req.destination,
            max_routes=5,
        )
    except OSRMError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    options: list[RouteOption] = [
        build_option(
            r,
            option_id=f"route_{i}",
            vehicle_type=req.vehicle_type,
            scenario_mode=req.scenario_mode,
        )
        for i, r in enumerate(routes)
    ]

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
        origin=req.origin.model_dump(),
        destination=req.destination.model_dump(),
        candidate_count=len(options),
        selected_id=selected.id,
        selected_metrics=selected.metrics.model_dump(),
        duration_ms=round((time.perf_counter() - t0) * 1000, 2),
    )

    return RouteResponse(selected=selected, candidates=options)


@app.post("/batch/pareto", response_model=BatchParetoResponse)
async def batch_pareto(req: BatchParetoRequest, osrm: OSRMDep) -> BatchParetoResponse:
    run_id = str(uuid.uuid4())
    t0 = time.perf_counter()

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

    manifest_path = write_manifest(
        run_id,
        {
            "type": "batch_pareto",
            "pair_count": len(req.pairs),
            "vehicle_type": req.vehicle_type,
            "scenario_mode": req.scenario_mode.value,
            "max_alternatives": req.max_alternatives,
            "batch_concurrency": settings.batch_concurrency,
            "duration_ms": duration_ms,
            "error_count": error_count,
        },
    )

    log_event(
        "batch_pareto_request",
        run_id=run_id,
        pair_count=len(req.pairs),
        vehicle_type=req.vehicle_type,
        scenario_mode=req.scenario_mode.value,
        duration_ms=duration_ms,
        error_count=error_count,
        manifest=str(manifest_path),
    )

    return BatchParetoResponse(run_id=run_id, results=results)


def _manifest_path_for_id(run_id: str) -> Path:
    try:
        uuid.UUID(run_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="invalid run_id") from e

    p = Path(settings.out_dir) / "manifests" / f"{run_id}.json"
    base = (Path(settings.out_dir) / "manifests").resolve()
    resolved = p.resolve()

    if not str(resolved).startswith(str(base)):
        raise HTTPException(status_code=400, detail="invalid run_id path")

    return resolved


@app.get("/runs/{run_id}/manifest")
async def get_manifest(run_id: str):
    path = _manifest_path_for_id(run_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="manifest not found")
    return FileResponse(str(path), media_type="application/json")
