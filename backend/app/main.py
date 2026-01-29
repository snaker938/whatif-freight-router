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
    LatLng,
    ParetoRequest,
    ParetoResponse,
    RouteMetrics,
    RouteOption,
    RouteRequest,
    RouteResponse,
    VehicleListResponse,
)
from .objectives_emissions import route_emissions_kg
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
    # With models.py importing VehicleProfile from vehicles.py, Pylance stops complaining.
    return VehicleListResponse(vehicles=list(VEHICLES.values()))


def _validate_osrm_geometry(geometry: Any) -> dict[str, Any]:
    if not isinstance(geometry, dict):
        raise HTTPException(status_code=502, detail="OSRM returned invalid geometry object")
    if geometry.get("type") != "LineString":
        raise HTTPException(status_code=502, detail="OSRM returned non-LineString geometry")
    coords = geometry.get("coordinates")
    if not isinstance(coords, list) or not coords:
        raise HTTPException(status_code=502, detail="OSRM returned empty geometry coordinates")
    return geometry


def build_option(
    route: dict[str, Any],
    *,
    option_id: str,
    vehicle_type: str,
    scenario_mode: ScenarioMode,
) -> RouteOption:
    vehicle = get_vehicle(vehicle_type)

    seg_d_m, seg_t_s = extract_segment_annotations(route)

    distance_m = float(route.get("distance", 0.0) or 0.0)
    base_duration_s = float(route.get("duration", 0.0) or 0.0)

    distance_km = distance_m / 1000.0
    base_duration_h = base_duration_s / 3600.0

    base_emissions = route_emissions_kg(
        vehicle=vehicle,
        segment_distances_m=seg_d_m,
        segment_durations_s=seg_t_s,
    )

    base_monetary = (distance_km * vehicle.cost_per_km) + (base_duration_h * vehicle.cost_per_hour)

    duration_s = apply_scenario_duration(base_duration_s, scenario_mode)
    extra_time_s = max(0.0, duration_s - base_duration_s)

    emissions = base_emissions + (extra_time_s / 3600.0) * vehicle.idle_emissions_kg_per_hour
    monetary = base_monetary + (extra_time_s / 3600.0) * vehicle.cost_per_hour

    duration_h = duration_s / 3600.0
    avg_speed_kmh = (distance_km / duration_h) if duration_h > 0 else 0.0

    geometry = _validate_osrm_geometry(route.get("geometry"))

    metrics = RouteMetrics(
        distance_km=distance_km,
        duration_s=duration_s,
        monetary_cost=monetary,
        emissions_kg=emissions,
        avg_speed_kmh=avg_speed_kmh,
    )

    try:
        return RouteOption(id=option_id, geometry=geometry, metrics=metrics)  # type: ignore[arg-type]
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=502, detail=f"Invalid route output from OSRM: {e}") from e


def _route_signature(route: dict[str, Any]) -> str:
    """Create a stable-ish hash to dedupe near-identical routes."""
    geom = route.get("geometry", {})
    coords = (geom or {}).get("coordinates", [])
    if not isinstance(coords, list) or not coords:
        raw = repr((route.get("distance"), route.get("duration")))
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    n = len(coords)
    step = max(1, n // 30)
    sample = coords[::step][:40]

    parts: list[str] = []
    for pt in sample:
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            continue
        try:
            lon = float(pt[0])
            lat = float(pt[1])
        except Exception:
            continue
        parts.append(f"{lon:.4f},{lat:.4f}")

    raw = "|".join(parts) + f"|n={n}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _midpoint(
    origin: LatLng, destination: LatLng, base_route: dict[str, Any] | None
) -> tuple[float, float]:
    """Return (lat, lon) midpoint using the base route geometry if available."""
    try:
        if base_route:
            geom = base_route.get("geometry", {})
            coords = (geom or {}).get("coordinates", [])
            if isinstance(coords, list) and coords:
                mid = coords[len(coords) // 2]
                if isinstance(mid, (list, tuple)) and len(mid) >= 2:
                    lon = float(mid[0])
                    lat = float(mid[1])
                    return lat, lon
    except Exception:
        pass

    return (origin.lat + destination.lat) / 2.0, (origin.lon + destination.lon) / 2.0


def _candidate_via_points(
    origin: LatLng, destination: LatLng, base_route: dict[str, Any] | None
) -> list[tuple[float, float]]:
    """Generate a few via points to force alternative paths (lat, lon)."""
    mid_lat, mid_lon = _midpoint(origin, destination, base_route)
    span = max(abs(destination.lat - origin.lat), abs(destination.lon - origin.lon))
    offset = max(0.08, min(span * 0.18, 0.6))

    def clamp_lat(x: float) -> float:
        return max(-89.9, min(89.9, x))

    def clamp_lon(x: float) -> float:
        return max(-179.9, min(179.9, x))

    return [
        (clamp_lat(mid_lat + offset), clamp_lon(mid_lon)),
        (clamp_lat(mid_lat - offset), clamp_lon(mid_lon)),
        (clamp_lat(mid_lat), clamp_lon(mid_lon + offset)),
        (clamp_lat(mid_lat), clamp_lon(mid_lon - offset)),
    ]


async def _collect_candidate_routes(
    *,
    osrm: OSRMClient,
    origin: LatLng,
    destination: LatLng,
    max_routes: int,
) -> list[dict[str, Any]]:
    """Get a small, diverse set of candidate routes.

    OSRM `alternatives=true` often returns only 1 route. To keep the sliders/plot useful,
    we also try:
      - excludable-class variants (motorway/toll/ferry/trunk), if supported
      - 1-via-point detours around the midpoint
    """
    max_routes = max(1, min(max_routes, 5))

    base = await osrm.fetch_routes(
        origin_lat=origin.lat,
        origin_lon=origin.lon,
        dest_lat=destination.lat,
        dest_lon=destination.lon,
        alternatives=max_routes,
    )

    raw: list[dict[str, Any]] = list(base)

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
        by_time = min(options, key=lambda o: o.metrics.duration_s)
        by_cost = min(options, key=lambda o: o.metrics.monetary_cost)
        by_co2 = min(options, key=lambda o: o.metrics.emissions_kg)
        unique_extremes = {r.id: r for r in (by_time, by_cost, by_co2)}
        pareto_sorted = sorted(unique_extremes.values(), key=lambda o: o.metrics.duration_s)

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
