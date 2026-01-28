from __future__ import annotations

import asyncio
import time
import uuid
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


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/vehicles", response_model=VehicleListResponse)
async def list_vehicles() -> VehicleListResponse:
    return VehicleListResponse(vehicles=list(VEHICLES.values()))


def _validate_osrm_geometry(geometry: Any) -> dict[str, Any]:
    # RouteOption.geometry is typed as GeoJSONLineString, but OSRM returns a dict that
    # should already be GeoJSON if we requested geojson format.
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

    # Base emissions from segment profile
    base_emissions = route_emissions_kg(
        vehicle=vehicle,
        segment_distances_m=seg_d_m,
        segment_durations_s=seg_t_s,
    )

    # Base monetary: distance + time
    base_monetary = (distance_km * vehicle.cost_per_km) + (base_duration_h * vehicle.cost_per_hour)

    # Scenario adjustment: extends cleanly later to real incident simulation
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


@app.post("/pareto", response_model=ParetoResponse)
async def compute_pareto(req: ParetoRequest, osrm: OSRMDep) -> ParetoResponse:
    request_id = str(uuid.uuid4())
    t0 = time.perf_counter()

    try:
        routes = await osrm.fetch_routes(
            origin_lat=req.origin.lat,
            origin_lon=req.origin.lon,
            dest_lat=req.destination.lat,
            dest_lon=req.destination.lon,
            alternatives=True,
        )
    except OSRMError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    routes = routes[: req.max_alternatives]
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

    # Practical demo UX: if the strict Pareto set collapses to a single route while
    # OSRM did return alternatives, the UI appears "broken" (sliders/plot can't change).
    # In that case, include objective-extremes (time/cost/CO2) so the user can still
    # explore trade-offs.
    if len(pareto_sorted) < 2 and len(options) > 1:
        by_time = min(options, key=lambda o: o.metrics.duration_s)
        by_cost = min(options, key=lambda o: o.metrics.monetary_cost)
        by_co2 = min(options, key=lambda o: o.metrics.emissions_kg)
        unique = {r.id: r for r in (by_time, by_cost, by_co2)}
        pareto_sorted = sorted(unique.values(), key=lambda o: o.metrics.duration_s)

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
        routes = await osrm.fetch_routes(
            origin_lat=req.origin.lat,
            origin_lon=req.origin.lon,
            dest_lat=req.destination.lat,
            dest_lon=req.destination.lon,
            alternatives=True,
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
                    alternatives=True,
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
    # Only allow UUID-looking IDs to avoid path traversal / filesystem probing.
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
