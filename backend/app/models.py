from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .scenario import ScenarioMode
from .vehicles import VehicleProfile


class LatLng(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)


class Weights(BaseModel):
    """User preference weights. Backend normalises to avoid UI mistakes."""

    time: float = Field(..., ge=0)
    money: float = Field(..., ge=0)
    co2: float = Field(..., ge=0)

    @field_validator("time", "money", "co2")
    @classmethod
    def finite(cls, v: float) -> float:
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError("weight must be finite")
        return v


class GeoJSONLineString(BaseModel):
    type: Literal["LineString"]
    coordinates: list[tuple[float, float]]  # [lon, lat]


class RouteRequest(BaseModel):
    origin: LatLng
    destination: LatLng
    vehicle_type: str = Field(default="rigid_hgv")
    scenario_mode: ScenarioMode = Field(default=ScenarioMode.NO_SHARING)
    weights: Weights = Field(default_factory=lambda: Weights(time=1, money=0, co2=0))


class ParetoRequest(BaseModel):
    origin: LatLng
    destination: LatLng
    vehicle_type: str = Field(default="rigid_hgv")
    scenario_mode: ScenarioMode = Field(default=ScenarioMode.NO_SHARING)
    # v0 default bumped to 5 so the UI more often has multiple routes to compare.
    # (OSRM will still cap the number of alternatives it can produce.)
    max_alternatives: int = Field(default=5, ge=1, le=5)


class ODPair(BaseModel):
    origin: LatLng
    destination: LatLng


class BatchParetoRequest(BaseModel):
    pairs: list[ODPair] = Field(..., min_length=1, max_length=500)
    vehicle_type: str = Field(default="rigid_hgv")
    scenario_mode: ScenarioMode = Field(default=ScenarioMode.NO_SHARING)
    max_alternatives: int = Field(default=5, ge=1, le=5)


class RouteMetrics(BaseModel):
    distance_km: float
    duration_s: float
    monetary_cost: float
    emissions_kg: float
    avg_speed_kmh: float


class RouteOption(BaseModel):
    id: str
    geometry: GeoJSONLineString
    metrics: RouteMetrics


class RouteResponse(BaseModel):
    selected: RouteOption
    candidates: list[RouteOption]


class ParetoResponse(BaseModel):
    routes: list[RouteOption]


class VehicleListResponse(BaseModel):
    vehicles: list[VehicleProfile]


class BatchParetoResult(BaseModel):
    origin: LatLng
    destination: LatLng
    routes: list[RouteOption] = Field(default_factory=list)
    error: str | None = None


class BatchParetoResponse(BaseModel):
    run_id: str
    results: list[BatchParetoResult]
