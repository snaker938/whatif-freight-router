from __future__ import annotations

from datetime import datetime
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


class CostToggles(BaseModel):
    """Optional cost-model controls with neutral defaults."""

    use_tolls: bool = True
    fuel_price_multiplier: float = Field(default=1.0, ge=0.0)
    carbon_price_per_kg: float = Field(default=0.0, ge=0.0)
    toll_cost_per_km: float = Field(default=0.0, ge=0.0)


ParetoMethod = Literal["dominance", "epsilon_constraint"]
TerrainProfile = Literal["flat", "rolling", "hilly"]
OptimizationMode = Literal["expected_value", "robust"]
FuelType = Literal["diesel", "petrol", "lng", "ev"]
EuroClass = Literal["euro4", "euro5", "euro6"]


class EpsilonConstraints(BaseModel):
    duration_s: float | None = Field(default=None, ge=0.0)
    monetary_cost: float | None = Field(default=None, ge=0.0)
    emissions_kg: float | None = Field(default=None, ge=0.0)


class EmissionsContext(BaseModel):
    fuel_type: FuelType = "diesel"
    euro_class: EuroClass = "euro6"
    ambient_temp_c: float = 15.0


class TimeWindowConstraints(BaseModel):
    earliest_arrival_utc: datetime | None = None
    latest_arrival_utc: datetime | None = None


class StochasticConfig(BaseModel):
    enabled: bool = False
    seed: int | None = None
    sigma: float = Field(default=0.08, ge=0.0, le=0.5)
    samples: int = Field(default=25, ge=5, le=200)


class GeoJSONLineString(BaseModel):
    type: Literal["LineString"]
    coordinates: list[tuple[float, float]]  # [lon, lat]


class RouteRequest(BaseModel):
    origin: LatLng
    destination: LatLng
    vehicle_type: str = Field(default="rigid_hgv")
    scenario_mode: ScenarioMode = Field(default=ScenarioMode.NO_SHARING)
    weights: Weights = Field(default_factory=lambda: Weights(time=1, money=0, co2=0))
    cost_toggles: CostToggles = Field(default_factory=CostToggles)
    terrain_profile: TerrainProfile = "flat"
    stochastic: StochasticConfig = Field(default_factory=StochasticConfig)
    optimization_mode: OptimizationMode = "expected_value"
    risk_aversion: float = Field(default=1.0, ge=0.0)
    emissions_context: EmissionsContext = Field(default_factory=EmissionsContext)
    departure_time_utc: datetime | None = None
    pareto_method: ParetoMethod = "dominance"
    epsilon: EpsilonConstraints | None = None


class ParetoRequest(BaseModel):
    origin: LatLng
    destination: LatLng
    vehicle_type: str = Field(default="rigid_hgv")
    scenario_mode: ScenarioMode = Field(default=ScenarioMode.NO_SHARING)
    # v0 default bumped to 5 so the UI more often has multiple routes to compare.
    # (OSRM will still cap the number of alternatives it can produce.)
    max_alternatives: int = Field(default=5, ge=1, le=5)
    cost_toggles: CostToggles = Field(default_factory=CostToggles)
    terrain_profile: TerrainProfile = "flat"
    stochastic: StochasticConfig = Field(default_factory=StochasticConfig)
    optimization_mode: OptimizationMode = "expected_value"
    risk_aversion: float = Field(default=1.0, ge=0.0)
    emissions_context: EmissionsContext = Field(default_factory=EmissionsContext)
    departure_time_utc: datetime | None = None
    pareto_method: ParetoMethod = "dominance"
    epsilon: EpsilonConstraints | None = None


class ODPair(BaseModel):
    origin: LatLng
    destination: LatLng


class BatchParetoRequest(BaseModel):
    pairs: list[ODPair] = Field(..., min_length=1, max_length=500)
    vehicle_type: str = Field(default="rigid_hgv")
    scenario_mode: ScenarioMode = Field(default=ScenarioMode.NO_SHARING)
    max_alternatives: int = Field(default=5, ge=1, le=5)
    cost_toggles: CostToggles = Field(default_factory=CostToggles)
    terrain_profile: TerrainProfile = "flat"
    stochastic: StochasticConfig = Field(default_factory=StochasticConfig)
    optimization_mode: OptimizationMode = "expected_value"
    risk_aversion: float = Field(default=1.0, ge=0.0)
    emissions_context: EmissionsContext = Field(default_factory=EmissionsContext)
    departure_time_utc: datetime | None = None
    pareto_method: ParetoMethod = "dominance"
    epsilon: EpsilonConstraints | None = None
    seed: int | None = None
    toggles: dict[str, bool | int | float | str] = Field(default_factory=dict)
    model_version: str | None = None


class BatchCSVImportRequest(BaseModel):
    csv_text: str = Field(..., min_length=1)
    vehicle_type: str = Field(default="rigid_hgv")
    scenario_mode: ScenarioMode = Field(default=ScenarioMode.NO_SHARING)
    max_alternatives: int = Field(default=5, ge=1, le=5)
    cost_toggles: CostToggles = Field(default_factory=CostToggles)
    terrain_profile: TerrainProfile = "flat"
    stochastic: StochasticConfig = Field(default_factory=StochasticConfig)
    optimization_mode: OptimizationMode = "expected_value"
    risk_aversion: float = Field(default=1.0, ge=0.0)
    emissions_context: EmissionsContext = Field(default_factory=EmissionsContext)
    departure_time_utc: datetime | None = None
    pareto_method: ParetoMethod = "dominance"
    epsilon: EpsilonConstraints | None = None
    seed: int | None = None
    toggles: dict[str, bool | int | float | str] = Field(default_factory=dict)
    model_version: str | None = None


class RouteMetrics(BaseModel):
    distance_km: float
    duration_s: float
    monetary_cost: float
    emissions_kg: float
    avg_speed_kmh: float
    energy_kwh: float | None = None


class RouteOption(BaseModel):
    id: str
    geometry: GeoJSONLineString
    metrics: RouteMetrics
    knee_score: float | None = None
    is_knee: bool = False
    eta_explanations: list[str] = Field(default_factory=list)
    eta_timeline: list[dict[str, float | str]] = Field(default_factory=list)
    segment_breakdown: list[dict[str, float | int]] = Field(default_factory=list)
    counterfactuals: list[dict[str, str | float | bool]] = Field(default_factory=list)
    uncertainty: dict[str, float] | None = None


class RouteResponse(BaseModel):
    selected: RouteOption
    candidates: list[RouteOption]


class ParetoResponse(BaseModel):
    routes: list[RouteOption]


class VehicleListResponse(BaseModel):
    vehicles: list[VehicleProfile]


class CustomVehicleListResponse(BaseModel):
    vehicles: list[VehicleProfile]


class VehicleMutationResponse(BaseModel):
    vehicle: VehicleProfile


class VehicleDeleteResponse(BaseModel):
    vehicle_id: str
    deleted: bool


class SignatureVerificationRequest(BaseModel):
    payload: dict[str, object] | list[object] | str
    signature: str = Field(..., min_length=1)
    secret: str | None = None


class SignatureVerificationResponse(BaseModel):
    valid: bool
    algorithm: str
    signature: str
    expected_signature: str


class BatchParetoResult(BaseModel):
    origin: LatLng
    destination: LatLng
    routes: list[RouteOption] = Field(default_factory=list)
    error: str | None = None
    fallback_used: bool = False


class BatchParetoResponse(BaseModel):
    run_id: str
    results: list[BatchParetoResult]


class ScenarioCompareRequest(BaseModel):
    origin: LatLng
    destination: LatLng
    vehicle_type: str = Field(default="rigid_hgv")
    scenario_mode: ScenarioMode | None = None
    weights: Weights = Field(default_factory=lambda: Weights(time=1, money=1, co2=1))
    max_alternatives: int = Field(default=5, ge=1, le=5)
    cost_toggles: CostToggles = Field(default_factory=CostToggles)
    terrain_profile: TerrainProfile = "flat"
    stochastic: StochasticConfig = Field(default_factory=StochasticConfig)
    optimization_mode: OptimizationMode = "expected_value"
    risk_aversion: float = Field(default=1.0, ge=0.0)
    emissions_context: EmissionsContext = Field(default_factory=EmissionsContext)
    departure_time_utc: datetime | None = None
    pareto_method: ParetoMethod = "dominance"
    epsilon: EpsilonConstraints | None = None


class ScenarioCompareResult(BaseModel):
    scenario_mode: ScenarioMode
    selected: RouteOption | None = None
    candidates: list[RouteOption] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    fallback_used: bool = False
    error: str | None = None


class ScenarioCompareResponse(BaseModel):
    run_id: str
    results: list[ScenarioCompareResult]
    deltas: dict[str, dict[str, float]]
    scenario_manifest_endpoint: str
    scenario_signature_endpoint: str


class DepartureOptimizeRequest(BaseModel):
    origin: LatLng
    destination: LatLng
    vehicle_type: str = Field(default="rigid_hgv")
    scenario_mode: ScenarioMode = Field(default=ScenarioMode.NO_SHARING)
    weights: Weights = Field(default_factory=lambda: Weights(time=1, money=1, co2=1))
    max_alternatives: int = Field(default=5, ge=1, le=5)
    cost_toggles: CostToggles = Field(default_factory=CostToggles)
    terrain_profile: TerrainProfile = "flat"
    stochastic: StochasticConfig = Field(default_factory=StochasticConfig)
    optimization_mode: OptimizationMode = "expected_value"
    risk_aversion: float = Field(default=1.0, ge=0.0)
    emissions_context: EmissionsContext = Field(default_factory=EmissionsContext)
    pareto_method: ParetoMethod = "dominance"
    epsilon: EpsilonConstraints | None = None
    time_window: TimeWindowConstraints | None = None
    window_start_utc: datetime
    window_end_utc: datetime
    step_minutes: int = Field(default=60, ge=5, le=720)


class DepartureOptimizeCandidate(BaseModel):
    departure_time_utc: str
    selected: RouteOption
    score: float
    warning_count: int = 0
    fallback_used: bool = False


class DepartureOptimizeResponse(BaseModel):
    best: DepartureOptimizeCandidate | None
    candidates: list[DepartureOptimizeCandidate]
    evaluated_count: int


class DutyChainStop(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    label: str | None = None


class DutyChainLegResult(BaseModel):
    leg_index: int
    origin: DutyChainStop
    destination: DutyChainStop
    selected: RouteOption | None = None
    candidates: list[RouteOption] = Field(default_factory=list)
    warning_count: int = 0
    fallback_used: bool = False
    error: str | None = None


class DutyChainRequest(BaseModel):
    stops: list[DutyChainStop] = Field(..., min_length=2, max_length=50)
    vehicle_type: str = Field(default="rigid_hgv")
    scenario_mode: ScenarioMode = Field(default=ScenarioMode.NO_SHARING)
    weights: Weights = Field(default_factory=lambda: Weights(time=1, money=1, co2=1))
    max_alternatives: int = Field(default=5, ge=1, le=5)
    cost_toggles: CostToggles = Field(default_factory=CostToggles)
    terrain_profile: TerrainProfile = "flat"
    stochastic: StochasticConfig = Field(default_factory=StochasticConfig)
    optimization_mode: OptimizationMode = "expected_value"
    risk_aversion: float = Field(default=1.0, ge=0.0)
    emissions_context: EmissionsContext = Field(default_factory=EmissionsContext)
    departure_time_utc: datetime | None = None
    pareto_method: ParetoMethod = "dominance"
    epsilon: EpsilonConstraints | None = None


class DutyChainResponse(BaseModel):
    legs: list[DutyChainLegResult]
    total_metrics: RouteMetrics
    leg_count: int
    successful_leg_count: int


class OracleFeedCheckInput(BaseModel):
    source: str = Field(..., min_length=1, max_length=120)
    schema_valid: bool
    signature_valid: bool | None = None
    freshness_s: float | None = Field(default=None, ge=0.0)
    latency_ms: float | None = Field(default=None, ge=0.0)
    record_count: int | None = Field(default=None, ge=0)
    observed_at_utc: datetime | None = None
    error: str | None = Field(default=None, max_length=500)


class OracleFeedCheckRecord(BaseModel):
    check_id: str
    source: str
    schema_valid: bool
    signature_valid: bool | None = None
    freshness_s: float | None = None
    latency_ms: float | None = None
    record_count: int | None = None
    observed_at_utc: str | None = None
    error: str | None = None
    passed: bool
    ingested_at_utc: str


class OracleQualitySourceSummary(BaseModel):
    source: str
    check_count: int
    pass_rate: float
    schema_failures: int
    signature_failures: int
    stale_count: int
    avg_latency_ms: float | None = None
    last_observed_at_utc: str | None = None


class OracleQualityDashboardResponse(BaseModel):
    total_checks: int
    source_count: int
    stale_threshold_s: float
    sources: list[OracleQualitySourceSummary]
    updated_at_utc: str


class ExperimentBundleInput(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=500)
    request: ScenarioCompareRequest


class ExperimentBundle(BaseModel):
    id: str
    name: str
    description: str | None = None
    request: ScenarioCompareRequest
    created_at: str
    updated_at: str


class ExperimentListResponse(BaseModel):
    experiments: list[ExperimentBundle]


class ExperimentCompareRequest(BaseModel):
    overrides: dict[str, object] = Field(default_factory=dict)
