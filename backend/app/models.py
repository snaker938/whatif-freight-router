from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .scenario import ScenarioMode
from .vehicles import VehicleProfile


class LatLng(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)


class Waypoint(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    label: str | None = None


class Weights(BaseModel):
    """User preference weights. Backend normalises to avoid UI mistakes."""

    time: float = Field(..., ge=0)
    money: float = Field(..., ge=0)
    co2: float = Field(..., ge=0)

    @model_validator(mode="before")
    @classmethod
    def accept_legacy_aliases(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if "money" not in data:
            for key in ("cost", "monetary_cost"):
                if key in data:
                    data["money"] = data[key]
                    break
        if "co2" not in data:
            for key in ("emissions", "emissions_kg", "co2e"):
                if key in data:
                    data["co2"] = data[key]
                    break
        return data

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
PipelineMode = Literal["legacy", "dccs", "dccs_refc", "voi", "tri_source"]
RouteRefinementPolicy = Literal["dccs", "first_n", "random_n", "corridor_uniform"]
TerrainProfile = Literal["flat", "rolling", "hilly"]
OptimizationMode = Literal["expected_value", "robust"]
FuelType = Literal["diesel", "petrol", "lng", "ev"]
EuroClass = Literal["euro4", "euro5", "euro6"]
WeatherProfile = Literal["clear", "rain", "storm", "snow", "fog"]
IncidentEventType = Literal["dwell", "accident", "closure"]
AmbiguityBudgetBand = Literal["low", "medium", "high", "unspecified"]


class EpsilonConstraints(BaseModel):
    duration_s: float | None = Field(default=None, ge=0.0)
    monetary_cost: float | None = Field(default=None, ge=0.0)
    emissions_kg: float | None = Field(default=None, ge=0.0)


class EmissionsContext(BaseModel):
    fuel_type: FuelType = "diesel"
    euro_class: EuroClass = "euro6"
    ambient_temp_c: float = 15.0


class WeatherImpactConfig(BaseModel):
    enabled: bool = False
    profile: WeatherProfile = "clear"
    intensity: float = Field(default=1.0, ge=0.0, le=2.0)
    apply_incident_uplift: bool = True


class IncidentSimulatorConfig(BaseModel):
    enabled: bool = False
    seed: int | None = None
    dwell_rate_per_100km: float = Field(default=0.8, ge=0.0)
    accident_rate_per_100km: float = Field(default=0.25, ge=0.0)
    closure_rate_per_100km: float = Field(default=0.05, ge=0.0)
    dwell_delay_s: float = Field(default=120.0, ge=0.0)
    accident_delay_s: float = Field(default=480.0, ge=0.0)
    closure_delay_s: float = Field(default=900.0, ge=0.0)
    max_events_per_route: int = Field(default=12, ge=0, le=1000)


class SimulatedIncidentEvent(BaseModel):
    event_id: str
    event_type: IncidentEventType
    segment_index: int = Field(..., ge=0)
    start_offset_s: float = Field(..., ge=0.0)
    delay_s: float = Field(..., ge=0.0)
    source: Literal["synthetic"] = "synthetic"


class TimeWindowConstraints(BaseModel):
    earliest_arrival_utc: datetime | None = None
    latest_arrival_utc: datetime | None = None


class StochasticConfig(BaseModel):
    enabled: bool = False
    seed: int | None = None
    sigma: float = Field(default=0.08, ge=0.0, le=0.5)
    samples: int = Field(default=25, ge=5, le=200)


class AmbiguityContextFields(BaseModel):
    od_ambiguity_index: float | None = Field(default=None, ge=0.0, le=1.0)
    od_ambiguity_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    od_engine_disagreement_prior: float | None = Field(default=None, ge=0.0, le=1.0)
    od_hard_case_prior: float | None = Field(default=None, ge=0.0, le=1.0)
    od_ambiguity_source_count: int | None = Field(default=None, ge=0, le=64)
    od_ambiguity_source_mix: str | None = None
    od_ambiguity_source_mix_count: int | None = Field(default=None, ge=0, le=64)
    od_ambiguity_source_entropy: float | None = Field(default=None, ge=0.0, le=1.0)
    od_ambiguity_support_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    od_ambiguity_prior_strength: float | None = Field(default=None, ge=0.0, le=1.0)
    od_ambiguity_family_density: float | None = Field(default=None, ge=0.0, le=1.0)
    od_ambiguity_margin_pressure: float | None = Field(default=None, ge=0.0, le=1.0)
    od_ambiguity_spread_pressure: float | None = Field(default=None, ge=0.0, le=1.0)
    od_ambiguity_toll_instability: float | None = Field(default=None, ge=0.0, le=1.0)
    od_candidate_path_count: int | None = Field(default=None, ge=0, le=512)
    od_corridor_family_count: int | None = Field(default=None, ge=0, le=128)
    od_objective_spread: float | None = Field(default=None, ge=0.0, le=1.0)
    od_nominal_margin_proxy: float | None = Field(default=None, ge=0.0, le=1.0)
    od_toll_disagreement_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    ambiguity_budget_prior: float | None = Field(default=None, ge=0.0, le=1.0)
    ambiguity_budget_band: AmbiguityBudgetBand | None = None


class GeoJSONLineString(BaseModel):
    type: Literal["LineString"]
    coordinates: list[tuple[float, float]]  # [lon, lat]


class RouteRequest(AmbiguityContextFields):
    origin: LatLng
    destination: LatLng
    waypoints: list[Waypoint] = Field(default_factory=list, max_length=48)
    vehicle_type: str = Field(default="rigid_hgv")
    scenario_mode: ScenarioMode = Field(default=ScenarioMode.NO_SHARING)
    max_alternatives: int = Field(default=24, ge=1, le=48)
    weights: Weights = Field(default_factory=lambda: Weights(time=1, money=0, co2=0))
    cost_toggles: CostToggles = Field(default_factory=CostToggles)
    terrain_profile: TerrainProfile = "flat"
    stochastic: StochasticConfig = Field(default_factory=StochasticConfig)
    optimization_mode: OptimizationMode = "expected_value"
    risk_aversion: float = Field(default=1.0, ge=0.0)
    emissions_context: EmissionsContext = Field(default_factory=EmissionsContext)
    weather: WeatherImpactConfig = Field(default_factory=WeatherImpactConfig)
    incident_simulation: IncidentSimulatorConfig = Field(default_factory=IncidentSimulatorConfig)
    departure_time_utc: datetime | None = None
    pareto_method: ParetoMethod = "dominance"
    epsilon: EpsilonConstraints | None = None
    pipeline_mode: PipelineMode | None = None
    refinement_policy: RouteRefinementPolicy | None = None
    pipeline_seed: int | None = None
    search_budget: int | None = Field(default=None, ge=1, le=128)
    evidence_budget: int | None = Field(default=None, ge=0, le=64)
    cert_world_count: int | None = Field(default=None, ge=10, le=500)
    certificate_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    tau_stop: float | None = Field(default=None, ge=0.0)
    evaluation_lean_mode: bool = False


class ParetoRequest(AmbiguityContextFields):
    origin: LatLng
    destination: LatLng
    waypoints: list[Waypoint] = Field(default_factory=list, max_length=48)
    vehicle_type: str = Field(default="rigid_hgv")
    scenario_mode: ScenarioMode = Field(default=ScenarioMode.NO_SHARING)
    # Default increased for richer strict-frontier candidate exploration.
    max_alternatives: int = Field(default=24, ge=1, le=48)
    weights: Weights = Field(default_factory=lambda: Weights(time=1, money=1, co2=1))
    cost_toggles: CostToggles = Field(default_factory=CostToggles)
    terrain_profile: TerrainProfile = "flat"
    stochastic: StochasticConfig = Field(default_factory=StochasticConfig)
    optimization_mode: OptimizationMode = "expected_value"
    risk_aversion: float = Field(default=1.0, ge=0.0)
    emissions_context: EmissionsContext = Field(default_factory=EmissionsContext)
    weather: WeatherImpactConfig = Field(default_factory=WeatherImpactConfig)
    incident_simulation: IncidentSimulatorConfig = Field(default_factory=IncidentSimulatorConfig)
    departure_time_utc: datetime | None = None
    pareto_method: ParetoMethod = "dominance"
    epsilon: EpsilonConstraints | None = None
    pipeline_mode: PipelineMode | None = None
    pipeline_seed: int | None = None


class ODPair(AmbiguityContextFields):
    origin: LatLng
    destination: LatLng


class BatchParetoRequest(AmbiguityContextFields):
    pairs: list[ODPair] = Field(..., min_length=1, max_length=500)
    waypoints: list[Waypoint] = Field(default_factory=list, max_length=48)
    vehicle_type: str = Field(default="rigid_hgv")
    scenario_mode: ScenarioMode = Field(default=ScenarioMode.NO_SHARING)
    max_alternatives: int = Field(default=24, ge=1, le=48)
    weights: Weights = Field(default_factory=lambda: Weights(time=1, money=1, co2=1))
    cost_toggles: CostToggles = Field(default_factory=CostToggles)
    terrain_profile: TerrainProfile = "flat"
    stochastic: StochasticConfig = Field(default_factory=StochasticConfig)
    optimization_mode: OptimizationMode = "expected_value"
    risk_aversion: float = Field(default=1.0, ge=0.0)
    emissions_context: EmissionsContext = Field(default_factory=EmissionsContext)
    weather: WeatherImpactConfig = Field(default_factory=WeatherImpactConfig)
    incident_simulation: IncidentSimulatorConfig = Field(default_factory=IncidentSimulatorConfig)
    departure_time_utc: datetime | None = None
    pareto_method: ParetoMethod = "dominance"
    epsilon: EpsilonConstraints | None = None
    seed: int | None = None
    toggles: dict[str, bool | int | float | str] = Field(default_factory=dict)
    model_version: str | None = None
    pipeline_mode: PipelineMode | None = None
    pipeline_seed: int | None = None
    search_budget: int | None = Field(default=None, ge=1, le=128)
    evidence_budget: int | None = Field(default=None, ge=0, le=64)
    cert_world_count: int | None = Field(default=None, ge=10, le=500)
    certificate_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    tau_stop: float | None = Field(default=None, ge=0.0)


class BatchCSVImportRequest(AmbiguityContextFields):
    csv_text: str = Field(..., min_length=1)
    waypoints: list[Waypoint] = Field(default_factory=list, max_length=48)
    vehicle_type: str = Field(default="rigid_hgv")
    scenario_mode: ScenarioMode = Field(default=ScenarioMode.NO_SHARING)
    max_alternatives: int = Field(default=24, ge=1, le=48)
    weights: Weights = Field(default_factory=lambda: Weights(time=1, money=1, co2=1))
    cost_toggles: CostToggles = Field(default_factory=CostToggles)
    terrain_profile: TerrainProfile = "flat"
    stochastic: StochasticConfig = Field(default_factory=StochasticConfig)
    optimization_mode: OptimizationMode = "expected_value"
    risk_aversion: float = Field(default=1.0, ge=0.0)
    emissions_context: EmissionsContext = Field(default_factory=EmissionsContext)
    weather: WeatherImpactConfig = Field(default_factory=WeatherImpactConfig)
    incident_simulation: IncidentSimulatorConfig = Field(default_factory=IncidentSimulatorConfig)
    departure_time_utc: datetime | None = None
    pareto_method: ParetoMethod = "dominance"
    epsilon: EpsilonConstraints | None = None
    seed: int | None = None
    toggles: dict[str, bool | int | float | str] = Field(default_factory=dict)
    model_version: str | None = None
    pipeline_mode: PipelineMode | None = None
    pipeline_seed: int | None = None
    search_budget: int | None = Field(default=None, ge=1, le=128)
    evidence_budget: int | None = Field(default=None, ge=0, le=64)
    cert_world_count: int | None = Field(default=None, ge=10, le=500)
    certificate_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    tau_stop: float | None = Field(default=None, ge=0.0)


class RouteMetrics(BaseModel):
    distance_km: float
    duration_s: float
    monetary_cost: float
    emissions_kg: float
    avg_speed_kmh: float
    energy_kwh: float | None = None
    weather_delay_s: float = 0.0
    incident_delay_s: float = 0.0


class TerrainSummaryPayload(BaseModel):
    source: Literal["dem_real", "missing", "unsupported_region"] = "missing"
    coverage_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    sample_spacing_m: float = Field(default=75.0, ge=1.0)
    ascent_m: float = 0.0
    descent_m: float = 0.0
    grade_histogram: dict[str, float] = Field(default_factory=dict)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    fail_closed_applied: bool = False
    version: str = "unknown"


class ScenarioSummary(BaseModel):
    mode: ScenarioMode
    context_key: str = "uk_default|mixed|rigid_hgv|weekday|clear"
    duration_multiplier: float
    incident_rate_multiplier: float
    incident_delay_multiplier: float
    fuel_consumption_multiplier: float
    emissions_multiplier: float
    stochastic_sigma_multiplier: float
    source: str
    version: str
    calibration_basis: str = "empirical"
    as_of_utc: str | None = None
    live_as_of_utc: str | None = None
    live_sources: str | None = None
    live_coverage_overall: float | None = None
    live_traffic_pressure: float | None = None
    live_incident_pressure: float | None = None
    live_weather_pressure: float | None = None
    scenario_edge_scaling_version: str | None = None
    mode_observation_source: str | None = None
    mode_projection_ratio: float | None = None


class EvidenceSourceRecord(BaseModel):
    family: str
    source: str
    active: bool = True
    freshness_timestamp_utc: str | None = None
    max_age_minutes: float | None = None
    signature: str | None = None
    confidence: float | None = None
    coverage_ratio: float | None = None
    fallback_used: bool = False
    fallback_source: str | None = None
    details: dict[str, str | float | int | bool] = Field(default_factory=dict)


class EvidenceProvenance(BaseModel):
    active_families: list[str] = Field(default_factory=list)
    families: list[EvidenceSourceRecord] = Field(default_factory=list)


class DecisionWorldSupportSummary(BaseModel):
    support_strength: float | None = Field(default=None, ge=0.0, le=1.0)
    source_support_strength: float | None = Field(default=None, ge=0.0, le=1.0)
    ambiguity_support_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    source_entropy: float | None = Field(default=None, ge=0.0, le=1.0)
    source_count: int = Field(default=0, ge=0)
    source_mix_count: int = Field(default=0, ge=0)
    family_density: float | None = Field(default=None, ge=0.0, le=1.0)
    margin_pressure: float | None = Field(default=None, ge=0.0, le=1.0)
    provenance_coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    live_family_count: int = Field(default=0, ge=0)
    snapshot_family_count: int = Field(default=0, ge=0)
    model_family_count: int = Field(default=0, ge=0)
    proxy_penalty: float | None = Field(default=None, ge=0.0, le=1.0)
    audit_correction: float | None = Field(default=None, ge=0.0, le=1.0)
    recommended_fidelity: str = "audit_first"
    support_sufficient: bool = False
    notes: list[str] = Field(default_factory=list)


class DecisionFidelitySummary(BaseModel):
    world_bundle_id: str | None = None
    audit_bundle_id: str | None = None
    multi_fidelity_mode: str = "unknown"
    policy: str = "unknown"
    world_count: int = Field(default=0, ge=0)
    unique_world_count: int = Field(default=0, ge=0)
    requested_world_count: int = Field(default=0, ge=0)
    effective_world_count: int = Field(default=0, ge=0)
    route_ids: list[str] = Field(default_factory=list)
    active_families: list[str] = Field(default_factory=list)
    world_kind_weights: dict[str, float] = Field(default_factory=dict)
    family_state_weights: dict[str, dict[str, float]] = Field(default_factory=dict)
    targeted_route_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    stress_world_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    proxy_world_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    refreshed_world_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    world_reuse_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    audited_families: list[str] = Field(default_factory=list)
    proxy_family_count: int = Field(default=0, ge=0)
    audited_world_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    correction_scale: float | None = Field(default=None, ge=0.0, le=1.0)
    correction_penalty: float | None = Field(default=None, ge=0.0, le=1.0)
    recommended_policy: str = "unknown"
    manifest_hash: str | None = None
    notes: list[str] = Field(default_factory=list)


class DecisionCertificationStateSummary(BaseModel):
    winner_id: str | None = None
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    certificate_map: dict[str, float] = Field(default_factory=dict)
    certificate_value: float | None = Field(default=None, ge=0.0, le=1.0)
    certified: bool = False
    certification_basis: str = "pending_runtime_wiring"
    world_bundle_id: str | None = None
    audit_bundle_id: str | None = None
    support_strength: float | None = Field(default=None, ge=0.0, le=1.0)
    winner_confidence: dict[str, Any] = Field(default_factory=dict)
    pairwise_gap_states: dict[str, dict[str, Any]] = Field(default_factory=dict)
    flip_radius: dict[str, Any] | None = None
    decision_region: dict[str, Any] = Field(default_factory=dict)
    certified_set: dict[str, Any] = Field(default_factory=dict)
    abstained: bool = False
    manifest_hash: str | None = None


class RouteCertificationSummary(BaseModel):
    route_id: str
    certificate: float = Field(ge=0.0, le=1.0)
    certified: bool = False
    threshold: float = Field(ge=0.0, le=1.0)
    active_families: list[str] = Field(default_factory=list)
    top_fragility_families: list[str] = Field(default_factory=list)
    top_competitor_route_id: str | None = None
    top_value_of_refresh_family: str | None = None
    ambiguity_context: dict[str, float | int | str | bool | None] | None = None
    world_support_summary: DecisionWorldSupportSummary | None = None
    world_fidelity_summary: DecisionFidelitySummary | None = None
    certification_state_summary: DecisionCertificationStateSummary | None = None


class VoiStopSummary(BaseModel):
    final_route_id: str
    certificate: float = Field(ge=0.0, le=1.0)
    certified: bool = False
    iteration_count: int = Field(ge=0)
    search_budget_used: int = Field(ge=0)
    evidence_budget_used: int = Field(ge=0)
    stop_reason: str
    best_rejected_action: str | None = None
    best_rejected_q: float | None = None
    search_completeness_score: float | None = Field(default=None, ge=0.0, le=1.0)
    search_completeness_gap: float | None = Field(default=None, ge=0.0)
    credible_search_uncertainty: bool | None = None


class PreferenceQuerySummary(BaseModel):
    key: str
    kind: str
    prompt: str
    rationale: str
    target: str | None = None
    options: list[str] = Field(default_factory=list)
    route_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DecisionPreferenceSummary(BaseModel):
    objective_id: Literal["minimum_monetary_cost"] = "minimum_monetary_cost"
    objective_field: Literal["monetary_cost"] = "monetary_cost"
    selector_policy: str = "tri_source_selective_minimum_cost"
    selective: bool = True
    tie_break_order: list[str] = Field(
        default_factory=lambda: ["certificate", "duration_s", "emissions_kg", "route_id"]
    )
    certified_only_required: bool = False
    time_guard_active: bool = False
    vetoed_targets: list[str] = Field(default_factory=list)
    compatible_route_ids: list[str] = Field(default_factory=list)
    certified_route_ids: list[str] = Field(default_factory=list)
    selected_route_id: str | None = None
    selected_certificate: float | None = Field(default=None, ge=0.0, le=1.0)
    stop_reason: str | None = None
    stop_hint_codes: list[str] = Field(default_factory=list)
    suggested_queries: list[PreferenceQuerySummary] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class DecisionSupportSourceRecord(BaseModel):
    source_id: str
    role: str | None = None
    required: bool = False
    present: bool = False
    status: str = "unknown"
    freshness_timestamp_utc: str | None = None
    provenance: str | None = None
    details: dict[str, str | float | int | bool] = Field(default_factory=dict)


class DecisionSupportSummary(BaseModel):
    support_mode: str = "tri_source_selective"
    required_source_count: int = Field(default=3, ge=0)
    observed_source_count: int = Field(default=0, ge=0)
    satisfied: bool = False
    sources: list[DecisionSupportSourceRecord] = Field(default_factory=list)
    source_mix: list[str] = Field(default_factory=list)
    missing_sources: list[str] = Field(default_factory=list)
    source_entropy: float | None = Field(default=None, ge=0.0, le=1.0)
    provenance_mode: str | None = None
    notes: list[str] = Field(default_factory=list)


class CertifiedSetSummary(BaseModel):
    objective_field: Literal["monetary_cost"] = "monetary_cost"
    selected_route_id: str | None = None
    minimum_cost_route_id: str | None = None
    certified_route_ids: list[str] = Field(default_factory=list)
    frontier_route_ids: list[str] = Field(default_factory=list)
    certificate_value: float | None = Field(default=None, ge=0.0, le=1.0)
    certificate_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    certificate_basis: str = "pending_runtime_wiring"
    certified: bool = False
    selective_gate_passed: bool = False


class DecisionAbstentionSummary(BaseModel):
    abstained: bool = False
    reason_code: str | None = None
    message: str | None = None
    blocking_sources: list[str] = Field(default_factory=list)
    retryable: bool = False
    abstention_type: str | None = None
    recommended_action: str | None = None


class DecisionWitnessSummary(BaseModel):
    primary_witness_route_id: str | None = None
    witness_route_ids: list[str] = Field(default_factory=list)
    challenger_route_ids: list[str] = Field(default_factory=list)
    witness_world_count: int | None = Field(default=None, ge=0)
    witness_source_ids: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class DecisionControllerSummary(BaseModel):
    controller_mode: str = "single_pass"
    engaged: bool = False
    iteration_count: int = Field(default=0, ge=0)
    action_count: int = Field(default=0, ge=0)
    stop_reason: str | None = None
    search_budget_used: int = Field(default=0, ge=0)
    evidence_budget_used: int = Field(default=0, ge=0)
    notes: list[str] = Field(default_factory=list)


class DecisionTheoremHookRecord(BaseModel):
    hook_id: str
    artifact_name: str | None = None
    status: str = "planned"
    note: str | None = None


class DecisionTheoremHookSummary(BaseModel):
    hooks: list[DecisionTheoremHookRecord] = Field(default_factory=list)


class DecisionLaneManifest(BaseModel):
    lane_id: str | None = None
    lane_name: str | None = None
    lane_version: str | None = None
    artifact_names: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class DecisionPackage(BaseModel):
    schema_version: str = "0.1.0"
    package_kind: Literal["decision_package"] = "decision_package"
    pipeline_mode: PipelineMode = "tri_source"
    terminal_kind: Literal["certified_singleton", "certified_set", "typed_abstention"] = "certified_singleton"
    selected_route_id: str | None = None
    preference_summary: DecisionPreferenceSummary = Field(default_factory=DecisionPreferenceSummary)
    support_summary: DecisionSupportSummary = Field(default_factory=DecisionSupportSummary)
    world_support_summary: DecisionWorldSupportSummary | None = None
    world_fidelity_summary: DecisionFidelitySummary | None = None
    certification_state_summary: DecisionCertificationStateSummary | None = None
    certified_set_summary: CertifiedSetSummary = Field(default_factory=CertifiedSetSummary)
    abstention_summary: DecisionAbstentionSummary | None = None
    witness_summary: DecisionWitnessSummary | None = None
    controller_summary: DecisionControllerSummary | None = None
    theorem_hook_summary: DecisionTheoremHookSummary | None = None
    lane_manifest: DecisionLaneManifest | None = None
    provenance: dict[str, str | float | int | bool | None] = Field(default_factory=dict)


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
    uncertainty_samples_meta: dict[str, str | float | int | bool] | None = None
    legs: list[dict[str, str | float | int | bool]] | None = None
    toll_confidence: float | None = None
    toll_metadata: dict[str, str | float | int | bool | list[str]] | None = None
    vehicle_profile_id: str | None = None
    vehicle_profile_version: int | None = None
    vehicle_profile_source: str | None = None
    scenario_summary: ScenarioSummary | None = None
    weather_summary: dict[str, float | str | bool] | None = None
    terrain_summary: TerrainSummaryPayload | None = None
    incident_events: list[SimulatedIncidentEvent] = Field(default_factory=list)
    evidence_provenance: EvidenceProvenance | None = None
    certification: RouteCertificationSummary | None = None


class DecisionPackageResponse(BaseModel):
    decision_package: DecisionPackage
    selected: RouteOption | None = None
    candidates: list[RouteOption] = Field(default_factory=list)
    run_id: str | None = None
    pipeline_mode: PipelineMode = "tri_source"
    manifest_endpoint: str | None = None
    artifacts_endpoint: str | None = None
    provenance_endpoint: str | None = None
    selected_certificate: RouteCertificationSummary | None = None
    voi_stop_summary: VoiStopSummary | None = None


class RouteResponse(BaseModel):
    selected: RouteOption
    candidates: list[RouteOption]
    run_id: str | None = None
    pipeline_mode: PipelineMode = "tri_source"
    manifest_endpoint: str | None = None
    artifacts_endpoint: str | None = None
    provenance_endpoint: str | None = None
    decision_package: DecisionPackage | None = None
    selected_certificate: RouteCertificationSummary | None = None
    voi_stop_summary: VoiStopSummary | None = None


class RouteBaselineResponse(BaseModel):
    baseline: RouteOption
    method: str
    compute_ms: float
    provider_mode: str | None = None
    baseline_policy: str | None = None
    asset_manifest_hash: str | None = None
    asset_recorded_at: str | None = None
    asset_freshness_status: str | None = None
    engine_manifest: dict[str, Any] | None = None
    notes: list[str] = Field(default_factory=list)


class ParetoResponse(BaseModel):
    routes: list[RouteOption]
    warnings: list[str] = Field(default_factory=list)
    diagnostics: dict[str, int | bool | float | str] = Field(default_factory=dict)


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


class BatchParetoResponse(BaseModel):
    run_id: str
    results: list[BatchParetoResult]


class ScenarioCompareRequest(BaseModel):
    origin: LatLng
    destination: LatLng
    waypoints: list[Waypoint] = Field(default_factory=list, max_length=48)
    vehicle_type: str = Field(default="rigid_hgv")
    scenario_mode: ScenarioMode | None = None
    weights: Weights = Field(default_factory=lambda: Weights(time=1, money=1, co2=1))
    max_alternatives: int = Field(default=24, ge=1, le=48)
    cost_toggles: CostToggles = Field(default_factory=CostToggles)
    terrain_profile: TerrainProfile = "flat"
    stochastic: StochasticConfig = Field(default_factory=StochasticConfig)
    optimization_mode: OptimizationMode = "expected_value"
    risk_aversion: float = Field(default=1.0, ge=0.0)
    emissions_context: EmissionsContext = Field(default_factory=EmissionsContext)
    weather: WeatherImpactConfig = Field(default_factory=WeatherImpactConfig)
    incident_simulation: IncidentSimulatorConfig = Field(default_factory=IncidentSimulatorConfig)
    departure_time_utc: datetime | None = None
    pareto_method: ParetoMethod = "dominance"
    epsilon: EpsilonConstraints | None = None


class ScenarioCompareResult(BaseModel):
    scenario_mode: ScenarioMode
    selected: RouteOption | None = None
    candidates: list[RouteOption] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None


class ScenarioCompareDelta(BaseModel):
    duration_s_delta: float | None = None
    monetary_cost_delta: float | None = None
    emissions_kg_delta: float | None = None
    duration_s_status: str = "ok"
    monetary_cost_status: str = "ok"
    emissions_kg_status: str = "ok"
    duration_s_reason_code: str | None = None
    monetary_cost_reason_code: str | None = None
    emissions_kg_reason_code: str | None = None
    duration_s_missing_source: str | None = None
    monetary_cost_missing_source: str | None = None
    emissions_kg_missing_source: str | None = None
    duration_s_reason_source: str | None = None
    monetary_cost_reason_source: str | None = None
    emissions_kg_reason_source: str | None = None


class ScenarioCompareResponse(BaseModel):
    run_id: str
    results: list[ScenarioCompareResult]
    deltas: dict[str, ScenarioCompareDelta]
    baseline_mode: ScenarioMode = ScenarioMode.NO_SHARING
    scenario_manifest_endpoint: str
    scenario_signature_endpoint: str


class DepartureOptimizeRequest(BaseModel):
    origin: LatLng
    destination: LatLng
    waypoints: list[Waypoint] = Field(default_factory=list, max_length=48)
    vehicle_type: str = Field(default="rigid_hgv")
    scenario_mode: ScenarioMode = Field(default=ScenarioMode.NO_SHARING)
    weights: Weights = Field(default_factory=lambda: Weights(time=1, money=1, co2=1))
    max_alternatives: int = Field(default=24, ge=1, le=48)
    cost_toggles: CostToggles = Field(default_factory=CostToggles)
    terrain_profile: TerrainProfile = "flat"
    stochastic: StochasticConfig = Field(default_factory=StochasticConfig)
    optimization_mode: OptimizationMode = "expected_value"
    risk_aversion: float = Field(default=1.0, ge=0.0)
    emissions_context: EmissionsContext = Field(default_factory=EmissionsContext)
    weather: WeatherImpactConfig = Field(default_factory=WeatherImpactConfig)
    incident_simulation: IncidentSimulatorConfig = Field(default_factory=IncidentSimulatorConfig)
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
    error: str | None = None


class DutyChainRequest(BaseModel):
    stops: list[DutyChainStop] = Field(..., min_length=2, max_length=50)
    vehicle_type: str = Field(default="rigid_hgv")
    scenario_mode: ScenarioMode = Field(default=ScenarioMode.NO_SHARING)
    weights: Weights = Field(default_factory=lambda: Weights(time=1, money=1, co2=1))
    max_alternatives: int = Field(default=24, ge=1, le=48)
    cost_toggles: CostToggles = Field(default_factory=CostToggles)
    terrain_profile: TerrainProfile = "flat"
    stochastic: StochasticConfig = Field(default_factory=StochasticConfig)
    optimization_mode: OptimizationMode = "expected_value"
    risk_aversion: float = Field(default=1.0, ge=0.0)
    emissions_context: EmissionsContext = Field(default_factory=EmissionsContext)
    weather: WeatherImpactConfig = Field(default_factory=WeatherImpactConfig)
    incident_simulation: IncidentSimulatorConfig = Field(default_factory=IncidentSimulatorConfig)
    departure_time_utc: datetime | None = None
    pareto_method: ParetoMethod = "dominance"
    epsilon: EpsilonConstraints | None = None


class DutyChainResponse(BaseModel):
    legs: list[DutyChainLegResult]
    total_metrics: RouteMetrics
    leg_count: int
    successful_leg_count: int


class CacheSnapshotStats(BaseModel):
    size: int = Field(default=0, ge=0)
    hits: int = Field(default=0, ge=0)
    misses: int = Field(default=0, ge=0)
    evictions: int = Field(default=0, ge=0)
    oversize_rejections: int = Field(default=0, ge=0)
    ttl_s: int = Field(default=0, ge=0)
    max_entries: int = Field(default=0, ge=0)
    estimated_bytes: int = Field(default=0, ge=0)
    max_estimated_bytes: int = Field(default=0, ge=0)
    schema_version: int | None = None
    invalidation_counters: dict[str, int] = Field(default_factory=dict)
    checkpoint_operations: int | None = None
    checkpointed_entries: int | None = None
    restore_operations: int | None = None
    restored_entries: int | None = None


class CacheStatsResponse(BaseModel):
    route_cache: CacheSnapshotStats
    hot_rerun_route_cache_checkpoint: CacheSnapshotStats
    certification_cache: CacheSnapshotStats
    k_raw_cache: CacheSnapshotStats
    route_option_cache: CacheSnapshotStats
    route_state_cache: CacheSnapshotStats
    voi_dccs_cache: CacheSnapshotStats


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
