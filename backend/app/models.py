from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .abstention import AbstentionRecord
from .certified_set import CertifiedSetState
from .certificate_witness import CertificateWitness
from .confidence_sequences import WinnerConfidenceState
from .decision_region import DecisionRegionState
from .flip_radius import FlipRadiusState
from .pairwise_gap_model import PairwiseGapState
from .preference_state import PreferenceState
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
PipelineMode = Literal["legacy", "dccs", "dccs_refc", "voi"]
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


class DecisionPackage(BaseModel):
    """Compatibility wrapper that exposes the certification-native decision shape."""

    model_config = ConfigDict(validate_assignment=True)

    terminal_type: Literal["certified_singleton", "certified_set", "typed_abstention"] = (
        "certified_singleton"
    )
    selected: "RouteOption | None" = None
    candidates: list["RouteOption"] = Field(default_factory=list)
    recommended_route: "RouteOption | None" = None
    certified_set: list["RouteOption"] = Field(default_factory=list)
    abstention: AbstentionRecord | None = None
    frontier_summary: dict[str, Any] = Field(default_factory=dict)
    selected_certificate: "RouteCertificationSummary | None" = None
    certificate_summary: dict[str, Any] | RouteCertificationSummary | None = None
    fixed_weight_certificate_state: dict[str, Any] = Field(default_factory=dict)
    stability_summary: dict[str, Any] = Field(default_factory=dict)
    winner_confidence_state: WinnerConfidenceState | dict[str, Any] | None = None
    pairwise_gap_states: list[PairwiseGapState | dict[str, Any]] = Field(default_factory=list)
    flip_radius_state: FlipRadiusState | dict[str, Any] | None = None
    decision_region_state: DecisionRegionState | dict[str, Any] | None = None
    certificate_witness: CertificateWitness | dict[str, Any] | None = None
    preference_summary: dict[str, Any] = Field(default_factory=dict)
    preference_state: PreferenceState = Field(default_factory=PreferenceState)
    preference_query_trace: dict[str, Any] = Field(default_factory=dict)
    support_summary: dict[str, Any] = Field(default_factory=dict)
    world_support_summary: dict[str, Any] = Field(default_factory=dict)
    abstention_summary: dict[str, Any] = Field(default_factory=dict)
    certified_set_summary: dict[str, Any] = Field(default_factory=dict)
    action_trace_summary: dict[str, Any] = Field(default_factory=dict)
    witness_summary: dict[str, Any] = Field(default_factory=dict)
    artifact_pointers: dict[str, str] = Field(default_factory=dict)
    selected_certificate_basis: str | None = None
    run_id: str | None = None
    pipeline_mode: PipelineMode | None = None
    manifest_endpoint: str | None = None
    artifacts_endpoint: str | None = None
    provenance_endpoint: str | None = None
    voi_stop_summary: "VoiStopSummary | None" = None


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


def _build_preference_summary(
    *,
    preference_state: PreferenceState | Mapping[str, Any] | None,
    selected_certificate_basis: str | None = None,
    pipeline_mode: str | None = None,
) -> dict[str, Any]:
    if isinstance(preference_state, PreferenceState):
        state_payload = preference_state.model_dump(mode="json")
    elif isinstance(preference_state, Mapping):
        state_payload = dict(preference_state)
    else:
        state_payload = {}

    compatible_set_summary = state_payload.get("compatible_set_summary")
    compatible_set_summary = (
        dict(compatible_set_summary) if isinstance(compatible_set_summary, Mapping) else {}
    )
    derived_invariants = state_payload.get("derived_invariants")
    derived_invariants = dict(derived_invariants) if isinstance(derived_invariants, Mapping) else {}
    contradiction_record = state_payload.get("contradiction_record")
    contradiction_record = dict(contradiction_record) if isinstance(contradiction_record, Mapping) else {}

    shrinkage_trace = state_payload.get("shrinkage_trace")
    shrinkage_rows = list(shrinkage_trace) if isinstance(shrinkage_trace, list) else []
    last_trace = shrinkage_rows[-1] if shrinkage_rows and isinstance(shrinkage_rows[-1], Mapping) else {}
    targeted_challenger_route_id = last_trace.get("target_route_id")
    query_count = int(state_payload.get("query_count", 0) or 0)
    preference_irrelevance_proven = bool(state_payload.get("preference_irrelevance_proven", False))
    compatible_set_size = int(compatible_set_summary.get("compatible_set_size", 0) or 0)
    support_flag = compatible_set_summary.get("support_flag")
    if query_count > 0:
        no_query_reason = None
    elif contradiction_record.get("contradiction_detected"):
        no_query_reason = "preference_contradiction_detected"
    elif support_flag is False:
        no_query_reason = "preference_support_insufficient"
    elif preference_irrelevance_proven:
        no_query_reason = "preference_irrelevance_proven"
    elif compatible_set_size <= 1:
        no_query_reason = "singleton_frontier"
    else:
        no_query_reason = state_payload.get("no_query_reason") or "no_preference_query_issued"
    query_selection_reason = last_trace.get("query_reason") or no_query_reason

    summary = {
        "terminal_type": state_payload.get("terminal_type"),
        "query_count": query_count,
        "compatible_set_summary": compatible_set_summary,
        "derived_invariants": derived_invariants,
        "contradiction_record": contradiction_record,
        "preference_irrelevance_proven": preference_irrelevance_proven,
        "no_query_reason": no_query_reason,
        "no_preference_query_reason": no_query_reason,
        "targeted_challenger_route_id": targeted_challenger_route_id,
        "query_selection_reason": query_selection_reason,
    }
    if selected_certificate_basis is not None:
        summary["selected_certificate_basis"] = selected_certificate_basis
    if pipeline_mode is not None:
        summary["pipeline_mode"] = pipeline_mode
    return summary


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


def _build_certified_set_summary(
    *,
    selected: RouteOption,
    candidates: list[RouteOption],
    certified_set: list[RouteOption],
    selected_certificate: RouteCertificationSummary | None,
    support_summary: dict[str, Any],
    terminal_type: str,
    base_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = dict(base_summary or {})
    member_route_ids = [route.id for route in certified_set]
    candidate_route_ids = [route.id for route in candidates]
    excluded_route_ids = [route.id for route in candidates if route.id not in member_route_ids]
    support_flag_value = support_summary.get("support_flag") if support_summary else None
    if support_flag_value is None and support_summary:
        support_flag_value = support_summary.get("supported")
    support_flag = bool(support_flag_value) if support_flag_value is not None else False
    witness = {
        "route_id": selected.id,
        "active_challenger_ids": excluded_route_ids[:1],
        "support_flag": support_flag,
    }
    payload.update(CertifiedSetState(
        member_route_ids=member_route_ids,
        excluded_route_ids=excluded_route_ids,
        exclusion_basis=["certificate_threshold", "frontier_selection"],
        certified=bool(
            selected_certificate is not None and selected_certificate.certified and bool(member_route_ids)
        ),
        threshold=float(selected_certificate.threshold) if selected_certificate is not None else 0.0,
        support_flag=support_flag,
        set_size=len(member_route_ids),
        witness=witness,
    ).as_dict())

    if terminal_type == "certified_set":
        payload.update(
            {
                "certified": bool(member_route_ids),
                "set_size": len(member_route_ids),
                "terminal_type": terminal_type,
                "not_applicable_reason": None,
            }
        )
        return payload

    normalized_excluded_route_ids = list(candidate_route_ids)
    if terminal_type == "certified_singleton":
        normalized_excluded_route_ids = [
            route_id for route_id in candidate_route_ids if route_id != selected.id
        ]

    payload.update(
        {
            "member_route_ids": [],
            "excluded_route_ids": normalized_excluded_route_ids,
            "certified": False,
            "set_size": 0,
            "terminal_type": terminal_type,
            "not_applicable_reason": (
                "singleton_terminal" if terminal_type == "certified_singleton" else "abstention_terminal"
            ),
            "selected_route_id": selected.id,
            "support_flag": support_flag,
        }
    )
    payload.setdefault("exclusion_basis", [])
    witness_payload = payload.get("witness")
    witness_payload = dict(witness_payload) if isinstance(witness_payload, Mapping) else {}
    witness_payload.setdefault("route_id", selected.id)
    payload["witness"] = witness_payload
    return payload


def _merge_support_summary(
    *,
    support_summary: Mapping[str, Any] | None,
    world_support_summary: Mapping[str, Any] | None,
    abstention: AbstentionRecord | None,
    selected_certificate: RouteCertificationSummary | None,
) -> dict[str, Any]:
    summary = dict(support_summary or {})
    world_summary = dict(world_support_summary or {})
    support_state = world_summary.get("support_state")
    support_state_map = dict(support_state) if isinstance(support_state, Mapping) else {}
    world_bundle_summary = world_summary.get("world_bundle_summary")
    world_bundle_summary = (
        dict(world_bundle_summary) if isinstance(world_bundle_summary, Mapping) else {}
    )
    risk_summary = world_summary.get("risk_summary")
    risk_summary = dict(risk_summary) if isinstance(risk_summary, Mapping) else {}

    support_flag = summary.get("support_flag")
    if support_flag is None:
        support_flag = summary.get("supported")
    if support_flag is None:
        support_flag = world_summary.get("support_flag")
    if support_flag is None:
        support_flag = support_state_map.get("support_flag")
    if support_flag is None and abstention is not None and abstention.support_flag is not None:
        support_flag = abstention.support_flag
    if support_flag is None and selected_certificate is not None:
        support_flag = bool(selected_certificate.certified)
    if support_flag is not None:
        normalized_flag = bool(support_flag)
        summary["support_flag"] = normalized_flag
        summary.setdefault("supported", normalized_flag)

    support_reason = summary.get("support_reason")
    if support_reason is None or str(support_reason).strip() == "":
        support_reason = world_summary.get("support_reason")
    if support_reason is None or str(support_reason).strip() == "":
        support_reason = support_state_map.get("out_of_support_reason")
    if (
        support_reason is None or str(support_reason).strip() == ""
    ) and abstention is not None and isinstance(abstention.detail, Mapping):
        support_reason = abstention.detail.get("support_reason")
    if support_reason is not None and str(support_reason).strip():
        summary["support_reason"] = str(support_reason).strip()

    for key in ("world_count", "unique_world_count", "world_reuse_rate", "calibration_bin", "support_bin"):
        if summary.get(key) is not None:
            continue
        value = world_summary.get(key)
        if value is None:
            value = support_state_map.get(key)
        if value is not None:
            summary[key] = value

    if not summary.get("active_families"):
        active_families = world_summary.get("active_families")
        if not isinstance(active_families, list) and selected_certificate is not None:
            active_families = list(selected_certificate.active_families)
        if isinstance(active_families, list):
            summary["active_families"] = list(active_families)

    multi_fidelity_summary = summary.get("multi_fidelity_summary")
    if not isinstance(multi_fidelity_summary, Mapping):
        multi_fidelity_summary = world_summary.get("multi_fidelity_summary")
    if not isinstance(multi_fidelity_summary, Mapping):
        multi_fidelity_summary = world_bundle_summary.get("multi_fidelity_summary")
    if not isinstance(multi_fidelity_summary, Mapping):
        multi_fidelity_summary = risk_summary.get("multi_fidelity_summary")
    if isinstance(multi_fidelity_summary, Mapping):
        compact_multi_fidelity = dict(multi_fidelity_summary)
        summary["multi_fidelity_summary"] = compact_multi_fidelity
        for key in (
            "proxy_world_count",
            "audit_world_count",
            "proxy_bias_model_version",
            "audit_propensity_version",
            "proxy_correction_active",
            "multi_fidelity_certificate_basis",
            "proxy_only_fraction",
            "audit_correction_mass",
            "positivity_diagnostics",
        ):
            if summary.get(key) is None and compact_multi_fidelity.get(key) is not None:
                summary[key] = compact_multi_fidelity.get(key)

    return summary


def _build_abstention_summary(
    *,
    abstention: AbstentionRecord | None,
    abstention_summary: Mapping[str, Any] | None,
    terminal_type: str,
) -> dict[str, Any]:
    summary = dict(abstention_summary or {})
    if abstention is None:
        summary.setdefault("reason_code", None)
        summary.setdefault("message", None)
        summary["terminal_type"] = terminal_type
        summary["has_typed_abstention"] = False
        return summary

    detail = summary.get("detail")
    detail_summary = dict(detail) if isinstance(detail, Mapping) else {}
    detail_summary.update(dict(abstention.detail))
    summary.update(
        {
            "reason_code": abstention.reason_code,
            "message": abstention.message,
            "terminal_type": abstention.terminal_type,
            "has_typed_abstention": True,
            "detail": detail_summary,
        }
    )
    if abstention.support_flag is not None:
        summary["support_flag"] = abstention.support_flag
    if abstention.evidence_family is not None:
        summary["evidence_family"] = abstention.evidence_family
    if abstention.budget_channel is not None:
        summary["budget_channel"] = abstention.budget_channel
    if abstention.model_assumption is not None:
        summary["model_assumption"] = abstention.model_assumption
    return summary


class RouteResponse(DecisionPackage):
    selected: RouteOption
    candidates: list[RouteOption]
    run_id: str | None = None
    pipeline_mode: PipelineMode = "legacy"
    manifest_endpoint: str | None = None
    artifacts_endpoint: str | None = None
    provenance_endpoint: str | None = None
    selected_certificate: RouteCertificationSummary | None = None
    voi_stop_summary: VoiStopSummary | None = None

    @model_validator(mode="after")
    def _sync_decision_package(self) -> "RouteResponse":
        if self.recommended_route is None:
            object.__setattr__(self, "recommended_route", self.selected)
        if self.abstention is None and not self.certified_set:
            object.__setattr__(self, "certified_set", [self.selected])
        elif self.abstention is not None and self.certified_set:
            object.__setattr__(self, "certified_set", [])
        if self.selected_certificate is not None and self.certificate_summary is None:
            object.__setattr__(self, "certificate_summary", self.selected_certificate)
        if self.abstention is not None:
            object.__setattr__(self, "terminal_type", "typed_abstention")
        elif len(self.certified_set) > 1:
            object.__setattr__(self, "terminal_type", "certified_set")
        else:
            object.__setattr__(self, "terminal_type", "certified_singleton")
        object.__setattr__(
            self,
            "support_summary",
            _merge_support_summary(
                support_summary=self.support_summary,
                world_support_summary=self.world_support_summary,
                abstention=self.abstention,
                selected_certificate=self.selected_certificate,
            ),
        )
        if not self.frontier_summary:
            object.__setattr__(self, "frontier_summary", {
                "candidate_count": len(self.candidates),
                "selected_route_id": self.selected.id,
            })
        if not self.stability_summary and self.selected_certificate is not None:
            object.__setattr__(self, "stability_summary", {
                "certificate": self.selected_certificate.certificate,
                "threshold": self.selected_certificate.threshold,
            })
        if not self.preference_summary:
            object.__setattr__(
                self,
                "preference_summary",
                _build_preference_summary(
                    preference_state=self.preference_state,
                    selected_certificate_basis=self.selected_certificate_basis,
                    pipeline_mode=self.pipeline_mode,
                ),
            )
        object.__setattr__(
            self,
            "abstention_summary",
            _build_abstention_summary(
                abstention=self.abstention,
                abstention_summary=self.abstention_summary,
                terminal_type=self.terminal_type,
            ),
        )
        if self.terminal_type != "certified_set" or not self.certified_set_summary:
            object.__setattr__(
                self,
                "certified_set_summary",
                _build_certified_set_summary(
                    selected=self.selected,
                    candidates=self.candidates,
                    certified_set=list(self.certified_set),
                    selected_certificate=self.selected_certificate,
                    support_summary=self.support_summary,
                    terminal_type=self.terminal_type,
                    base_summary=self.certified_set_summary,
                ),
            )
        if not self.action_trace_summary:
            object.__setattr__(self, "action_trace_summary", {
                "pipeline_mode": self.pipeline_mode,
                "selected_candidate_count": len(self.candidates),
            })
        if not self.witness_summary:
            object.__setattr__(self, "witness_summary", {
                "route_id": self.selected.id,
                "selected_certificate_basis": self.selected_certificate_basis,
            })
        if not self.artifact_pointers:
            object.__setattr__(self, "artifact_pointers", {
                "manifest_endpoint": self.manifest_endpoint or "",
                "artifacts_endpoint": self.artifacts_endpoint or "",
                "provenance_endpoint": self.provenance_endpoint or "",
            })
        if self.selected_certificate_basis is None and self.selected_certificate is not None:
            object.__setattr__(self, "selected_certificate_basis", "selected_certificate")
        if not self.preference_query_trace:
            object.__setattr__(
                self,
                "preference_query_trace",
                {
                    "schema_version": "preference-query-trace-v1",
                    "selected_route_id": self.selected.id,
                    "selected_certificate_basis": self.selected_certificate_basis
                    or ("selected_certificate" if self.selected_certificate is not None else "empirical"),
                    "terminal_type": self.preference_state.terminal_type,
                    "query_count": int(self.preference_state.query_count),
                    "query_history": [
                        query.model_dump(mode="json") for query in self.preference_state.query_history
                    ],
                    "shrinkage_trace": [
                        trace.model_dump(mode="json") for trace in self.preference_state.shrinkage_trace
                    ],
                    "compatible_set_summary": self.preference_state.compatible_set_summary.model_dump(mode="json"),
                    "derived_invariants": dict(self.preference_state.derived_invariants),
                    "contradiction_record": self.preference_state.contradiction_record.model_dump(mode="json"),
                    "preference_irrelevance_proven": bool(self.preference_state.preference_irrelevance_proven),
                    "no_query_reason": self.preference_state.no_query_reason,
                    "no_preference_query_reason": self.preference_state.no_query_reason,
                    "targeted_challenger_route_id": None,
                    "query_selection_reason": self.preference_state.no_query_reason,
                    "provenance": {
                        "selected_route_id": self.selected.id,
                        "pipeline_mode": self.pipeline_mode,
                    },
                },
            )
        return self


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
