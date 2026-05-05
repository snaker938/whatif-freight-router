export type ScenarioMode = 'no_sharing' | 'partial_sharing' | 'full_sharing';
export type ParetoMethod = 'dominance' | 'epsilon_constraint';
export type PipelineMode = 'legacy' | 'dccs' | 'dccs_refc' | 'voi';
export type TerrainProfile = 'flat' | 'rolling' | 'hilly';
export type OptimizationMode = 'expected_value' | 'robust';
export type ComputeMode = 'pareto_stream' | 'pareto_json' | 'route_single';
export type FuelType = 'diesel' | 'petrol' | 'lng' | 'ev';
export type EuroClass = 'euro4' | 'euro5' | 'euro6';
export type WeatherProfile = 'clear' | 'rain' | 'storm' | 'snow' | 'fog';

export type CostToggles = {
  use_tolls: boolean;
  fuel_price_multiplier: number;
  carbon_price_per_kg: number;
  toll_cost_per_km: number;
};

export type EpsilonConstraints = {
  duration_s?: number;
  monetary_cost?: number;
  emissions_kg?: number;
};

export type StochasticConfig = {
  enabled?: boolean;
  seed?: number | null;
  sigma?: number;
  samples?: number;
};

export type EmissionsContext = {
  fuel_type?: FuelType;
  euro_class?: EuroClass;
  ambient_temp_c?: number;
};

export type WeatherImpactConfig = {
  enabled?: boolean;
  profile?: WeatherProfile;
  intensity?: number;
  apply_incident_uplift?: boolean;
};

export type IncidentSimulatorConfig = {
  enabled?: boolean;
  seed?: number | null;
  dwell_rate_per_100km?: number;
  accident_rate_per_100km?: number;
  closure_rate_per_100km?: number;
  dwell_delay_s?: number;
  accident_delay_s?: number;
  closure_delay_s?: number;
  max_events_per_route?: number;
};

export type TimeWindowConstraints = {
  earliest_arrival_utc?: string;
  latest_arrival_utc?: string;
};

export type LatLng = { lat: number; lon: number };
export type Waypoint = {
  lat: number;
  lon: number;
  label?: string | null;
};
export type PinNodeKind = 'origin' | 'destination' | 'stop';
export type PinSelectionId = 'origin' | 'destination' | 'stop-1';

export type ManagedStop = {
  id: 'stop-1';
  lat: number;
  lon: number;
  label: string;
};

export type PinDisplayNode = {
  id: PinSelectionId;
  kind: PinNodeKind;
  lat: number;
  lon: number;
  label: string;
  order: number;
  color: string;
};

export type PinFocusRequest = {
  id: PinSelectionId;
  nonce: number;
  zoom?: number;
  openPopup?: boolean;
};

export type MapFailureOverlay = {
  reason_code: string;
  message: string;
  stage?: string | null;
  stage_detail?: string | null;
};

export type TutorialGuideTarget = {
  lat: number;
  lon: number;
  radius_km: number;
  label: string;
  stage: number;
  pan_nonce: number;
  zoom: number;
};

export type IncidentEventType = 'dwell' | 'accident' | 'closure';

export type RouteMetrics = {
  distance_km: number;
  duration_s: number;
  monetary_cost: number;
  emissions_kg: number;
  avg_speed_kmh: number;
  energy_kwh?: number | null;
  weather_delay_s?: number;
  incident_delay_s?: number;
};

export type SimulatedIncidentEvent = {
  event_id: string;
  event_type: IncidentEventType;
  segment_index: number;
  start_offset_s: number;
  delay_s: number;
  source: 'synthetic';
};

export type RouteSegmentBreakdownRow = {
  segment_index: number;
  distance_km: number;
  duration_s: number;
  incident_delay_s?: number;
  avg_speed_kmh?: number;
  emissions_kg: number;
  monetary_cost: number;
  time_cost?: number;
  fuel_cost?: number;
  toll_cost?: number;
  carbon_cost?: number;
  energy_kwh?: number;
  fuel_liters?: number;
  grade_pct?: number;
};

export type WeatherSummary = {
  enabled: boolean;
  profile: string;
  intensity: number;
  apply_incident_uplift: boolean;
  speed_multiplier: number;
  incident_multiplier: number;
  weather_delay_s?: number;
  incident_rate_multiplier?: number;
  terrain_source?: string;
  terrain_ascent_m?: number;
  terrain_descent_m?: number;
  terrain_coverage_ratio?: number;
  terrain_confidence?: number;
  terrain_dem_version?: string;
};

export type TerrainSummary = {
  source: 'dem_real' | 'missing' | 'unsupported_region';
  coverage_ratio: number;
  sample_spacing_m: number;
  ascent_m: number;
  descent_m: number;
  grade_histogram: Record<string, number>;
  confidence: number;
  fail_closed_applied: boolean;
  version: string;
};

export type ScenarioSummary = {
  mode: ScenarioMode;
  context_key?: string;
  duration_multiplier: number;
  incident_rate_multiplier: number;
  incident_delay_multiplier: number;
  fuel_consumption_multiplier: number;
  emissions_multiplier: number;
  stochastic_sigma_multiplier: number;
  source: string;
  version: string;
  calibration_basis?: string;
  as_of_utc?: string;
  live_as_of_utc?: string;
  live_sources?: string;
  live_coverage_overall?: number;
  live_traffic_pressure?: number;
  live_incident_pressure?: number;
  live_weather_pressure?: number;
  scenario_edge_scaling_version?: string;
  mode_observation_source?: string;
  mode_projection_ratio?: number;
};

export type EvidenceSourceRecord = {
  family: string;
  source: string;
  active: boolean;
  freshness_timestamp_utc?: string | null;
  max_age_minutes?: number | null;
  signature?: string | null;
  confidence?: number | null;
  coverage_ratio?: number | null;
  fallback_used?: boolean;
  fallback_source?: string | null;
  details?: Record<string, string | number | boolean>;
};

export type EvidenceProvenance = {
  active_families: string[];
  families: EvidenceSourceRecord[];
};

export type PreferenceTerminalType = 'open' | 'certified' | 'abstained';
export type PreferenceQueryType = 'pairwise' | 'threshold' | 'ratio' | 'veto' | 'time_guard';

export type PreferenceWeightsSummary = {
  time?: number | null;
  money?: number | null;
  co2?: number | null;
};

export type PreferenceCompatibleSetSummary = {
  route_ids?: string[];
  compatible_set_size?: number | null;
  compatible_set_volume_proxy?: number | null;
  necessary_best_prob?: number | null;
  possible_best_prob?: number | null;
  necessary_best_route_ids?: string[];
  possible_best_route_ids?: string[];
  support_flag?: boolean | null;
  support_reason?: string | null;
};

export type PairwisePreferenceQuery = {
  query_type: 'pairwise';
  preferred_route_id: string;
  challenger_route_id: string;
  reason?: string | null;
  weight_hint?: Record<string, number> | null;
};

export type ThresholdPreferenceQuery = {
  query_type: 'threshold';
  route_id: string;
  metric_name: string;
  threshold_value: number;
  direction?: 'gte' | 'lte';
  reason?: string | null;
};

export type RatioPreferenceQuery = {
  query_type: 'ratio';
  route_id: string;
  numerator_metric: string;
  denominator_metric: string;
  minimum_ratio: number;
  reason?: string | null;
};

export type VetoPreferenceQuery = {
  query_type: 'veto';
  route_id: string;
  veto_name: string;
  active?: boolean | null;
  reason?: string | null;
};

export type TimeGuardPreferenceQuery = {
  query_type: 'time_guard';
  route_id: string;
  latest_arrival_utc?: string | null;
  max_travel_time_s?: number | null;
  preserve_time_budget_s?: number | null;
  reason?: string | null;
};

export type PreferenceQuery =
  | PairwisePreferenceQuery
  | ThresholdPreferenceQuery
  | RatioPreferenceQuery
  | VetoPreferenceQuery
  | TimeGuardPreferenceQuery;

export type PreferenceShrinkageTrace = {
  query_index: number;
  query_type: PreferenceQueryType;
  before_size: number;
  after_size: number;
  before_volume_proxy: number;
  after_volume_proxy: number;
  predicted_shrinkage?: number | null;
  realized_shrinkage?: number | null;
  target_route_id?: string | null;
  query_reason?: string | null;
  preference_irrelevance?: boolean | null;
};

export type PreferenceContradictionRecord = {
  contradiction_detected?: boolean | null;
  contradiction_reasons?: string[];
};

export type PreferenceTraceProvenance = {
  selected_route_id?: string | null;
  pipeline_mode?: PipelineMode | null;
};

export type PreferenceState = {
  compatible_set_summary?: PreferenceCompatibleSetSummary | null;
  compatible_weights?: Array<Record<string, number>>;
  pairwise_constraints?: PairwisePreferenceQuery[];
  threshold_constraints?: ThresholdPreferenceQuery[];
  ratio_constraints?: RatioPreferenceQuery[];
  veto_rules?: VetoPreferenceQuery[];
  time_preserving_guard_rules?: TimeGuardPreferenceQuery[];
  query_history?: PreferenceQuery[];
  shrinkage_trace?: PreferenceShrinkageTrace[];
  contradiction_record?: PreferenceContradictionRecord | null;
  derived_invariants?: Record<string, boolean> | null;
  terminal_type?: PreferenceTerminalType | null;
  preference_irrelevance_proven?: boolean | null;
  no_query_reason?: string | null;
  no_preference_query_reason?: string | null;
  query_count?: number | null;
};

export type PreferenceQueryTrace = {
  schema_version?: string | null;
  selected_route_id?: string | null;
  selected_certificate_basis?: string | null;
  terminal_type?: PreferenceTerminalType | null;
  query_count?: number | null;
  query_history?: PreferenceQuery[];
  shrinkage_trace?: PreferenceShrinkageTrace[];
  compatible_set_summary?: PreferenceCompatibleSetSummary | null;
  derived_invariants?: Record<string, boolean> | null;
  contradiction_record?: PreferenceContradictionRecord | null;
  preference_irrelevance_proven?: boolean | null;
  no_query_reason?: string | null;
  no_preference_query_reason?: string | null;
  targeted_challenger_route_id?: string | null;
  query_selection_reason?: string | null;
  provenance?: PreferenceTraceProvenance | null;
};

export type PreferenceSummary = {
  selected_certificate_basis?: string | null;
  pipeline_mode?: PipelineMode | null;
  objective_field?: string | null;
  selector_policy?: string | null;
  selective?: boolean | null;
  tie_break_order?: string[] | null;
  weights?: PreferenceWeightsSummary | null;
  preference_state?: PreferenceState | null;
  compatible_set_summary?: PreferenceCompatibleSetSummary | null;
  derived_invariants?: Record<string, boolean> | null;
  contradiction_record?: PreferenceContradictionRecord | null;
  preference_irrelevance_proven?: boolean | null;
  no_query_reason?: string | null;
  no_preference_query_reason?: string | null;
  targeted_challenger_route_id?: string | null;
  query_selection_reason?: string | null;
  query_count?: number | null;
};

export type PreferenceRuntimeUpdateRequest = {
  candidate_routes: RouteOption[];
  selected_route_id?: string | null;
  selected_certificate_basis?: string | null;
  pipeline_mode?: PipelineMode | null;
  support_flag?: boolean | null;
  support_reason?: string | null;
  preference_state: PreferenceState;
};

export type PreferenceRuntimeUpdateResponse = {
  selected_route_id?: string | null;
  selected_certificate_basis?: string | null;
  pipeline_mode?: PipelineMode | null;
  terminal_type?: PreferenceTerminalType | null;
  preference_state: PreferenceState;
  preference_query_trace: PreferenceQueryTrace;
  preference_summary: PreferenceSummary;
};

export type SupportStateSummary = {
  schema_version?: string;
  support_flag?: boolean | null;
  support_status?: string | null;
  support_reason?: string | null;
  support_score?: number | null;
  support_ratio?: number | null;
  support_bin?: string | null;
  calibration_bin?: string | null;
  support_source?: string | null;
  out_of_support_reason?: string | null;
  coverage_ratio?: number | null;
  confidence?: number | null;
  provenance?: Record<string, unknown> | null;
};

export type ProbabilisticWorldBundleSummary = {
  world_count?: number | null;
  unique_world_count?: number | null;
  active_families?: string[];
  state_catalog?: string[];
  state_weights?: Record<string, Record<string, number>>;
  world_reuse_rate?: number | null;
  world_reuse_rate_within_manifest?: number | null;
  world_reuse_rate_cross_request?: number | null;
  certification_cache_reuse_origin?: string | null;
  certification_cache_reuse_applied?: boolean | null;
  manifest_hash?: string | null;
  support_state?: SupportStateSummary | null;
};

export type AuditWorldBundleSummary = {
  audit_world_count?: number | null;
  audited_route_pair_count?: number | null;
  partially_audited_world_count?: number | null;
  fully_audited_world_count?: number | null;
  reused_world_count?: number | null;
  corrected_world_count?: number | null;
  support_condition?: string | null;
  calibration_version?: string | null;
  propensity_version?: string | null;
  diagnostics?: Record<string, unknown> | null;
};

export type WorldBundleSummary = {
  regime_id?: string | null;
  copula_id?: string | null;
  calibration_version?: string | null;
  as_of_utc?: string | null;
  support_state?: SupportStateSummary | null;
  probabilistic_world_bundle?: ProbabilisticWorldBundleSummary | null;
  audit_world_bundle?: AuditWorldBundleSummary | null;
  uncertainty_summary?: Record<string, unknown> | null;
  provenance?: Record<string, unknown> | null;
};

export type WorldSupportSummary = {
  schema_version?: string | null;
  selected_route_id?: string | null;
  selected_certificate_basis?: string | null;
  support_strength?: number | string | null;
  source_support_strength?: number | string | null;
  recommended_fidelity?: string | null;
  proxy_penalty?: number | null;
  audit_correction?: number | null;
  support_sufficient?: boolean | null;
  support_state?: SupportStateSummary | null;
  world_bundle_summary?: WorldBundleSummary | null;
  scenario_summary?: ScenarioSummary | null;
  risk_summary?: Record<string, unknown> | null;
  provenance?: Record<string, unknown> | null;
  support_flag?: boolean | null;
  support_reason?: string | null;
  world_count?: number | null;
  unique_world_count?: number | null;
  world_reuse_rate?: number | null;
  calibration_bin?: string | null;
  support_bin?: string | null;
  active_families?: string[];
};

export type ActionTraceSummary = {
  stop_reason?: string | null;
  search_completeness_score?: number | null;
  search_completeness_gap?: number | null;
  pipeline_mode?: PipelineMode | null;
  selected_candidate_count?: number | null;
};

export type WitnessSummary = {
  witness_size?: number | null;
  active_challenger_ids?: string[];
  active_evidence_families?: string[];
  route_id?: string | null;
  primary_witness_route_id?: string | null;
  witness_route_ids?: string[];
  challenger_route_ids?: string[];
  witness_source_ids?: string[];
  witness_world_count?: number | null;
  selected_certificate_basis?: string | null;
};

export type RouteFragilityMapArtifact = Record<string, Record<string, number>>;

export type FlipRadiusSummaryArtifact = {
  route_id?: string | null;
  minimum_flip_budget?: number | null;
  dominant_fragility_family?: string | null;
  adversarial_degradation_curve?: Array<Record<string, unknown>> | null;
  provenance?: Record<string, unknown> | null;
};

export type DecisionRegionSummaryArtifact = {
  route_id?: string | null;
  nearest_certificate_boundary?: string | null;
  active_challenger_id?: string | null;
  dominant_evidence_family?: string | null;
  most_fragile_preference_direction?: string | null;
  minimum_joint_perturbation?: number | null;
  nearest_threat_axis?: string | null;
  support_flag?: boolean | null;
  provenance?: Record<string, unknown> | null;
};

export type ValueOfRefreshMarginSummary = {
  world_count?: number | null;
  mean_runner_up_gap?: number | null;
  min_runner_up_gap?: number | null;
  max_runner_up_gap?: number | null;
  positive_world_share?: number | null;
  margin_stability_signal?: number | null;
};

export type ValueOfRefreshRankingEntry = {
  family: string;
  vor?: number | null;
  controller_score?: number | null;
  empirical_vor?: number | null;
  raw_refresh_gain?: number | null;
  basis?: string | null;
};

export type ValueOfRefreshArtifact = {
  selected_route_id?: string | null;
  baseline_certificate?: number | null;
  empirical_baseline_certificate?: number | null;
  controller_baseline_certificate?: number | null;
  baseline_margin_summary?: ValueOfRefreshMarginSummary | null;
  fragility_stress_state?: string | null;
  per_family_certificate?: Record<string, number>;
  per_family_margin_summary?: Record<string, ValueOfRefreshMarginSummary>;
  ranking?: ValueOfRefreshRankingEntry[];
  top_refresh_family?: string | null;
  top_refresh_gain?: number | null;
  controller_ranking_basis?: string | null;
  controller_ranking?: ValueOfRefreshRankingEntry[];
  top_refresh_family_controller?: string | null;
  top_refresh_gain_controller?: number | null;
  controller_refresh_frontier_mode?: string | null;
  controller_refresh_frontier_route_ids?: string[];
  controller_refresh_frontier_count?: number | null;
  single_frontier_certificate_cap?: number | null;
  single_frontier_certificate_cap_applied?: boolean | null;
  single_frontier_requires_full_stress?: boolean | null;
  single_frontier_observed_coverage_ratio?: number | null;
  single_frontier_observed_coverage_relief?: number | null;
  single_frontier_observed_coverage_ceiling?: number | null;
  single_frontier_observed_stress_fraction?: number | null;
};

export type SampledWorldManifestWorld = {
  world_id: string;
  states?: Record<string, string>;
  stress_factor?: number | null;
  world_kind?: string | null;
  target_route_id?: string | null;
  target_route_ids?: Record<string, string> | null;
};

export type SampledWorldManifestArtifact = {
  seed?: number | null;
  requested_world_count?: number | null;
  sampler_requested_world_count?: number | null;
  world_count?: number | null;
  unique_world_count?: number | null;
  active_families?: string[];
  state_catalog?: string[];
  state_weights?: Record<string, Record<string, number>>;
  ambiguity_context?: Record<string, unknown> | null;
  hard_case_stress_pack_count?: number | null;
  supported_ambiguity_stress_pack_count?: number | null;
  targeted_stress_pack_count?: number | null;
  mixed_targeted_stress_pack_count?: number | null;
  single_family_targeted_stress_pack_count?: number | null;
  world_reuse_rate?: number | null;
  stress_world_fraction?: number | null;
  refc_stress_world_fraction?: number | null;
  hard_case_stress_world_fraction?: number | null;
  supported_ambiguity_stress_world_fraction?: number | null;
  worlds?: SampledWorldManifestWorld[];
  manifest_hash?: string | null;
  forced_refreshed_families?: string[];
  selected_route_id?: string | null;
  effective_world_count?: number | null;
  world_count_policy?: string | null;
};

export type TypedAbstentionReason =
  | 'uncertified_due_to_search'
  | 'uncertified_due_to_evidence'
  | 'uncertified_due_to_preference'
  | 'uncertified_due_to_out_of_support_world_model'
  | 'uncertified_due_to_budget'
  | 'uncertified_due_to_model_assumption';

export type AbstentionRecord = {
  reason_code: TypedAbstentionReason;
  message: string;
  detail?: Record<string, unknown>;
  support_flag?: boolean | null;
  evidence_family?: string | null;
  budget_channel?: string | null;
  model_assumption?: string | null;
  terminal_type?: 'typed_abstention';
};

export type ProofAbstentionProvenance = {
  reason_code?: TypedAbstentionReason | null;
  message?: string | null;
  detail?: Record<string, unknown> | null;
  support_flag?: boolean | null;
  evidence_family?: string | null;
  budget_channel?: string | null;
  model_assumption?: string | null;
  terminal_type?: 'typed_abstention' | null;
};

export type RouteCertificationSummary = {
  route_id: string;
  certificate: number;
  certified: boolean;
  threshold: number;
  certificate_lcb?: number | null;
  certificate_ucb?: number | null;
  minimum_pairwise_gap_lcb?: number | null;
  active_families?: string[];
  top_fragility_families?: string[];
  top_competitor_route_id?: string | null;
  top_value_of_refresh_family?: string | null;
};

export type SupportSummary = {
  satisfied?: boolean | null;
  supported?: boolean | null;
  support_flag?: boolean | null;
  observed_source_count?: number | null;
  required_source_count?: number | null;
  source_mix?: string[] | null;
  missing_sources?: string[] | null;
  provenance_mode?: string | null;
  [key: string]: unknown;
};

export type CertifiedSetSummary = {
  certified?: boolean | null;
  selected_route_id?: string | null;
  certified_route_ids?: string[] | null;
  frontier_route_ids?: string[] | null;
  certificate_basis?: string | null;
  minimum_cost_route_id?: string | null;
  [key: string]: unknown;
};

export type AbstentionSummary = {
  abstained?: boolean | null;
  reason_code?: string | null;
  blocking_sources?: string[] | null;
  retryable?: boolean | null;
  [key: string]: unknown;
};

export type WorldFidelitySummary = {
  multi_fidelity_mode?: string | null;
  policy?: string | null;
  effective_world_count?: number | null;
  world_count?: number | null;
  proxy_world_fraction?: number | null;
  stress_world_fraction?: number | null;
  world_reuse_rate?: number | null;
  recommended_policy?: string | null;
  [key: string]: unknown;
};

export type CertificationStateSummary = {
  winner_id?: string | null;
  certification_basis?: string | null;
  certified?: boolean | null;
  abstained?: boolean | null;
  support_strength?: number | string | null;
  certified_set?: { safe?: boolean | null; [key: string]: unknown } | null;
  [key: string]: unknown;
};

export type ControllerSummary = {
  controller_mode?: string | null;
  engaged?: boolean | null;
  iteration_count?: number | null;
  action_count?: number | null;
  stop_reason?: string | null;
  search_budget_used?: number | null;
  evidence_budget_used?: number | null;
  [key: string]: unknown;
};

export type TheoremHookSummary = {
  hooks?: Array<{
    hook_id?: string | null;
    status?: string | null;
    artifact_name?: string | null;
    [key: string]: unknown;
  }>;
  [key: string]: unknown;
};

export type LaneManifestSummary = {
  lane_id?: string | null;
  lane_name?: string | null;
  lane_version?: string | null;
  artifact_names?: string[] | null;
  [key: string]: unknown;
};

export type DecisionProofContext = {
  selected_certificate_basis?: string | null;
  support_flag?: boolean | null;
  out_of_support_reason?: string | null;
  typed_abstention?: ProofAbstentionProvenance | AbstentionRecord | null;
  controller_boundary_summary?: VoiControllerBoundarySummary | null;
  controller_state?: VoiControllerState | null;
  witness_summary?: WitnessSummary | null;
  world_support_summary?: WorldSupportSummary | null;
  certificate_summary?: RouteCertificationSummary | Record<string, unknown> | null;
  support_summary?: SupportSummary | null;
  abstention_summary?: AbstentionSummary | null;
  action_trace_summary?: ActionTraceSummary | null;
};

export type VoiStopSummary = {
  final_route_id: string;
  certificate: number;
  certified: boolean;
  iteration_count: number;
  search_budget_used: number;
  evidence_budget_used: number;
  stop_reason: string;
  best_rejected_action?: string | null;
  best_rejected_q?: number | null;
  credible_search_uncertainty?: boolean | null;
};

export type VoiTraceAction = {
  action_id?: string | null;
  kind?: string | null;
  target?: string | null;
  action_family?: string | null;
  action_modality?: string | null;
  cost_search?: number | null;
  cost_evidence?: number | null;
  predicted_delta_certificate?: number | null;
  predicted_delta_margin?: number | null;
  predicted_delta_frontier?: number | null;
  predicted_delta_search_completeness?: number | null;
  predicted_winner_lcb_gain?: number | null;
  predicted_gap_lcb_gain?: number | null;
  predicted_radius_or_flip_budget_gain?: number | null;
  predicted_unresolved_mass_reduction?: number | null;
  predicted_preference_ambiguity_reduction?: number | null;
  predicted_boundary_contraction?: number | null;
  predicted_delta_radius_or_flip_budget?: number | null;
  predicted_preference_shrinkage?: number | null;
  predicted_certified_set_contraction?: number | null;
  q_score?: number | null;
  feasible?: boolean | null;
  preconditions?: string[];
  reason?: string | null;
  metadata?: Record<string, unknown> | null;
};

export type VoiTraceIteration = {
  iteration?: number | null;
  selected_route_id?: string | null;
  selected_certificate?: number | null;
  remaining_search_budget?: number | null;
  remaining_evidence_budget?: number | null;
  frontier_size?: number | null;
  feasible_actions?: VoiTraceAction[];
  chosen_action?: VoiTraceAction | null;
  best_rejected_action?: VoiTraceAction | null;
  next_best_unused_action?: VoiTraceAction | null;
  realized_certificate_before?: number | null;
  realized_certificate_after?: number | null;
  realized_certificate_delta?: number | null;
  realized_frontier_gain?: number | null;
  realized_selected_route_changed?: boolean | null;
  realized_selected_score_delta?: number | null;
  realized_runner_up_gap_before?: number | null;
  realized_runner_up_gap_after?: number | null;
  realized_runner_up_gap_delta?: number | null;
  realized_evidence_uncertainty_before?: number | null;
  realized_evidence_uncertainty_after?: number | null;
  realized_evidence_uncertainty_delta?: number | null;
  realized_productive?: boolean | null;
};

export type VoiControllerAuditPropensitySummary = {
  audit_coverage_ratio?: number | null;
  coverage_ratio?: number | null;
  minimum_propensity?: number | null;
  mean_propensity?: number | null;
  mean_audit_probability?: number | null;
  positivity_ok?: boolean | null;
  weak_overlap_detected?: boolean | null;
  leakage_safe_training?: boolean | null;
  correction_path_estimator?: string | null;
  certification_evaluation_tag?: string | null;
  propensity_model_version?: string | null;
  correction_model_version?: string | null;
  [key: string]: unknown;
};

export type VoiControllerBoundarySummary = {
  active_challenger_id?: string | null;
  active_challenger_ids?: string[];
  top_competitor_route_id?: string | null;
  boundary_count?: number | null;
  challenger_count?: number | null;
  boundary_status?: string | null;
  certificate_boundary_kind?: string | null;
  [key: string]: unknown;
};

export type VoiControllerState = {
  iteration_index?: number | null;
  winner_id?: string | null;
  selected_route_id?: string | null;
  remaining_search_budget?: number | null;
  remaining_evidence_budget?: number | null;
  certificate_lcb?: number | null;
  certificate_ucb?: number | null;
  necessary_best_probability?: number | null;
  possible_best_probability?: number | null;
  minimum_pairwise_gap_lcb?: number | null;
  deterministic_local_flip_radius?: number | null;
  probabilistic_flip_radius?: number | null;
  minimum_flip_budget?: number | null;
  certified_set_size?: number | null;
  weight_set_volume?: number | null;
  weight_set_shrinkage?: number | null;
  preference_irrelevance_proven?: boolean | null;
  no_preference_query_reason?: string | null;
  unresolved_possible_frontier_mass?: number | null;
  unresolved_possible_winner_mass?: number | null;
  unresolved_certificate_critical_mass?: number | null;
  support_flag?: boolean | null;
  out_of_support_reason?: string | null;
  proxy_only_fraction?: number | null;
  audit_propensity_summary?: VoiControllerAuditPropensitySummary | null;
  active_certificate_boundary_summary?: VoiControllerBoundarySummary | null;
  support_richness?: number | null;
  ambiguity_pressure?: number | null;
  search_completeness_score?: number | null;
  search_completeness_gap?: number | null;
  pending_challenger_mass?: number | null;
  best_pending_flip_probability?: number | null;
  top_refresh_gain?: number | null;
  competitor_pressure?: number | null;
  feasible_actions?: VoiTraceAction[];
  chosen_action?: VoiTraceAction | null;
  best_rejected_action?: VoiTraceAction | null;
  [key: string]: unknown;
};

export type VoiActionTraceArtifact = {
  pipeline_mode?: PipelineMode | null;
  selected_route_id?: string | null;
  actions?: VoiTraceIteration[];
};

export type VoiStopCertificateArtifact = {
  pipeline_mode?: PipelineMode | null;
  status?: string | null;
  final_winner_route_id?: string | null;
  selected_route_id?: string | null;
  certificate_value?: number | null;
  certified?: boolean | null;
  search_budget_used?: number | null;
  search_budget_remaining?: number | null;
  evidence_budget_used?: number | null;
  evidence_budget_remaining?: number | null;
  search_completeness_score?: number | null;
  search_completeness_gap?: number | null;
  credible_search_uncertainty?: boolean | null;
  credible_evidence_uncertainty?: boolean | null;
  stop_reason?: string | null;
  action_trace?: VoiTraceIteration[];
  best_rejected_action?: VoiTraceAction | null;
  ambiguity_summary?: Record<string, unknown> | null;
  terminal_action_id?: string | null;
  terminal_action_kind?: string | null;
  terminal_action_family?: string | null;
  terminal_action_modality?: string | null;
  predicted_delta_radius_or_flip_budget?: number | null;
  realized_delta_radius_or_flip_budget?: number | null;
  predicted_preference_shrinkage?: number | null;
  realized_preference_shrinkage?: number | null;
  predicted_certified_set_contraction?: number | null;
  realized_certified_set_contraction?: number | null;
  hindsight_necessity_label?: string | null;
  metric_semantics?: Record<string, string> | null;
  iteration_count?: number | null;
  controller_state?: VoiControllerState | null;
};

export type GeoJSONLineString = {
  type: 'LineString';
  coordinates: [number, number][];
};

export type RouteOption = {
  id: string;
  geometry: GeoJSONLineString;
  metrics: RouteMetrics;
  knee_score?: number | null;
  is_knee?: boolean;
  eta_explanations?: string[];
  eta_timeline?: Array<Record<string, string | number>>;
  segment_breakdown?: RouteSegmentBreakdownRow[];
  counterfactuals?: Array<Record<string, string | number | boolean>>;
  uncertainty?: Record<string, number> | null;
  uncertainty_samples_meta?: Record<string, string | number | boolean> | null;
  legs?: Array<Record<string, string | number | boolean>> | null;
  toll_confidence?: number | null;
  toll_metadata?: Record<string, string | number | boolean | string[]> | null;
  vehicle_profile_id?: string | null;
  vehicle_profile_version?: number | null;
  vehicle_profile_source?: string | null;
  scenario_summary?: ScenarioSummary | null;
  incident_events?: SimulatedIncidentEvent[];
  weather_summary?: WeatherSummary | null;
  terrain_summary?: TerrainSummary | null;
  evidence_provenance?: EvidenceProvenance | null;
  certification?: RouteCertificationSummary | null;
};

export type RouteResponse = {
  selected: RouteOption;
  candidates: RouteOption[];
  recommended_route?: RouteOption | null;
  certified_set?: RouteOption[] | null;
  abstention?: AbstentionRecord | null;
  terminal_type?: 'certified_singleton' | 'certified_set' | 'typed_abstention';
  run_id?: string | null;
  pipeline_mode?: PipelineMode;
  manifest_endpoint?: string | null;
  artifacts_endpoint?: string | null;
  provenance_endpoint?: string | null;
  selected_certificate?: RouteCertificationSummary | null;
  selected_certificate_basis?: string | null;
  voi_stop_summary?: VoiStopSummary | null;
  artifact_pointers?: Record<string, string | null> | null;
  preference_state?: PreferenceState | null;
  preference_query_trace?: PreferenceQueryTrace | null;
  frontier_summary?: Record<string, unknown> | null;
  certificate_summary?: RouteCertificationSummary | Record<string, unknown> | null;
  stability_summary?: Record<string, unknown> | null;
  preference_summary?: PreferenceSummary | null;
  support_summary?: SupportSummary | null;
  abstention_summary?: AbstentionSummary | null;
  certified_set_summary?: CertifiedSetSummary | null;
  action_trace_summary?: ActionTraceSummary | null;
  witness_summary?: WitnessSummary | null;
  world_support_summary?: WorldSupportSummary | null;
};

export type DecisionPackage = {
  package_kind?: string | null;
  schema_version?: string | number | null;
  selected?: RouteOption | null;
  candidates?: RouteOption[] | null;
  recommended_route?: RouteOption | null;
  certified_set?: RouteOption[] | null;
  abstention?: AbstentionRecord | null;
  terminal_type?: 'certified_singleton' | 'certified_set' | 'typed_abstention';
  terminal_kind?: string | null;
  selected_route_id?: string | null;
  frontier_summary?: Record<string, unknown> | null;
  certificate_summary?: RouteCertificationSummary | Record<string, unknown> | null;
  selected_certificate?: RouteCertificationSummary | null;
  stability_summary?: Record<string, unknown> | null;
  preference_summary?: PreferenceSummary | null;
  support_summary?: SupportSummary | null;
  abstention_summary?: AbstentionSummary | null;
  certified_set_summary?: CertifiedSetSummary | null;
  action_trace_summary?: ActionTraceSummary | null;
  witness_summary?: WitnessSummary | null;
  world_fidelity_summary?: WorldFidelitySummary | null;
  certification_state_summary?: CertificationStateSummary | null;
  controller_summary?: ControllerSummary | null;
  theorem_hook_summary?: TheoremHookSummary | null;
  lane_manifest?: LaneManifestSummary | null;
  provenance?: Record<string, unknown> | null;
  proof_context?: DecisionProofContext | null;
  artifact_pointers?: Record<string, string | null> | null;
  run_id?: string | null;
  pipeline_mode?: PipelineMode;
  manifest_endpoint?: string | null;
  artifacts_endpoint?: string | null;
  provenance_endpoint?: string | null;
  selected_certificate_basis?: string | null;
  voi_stop_summary?: VoiStopSummary | null;
  preference_state?: PreferenceState | null;
  preference_query_trace?: PreferenceQueryTrace | null;
  world_support_summary?: WorldSupportSummary | null;
};

export type DecisionPackageResponse = {
  decision_package: DecisionPackage;
};

export type RouteResponsePayload = RouteResponse | DecisionPackage | DecisionPackageResponse;

export type RouteBaselineResponse = {
  baseline: RouteOption;
  method: 'osrm_quick_baseline' | 'ors_reference' | 'ors_proxy_baseline';
  compute_ms: number;
  notes?: string[];
};

export type ParetoResponse = {
  routes: RouteOption[];
  warnings?: string[];
  diagnostics?: Record<string, string | number | boolean>;
};

export type CandidateDiagnostics = {
  [key: string]: unknown;
  prefetch_ms?: number;
  scenario_context_ms?: number;
  graph_search_ms_initial?: number;
  graph_search_ms_retry?: number;
  graph_search_ms_rescue?: number;
  osrm_refine_ms?: number;
  build_options_ms?: number;
};

export type ParetoStreamMetaEvent = {
  type: 'meta';
  total: number;
  done?: number;
  request_id?: string;
  stage?: string;
  stage_detail?: string;
  elapsed_ms?: number;
  stage_elapsed_ms?: number;
  heartbeat?: number;
  candidate_done?: number;
  candidate_total?: number;
  candidate_diagnostics?: CandidateDiagnostics | null;
};

export type ParetoStreamRouteEvent = {
  type: 'route';
  done: number;
  total: number;
  route: RouteOption;
};

export type ParetoStreamErrorEvent = {
  type: 'error';
  done: number;
  total: number;
  message: string;
};

export type StrictReasonCode =
  | 'routing_graph_no_path'
  | 'routing_graph_unavailable'
  | 'routing_graph_fragmented'
  | 'routing_graph_disconnected_od'
  | 'routing_graph_coverage_gap'
  | 'routing_graph_precheck_timeout'
  | 'routing_graph_warming_up'
  | 'routing_graph_warmup_failed'
  | 'live_source_refresh_failed'
  | (string & {});

export type ParetoStreamFatalEvent = {
  type: 'fatal';
  message: string;
  reason_code?: StrictReasonCode;
  warnings?: string[];
  request_id?: string;
  stage?: string;
  stage_detail?: string;
  elapsed_ms?: number;
  stage_elapsed_ms?: number;
  candidate_done?: number;
  candidate_total?: number;
  candidate_diagnostics?: Record<string, unknown> | null;
  failure_chain?: Record<string, unknown> | null;
};

export type ParetoStreamDoneEvent = {
  type: 'done';
  done: number;
  total: number;
  routes: RouteOption[];
  warning_count?: number;
  warnings?: string[];
  candidate_diagnostics?: Record<string, unknown> | null;
};

export type ParetoStreamEvent =
  | ParetoStreamMetaEvent
  | ParetoStreamRouteEvent
  | ParetoStreamErrorEvent
  | ParetoStreamFatalEvent
  | ParetoStreamDoneEvent;

export type VehicleProfile = {
  id: string;
  label: string;
  mass_tonnes: number;
  emission_factor_kg_per_tkm: number;
  cost_per_km: number;
  cost_per_hour: number;
  idle_emissions_kg_per_hour: number;
  powertrain?: 'ice' | 'ev';
  ev_kwh_per_km?: number | null;
  grid_co2_kg_per_kwh?: number | null;
  schema_version?: number;
  vehicle_class?: 'van' | 'rigid_hgv' | 'artic_hgv' | 'ev';
  toll_vehicle_class?: string;
  toll_axle_class?: string;
  fuel_surface_class?: 'van' | 'rigid_hgv' | 'artic_hgv' | 'ev';
  risk_bucket?: string;
  stochastic_bucket?: string;
  terrain_params?: {
    mass_kg: number;
    c_rr: number;
    drag_area_m2: number;
    drivetrain_efficiency: number;
    regen_efficiency: number;
  };
  aliases?: string[];
  profile_source?: string;
  profile_as_of_utc?: string | null;
};

export type VehicleListResponse = { vehicles: VehicleProfile[] };

export type ScenarioCompareResult = {
  scenario_mode: ScenarioMode;
  selected: RouteOption | null;
  candidates: RouteOption[];
  warnings: string[];
  error?: string | null;
};

export type ScenarioCompareResponse = {
  run_id: string;
  results: ScenarioCompareResult[];
  deltas: Record<
    string,
    {
      duration_s_delta?: number | null;
      monetary_cost_delta?: number | null;
      emissions_kg_delta?: number | null;
      duration_s_status?: 'ok' | 'missing' | string;
      monetary_cost_status?: 'ok' | 'missing' | string;
      emissions_kg_status?: 'ok' | 'missing' | string;
      duration_s_reason_code?: string | null;
      monetary_cost_reason_code?: string | null;
      emissions_kg_reason_code?: string | null;
      duration_s_missing_source?: string | null;
      monetary_cost_missing_source?: string | null;
      emissions_kg_missing_source?: string | null;
      duration_s_reason_source?: string | null;
      monetary_cost_reason_source?: string | null;
      emissions_kg_reason_source?: string | null;
    }
  >;
  baseline_mode?: ScenarioMode;
  scenario_manifest_endpoint: string;
  scenario_signature_endpoint: string;
};

export type ScenarioCompareRequest = {
  origin: LatLng;
  destination: LatLng;
  waypoints?: Waypoint[];
  vehicle_type?: string;
  scenario_mode?: ScenarioMode | null;
  weights?: { time: number; money: number; co2: number };
  max_alternatives?: number;
  cost_toggles?: CostToggles;
  terrain_profile?: TerrainProfile;
  departure_time_utc?: string;
  stochastic?: StochasticConfig;
  optimization_mode?: OptimizationMode;
  risk_aversion?: number;
  emissions_context?: EmissionsContext;
  weather?: WeatherImpactConfig;
  incident_simulation?: IncidentSimulatorConfig;
  pareto_method?: ParetoMethod;
  epsilon?: EpsilonConstraints;
};

export type RouteRequest = {
  origin: LatLng;
  destination: LatLng;
  waypoints?: Waypoint[];
  vehicle_type?: string;
  scenario_mode?: ScenarioMode;
  max_alternatives?: number;
  weights?: { time: number; money: number; co2: number };
  cost_toggles?: CostToggles;
  terrain_profile?: TerrainProfile;
  stochastic?: StochasticConfig;
  optimization_mode?: OptimizationMode;
  risk_aversion?: number;
  emissions_context?: EmissionsContext;
  weather?: WeatherImpactConfig;
  incident_simulation?: IncidentSimulatorConfig;
  departure_time_utc?: string;
  pareto_method?: ParetoMethod;
  epsilon?: EpsilonConstraints;
  pipeline_mode?: PipelineMode | null;
  pipeline_seed?: number | null;
  search_budget?: number | null;
  evidence_budget?: number | null;
  cert_world_count?: number | null;
  certificate_threshold?: number | null;
  tau_stop?: number | null;
};

export type ParetoRequest = RouteRequest & {
  max_alternatives?: number;
};

export type ExperimentBundle = {
  id: string;
  name: string;
  description?: string | null;
  request: ScenarioCompareRequest;
  created_at: string;
  updated_at: string;
};

export type ExperimentListResponse = {
  experiments: ExperimentBundle[];
};

export type ExperimentCatalogSort = 'updated_desc' | 'updated_asc' | 'name_asc' | 'name_desc';

export type DepartureOptimizeRequest = {
  origin: LatLng;
  destination: LatLng;
  waypoints?: Waypoint[];
  vehicle_type?: string;
  scenario_mode?: ScenarioMode;
  weights?: { time: number; money: number; co2: number };
  max_alternatives?: number;
  cost_toggles?: CostToggles;
  terrain_profile?: TerrainProfile;
  stochastic?: StochasticConfig;
  optimization_mode?: OptimizationMode;
  risk_aversion?: number;
  emissions_context?: EmissionsContext;
  weather?: WeatherImpactConfig;
  incident_simulation?: IncidentSimulatorConfig;
  pareto_method?: ParetoMethod;
  epsilon?: EpsilonConstraints;
  time_window?: TimeWindowConstraints;
  window_start_utc: string;
  window_end_utc: string;
  step_minutes: number;
};

export type DepartureOptimizeCandidate = {
  departure_time_utc: string;
  selected: RouteOption;
  score: number;
  warning_count: number;
};

export type DepartureOptimizeResponse = {
  best: DepartureOptimizeCandidate | null;
  candidates: DepartureOptimizeCandidate[];
  evaluated_count: number;
};

export type DutyChainStop = {
  lat: number;
  lon: number;
  label?: string | null;
};

export type DutyChainLegResult = {
  leg_index: number;
  origin: DutyChainStop;
  destination: DutyChainStop;
  selected: RouteOption | null;
  candidates: RouteOption[];
  warning_count: number;
  error?: string | null;
};

export type DutyChainRequest = {
  stops: DutyChainStop[];
  vehicle_type?: string;
  scenario_mode?: ScenarioMode;
  weights?: { time: number; money: number; co2: number };
  max_alternatives?: number;
  cost_toggles?: CostToggles;
  terrain_profile?: TerrainProfile;
  stochastic?: StochasticConfig;
  optimization_mode?: OptimizationMode;
  risk_aversion?: number;
  emissions_context?: EmissionsContext;
  weather?: WeatherImpactConfig;
  incident_simulation?: IncidentSimulatorConfig;
  departure_time_utc?: string;
  pareto_method?: ParetoMethod;
  epsilon?: EpsilonConstraints;
};

export type ODPair = {
  origin: LatLng;
  destination: LatLng;
};

export type BatchParetoRequest = {
  pairs: ODPair[];
  waypoints?: Waypoint[];
  vehicle_type?: string;
  scenario_mode?: ScenarioMode;
  max_alternatives?: number;
  weights?: { time: number; money: number; co2: number };
  cost_toggles?: CostToggles;
  terrain_profile?: TerrainProfile;
  stochastic?: StochasticConfig;
  optimization_mode?: OptimizationMode;
  risk_aversion?: number;
  emissions_context?: EmissionsContext;
  weather?: WeatherImpactConfig;
  incident_simulation?: IncidentSimulatorConfig;
  departure_time_utc?: string;
  pareto_method?: ParetoMethod;
  epsilon?: EpsilonConstraints;
  seed?: number | null;
  toggles?: Record<string, string | number | boolean>;
  model_version?: string | null;
};

export type BatchCSVImportRequest = Omit<BatchParetoRequest, 'pairs'> & {
  csv_text: string;
};

export type BatchParetoResult = {
  origin: LatLng;
  destination: LatLng;
  routes: RouteOption[];
  error?: string | null;
};

export type BatchParetoResponse = {
  run_id: string;
  results: BatchParetoResult[];
};

export type DutyChainResponse = {
  legs: DutyChainLegResult[];
  total_metrics: RouteMetrics;
  leg_count: number;
  successful_leg_count: number;
};

export type OracleFeedCheckInput = {
  source: string;
  schema_valid: boolean;
  signature_valid?: boolean | null;
  freshness_s?: number | null;
  latency_ms?: number | null;
  record_count?: number | null;
  observed_at_utc?: string | null;
  error?: string | null;
};

export type OracleFeedCheckRecord = {
  check_id: string;
  source: string;
  schema_valid: boolean;
  signature_valid?: boolean | null;
  freshness_s?: number | null;
  latency_ms?: number | null;
  record_count?: number | null;
  observed_at_utc?: string | null;
  error?: string | null;
  passed: boolean;
  ingested_at_utc: string;
};

export type OracleQualitySourceSummary = {
  source: string;
  check_count: number;
  pass_rate: number;
  schema_failures: number;
  signature_failures: number;
  stale_count: number;
  avg_latency_ms?: number | null;
  last_observed_at_utc?: string | null;
};

export type OracleQualityDashboardResponse = {
  total_checks: number;
  source_count: number;
  stale_threshold_s: number;
  sources: OracleQualitySourceSummary[];
  updated_at_utc: string;
};

export type HealthResponse = {
  status: string;
};

export type HealthReadyResponse = {
  status: 'ready' | 'not_ready';
  strict_route_ready: boolean;
  recommended_action?: 'wait' | 'retry' | 'ready' | 'rebuild_graph' | 'refresh_live_sources' | string;
  route_graph: {
    ok?: boolean;
    status?: string;
    state?: 'idle' | 'loading' | 'ready' | 'failed' | string;
    phase?: string;
    elapsed_ms?: number | null;
    timeout_s?: number | null;
    timed_out?: boolean;
    last_error?: string | null;
    asset_path?: string | null;
    asset_exists?: boolean;
    asset_size_mb?: number | null;
    nodes_seen?: number;
    nodes_kept?: number;
    edges_seen?: number;
    edges_kept?: number;
    thread_alive?: boolean;
    cache_loaded?: boolean;
    [key: string]: unknown;
  };
  strict_live?: {
    ok: boolean;
    status?: 'ok' | 'stale' | 'unavailable' | 'disabled' | string;
    reason_code?: string;
    message?: string;
    as_of_utc?: string | null;
    age_minutes?: number | null;
    max_age_minutes?: number | null;
    checked_at_utc?: string | null;
    [key: string]: unknown;
  };
};

export type LiveCallEntry = {
  entry_id: number;
  request_id: string;
  at_utc: string;
  source_key: string;
  source_family?: string;
  component: string;
  url: string;
  method: string;
  requested: boolean;
  success: boolean;
  status_code?: number | null;
  fetch_error?: string | null;
  cache_hit?: boolean;
  stale_cache_used?: boolean;
  retry_attempts?: number;
  retry_count?: number;
  retry_total_backoff_ms?: number;
  retry_last_error?: string | null;
  retry_last_status_code?: number | null;
  retry_deadline_exceeded?: boolean;
  duration_ms?: number | null;
  headers?: Record<string, unknown> | null;
  request_headers_raw?: Record<string, unknown> | null;
  response_headers_raw?: Record<string, unknown> | null;
  response_body_raw?: string | null;
  response_body_truncated?: boolean;
  response_body_content_type?: string | null;
  response_body_bytes?: number | null;
  extra?: Record<string, unknown> | null;
};

export type LiveCallExpectedRow = {
  source_key: string;
  source_family?: string;
  component: string;
  url: string;
  method: string;
  required: boolean;
  description?: string | null;
  phase?: string | null;
  gate?: string | null;
};

export type LiveCallExpectedRollup = LiveCallExpectedRow & {
  observed_calls: number;
  requested_calls: number;
  success_count: number;
  failure_count: number;
  last_status_code?: number | null;
  last_fetch_error?: string | null;
  blocked?: boolean;
  blocked_reason?: string | null;
  blocked_stage?: string | null;
  blocked_detail?: string | null;
  satisfied: boolean;
  status?: 'ok' | 'blocked' | 'not_reached' | 'miss' | string;
};

export type LiveCallTraceSummary = {
  total_calls: number;
  requested_calls: number;
  successful_calls: number;
  failed_calls: number;
  cache_hit_calls: number;
  stale_cache_calls: number;
  expected_total: number;
  expected_satisfied: number;
  expected_ok_count?: number;
  expected_blocked_count?: number;
  expected_not_reached_count?: number;
  expected_miss_count?: number;
  dropped_entries: number;
};

export type LiveCallTraceResponse = {
  request_id: string;
  endpoint: string;
  status: string;
  error_reason?: string | null;
  started_at_utc: string;
  finished_at_utc?: string | null;
  expected_calls: LiveCallExpectedRow[];
  expected_rollup: LiveCallExpectedRollup[];
  observed_calls: LiveCallEntry[];
  summary: LiveCallTraceSummary;
};

export type CacheStatsResponse = {
  hits: number;
  misses: number;
  entries: number;
};

export type MetricsResponse = Record<string, unknown>;

export type CacheClearResponse = {
  cleared: number;
};

export type CustomVehicleListResponse = {
  vehicles: VehicleProfile[];
};

export type VehicleMutationResponse = {
  vehicle: VehicleProfile;
};

export type VehicleDeleteResponse = {
  vehicle_id: string;
  deleted: boolean;
};

export type SignatureVerificationRequest = {
  payload: Record<string, unknown> | unknown[] | string;
  signature: string;
  secret?: string | null;
};

export type SignatureVerificationResponse = {
  valid: boolean;
  algorithm: string;
  signature: string;
  expected_signature: string;
};

export type RunManifestSummary = {
  run_id: string;
  signature?: Record<string, unknown>;
  [key: string]: unknown;
};

export type ProofDashboardSliceId =
  | 'v0'
  | 'a'
  | 'b'
  | 'c'
  | 'broad'
  | 'focused'
  | 'cold_hot'
  | 'osrm_ors'
  | 'theorem_artifact';

export type ProofArtifactLink = {
  label: string;
  href?: string | null;
  artifact?: string | null;
};

export type ProofDemoPresetId =
  | 'safe_singleton'
  | 'certified_set'
  | 'support_abstention'
  | 'preference_sensitive'
  | 'collapse_prone'
  | 'hot_rerun';

export type ProofDemoPreset = {
  id: ProofDemoPresetId;
  title: string;
  subtitle: string;
  focus: string;
};

export type RunArtifactsListResponse = {
  run_id: string;
  artifacts: Array<{
    name: string;
    endpoint: string;
    size_bytes: number;
  }>;
  provenance_endpoint: string;
};

export type FrontendArtifactInspectionItem = {
  name: string;
  label: string;
  description: string;
  expectation: 'guaranteed' | 'conditional';
  present: boolean;
  listed: boolean;
  sizeBytes?: number | null;
};

export type FrontendArtifactInspectionGroup = {
  id: string;
  label: string;
  description: string;
  items: FrontendArtifactInspectionItem[];
  presentCount: number;
  listedCount: number;
  missingExpectedCount: number;
};

type FrontendArtifactSpec = Omit<FrontendArtifactInspectionItem, 'present' | 'listed' | 'sizeBytes'>;

const FRONTEND_ARTIFACT_GROUP_SPECS: Array<{
  id: string;
  label: string;
  description: string;
  items: FrontendArtifactSpec[];
}> = [
  {
    id: 'core',
    label: 'Core run documents',
    description: 'Decision package, manifest, provenance, and signature documents that anchor the run.',
    items: [
      {
        name: 'decision_package.json',
        label: 'Decision package',
        description: 'Route decision payload with selected route, certification state, support state, and preference state',
        expectation: 'guaranteed',
      },
      {
        name: 'manifest.json',
        label: 'Manifest',
        description: 'Run manifest with artifact pointers and reproducibility metadata',
        expectation: 'guaranteed',
      },
      {
        name: 'provenance.json',
        label: 'Provenance',
        description: 'Evidence-source and live-data provenance used for the decision',
        expectation: 'guaranteed',
      },
      {
        name: 'signature.json',
        label: 'Signature',
        description: 'Signed reproducibility signature for the run package',
        expectation: 'conditional',
      },
    ],
  },
  {
    id: 'certification',
    label: 'Certification artifacts',
    description: 'Artifacts explaining why the selected route or certified set is defensible.',
    items: [
      {
        name: 'route_fragility_map.json',
        label: 'Route fragility',
        description: 'Per-route fragility by evidence family and perturbation axis',
        expectation: 'conditional',
      },
      {
        name: 'competitor_fragility_map.json',
        label: 'Competitor fragility',
        description: 'Fragility evidence for active challenger routes',
        expectation: 'conditional',
      },
      {
        name: 'decision_region_summary.json',
        label: 'Decision region',
        description: 'Nearest certificate boundary and active challenge directions',
        expectation: 'conditional',
      },
      {
        name: 'flip_radius_summary.json',
        label: 'Flip radius',
        description: 'Minimum perturbation budget needed to overturn the selected route',
        expectation: 'conditional',
      },
    ],
  },
  {
    id: 'worlds',
    label: 'World and refresh artifacts',
    description: 'Sampled-world, evidence-refresh, and controller artifacts used by DCCS/REFC/VOI lanes.',
    items: [
      {
        name: 'sampled_world_manifest.json',
        label: 'Sampled worlds',
        description: 'Stress and uncertainty worlds sampled for certification',
        expectation: 'conditional',
      },
      {
        name: 'value_of_refresh.json',
        label: 'Value of refresh',
        description: 'Expected certificate or margin gain from refreshing evidence families',
        expectation: 'conditional',
      },
      {
        name: 'controller_trace.json',
        label: 'Controller trace',
        description: 'VOI controller actions, budgets, and stop reasons',
        expectation: 'conditional',
      },
      {
        name: 'theorem_hook_summary.json',
        label: 'Theorem hooks',
        description: 'Claim-map hooks linking run evidence to thesis theorem obligations',
        expectation: 'conditional',
      },
    ],
  },
];

export function buildFrontendArtifactInspectionGroups({
  decisionPackage,
  listedArtifactNames,
  artifacts,
}: {
  decisionPackage?: DecisionPackage | null;
  listedArtifactNames?: string[] | null;
  artifacts?: RunArtifactsListResponse | null;
}): FrontendArtifactInspectionGroup[] {
  const listedNames = new Set((listedArtifactNames ?? []).filter(Boolean));
  const artifactByName = new Map((artifacts?.artifacts ?? []).map((artifact) => [artifact.name, artifact]));
  const packagePointers = decisionPackage?.artifact_pointers ?? {};
  const pointerNames = new Set(Object.values(packagePointers).filter((value): value is string => Boolean(value)));
  const groups = FRONTEND_ARTIFACT_GROUP_SPECS.map((group) => {
    const items = group.items.map((item) => {
      const present = artifactByName.has(item.name) || pointerNames.has(item.name);
      const listed = listedNames.has(item.name) || present;
      return {
        ...item,
        present,
        listed,
        sizeBytes: artifactByName.get(item.name)?.size_bytes ?? null,
      };
    });
    return {
      ...group,
      items,
      presentCount: items.filter((item) => item.present).length,
      listedCount: items.filter((item) => item.listed).length,
      missingExpectedCount: items.filter((item) => item.expectation === 'guaranteed' && !item.present).length,
    };
  });

  const knownNames = new Set(groups.flatMap((group) => group.items.map((item) => item.name)));
  const extraItems = (artifacts?.artifacts ?? [])
    .filter((artifact) => !knownNames.has(artifact.name))
    .map((artifact) => ({
      name: artifact.name,
      label: artifact.name,
      description: 'Additional run-store artifact returned by the backend',
      expectation: 'conditional' as const,
      present: true,
      listed: listedNames.has(artifact.name),
      sizeBytes: artifact.size_bytes,
    }));
  if (extraItems.length) {
    groups.push({
      id: 'additional',
      label: 'Additional artifacts',
      description: 'Artifacts returned by the backend that are not part of the fixed frontend inspection catalogue.',
      items: extraItems,
      presentCount: extraItems.length,
      listedCount: extraItems.filter((item) => item.listed).length,
      missingExpectedCount: 0,
    });
  }

  if (!decisionPackage && !artifacts && listedNames.size === 0) {
    return [];
  }
  return groups.filter((group) => group.items.length > 0);
}

export type StrictErrorDetail = {
  reason_code?: StrictReasonCode;
  message?: string;
  warnings?: string[];
  [key: string]: unknown;
};
