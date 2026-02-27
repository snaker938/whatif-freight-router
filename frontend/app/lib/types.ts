export type ScenarioMode = 'no_sharing' | 'partial_sharing' | 'full_sharing';
export type ParetoMethod = 'dominance' | 'epsilon_constraint';
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
};

export type RouteResponse = {
  selected: RouteOption;
  candidates: RouteOption[];
};

export type ParetoResponse = {
  routes: RouteOption[];
  warnings?: string[];
  diagnostics?: Record<string, string | number | boolean>;
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
  candidate_diagnostics?: Record<string, unknown> | null;
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

export type RunArtifactsListResponse = {
  run_id: string;
  artifacts: Array<{
    name: string;
    endpoint: string;
    size_bytes: number;
  }>;
  provenance_endpoint: string;
};

export type StrictErrorDetail = {
  reason_code?: StrictReasonCode;
  message?: string;
  warnings?: string[];
  [key: string]: unknown;
};
