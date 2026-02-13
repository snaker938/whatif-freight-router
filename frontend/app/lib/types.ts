export type ScenarioMode = 'no_sharing' | 'partial_sharing' | 'full_sharing';
export type ParetoMethod = 'dominance' | 'epsilon_constraint';
export type TerrainProfile = 'flat' | 'rolling' | 'hilly';
export type OptimizationMode = 'expected_value' | 'robust';

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

export type TimeWindowConstraints = {
  earliest_arrival_utc?: string;
  latest_arrival_utc?: string;
};

export type LatLng = { lat: number; lon: number };
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
  incident_events?: SimulatedIncidentEvent[];
  weather_summary?: WeatherSummary | null;
};

export type ParetoResponse = { routes: RouteOption[] };

export type ParetoStreamMetaEvent = {
  type: 'meta';
  total: number;
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

export type ParetoStreamFatalEvent = {
  type: 'fatal';
  message: string;
};

export type ParetoStreamDoneEvent = {
  type: 'done';
  done: number;
  total: number;
  routes: RouteOption[];
  warning_count?: number;
  warnings?: string[];
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
};

export type VehicleListResponse = { vehicles: VehicleProfile[] };

export type ScenarioCompareResult = {
  scenario_mode: ScenarioMode;
  selected: RouteOption | null;
  candidates: RouteOption[];
  warnings: string[];
  fallback_used: boolean;
  error?: string | null;
};

export type ScenarioCompareResponse = {
  run_id: string;
  results: ScenarioCompareResult[];
  deltas: Record<string, Record<string, number>>;
  scenario_manifest_endpoint: string;
  scenario_signature_endpoint: string;
};

export type ScenarioCompareRequest = {
  origin: LatLng;
  destination: LatLng;
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
  pareto_method?: ParetoMethod;
  epsilon?: EpsilonConstraints;
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
  vehicle_type?: string;
  scenario_mode?: ScenarioMode;
  weights?: { time: number; money: number; co2: number };
  max_alternatives?: number;
  cost_toggles?: CostToggles;
  terrain_profile?: TerrainProfile;
  stochastic?: StochasticConfig;
  optimization_mode?: OptimizationMode;
  risk_aversion?: number;
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
  fallback_used: boolean;
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
  fallback_used: boolean;
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
  departure_time_utc?: string;
  pareto_method?: ParetoMethod;
  epsilon?: EpsilonConstraints;
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
