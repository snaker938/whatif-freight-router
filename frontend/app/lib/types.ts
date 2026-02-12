export type ScenarioMode = 'no_sharing' | 'partial_sharing' | 'full_sharing';
export type ParetoMethod = 'dominance' | 'epsilon_constraint';
export type TerrainProfile = 'flat' | 'rolling' | 'hilly';

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

export type LatLng = { lat: number; lon: number };

export type RouteMetrics = {
  distance_km: number;
  duration_s: number;
  monetary_cost: number;
  emissions_kg: number;
  avg_speed_kmh: number;
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
  segment_breakdown?: Array<Record<string, string | number>>;
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
  weights?: { time: number; money: number; co2: number };
  max_alternatives?: number;
  cost_toggles?: CostToggles;
  terrain_profile?: TerrainProfile;
  departure_time_utc?: string;
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

export type DepartureOptimizeRequest = {
  origin: LatLng;
  destination: LatLng;
  vehicle_type?: string;
  scenario_mode?: ScenarioMode;
  weights?: { time: number; money: number; co2: number };
  max_alternatives?: number;
  cost_toggles?: CostToggles;
  terrain_profile?: TerrainProfile;
  pareto_method?: ParetoMethod;
  epsilon?: EpsilonConstraints;
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
