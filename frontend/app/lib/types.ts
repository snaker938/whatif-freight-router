export type ScenarioMode = 'no_sharing' | 'partial_sharing' | 'full_sharing';

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
