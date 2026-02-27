import type {
  DepartureOptimizeRequest,
  DutyChainRequest,
  EpsilonConstraints,
  IncidentSimulatorConfig,
  LatLng,
  OptimizationMode,
  ParetoMethod,
  ParetoRequest,
  RouteRequest,
  ScenarioCompareRequest,
  ScenarioMode,
  StochasticConfig,
  TerrainProfile,
  TimeWindowConstraints,
  Waypoint,
  WeatherImpactConfig,
  EmissionsContext,
  CostToggles,
} from './types';

export type RoutingAdvancedPatch = {
  pareto_method?: ParetoMethod;
  epsilon?: EpsilonConstraints;
  departure_time_utc?: string;
  cost_toggles?: CostToggles;
  terrain_profile?: TerrainProfile;
  stochastic?: StochasticConfig;
  optimization_mode?: OptimizationMode;
  risk_aversion?: number;
  emissions_context?: EmissionsContext;
  weather?: WeatherImpactConfig;
  incident_simulation?: IncidentSimulatorConfig;
};

type RoutingCommonInput = {
  vehicle_type: string;
  scenario_mode: ScenarioMode;
  weights: { time: number; money: number; co2: number };
  max_alternatives: number;
  advanced: RoutingAdvancedPatch;
};

type ODRoutingInput = RoutingCommonInput & {
  origin: LatLng;
  destination: LatLng;
  waypoints: Waypoint[];
};

export function buildRouteRequest(input: ODRoutingInput): RouteRequest {
  return {
    origin: input.origin,
    destination: input.destination,
    waypoints: input.waypoints,
    vehicle_type: input.vehicle_type,
    scenario_mode: input.scenario_mode,
    max_alternatives: input.max_alternatives,
    weights: input.weights,
    ...input.advanced,
  };
}

export function buildParetoRequest(input: ODRoutingInput): ParetoRequest {
  return {
    origin: input.origin,
    destination: input.destination,
    waypoints: input.waypoints,
    vehicle_type: input.vehicle_type,
    scenario_mode: input.scenario_mode,
    max_alternatives: input.max_alternatives,
    weights: input.weights,
    ...input.advanced,
  };
}

export function buildScenarioCompareRequest(input: ODRoutingInput): ScenarioCompareRequest {
  return {
    origin: input.origin,
    destination: input.destination,
    waypoints: input.waypoints,
    vehicle_type: input.vehicle_type,
    scenario_mode: input.scenario_mode,
    max_alternatives: input.max_alternatives,
    weights: input.weights,
    ...input.advanced,
  };
}

type DepartureOptimizeInput = ODRoutingInput & {
  window_start_utc: string;
  window_end_utc: string;
  step_minutes: number;
  time_window?: TimeWindowConstraints;
};

export function buildDepartureOptimizeRequest(
  input: DepartureOptimizeInput,
): DepartureOptimizeRequest {
  return {
    origin: input.origin,
    destination: input.destination,
    waypoints: input.waypoints,
    vehicle_type: input.vehicle_type,
    scenario_mode: input.scenario_mode,
    max_alternatives: input.max_alternatives,
    weights: input.weights,
    window_start_utc: input.window_start_utc,
    window_end_utc: input.window_end_utc,
    step_minutes: input.step_minutes,
    ...(input.time_window ? { time_window: input.time_window } : {}),
    ...input.advanced,
  };
}

type DutyChainInput = RoutingCommonInput & {
  stops: DutyChainRequest['stops'];
};

export function buildDutyChainRequest(input: DutyChainInput): DutyChainRequest {
  return {
    stops: input.stops,
    vehicle_type: input.vehicle_type,
    scenario_mode: input.scenario_mode,
    max_alternatives: input.max_alternatives,
    weights: input.weights,
    ...input.advanced,
  };
}
