export const SIDEBAR_SECTION_HINTS = {
  setup: 'Set the core run context before computing routes.',
  advancedParameters: 'Fine-tune optimization behavior and cost assumptions.',
  preferences: 'Set objective weights and generate candidate routes.',
  selectedRoute: 'Live summary for the currently selected route.',
  routes: 'Review generated options and select one to inspect.',
  compareScenarios: 'See No/Partial/Full sharing deltas side by side.',
  departureOptimization: 'Sweep a time window to find a better departure slot.',
  dutyChainPlanner: 'Run one vehicle across ordered multi-leg stops.',
  oracleQualityDashboard: 'Track feed validation, freshness, and reliability signals.',
  experiments: 'Save, filter, reload, and replay scenario bundles.',
  scenarioTimeLapse: 'Animate progress along the selected route.',
  devTools: 'Backend endpoint coverage controls and diagnostics.',
} as const;

export const SIDEBAR_FIELD_HELP = {
  vehicleType: 'Chooses profile factors used in cost and emissions.',
  scenarioMode:
    'Applies strict scenario policy multipliers to duration, incidents, fuel/energy, emissions, and uncertainty.',
  language: 'Controls UI language and locale formatting.',
  optimizationMode: 'Expected Value favors mean outcome; Robust penalizes variance.',
  riskAversion: 'Higher values prefer safer, lower-variance choices.',
  paretoMethod: 'Choose dominance-only or epsilon-constrained filtering.',
  epsilonDuration: 'Optional maximum allowed route duration in seconds.',
  epsilonMonetaryCost: 'Optional maximum allowed monetary proxy value.',
  epsilonEmissions: 'Optional maximum allowed emissions in kilograms.',
  departureTimeUtc: 'Sets route evaluation time for profile effects.',
  stochasticEnabled: 'Simulates travel-time uncertainty using repeated samples.',
  stochasticSeed: 'Fixes random sampling for reproducible runs.',
  stochasticSigma: 'Controls spread of sampled travel-time noise.',
  stochasticSamples: 'Number of simulated draws per candidate route.',
  terrainProfile: 'Approximates road-gradient impact on duration/emissions.',
  useTolls: 'Includes toll charges in monetary proxy calculations.',
  fuelPriceMultiplier: 'Scales fuel-cost contribution up or down.',
  carbonPrice: 'Adds emissions-linked carbon cost to the money objective.',
  tollCostPerKm: 'Per-kilometer toll amount when tolls are enabled.',
  maxAlternatives: 'Sets route candidate breadth before strict Pareto filtering.',
  fuelType: 'Fuel basis used by emissions and energy calculations.',
  euroClass: 'Vehicle emissions compliance class for context adjustments.',
  ambientTempC: 'Ambient temperature used in fuel/energy context.',
  weatherEnabled: 'Enables weather-aware travel-time and incident adjustments.',
  weatherProfile: 'Selects weather regime profile used by backend calibration.',
  weatherIntensity: 'Scales weather severity from 0 (none) to 2 (severe).',
  incidentSimulationEnabled: 'Enables synthetic incident simulation in backend.',
  incidentSeed: 'Optional deterministic seed for incident generation.',
  incidentDwellRate: 'Expected dwell events per 100km.',
  incidentAccidentRate: 'Expected accident events per 100km.',
  incidentClosureRate: 'Expected closure events per 100km.',
  incidentDwellDelay: 'Per-event dwell delay in seconds.',
  incidentAccidentDelay: 'Per-event accident delay in seconds.',
  incidentClosureDelay: 'Per-event closure delay in seconds.',
  incidentMaxEvents: 'Maximum simulated incidents allowed per route.',
  windowStartEnd: 'Defines the departure search interval in UTC.',
  stepMinutes: 'Sampling interval between evaluated departures.',
  earliestLatestArrival: 'Optional arrival feasibility constraints.',
  dutyStopsTextarea: 'One stop per line: lat,lon,label(optional).',
  oracleSource: 'Feed/source identifier used in quality aggregation.',
  schemaValid: 'Whether payload matched expected schema rules.',
  signatureState: 'Known signature validation status for the check.',
  freshnessSeconds: 'Age of source data at ingestion time.',
  latencyMs: 'Source delivery latency in milliseconds.',
  recordCount: 'Number of records observed in this check.',
  errorNote: 'Optional detail about failure or degradation.',
  catalogSearch: 'Find experiments by name, ID, or description.',
  filterVehicle: 'Show only bundles using selected vehicle profile.',
  filterScenario: 'Show only bundles for a selected scenario mode.',
  sort: 'Controls experiment list ordering.',
  experimentName: 'Short title for the saved bundle.',
  experimentDescription: 'Optional context for future recall.',
  playbackSpeed: 'Choose how quickly the route animation progresses.',
  computeMode: 'Choose stream Pareto, JSON Pareto, or single-route endpoint.',
} as const;

export const SIDEBAR_DROPDOWN_OPTIONS_HELP = {
  scenarioMode:
    'Options: No Sharing (baseline), Partial Sharing (moderate coordination), Full Sharing (maximum coordination).',
  optimizationMode:
    'Options: Expected Value (average objective ranking), Robust (average plus risk penalty).',
  paretoMethod:
    'Options: Dominance (non-dominated only), Epsilon Constraint (apply caps then non-dominance).',
  terrainProfile:
    'Options: Flat (minimal gradient effect), Rolling (moderate), Hilly (higher penalty).',
  signatureState:
    'Options: Unknown (not provided), Valid (passed), Invalid (failed).',
  experimentFilterScenario:
    'Options: All Scenarios, No Sharing, Partial Sharing, Full Sharing.',
  experimentSort:
    'Options: Updated Newest, Updated Oldest, Name A-Z, Name Z-A.',
  timeLapseSpeed:
    'Options: x0.5 (slower), x1 (normal), x2 (faster), x4 (fastest).',
} as const;

export function vehicleDescriptionFromId(vehicleId: string): string {
  if (vehicleId === 'van') return 'Light vehicle profile.';
  if (vehicleId === 'rigid_hgv') return 'Medium heavy-goods profile.';
  if (vehicleId === 'artic_hgv') return 'High-capacity articulated profile.';
  return 'Custom profile from configured vehicle store.';
}
