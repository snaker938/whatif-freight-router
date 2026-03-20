export const SIDEBAR_SECTION_HINTS = {
  setup:
    'Set the basic trip first: start, end, vehicle, and scenario mode. If these are wrong, every result after this will be misleading.',
  advancedParameters:
    'This is where you control how strict, cautious, and realistic the model should be. Small changes here can change both the route shape and the route score.',
  preferences:
    'Set how much you care about time, money, and CO2, then run compute. These sliders decide which route is selected when several good options exist.',
  selectedRoute:
    'Shows the route you are currently focused on, with timing, cost, emissions, and segment-level detail. Use this to explain why that route was chosen.',
  routes:
    'Review all generated candidates, sort them, and pick one for deeper inspection. This is your route short-listing area.',
  compareScenarios:
    'Run the same trip under different sharing assumptions and compare outcomes side by side. Useful for quickly showing policy impact.',
  departureOptimization:
    'Search different departure times to find a better slot. This is useful when traffic patterns matter more than path shape.',
  dutyChainPlanner:
    'Plan one vehicle across many stops in order. This helps with practical multi-leg operations, not just one origin-destination pair.',
  oracleQualityDashboard:
    'Track source quality checks like freshness, schema validity, and signature state. Use this to judge whether live data can be trusted.',
  experiments:
    'Save complete run setups and reload them later. This makes comparisons repeatable and easier to share.',
  scenarioTimeLapse:
    'Animate travel progress across the selected route. Good for quickly seeing where time is spent along the corridor.',
  devTools:
    'Operational tools and diagnostics for backend behavior, live-call traces, and cache/health debugging.',
} as const;

export const SIDEBAR_FIELD_HELP = {
  vehicleType:
    'This tells the model what kind of vehicle you are routing. A heavier or less efficient vehicle usually costs more to run and emits more CO2, so route rankings can change when you switch this.',
  scenarioMode:
    'This controls the operating context used by the smart model. In practice, it changes how optimistic or conservative the route scoring is for time, incidents, fuel use, and uncertainty.',
  language:
    'Changes interface text and number/date formatting only. It does not change route math.',
  optimizationMode:
    'Expected Value favors the route with the best average outcome. Robust mode gives extra weight to consistency, so it can avoid routes that are fast on average but risky on bad days.',
  riskAversion:
    'Used when optimization mode is Robust. Higher values push selection toward steadier routes and away from routes with wider swings.',
  paretoMethod:
    'Controls how candidates are kept or dropped before final selection. Dominance keeps routes that are not clearly worse than another route. Epsilon mode first applies hard limits you set, then does the same dominance check.',
  epsilonDuration:
    'Maximum allowed trip time in seconds when Epsilon mode is active. Any route slower than this is removed before ranking.',
  epsilonMonetaryCost:
    'Maximum allowed money score when Epsilon mode is active. Use this when you need to enforce a hard budget ceiling.',
  epsilonEmissions:
    'Maximum allowed CO2 in kilograms when Epsilon mode is active. Use this when you must stay under an emissions cap.',
  departureTimeUtc:
    'Anchors the route to a specific departure time. This matters because traffic-like effects and live context can vary by time of day.',
  stochasticEnabled:
    'Turns on sampled travel-time uncertainty. With it on, the model runs many possible travel-time outcomes and can favor routes that are more reliable, not just fastest on paper.',
  stochasticSeed:
    'Locks the random sampling so repeated runs are reproducible. Useful when comparing settings and wanting the same random pattern each time.',
  stochasticSigma:
    'Controls how spread out the sampled travel times are. Low values keep outcomes close together; higher values create wider best-to-worst swings.',
  stochasticSamples:
    'How many sampled outcomes are simulated per candidate route. More samples usually give a steadier estimate but take more compute time.',
  terrainProfile:
    'Simple terrain shape assumption used by the cost and energy model. Flatter profiles generally reduce fuel and time penalties; hillier profiles increase them.',
  useTolls:
    'Includes toll charges in money scoring. Turn this off to compare purely non-toll economics.',
  fuelPriceMultiplier:
    'Scales fuel-related cost up or down to run stress tests. For example, use values above 1.0 to model expensive fuel periods.',
  carbonPrice:
    'Adds a money penalty for emissions. This lets you test how route choice changes when your business assigns financial value to CO2.',
  tollCostPerKm:
    'Fallback toll rate per kilometer for tolled segments when exact toll mapping is not available. In strict, healthy toll-data runs this usually has little or no effect.',
  maxAlternatives:
    'How many candidate routes are explored before filtering and ranking. Higher values can discover more options but increase runtime.',
  fuelType:
    'Sets the base fuel/energy behavior used in money and emissions calculations. Switching fuel type can significantly change route scores.',
  euroClass:
    'Vehicle emissions standard (Euro 4, 5, or 6). In UK context this reflects cleaner vs older engine standards; cleaner classes generally align with lower emissions assumptions.',
  ambientTempC:
    'Outside temperature used by the fuel/energy model. Extreme cold or heat can shift energy use and therefore cost and emissions.',
  weatherEnabled:
    'Applies weather effects to route timing and optionally incident pressure. This can make routes with fewer weak points become more attractive.',
  weatherProfile:
    'Weather pattern to simulate (clear, rain, storm, snow, fog). Different profiles apply different slow-down and disruption behavior.',
  weatherIntensity:
    'Strength of weather impact from 0 to 2. Higher values increase travel-time impact and, if enabled, can increase incident pressure.',
  incidentSimulationEnabled:
    'Adds simulated events such as dwell, accidents, and closures on top of baseline travel time. Useful for stress-testing reliability behavior.',
  incidentSeed:
    'Locks simulated incident placement and timing for repeatable comparisons.',
  incidentDwellRate:
    'Expected number of dwell-type disruptions per 100 km. Higher values create more short stops.',
  incidentAccidentRate:
    'Expected number of accident disruptions per 100 km. Higher values create more medium-delay events.',
  incidentClosureRate:
    'Expected number of closure disruptions per 100 km. Higher values create more severe detours or delays.',
  incidentDwellDelay:
    'Average delay in seconds added when a dwell event occurs.',
  incidentAccidentDelay:
    'Average delay in seconds added when an accident event occurs.',
  incidentClosureDelay:
    'Average delay in seconds added when a closure event occurs.',
  incidentMaxEvents:
    'Safety cap on total simulated events per route. Prevents extreme outlier runs from adding unrealistic numbers of disruptions.',
  windowStartEnd:
    'Start and end of the departure-time search window. The optimizer tests departures inside this interval.',
  stepMinutes:
    'Spacing between tested departure times. Smaller steps search more precisely but run more evaluations.',
  earliestLatestArrival:
    'Optional arrival bounds. Use these to reject departures that arrive too early or too late for operational constraints.',
  dutyStopsTextarea:
    'Multi-stop input area. Enter one stop per line as lat,lon,label. The planner runs the legs in order and returns totals across the full chain.',
  oracleSource:
    'Name of the feed/source being quality-checked. This is used for grouping and tracking reliability over time.',
  schemaValid:
    'Marks whether the incoming data matched the expected shape. If false, downstream route logic should treat this source carefully.',
  signatureState:
    'Records whether payload authenticity checks passed. Useful for governance and trust audits.',
  freshnessSeconds:
    'How old the source data was when checked. Lower is fresher.',
  latencyMs:
    'How long the source took to respond. Persistent high latency can lead to compute slowdowns or timeout risk.',
  recordCount:
    'Number of records seen in this check. Useful to spot empty or unexpectedly small payloads.',
  errorNote:
    'Optional human note for degraded checks. Use it to capture context that raw metrics do not explain.',
  catalogSearch:
    'Search saved experiments by name, ID, or description text.',
  filterVehicle:
    'Limit the experiment list to one vehicle type so comparisons stay fair.',
  filterScenario:
    'Limit the experiment list to one scenario mode for focused review.',
  sort:
    'Changes how saved experiments are ordered in the list.',
  experimentName:
    'Short, clear title for a saved setup. Good names make later comparison much faster.',
  experimentDescription:
    'Longer note for intent, assumptions, or outcome. Helpful for team handoff.',
  playbackSpeed:
    'Controls how fast the time-lapse marker moves along the selected route.',
  computeMode:
    'Choose how route computation is requested. Stream mode gives live progress updates, JSON mode returns one final response, and single-route mode asks only for a focused route response.',
} as const;

export const SIDEBAR_DROPDOWN_OPTIONS_HELP = {
  scenarioMode:
    'No Sharing is the baseline with minimal coordination assumptions. Partial Sharing applies moderate coordination effects. Full Sharing applies the strongest coordination assumptions and can shift both route ranking and route shape.',
  optimizationMode:
    'Expected Value picks what looks best on average. Robust gives extra weight to consistency and can prefer routes that are slightly slower but less likely to be badly delayed.',
  paretoMethod:
    'Dominance keeps routes that are not clearly worse than another route across all tracked goals. Epsilon Constraint first removes routes that break your hard limits, then keeps only the best remaining trade-off routes.',
  terrainProfile:
    'Flat assumes mild terrain impact, Rolling applies medium impact, and Hilly applies stronger grade penalties. This can change fuel use, cost, and sometimes final route order.',
  signatureState:
    'Unknown means no signature evidence was provided. Valid means checks passed. Invalid means checks failed and data trust should be treated carefully.',
  experimentFilterScenario:
    'All Scenarios shows everything. The other options narrow the list so you can compare only runs done under one scenario mode.',
  experimentSort:
    'Updated Newest is best for recent work. Updated Oldest helps replay older baselines. Name sorting is useful when you follow strict naming conventions.',
  timeLapseSpeed:
    'x0.5 is easiest to inspect closely, x1 is normal pace, x2 is quick review, and x4 is a fast pass for long routes.',
} as const;

export function vehicleDescriptionFromId(vehicleId: string): string {
  if (vehicleId === 'van') return 'Light commercial profile for smaller payload operations.';
  if (vehicleId === 'rigid_hgv') return 'Rigid heavy-goods profile with mid-to-high freight capacity.';
  if (vehicleId === 'artic_hgv') return 'Articulated heavy-goods profile for high-capacity long-haul work.';
  return 'Custom vehicle profile loaded from your configured vehicle catalog.';
}
