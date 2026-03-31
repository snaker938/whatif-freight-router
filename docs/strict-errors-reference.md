# Strict Error Contract Reference

Last Updated: 2026-03-31  
Applies To: `backend/app/main.py`, `backend/app/model_data_errors.py`

This document describes the current frozen strict reason-code failures used across route-producing APIs.

## Canonical Error Shape (`422`)

```json
{
  "detail": {
    "reason_code": "terrain_dem_coverage_insufficient",
    "message": "Terrain DEM coverage below threshold.",
    "warnings": [
      "Coverage 0.91 < required 0.96"
    ]
  }
}
```

Optional keys are included when relevant, for example terrain coverage fields or strict-live freshness diagnostics.

Strict-runtime behavior is also shaped by settings such as:

- `STRICT_LIVE_DATA_REQUIRED`
- `LIVE_ROUTE_COMPUTE_REFRESH_MODE`
- `LIVE_ROUTE_COMPUTE_REQUIRE_ALL_EXPECTED`
- `LIVE_ROUTE_COMPUTE_PROBE_TERRAIN`
- `ROUTE_GRAPH_FAST_STARTUP_ENABLED`

In current strict paths, the backend forces live-data freshness, requires the expected live route-compute sources, probes terrain when full strict evidence is needed, and disables route-graph fast startup.

## Canonical Stream Fatal Shape

```json
{
  "type": "fatal",
  "reason_code": "epsilon_infeasible",
  "message": "No routes satisfy epsilon constraints for this request.",
  "warnings": []
}
```

## Frozen Reason Codes

The backend normalizes route-producing failures to the current `FROZEN_REASON_CODES` set:

- `routing_graph_unavailable`
- `routing_graph_fragmented`
- `routing_graph_disconnected_od`
- `routing_graph_coverage_gap`
- `routing_graph_no_path`
- `routing_graph_precheck_timeout`
- `routing_graph_deferred_load`
- `routing_graph_warming_up`
- `routing_graph_warmup_failed`
- `live_source_refresh_failed`
- `route_compute_timeout`
- `departure_profile_unavailable`
- `holiday_data_unavailable`
- `stochastic_calibration_unavailable`
- `scenario_profile_unavailable`
- `scenario_profile_invalid`
- `risk_normalization_unavailable`
- `risk_prior_unavailable`
- `terrain_region_unsupported`
- `terrain_dem_asset_unavailable`
- `terrain_dem_coverage_insufficient`
- `toll_topology_unavailable`
- `toll_tariff_unavailable`
- `toll_tariff_unresolved`
- `fuel_price_auth_unavailable`
- `fuel_price_source_unavailable`
- `vehicle_profile_unavailable`
- `vehicle_profile_invalid`
- `carbon_policy_unavailable`
- `carbon_intensity_unavailable`
- `epsilon_infeasible`
- `no_route_candidates`
- `baseline_route_unavailable`
- `baseline_provider_unconfigured`
- `model_asset_unavailable`

## Primary Failure Families

### Graph Readiness and Connectivity

- `routing_graph_unavailable`: graph asset missing or unusable
- `routing_graph_fragmented`: graph loaded but failed connectivity quality gates
- `routing_graph_disconnected_od`: origin and destination are disconnected in the strict graph
- `routing_graph_coverage_gap`: graph coverage does not adequately support the requested OD
- `routing_graph_no_path`: graph search completed but found no route
- `routing_graph_precheck_timeout`: bounded graph precheck timed out
- `routing_graph_deferred_load`: graph not yet loaded for the requested path
- `routing_graph_warming_up`: startup warmup still in progress
- `routing_graph_warmup_failed`: warmup exceeded timeout or failed

### Live Refresh and Runtime Timeouts

- `live_source_refresh_failed`: strict route-compute refresh could not obtain the required live data
- `route_compute_timeout`: route-producing attempt exceeded its configured timeout budget

### Scenario and Stochastic Inputs

- `scenario_profile_unavailable`: scenario profile or required live context not retrievable or too stale
- `scenario_profile_invalid`: schema, transform, or monotonicity validation failed
- `stochastic_calibration_unavailable`: strict stochastic calibration missing or invalid
- `departure_profile_unavailable`: departure profile unavailable
- `holiday_data_unavailable`: holiday calendar input unavailable

### Risk and Asset Inputs

- `risk_normalization_unavailable`: normalization artifact unavailable
- `risk_prior_unavailable`: uncertainty prior unavailable
- `model_asset_unavailable`: generic strict model-asset availability failure

### Terrain, Toll, Fuel, and Carbon Inputs

- `terrain_region_unsupported`: request outside supported terrain region
- `terrain_dem_asset_unavailable`: DEM asset or live tile unavailable
- `terrain_dem_coverage_insufficient`: DEM coverage below strict threshold
- `toll_topology_unavailable`: toll topology unavailable
- `toll_tariff_unavailable`: toll tariff source unavailable
- `toll_tariff_unresolved`: toll tariff could not be resolved for a route segment
- `fuel_price_auth_unavailable`: required fuel auth or credentials missing
- `fuel_price_source_unavailable`: fuel source missing, stale, or invalid
- `carbon_policy_unavailable`: carbon policy schedule unavailable
- `carbon_intensity_unavailable`: carbon intensity input unavailable

### Vehicle and Baseline Inputs

- `vehicle_profile_unavailable`: unknown or missing vehicle profile
- `vehicle_profile_invalid`: vehicle profile failed validation
- `baseline_route_unavailable`: requested baseline route could not be produced
- `baseline_provider_unconfigured`: baseline provider is not configured for the requested path

### Feasibility and Selection

- `epsilon_infeasible`: no route satisfies the supplied epsilon constraints
- `no_route_candidates`: no viable candidates remained after strict filtering

## Notes for Consumers

- batch per-pair failures are serialized into `error` text using `reason_code:<code>; message:<message>`
- stream fatal events preserve reason code and warnings for machine parsing
- unknown or internal error labels are normalized into the frozen set before emission
- readiness-related failures should be interpreted alongside `GET /health/ready`

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Backend APIs and Tooling](backend-api-tools.md)
- [Model Assets and Data Sources](model-assets-and-data-sources.md)
- [API Cookbook](api-cookbook.md)
