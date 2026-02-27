# Strict Error Contract Reference

Last Updated: 2026-02-23  
Applies To: `backend/app/main.py`, `backend/app/model_data_errors.py`

This document describes strict reason-code failures used across route-producing APIs.

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

Optional keys are included when relevant:

- `terrain_dem_version`
- `terrain_coverage_required`
- `terrain_coverage_min_observed`

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

The backend normalizes to the frozen set below (`FROZEN_REASON_CODES`).

- `routing_graph_unavailable`
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
- `model_asset_unavailable`

## Primary Failure Families

### Terrain and Graph

- `terrain_region_unsupported`: non-UK request under UK-only terrain policy.
- `terrain_dem_asset_unavailable`: DEM asset/load/live tile unavailable.
- `terrain_dem_coverage_insufficient`: coverage below strict threshold.
- `routing_graph_unavailable`: routing graph missing/invalid for strict run.

### Toll and Cost Inputs

- `toll_topology_unavailable`: live or strict topology unavailable.
- `toll_tariff_unavailable`: tariff source unavailable.
- `toll_tariff_unresolved`: candidate could not resolve toll tariff mapping.

### Fuel and Carbon Inputs

- `fuel_price_auth_unavailable`: required auth/API credentials missing/invalid.
- `fuel_price_source_unavailable`: missing/stale/invalid price payload.
- `carbon_policy_unavailable`: carbon policy schedule unavailable.
- `carbon_intensity_unavailable`: carbon intensity lookup unavailable.

### Scenario and Stochastic Inputs

- `scenario_profile_unavailable`: scenario profile/context not retrievable or too stale.
- `scenario_profile_invalid`: schema/monotonicity/transform violations.
- `stochastic_calibration_unavailable`: strict stochastic asset missing/invalid.

For stale scenario coefficients, diagnostics now prioritize freshness fields (`as_of_utc`, `age_minutes`, `max_age_minutes`) and only include scenario coverage-gate summaries when those fields are actually present in the failure payload.

### Vehicle and Risk Inputs

- `vehicle_profile_unavailable`: unknown/missing vehicle profile in strict flow.
- `vehicle_profile_invalid`: strict validation failed for vehicle profile payload.
- `risk_normalization_unavailable`: normalization artifact unavailable.
- `risk_prior_unavailable`: uncertainty prior missing for strict requirements.

### Selection and Feasibility

- `epsilon_infeasible`: no candidate satisfies provided epsilon constraints.
- `no_route_candidates`: no viable candidates after strict filtering.
- `model_asset_unavailable`: generic strict model asset availability failure.

## Notes for Batch and Stream Consumers

- Batch per-pair failures are serialized into `error` text in strict format:
  `reason_code:<code>; message:<message>`
- Stream fatal events preserve reason code and warning list for machine parsing.
- Unknown/internal codes are normalized to the frozen set before emission.
- Route readiness should be checked via `GET /health/ready`; strict mode now includes `strict_live` and may return `recommended_action=refresh_live_sources` when scenario coefficients are stale.

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Backend APIs and Tooling](backend-api-tools.md)
- [Model Assets and Data Sources](model-assets-and-data-sources.md)
- [API Cookbook](api-cookbook.md)

