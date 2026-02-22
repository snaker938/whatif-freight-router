# Strict Error Contract Reference

Last Updated: 2026-02-21  
Applies To: all route-producing backend flows

## Non-Stream Error Shape (`422`)

```json
{
  "detail": {
    "reason_code": "terrain_dem_coverage_insufficient",
    "message": "Terrain DEM coverage below threshold.",
    "warnings": ["Coverage 0.91 < required 0.98"]
  }
}
```

## Stream Fatal Shape

```json
{
  "type": "fatal",
  "reason_code": "epsilon_infeasible",
  "message": "No candidates satisfy epsilon constraints.",
  "warnings": []
}
```

## Frozen Reason Codes

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

## Related Docs

Fuel strict semantics:

- missing/invalid live auth when required maps to `fuel_price_auth_unavailable`
- missing/stale/invalid source data maps to `fuel_price_source_unavailable`
- signed fallback is only accepted when `LIVE_FUEL_ALLOW_SIGNED_FALLBACK=true` and signature/freshness checks pass
- unknown vehicle id maps to `vehicle_profile_unavailable`
- invalid/stale vehicle profile asset maps to `vehicle_profile_invalid`
- missing/stale scenario policy asset maps to `scenario_profile_unavailable`
- missing required `LIVE_SCENARIO_COEFFICIENT_URL` in strict runtime maps to `scenario_profile_unavailable`
- invalid/non-monotonic scenario policy asset maps to `scenario_profile_invalid`
- missing/stale/incomplete free live scenario context (WebTRIS, Traffic England, DfT, Open-Meteo) maps to `scenario_profile_unavailable`
- invalid scenario live payload structure or trusted-host policy failures map to `scenario_profile_invalid`
- missing stochastic posterior context model or missing shock quantile mappings in strict runtime map to `stochastic_calibration_unavailable`

- [Documentation Index](README.md)
- [Backend APIs and Tooling](backend-api-tools.md)
- [API Cookbook](api-cookbook.md)
- [Quality Gates and Benchmarks](quality-gates-and-benchmarks.md)
