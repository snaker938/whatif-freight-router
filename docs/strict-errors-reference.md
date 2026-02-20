# Strict Error Contract Reference

Last Updated: 2026-02-19  
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
- `terrain_region_unsupported`
- `terrain_dem_asset_unavailable`
- `terrain_dem_coverage_insufficient`
- `toll_topology_unavailable`
- `toll_tariff_unavailable`
- `toll_tariff_unresolved`
- `fuel_price_auth_unavailable`
- `fuel_price_source_unavailable`
- `carbon_policy_unavailable`
- `carbon_intensity_unavailable`
- `epsilon_infeasible`
- `no_route_candidates`

## Related Docs

- [Documentation Index](README.md)
- [Backend APIs and Tooling](backend-api-tools.md)
- [API Cookbook](api-cookbook.md)
- [Quality Gates and Benchmarks](quality-gates-and-benchmarks.md)
