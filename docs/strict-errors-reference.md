# Strict Error Contract Reference

Last Updated: 2026-04-09
Applies To: `backend/app/main.py`, `backend/app/model_data_errors.py`, `backend/app/models.py`

This document describes strict reason-code failures used across route-producing APIs.

## Canonical Error Shape (`422`)

```json
{
  "detail": {
    "reason_code": "terrain_dem_coverage_insufficient",
    "message": "Terrain DEM coverage below threshold.",
    "warnings": [
      "Coverage 0.91 < required 0.96"
    ],
    "stage": "collecting_candidates",
    "stage_detail": "terrain_fail_closed",
    "terrain_dem_version": "uk_dem_v1",
    "terrain_coverage_required": 0.96,
    "terrain_coverage_min_observed": 0.91
  }
}
```

Optional keys are included when relevant:

- `stage`
- `stage_detail`
- `terrain_dem_version`
- `terrain_coverage_required`
- `terrain_coverage_min_observed`
- `candidate_diagnostics`
- `retry_after_seconds`
- `retry_hint`

## Canonical Stream Fatal Shape

```json
{
  "type": "fatal",
  "reason_code": "epsilon_infeasible",
  "message": "No routes satisfy epsilon constraints for this request.",
  "warnings": []
}
```

Stream fatal events can also carry `stage`, `stage_detail`, `stage_elapsed_ms`, `last_error`, `retry_after_seconds`, `retry_hint`, and `candidate_diagnostics` when the underlying failure provides that context.

## Frozen Reason Codes

The backend normalizes to the frozen set below (`FROZEN_REASON_CODES`).

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

### Terrain And Graph

- `terrain_region_unsupported`: non-UK request under UK-only terrain policy.
- `terrain_dem_asset_unavailable`: DEM asset/load/live tile unavailable.
- `terrain_dem_coverage_insufficient`: coverage below strict threshold.
- `routing_graph_unavailable`: routing graph missing/invalid for strict run.
- `routing_graph_deferred_load`: deferred graph load did not complete in time.
- `routing_graph_warming_up`: startup warmup still in progress.
- `routing_graph_warmup_failed`: warmup failed before readiness.
- `routing_graph_fragmented`: the loaded graph is fragmented.
- `routing_graph_disconnected_od`: origin/destination disconnected in the graph.
- `routing_graph_coverage_gap`: origin/destination neighborhood coverage is insufficient.
- `routing_graph_precheck_timeout`: graph feasibility precheck exceeded budget.
- `routing_graph_no_path`: no path survived the strict graph search.

### Toll And Cost Inputs

- `toll_topology_unavailable`: live or strict topology unavailable.
- `toll_tariff_unavailable`: tariff source unavailable.
- `toll_tariff_unresolved`: candidate could not resolve toll tariff mapping.

### Fuel And Carbon Inputs

- `fuel_price_auth_unavailable`: required auth/API credentials missing/invalid.
- `fuel_price_source_unavailable`: missing/stale/invalid price payload.
- `carbon_policy_unavailable`: carbon policy schedule unavailable.
- `carbon_intensity_unavailable`: carbon intensity lookup unavailable.

### Scenario And Stochastic Inputs

- `scenario_profile_unavailable`: scenario profile/context not retrievable or too stale.
- `scenario_profile_invalid`: schema/monotonicity/transform violations.
- `stochastic_calibration_unavailable`: strict stochastic asset missing/invalid.
- `live_source_refresh_failed`: live source refresh gate failed before route candidate search could continue.

For stale scenario coefficients, diagnostics prioritize freshness fields (`as_of_utc`, `age_minutes`, `max_age_minutes`) and only include scenario coverage-gate summaries when those fields are actually present in the failure payload.

### Vehicle And Risk Inputs

- `vehicle_profile_unavailable`: unknown/missing vehicle profile in strict flow.
- `vehicle_profile_invalid`: strict validation failed for vehicle profile payload.
- `risk_normalization_unavailable`: normalization artifact unavailable.
- `risk_prior_unavailable`: uncertainty prior missing for strict requirements.

### Selection And Feasibility

- `epsilon_infeasible`: no candidate satisfies provided epsilon constraints.
- `no_route_candidates`: no viable candidates after strict filtering.
- `route_compute_timeout`: strict route attempt exceeded its timeout budget.
- `baseline_route_unavailable`: OSRM or ORS baseline route could not be computed.
- `baseline_provider_unconfigured`: baseline provider is not configured for the requested path.
- `model_asset_unavailable`: generic strict model asset availability failure.

## Error Payload Notes

- `POST /route`, `POST /pareto`, and related strict flows may include `stage` and `stage_detail` in the error detail.
- `POST /route` and `POST /pareto` stream fatal events may additionally include `stage_elapsed_ms`, `retry_hint`, `retry_after_seconds`, and `candidate_diagnostics` when that context is available.
- `GET /health/ready` reports graph readiness separately from live-source readiness, so a request can be graph-ready but still return `recommended_action=refresh_live_sources`.
- Only reason codes in `FROZEN_REASON_CODES` are emitted as canonical strict codes; internal loader labels that are not frozen normalize to `model_asset_unavailable`.

## Notes For Batch And Stream Consumers

- Batch per-pair failures are serialized into `error` text in strict format:
  `reason_code:<code>; message:<message>` with an optional `; warning=<first warning>` suffix.
- Stream fatal events preserve reason code and warning list for machine parsing.
- Unknown/internal codes are normalized to the frozen set before emission.
- Route readiness should be checked via `GET /health/ready`; strict mode now includes `strict_live` and may return `recommended_action=refresh_live_sources` when scenario coefficients are stale.

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Backend APIs and Tooling](backend-api-tools.md)
- [Model Assets and Data Sources](model-assets-and-data-sources.md)
- [API Cookbook](api-cookbook.md)
