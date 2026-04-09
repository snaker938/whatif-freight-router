# Synthetic Incidents and Weather

Last Updated: 2026-04-09
Applies To: route-producing backend endpoints

This document defines weather and synthetic incident request fields used for controlled scenario testing and for the current route-scoring request models.

## Supported Endpoints

- `POST /route`
- `POST /pareto`
- `POST /pareto/stream`
- `POST /batch/pareto`
- `POST /batch/import/csv`
- `POST /scenario/compare`
- `POST /departure/optimize`
- `POST /duty/chain`

## Current Request Fields

`backend/app/models.py` currently exposes:

- `WeatherImpactConfig(enabled, profile, intensity, apply_incident_uplift)`
- `IncidentSimulatorConfig(enabled, seed, dwell_rate_per_100km, accident_rate_per_100km, closure_rate_per_100km, dwell_delay_s, accident_delay_s, closure_delay_s, max_events_per_route)`

The current weather profile literal is `clear | rain | storm | snow | fog`, intensity is bounded to `0.0..2.0`, and the default incident cap is per route rather than a global batch count.

## Weather Block

```json
{
  "weather": {
    "enabled": true,
    "profile": "rain",
    "intensity": 1.2,
    "apply_incident_uplift": true
  }
}
```

Weather is still a request-level multiplier rather than a separate route mode. In the current code path, `profile` feeds the weather multiplier logic, and `apply_incident_uplift` keeps weather effects coupled to the incident-adjusted travel-time path unless explicitly turned off.

## Incident Simulation Block

```json
{
  "incident_simulation": {
    "enabled": true,
    "seed": 123,
    "dwell_rate_per_100km": 1.0,
    "accident_rate_per_100km": 0.3,
    "closure_rate_per_100km": 0.05,
    "dwell_delay_s": 120.0,
    "accident_delay_s": 480.0,
    "closure_delay_s": 900.0,
    "max_events_per_route": 8
  }
}
```

The simulator produces deterministic `dwell`, `accident`, and `closure` events from the supplied seed, then sorts them by start offset, segment index, event type, and event id. The defaults in the request model are `0.8`, `0.25`, and `0.05` events per 100 km, with delays of `120`, `480`, and `900` seconds and a `max_events_per_route` cap of `12`.

## Determinism

- Use explicit `seed` for repeatability.
- Keep identical route inputs and model assets to reproduce outcomes.
- In strict live runtime, synthetic incident generation is forcibly disabled when `STRICT_LIVE_DATA_REQUIRED=true`, so this block is for controlled test and scenario-analysis paths rather than production live-evidence runs.

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Backend APIs and Tooling](backend-api-tools.md)
- [API Cookbook](api-cookbook.md)
- [Quality Gates and Benchmarks](quality-gates-and-benchmarks.md)

