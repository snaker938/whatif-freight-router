# Synthetic Incidents and Weather

Last Updated: 2026-03-31  
Applies To: route-producing backend endpoints

This document defines the current weather and synthetic-incident request fields used for controlled scenario testing.

The fields below correspond to `backend/app/models.py` request models. Weather controls use `WeatherImpactConfig`; incident controls use `IncidentSimulatorConfig`.

## Supported Endpoints

- `POST /route`
- `POST /pareto`
- `POST /pareto/stream`
- `POST /batch/pareto`
- `POST /batch/import/csv`
- `POST /scenario/compare`
- `POST /departure/optimize`
- `POST /duty/chain`

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

Current weather settings are intentionally bounded. `profile` accepts the configured weather profile enum, `intensity` is clamped between `0.0` and `2.0`, and `apply_incident_uplift` decides whether weather also scales incident severity.

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
    "max_events_per_route": 12
  }
}
```

The incident simulator currently supports per-100km event rates and fixed delays for dwell, accident, and closure events. `max_events_per_route` caps the total synthetic incident count for a route.

## Determinism

- use explicit `seed` for repeatability
- keep identical route inputs and model assets to reproduce outcomes
- these controls are useful for controlled local experiments, not as substitutes for live evidence in strict runtime
- the simulator defaults are defined in `backend/app/models.py` and should be treated as the canonical source when adjusting examples

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Backend APIs and Tooling](backend-api-tools.md)
- [API Cookbook](api-cookbook.md)
- [Quality Gates and Benchmarks](quality-gates-and-benchmarks.md)
