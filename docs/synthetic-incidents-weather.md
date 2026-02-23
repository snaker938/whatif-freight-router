# Synthetic Incidents and Weather

Last Updated: 2026-02-19  
Applies To: route-producing backend endpoints

This document defines weather and synthetic incident request fields used for controlled scenario testing.

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

## Incident Simulation Block

```json
{
  "incident_simulation": {
    "enabled": true,
    "seed": 123,
    "dwell_rate_per_100km": 1.0,
    "accident_rate_per_100km": 0.3,
    "closure_rate_per_100km": 0.05,
    "max_events": 8
  }
}
```

## Determinism

- Use explicit `seed` for repeatability.
- Keep identical route inputs and model assets to reproduce outcomes.

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Backend APIs and Tooling](backend-api-tools.md)
- [API Cookbook](api-cookbook.md)
- [Quality Gates and Benchmarks](quality-gates-and-benchmarks.md)

