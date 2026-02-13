# Synthetic Incident Simulator and Weather Adapter

This document describes the synthetic weather and incident features added to the compute APIs.

## Supported Endpoints

The following request models now accept additive `weather` and `incident_simulation` fields:

- `POST /route`
- `POST /pareto`
- `POST /pareto/stream`
- `POST /batch/pareto`
- `POST /batch/import/csv`
- `POST /scenario/compare`
- `POST /departure/optimize`
- `POST /duty/chain`

When omitted, both features are disabled by default and behavior stays backward compatible.

## Request Schema

`weather`:

```json
{
  "enabled": false,
  "profile": "clear",
  "intensity": 1.0,
  "apply_incident_uplift": true
}
```

`profile` must be one of: `clear`, `rain`, `storm`, `snow`, `fog`.

`incident_simulation`:

```json
{
  "enabled": false,
  "seed": null,
  "dwell_rate_per_100km": 0.8,
  "accident_rate_per_100km": 0.25,
  "closure_rate_per_100km": 0.05,
  "dwell_delay_s": 120.0,
  "accident_delay_s": 480.0,
  "closure_delay_s": 900.0,
  "max_events_per_route": 12
}
```

## Computation Order

Duration stages are applied in this order:

1. `baseline`
2. `time_of_day`
3. `scenario`
4. `weather`
5. `gradient`
6. `incidents`

This stage order is reflected in `eta_timeline` and `eta_explanations`.

## Weather Adapter Logic

Weather speed multipliers are table-driven by profile and scaled by intensity:

- `clear`: `1.00`
- `rain`: `1.08`
- `storm`: `1.20`
- `snow`: `1.28`
- `fog`: `1.12`

Incident-rate uplift multipliers (when `apply_incident_uplift=true`):

- `clear`: `1.00`
- `rain`: `1.15`
- `storm`: `1.50`
- `snow`: `1.80`
- `fog`: `1.25`

Intensity is clamped to `[0.0, 2.0]`. Multipliers are computed as:

- `1 + (profile_base - 1) * intensity`

## Synthetic Incident Model

Event types:

- `dwell`
- `accident`
- `closure`

For each segment and event type, event probability is:

- `min(0.98, rate_per_100km * weather_factor * segment_km / 100)`

If an event occurs:

- delay is `base_delay_s * random_scale`, where `random_scale` is in `[0.75, 1.25]`
- event start time is a random offset within the segment’s running timeline
- `event_id` is deterministic for the same route + seed + generated event values

Generated events are sorted by start time and truncated by `max_events_per_route`.

## Response Fields

`RouteMetrics` adds:

- `weather_delay_s`
- `incident_delay_s`

`RouteOption` adds:

- `weather_summary` (when weather is enabled)
- `incident_events`

Each segment entry in `segment_breakdown` includes:

- `incident_delay_s`

## Determinism Rules

- Weather multipliers are deterministic for the same `profile` + `intensity`.
- Incident generation is deterministic for the same `seed` and route geometry/signature.
- Different seeds can produce different event sets and delays.

## Manifest and Provenance

For batch runs, submitted weather and incident simulator configs are persisted additively in:

- run manifest `request` and `execution`
- artifact metadata
- provenance `input_received` event context
- structured request logs for compute endpoints

## Assumptions and Limitations

- Incidents are synthetic and parameterized; no live incident feed is used in this wave.
- Weather profiles are static request-time inputs; no live weather provider integration is used.
- Event semantics model delay only (no topology edits or live closure rerouting).
- This wave does not add frontend map overlays for incidents.
