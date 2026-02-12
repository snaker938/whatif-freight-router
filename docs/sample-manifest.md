# Sample Manifest and Expected Output

This document shows one canonical `batch/pareto` input and example outputs for
documentation and reproducibility.

## Canonical OD input

- Origin: `{ "lat": 52.4862, "lon": -1.8904 }`
- Destination: `{ "lat": 51.5072, "lon": -0.1276 }`
- Vehicle: `rigid_hgv`
- Scenario: `no_sharing`

## Sample request

File: `docs/examples/sample_batch_request.json`

```json
{
  "pairs": [
    {
      "origin": { "lat": 52.4862, "lon": -1.8904 },
      "destination": { "lat": 51.5072, "lon": -0.1276 }
    }
  ],
  "vehicle_type": "rigid_hgv",
  "scenario_mode": "no_sharing",
  "max_alternatives": 5
}
```

## Expected batch response shape

File: `docs/examples/sample_batch_response.json`

```json
{
  "run_id": "<uuid>",
  "results": [
    {
      "origin": { "lat": 52.4862, "lon": -1.8904 },
      "destination": { "lat": 51.5072, "lon": -0.1276 },
      "routes": [
        {
          "id": "pair0_route0",
          "geometry": {
            "type": "LineString",
            "coordinates": [
              [-1.8904, 52.4862],
              [-0.1276, 51.5072]
            ]
          },
          "metrics": {
            "distance_km": 190.321,
            "duration_s": 11520.44,
            "monetary_cost": 171.84,
            "emissions_kg": 242.602,
            "avg_speed_kmh": 59.48
          }
        }
      ],
      "error": null
    }
  ]
}
```

`run_id` is generated per run, so docs use the placeholder `"<uuid>"`.

## Expected manifest shape

File: `docs/examples/sample_manifest.json`

```json
{
  "run_id": "<uuid>",
  "created_at": "<iso8601-utc>",
  "type": "batch_pareto",
  "pair_count": 1,
  "vehicle_type": "rigid_hgv",
  "scenario_mode": "no_sharing",
  "max_alternatives": 5,
  "batch_concurrency": 8,
  "duration_ms": 482.37,
  "error_count": 0
}
```

`created_at` is generated at runtime in UTC, so docs use the placeholder
`"<iso8601-utc>"`.

## Fetching a manifest

After calling `POST /batch/pareto`, read the returned `run_id` and fetch:

- `GET /runs/{run_id}/manifest`

Example:

```text
GET /runs/0f4c8b80-2c88-4f61-9954-2b67f6f90af4/manifest
```

## Assumptions and limitations

- Numbers in sample outputs are illustrative and may not match live OSRM runs.
- `run_id` and `created_at` are intentionally non-deterministic.
- The manifest currently stores run metadata only, not full route geometry.
