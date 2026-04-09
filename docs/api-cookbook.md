# API Cookbook

Last Updated: 2026-04-09
Applies To: local backend API usage from PowerShell

This page provides reproducible CLI examples for the strict backend API.

## 1) Base Payload

```powershell
$base = @{
  origin = @{ lat = 52.4862; lon = -1.8904 }
  destination = @{ lat = 51.5072; lon = -0.1276 }
  vehicle_type = "rigid_hgv"
  scenario_mode = "no_sharing"
  weights = @{ time = 1; money = 1; co2 = 1 }
  max_alternatives = 24
  pareto_method = "dominance"
  pipeline_mode = "legacy"
} | ConvertTo-Json -Depth 10
```

## 2) Single Route (`POST /route`)

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/route" -Method Post -ContentType "application/json" -Body $base
```

The response includes `selected`, `candidates`, `run_id`, `pipeline_mode`, and the run artifact pointers (`manifest_endpoint`, `artifacts_endpoint`, `provenance_endpoint`) when route execution succeeds.

## 3) Pareto (`POST /pareto`)

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/pareto" -Method Post -ContentType "application/json" -Body $base
```

The Pareto response includes `routes`, `warnings`, and a `diagnostics` object with the current candidate, precheck, and cache metrics.

## 4) Pareto Stream (`POST /pareto/stream`)

`/pareto/stream` and `/api/pareto/stream` emit NDJSON event lines.

```powershell
Invoke-WebRequest -Uri "http://localhost:8000/pareto/stream" -Method Post -ContentType "application/json" -Body $base
```

## 5) Scenario Compare (`POST /scenario/compare`)

```powershell
$compare = @{
  origin = @{ lat = 52.4862; lon = -1.8904 }
  destination = @{ lat = 53.4808; lon = -2.2426 }
  vehicle_type = "rigid_hgv"
  scenario_mode = "partial_sharing"
  max_alternatives = 24
} | ConvertTo-Json -Depth 10
Invoke-RestMethod -Uri "http://localhost:8000/scenario/compare" -Method Post -ContentType "application/json" -Body $compare
```

The `deltas` payload carries per-metric deltas and per-metric `*_status`, `*_reason_code`, `*_missing_source`, and `*_reason_source` fields.

## 6) Batch Pareto (`POST /batch/pareto`)

```powershell
$batch = Get-Content docs/examples/sample_batch_request.json -Raw
$batchResp = Invoke-RestMethod -Uri "http://localhost:8000/batch/pareto" -Method Post -ContentType "application/json" -Body $batch
$runId = $batchResp.run_id
$runId
```

The paired result objects contain `origin`, `destination`, `routes`, and `error`.

## 7) Batch CSV Import (`POST /batch/import/csv`)

```powershell
$csvText = @"
origin_lat,origin_lon,destination_lat,destination_lon
52.4862,-1.8904,51.5072,-0.1276
53.4808,-2.2426,53.8008,-1.5491
"@
$csvReq = @{
  csv_text = $csvText
  vehicle_type = "rigid_hgv"
  scenario_mode = "no_sharing"
  max_alternatives = 6
  seed = 20260409
  model_version = "thesis-script-v3"
} | ConvertTo-Json -Depth 10
Invoke-RestMethod -Uri "http://localhost:8000/batch/import/csv" -Method Post -ContentType "application/json" -Body $csvReq
```

## 8) Duty Chain (`POST /duty/chain`)

```powershell
$duty = @{
  stops = @(
    @{ lat = 52.4862; lon = -1.8904; label = "Birmingham" }
    @{ lat = 52.2053; lon = 0.1218; label = "Cambridge" }
    @{ lat = 51.5072; lon = -0.1276; label = "London" }
  )
  vehicle_type = "rigid_hgv"
  scenario_mode = "no_sharing"
  max_alternatives = 6
} | ConvertTo-Json -Depth 12
Invoke-RestMethod -Uri "http://localhost:8000/duty/chain" -Method Post -ContentType "application/json" -Body $duty
```

The response includes `legs`, `total_metrics`, `leg_count`, and `successful_leg_count`.

## 9) Departure Optimize (`POST /departure/optimize`)

```powershell
$dep = @{
  origin = @{ lat = 52.4862; lon = -1.8904 }
  destination = @{ lat = 51.5072; lon = -0.1276 }
  vehicle_type = "rigid_hgv"
  scenario_mode = "no_sharing"
  window_start_utc = "2026-04-09T06:00:00Z"
  window_end_utc = "2026-04-09T12:00:00Z"
  step_minutes = 60
  time_window = @{
    earliest_arrival_utc = "2026-04-09T09:00:00Z"
    latest_arrival_utc = "2026-04-09T16:00:00Z"
  }
} | ConvertTo-Json -Depth 12
Invoke-RestMethod -Uri "http://localhost:8000/departure/optimize" -Method Post -ContentType "application/json" -Body $dep
```

## 10) Route Baselines

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/route/baseline?realism=true" -Method Post -ContentType "application/json" -Body $base
Invoke-RestMethod -Uri "http://localhost:8000/route/baseline?realism=false" -Method Post -ContentType "application/json" -Body $base
Invoke-RestMethod -Uri "http://localhost:8000/route/baseline/ors?realism=true" -Method Post -ContentType "application/json" -Body $base
```

The baseline response includes `method`, `compute_ms`, `provider_mode`, `baseline_policy`, `asset_manifest_hash`, `asset_recorded_at`, `asset_freshness_status`, and optional `engine_manifest` plus `notes`.

## 11) Run Artifacts + Signatures

```powershell
$runId = "<run_id>"
Invoke-RestMethod -Uri "http://localhost:8000/runs/$runId/manifest"
Invoke-RestMethod -Uri "http://localhost:8000/runs/$runId/scenario-manifest"
Invoke-RestMethod -Uri "http://localhost:8000/runs/$runId/signature"
Invoke-RestMethod -Uri "http://localhost:8000/runs/$runId/scenario-signature"
Invoke-RestMethod -Uri "http://localhost:8000/runs/$runId/artifacts"
Invoke-RestMethod -Uri "http://localhost:8000/runs/$runId/artifacts/results.json"
Invoke-WebRequest -Uri "http://localhost:8000/runs/$runId/artifacts/report.pdf" -OutFile ".\report_$runId.pdf"
```

Use `GET /runs/{run_id}/artifacts/{artifact_name}` to retrieve any file in the run folder, including thesis-specific outputs such as thesis_report.md, thesis_summary.json, and thesis_metrics.json.

## 12) Signature Verify (`POST /verify/signature`)

```powershell
$verify = @{
  payload = @{ hello = "world" }
  signature = "abcdef"
  secret = "dev-manifest-signing-secret"
} | ConvertTo-Json -Depth 8
Invoke-RestMethod -Uri "http://localhost:8000/verify/signature" -Method Post -ContentType "application/json" -Body $verify
```

## 13) Readiness And Diagnostics

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health/ready"
Invoke-RestMethod -Uri "http://localhost:8000/metrics"
Invoke-RestMethod -Uri "http://localhost:8000/cache/stats"
Invoke-RestMethod -Uri "http://localhost:8000/cache?scope=thesis_cold" -Method Delete
Invoke-RestMethod -Uri "http://localhost:8000/cache/hot-rerun/restore" -Method Post
```

If a strict route call returns `x-route-request-id`, you can fetch the trace payload with:

```powershell
$requestId = "<x-route-request-id>"
Invoke-RestMethod -Uri "http://localhost:8000/debug/live-calls/$requestId"
```

## 14) Oracle Quality

```powershell
$check = @{
  source = "oracle_demo"
  schema_valid = $true
  signature_valid = $true
  freshness_s = 45
  latency_ms = 120
  record_count = 10
  observed_at_utc = "2026-04-09T10:00:00Z"
} | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8000/oracle/quality/check" -Method Post -ContentType "application/json" -Body $check
Invoke-RestMethod -Uri "http://localhost:8000/oracle/quality/dashboard" -Method Get
Invoke-RestMethod -Uri "http://localhost:8000/oracle/quality/dashboard.csv" -Method Get
```

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Backend APIs and Tooling](backend-api-tools.md)
- [Strict Error Contract Reference](strict-errors-reference.md)
- [Sample Manifest and Outputs](sample-manifest.md)
