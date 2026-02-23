# API Cookbook

Last Updated: 2026-02-23  
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
  max_alternatives = 8
  pareto_method = "dominance"
} | ConvertTo-Json -Depth 10
```

## 2) Single Route (`POST /route`)

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/route" -Method Post -ContentType "application/json" -Body $base
```

## 3) Pareto (`POST /pareto`)

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/pareto" -Method Post -ContentType "application/json" -Body $base
```

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
  max_alternatives = 8
} | ConvertTo-Json -Depth 10
Invoke-RestMethod -Uri "http://localhost:8000/scenario/compare" -Method Post -ContentType "application/json" -Body $compare
```

## 6) Batch Pareto (`POST /batch/pareto`)

```powershell
$batch = Get-Content docs/examples/sample_batch_request.json -Raw
$batchResp = Invoke-RestMethod -Uri "http://localhost:8000/batch/pareto" -Method Post -ContentType "application/json" -Body $batch
$runId = $batchResp.run_id
$runId
```

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

## 9) Departure Optimize (`POST /departure/optimize`)

```powershell
$dep = @{
  origin = @{ lat = 52.4862; lon = -1.8904 }
  destination = @{ lat = 51.5072; lon = -0.1276 }
  vehicle_type = "rigid_hgv"
  scenario_mode = "no_sharing"
  window_start_utc = "2026-02-24T06:00:00Z"
  window_end_utc = "2026-02-24T12:00:00Z"
  step_minutes = 60
} | ConvertTo-Json -Depth 12
Invoke-RestMethod -Uri "http://localhost:8000/departure/optimize" -Method Post -ContentType "application/json" -Body $dep
```

## 10) Run Artifacts + Signatures

```powershell
$runId = "<run_id>"
Invoke-RestMethod -Uri "http://localhost:8000/runs/$runId/manifest"
Invoke-RestMethod -Uri "http://localhost:8000/runs/$runId/scenario-manifest"
Invoke-RestMethod -Uri "http://localhost:8000/runs/$runId/signature"
Invoke-RestMethod -Uri "http://localhost:8000/runs/$runId/scenario-signature"
Invoke-RestMethod -Uri "http://localhost:8000/runs/$runId/artifacts"
Invoke-WebRequest -Uri "http://localhost:8000/runs/$runId/artifacts/report.pdf" -OutFile ".\\report_$runId.pdf"
```

## 11) Signature Verify (`POST /verify/signature`)

```powershell
$verify = @{
  payload = @{ hello = "world" }
  signature = "abcdef"
  secret = "dev-manifest-signing-secret"
} | ConvertTo-Json -Depth 8
Invoke-RestMethod -Uri "http://localhost:8000/verify/signature" -Method Post -ContentType "application/json" -Body $verify
```

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Backend APIs and Tooling](backend-api-tools.md)
- [Strict Error Contract Reference](strict-errors-reference.md)
- [Sample Manifest and Outputs](sample-manifest.md)

