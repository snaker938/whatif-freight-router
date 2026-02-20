# API Cookbook

Last Updated: 2026-02-19  
Applies To: local backend API usage from CLI

This replaces notebook workflows with reproducible terminal-first examples.

## 1) Single Route

```powershell
$body = @{
  origin = @{ lat = 52.4862; lon = -1.8904 }
  destination = @{ lat = 51.5072; lon = -0.1276 }
  vehicle_type = "rigid_hgv"
  scenario_mode = "no_sharing"
} | ConvertTo-Json -Depth 8
Invoke-RestMethod -Uri "http://localhost:8000/route" -Method Post -ContentType "application/json" -Body $body
```

## 2) Pareto JSON

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/pareto" -Method Post -ContentType "application/json" -Body $body
```

## 3) Batch Pareto

```powershell
$batch = Get-Content docs/examples/sample_batch_request.json -Raw
Invoke-RestMethod -Uri "http://localhost:8000/batch/pareto" -Method Post -ContentType "application/json" -Body $batch
```

## 4) Duty Chain

```powershell
$duty = @{
  stops = @(
    @{ lat = 52.4862; lon = -1.8904; label = "Birmingham" }
    @{ lat = 52.2053; lon = 0.1218; label = "Cambridge" }
    @{ lat = 51.5072; lon = -0.1276; label = "London" }
  )
  vehicle_type = "rigid_hgv"
  scenario_mode = "no_sharing"
} | ConvertTo-Json -Depth 10
Invoke-RestMethod -Uri "http://localhost:8000/duty/chain" -Method Post -ContentType "application/json" -Body $duty
```

## 5) Signature Verify

```powershell
$verify = @{
  payload = "{\"hello\":\"world\"}"
  signature = "abcdef"
  secret = "dev-manifest-signing-secret"
} | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8000/verify/signature" -Method Post -ContentType "application/json" -Body $verify
```

## Related Docs

- [Documentation Index](README.md)
- [Backend APIs and Tooling](backend-api-tools.md)
- [Strict Error Contract Reference](strict-errors-reference.md)
- [Sample Manifest and Outputs](sample-manifest.md)
