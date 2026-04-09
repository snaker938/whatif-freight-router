# API Cookbook

Last Updated: 2026-04-09
Applies To: local backend API usage from PowerShell

This page provides reproducible local examples for the current strict backend API.

## 1) Minimal Base Payload

```powershell
$base = @{
  origin = @{ lat = 52.4862; lon = -1.8904 }
  destination = @{ lat = 51.5072; lon = -0.1276 }
  vehicle_type = "rigid_hgv"
  scenario_mode = "no_sharing"
  weights = @{ time = 1; money = 1; co2 = 1 }
  max_alternatives = 8
  pareto_method = "dominance"
} | ConvertTo-Json -Depth 12
```

If you omit `pipeline_mode`, the public `POST /route` lane now defaults to `tri_source`. Use an explicit `pipeline_mode` only when you want to force a specific runtime variant or reproduce a non-default evaluation path.

## 2) Explicit Pipeline Override Payload

```powershell
$thesis = @{
  origin = @{ lat = 52.4862; lon = -1.8904 }
  destination = @{ lat = 53.4808; lon = -2.2426 }
  vehicle_type = "rigid_hgv"
  scenario_mode = "partial_sharing"
  max_alternatives = 12
  weights = @{ time = 1; money = 1; co2 = 1 }
  pareto_method = "dominance"
  pipeline_mode = "voi"
  pipeline_seed = 20260331
  search_budget = 8
  evidence_budget = 3
  cert_world_count = 64
  certificate_threshold = 0.75
  tau_stop = 0.02
  stochastic = @{ enabled = $true; seed = 11; sigma = 0.08; samples = 25 }
  weather = @{ enabled = $true; profile = "rain"; intensity = 1.1; apply_incident_uplift = $true }
  incident_simulation = @{
    enabled = $true
    seed = 17
    dwell_rate_per_100km = 0.8
    accident_rate_per_100km = 0.25
    closure_rate_per_100km = 0.05
    dwell_delay_s = 120
    accident_delay_s = 480
    closure_delay_s = 900
    max_events_per_route = 12
  }
} | ConvertTo-Json -Depth 16
```

Use an explicit override like this for targeted evaluation or debugging. Normal `POST /route` calls should usually omit `pipeline_mode` and use the public `tri_source` default.

## 3) Single Route (`POST /route`)

```powershell
$routeResp = Invoke-RestMethod -Uri "http://localhost:8000/route" -Method Post -ContentType "application/json" -Body $base
$routeResp.pipeline_mode
$routeResp.run_id
$routeResp.manifest_endpoint
$routeResp.artifacts_endpoint
$routeResp.provenance_endpoint
$routeResp.decision_package.package_kind
$routeResp.decision_package.pipeline_mode
$routeResp.decision_package.terminal_kind
$routeResp.decision_package.preference_summary
$routeResp.decision_package.certified_set_summary
$routeResp.decision_package.abstention_summary
$routeResp.decision_package.preference_summary.suggested_queries
$routeResp.decision_package.abstention_summary.abstention_type
```

`POST /route` defaults to the public `tri_source` lane when `pipeline_mode` is omitted. The returned `RouteResponse` still includes `selected`, `candidates`, `selected_certificate`, and `voi_stop_summary`, and it now also includes `decision_package`.

`decision_package.pipeline_mode` is the public lane name. For single-leg requests, the current coordinator executes `tri_source` through `voi` internals while keeping the public lane name stable. If you send waypoint requests, the route runtime currently falls back to the explicit legacy compatibility path.

`decision_package` is the public decision summary mirror, not a second API surface. Its public `terminal_kind` values are exactly `certified_singleton`, `certified_set`, and `typed_abstention`. Waypoint compatibility fallback is represented through `pipeline_mode`, provenance, warnings, and `abstention_summary.reason_code=legacy_runtime_selected`, not through a fourth public terminal kind. The package also carries the landed summary objects for preference, support, world support, world fidelity, certification state, certified-set membership, abstention, witness, controller, theorem-hook, and lane-manifest state.

`decision_package.preference_summary` is the currently landed preference bridge. It is summary-only. The stable fields are selector metadata (`objective_id`, `objective_field`, `selector_policy`, `selective`, `tie_break_order`), stop metadata (`certified_only_required`, `time_guard_active`, `vetoed_targets`, `compatible_route_ids`, `certified_route_ids`, `selected_route_id`, `selected_certificate`, `stop_reason`, `stop_hint_codes`), and `suggested_queries`, which are populated from current runtime facts. There are no public preference query/update endpoints yet.

The `RouteRequest` shape also accepts `waypoints`, `refinement_policy`, `pipeline_mode`, `pipeline_seed`, `search_budget`, `evidence_budget`, `cert_world_count`, `certificate_threshold`, `tau_stop`, and `evaluation_lean_mode` when you need the thesis controls enabled explicitly.

In the frontend workflow, the normal path is: inspect the returned `decision_package` in the route result, use the certification panel to read terminal/support/controller state, then open Run Inspector to browse grouped route-core, decision-package, support, DCCS/REFC, controller/VOI, witness/fragility, theorem/lane, and evaluation artifact families.

## 3b) OSRM Baseline (`POST /route/baseline`)

```powershell
$baselineResp = Invoke-RestMethod -Uri "http://localhost:8000/route/baseline" -Method Post -ContentType "application/json" -Body $base
$baselineResp.method
$baselineResp.provider_mode
$baselineResp.compute_ms
$baselineResp.baseline_policy
$baselineResp.asset_freshness_status
$baselineResp.notes
```

Use this when you want the backend’s OSRM-aligned comparator route without running the full smart-router selection stack.

## 3c) ORS Baseline (`POST /route/baseline/ors`)

```powershell
$orsBaselineResp = Invoke-RestMethod -Uri "http://localhost:8000/route/baseline/ors" -Method Post -ContentType "application/json" -Body $base
$orsBaselineResp.method
$orsBaselineResp.provider_mode
$orsBaselineResp.compute_ms
$orsBaselineResp.baseline_policy
$orsBaselineResp.asset_manifest_hash
$orsBaselineResp.notes
```

Use this when you want the ORS/reference-route comparator. The frontend baseline panels compare against both OSRM and ORS, but the backend endpoints remain separate.

## 4) Pareto (`POST /pareto`)

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/pareto" -Method Post -ContentType "application/json" -Body $base
```

## 5) Pareto Stream (`POST /pareto/stream`)

The direct backend stream is `POST /pareto/stream`. If the frontend is running, the browser-facing proxy path is `POST /api/pareto/stream`.

```powershell
Invoke-WebRequest -Uri "http://localhost:8000/pareto/stream" -Method Post -ContentType "application/json" -Body $base
Invoke-WebRequest -Uri "http://localhost:3000/api/pareto/stream" -Method Post -ContentType "application/json" -Body $base
```

## 6) Scenario Compare (`POST /scenario/compare`)

```powershell
$compare = @{
  origin = @{ lat = 52.4862; lon = -1.8904 }
  destination = @{ lat = 53.4808; lon = -2.2426 }
  vehicle_type = "rigid_hgv"
  scenario_mode = "partial_sharing"
  max_alternatives = 8
} | ConvertTo-Json -Depth 12
$compareResp = Invoke-RestMethod -Uri "http://localhost:8000/scenario/compare" -Method Post -ContentType "application/json" -Body $compare
$compareResp.run_id
$compareResp.scenario_manifest_endpoint
$compareResp.scenario_signature_endpoint
```

## 7) Batch Pareto (`POST /batch/pareto`)

```powershell
$batch = Get-Content docs/examples/sample_batch_request.json -Raw
$batchResp = Invoke-RestMethod -Uri "http://localhost:8000/batch/pareto" -Method Post -ContentType "application/json" -Body $batch
$runId = $batchResp.run_id
$runId
```

The batch request family accepts `pairs`, `waypoints`, `seed`, `toggles`, `model_version`, `pipeline_mode`, `pipeline_seed`, `search_budget`, `evidence_budget`, `cert_world_count`, `certificate_threshold`, and `tau_stop` in the current code.

The live `BatchParetoResponse` contract is intentionally narrow: top-level `run_id` plus `results`. Use `run_id` with the run-store endpoints when you need manifests, provenance, or downloaded artifacts.

## 8) Batch CSV Import (`POST /batch/import/csv`)

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
  pipeline_mode = "dccs_refc"
  search_budget = 6
  cert_world_count = 48
  certificate_threshold = 0.75
} | ConvertTo-Json -Depth 12
Invoke-RestMethod -Uri "http://localhost:8000/batch/import/csv" -Method Post -ContentType "application/json" -Body $csvReq
```

The CSV import request shares the same route and pipeline controls as `POST /batch/pareto`, but it uses `csv_text` instead of `pairs`.

## 9) Duty Chain (`POST /duty/chain`)

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

## 10) Departure Optimize (`POST /departure/optimize`)

```powershell
$dep = @{
  origin = @{ lat = 52.4862; lon = -1.8904 }
  destination = @{ lat = 51.5072; lon = -0.1276 }
  vehicle_type = "rigid_hgv"
  scenario_mode = "no_sharing"
  window_start_utc = "2026-03-31T06:00:00Z"
  window_end_utc = "2026-03-31T12:00:00Z"
  step_minutes = 60
} | ConvertTo-Json -Depth 12
Invoke-RestMethod -Uri "http://localhost:8000/departure/optimize" -Method Post -ContentType "application/json" -Body $dep
```

## 11) Readiness and Cache Ops

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health/ready"
Invoke-RestMethod -Uri "http://localhost:8000/cache/stats"
Invoke-RestMethod -Uri "http://localhost:8000/cache/hot-rerun/restore" -Method Post
Invoke-RestMethod -Uri "http://localhost:8000/cache" -Method Delete
```

## 12) Dev Live-Call Trace

```powershell
$requestId = "<request_id>"
Invoke-RestMethod -Uri "http://localhost:8000/debug/live-calls/$requestId"
```

This endpoint is development-gated and only useful when live-call tracing is enabled.

## 13) Run Artifacts, Manifests, and Signatures

```powershell
$runId = "<run_id>"
Invoke-RestMethod -Uri "http://localhost:8000/runs/$runId/manifest"
Invoke-RestMethod -Uri "http://localhost:8000/runs/$runId/scenario-manifest"
Invoke-RestMethod -Uri "http://localhost:8000/runs/$runId/provenance"
Invoke-RestMethod -Uri "http://localhost:8000/runs/$runId/signature"
Invoke-RestMethod -Uri "http://localhost:8000/runs/$runId/scenario-signature"
Invoke-RestMethod -Uri "http://localhost:8000/runs/$runId/artifacts"
Invoke-WebRequest -Uri "http://localhost:8000/runs/$runId/artifacts/decision_package.json" -OutFile ".\\$runId-decision_package.json"
Invoke-WebRequest -Uri "http://localhost:8000/runs/$runId/artifacts/preference_summary.json" -OutFile ".\\$runId-preference_summary.json"
Invoke-WebRequest -Uri "http://localhost:8000/runs/$runId/artifacts/certified_set.json" -OutFile ".\\$runId-certified_set.json"
Invoke-WebRequest -Uri "http://localhost:8000/runs/$runId/artifacts/voi_controller_trace_summary.json" -OutFile ".\\$runId-voi_controller_trace_summary.json"
Invoke-WebRequest -Uri "http://localhost:8000/runs/$runId/artifacts/voi_replay_oracle_summary.json" -OutFile ".\\$runId-voi_replay_oracle_summary.json"
Invoke-WebRequest -Uri "http://localhost:8000/runs/$runId/artifacts/evaluation_manifest.json" -OutFile ".\\$runId-evaluation_manifest.json"
Invoke-WebRequest -Uri "http://localhost:8000/runs/$runId/artifacts/thesis_report.md" -OutFile ".\\$runId-thesis_report.md"
```

Current route runs may publish decision-package artifacts alongside the older route/certification artifacts:

- guaranteed when a `decision_package` is present:
  - `decision_package.json`
  - `preference_summary.json`
  - `support_summary.json`
  - `support_provenance.json`
  - `certified_set.json`
  - `certified_set_routes.jsonl`
  - `final_route_trace.json`
- conditional follow-on artifacts:
  - `support_trace.jsonl`
- `abstention_summary.json` when the package includes an abstention summary
- `witness_summary.json` and `witness_routes.jsonl` when witness state is available
- `controller_summary.json` and `controller_trace.jsonl` when controller state is available
- `voi_controller_trace_summary.json` when controller trace summary state is available
- `voi_replay_oracle_summary.json` when replay-oracle traces are available
- `theorem_hook_map.json`
- `lane_manifest.json`

The frontend Run Inspector uses only the artifact names above plus the current `decision_package` to highlight which files are expected for the run and which remain conditional.

For detached frontend browsing, remember that the proxy allowlist is narrower than the backend run-store allowlist. If you need `decision_package.json`, `preference_summary.json`, `support_summary.json`, `certified_set.json`, controller traces, theorem hooks, lane manifests, or `evaluation_manifest.json`, fetch them from the backend run-store endpoints directly unless the current active route already exposed the corresponding summary through `decision_package`.

## 14) Signature Verify (`POST /verify/signature`)

```powershell
$verify = @{
  payload = @{
    run_id = "example-run-id"
    created_at = "2026-01-01T00:00:00Z"
  }
  signature = "abcdef"
  secret = "dev-manifest-signing-secret"
} | ConvertTo-Json -Depth 8
Invoke-RestMethod -Uri "http://localhost:8000/verify/signature" -Method Post -ContentType "application/json" -Body $verify
```

`payload` can also be a plain JSON array or a string when you are verifying detached text or list-based manifests.

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Backend APIs and Tooling](backend-api-tools.md)
- [Strict Error Contract Reference](strict-errors-reference.md)
- [Sample Manifest and Outputs](sample-manifest.md)
