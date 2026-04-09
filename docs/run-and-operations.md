# Run and Operations Guide

Last Updated: 2026-02-25
Applies To: local backend/frontend runtime, full rebuilds, docs checks, and low-resource test execution

This runbook is the operational reference for running and rebuilding the project safely.

## Prerequisites

- PowerShell (Windows)
- Docker Desktop (for OSRM and compose workflows)
- Python + `uv` (backend)
- Node.js + `pnpm` (frontend)

## Environment Setup

From repo root:

```powershell
if (!(Test-Path .env)) { Copy-Item .env.example .env }
```

Then review strict live URL/auth settings in `.env`. The checked-in defaults in `.env.example` are the authoritative current matrix:

- live runtime switches: `LIVE_RUNTIME_DATA_ENABLED=true`, `STRICT_LIVE_DATA_REQUIRED=true`
- live scenario refresh policy: `LIVE_ROUTE_COMPUTE_REFRESH_MODE=route_compute`, `LIVE_ROUTE_COMPUTE_REQUIRE_ALL_EXPECTED=true`, `LIVE_ROUTE_COMPUTE_FORCE_NO_CACHE_HEADERS=false`, `LIVE_ROUTE_COMPUTE_FORCE_UNCACHED=false`, `LIVE_ROUTE_COMPUTE_PREFETCH_TIMEOUT_MS=300000`, `LIVE_ROUTE_COMPUTE_PREFETCH_MAX_CONCURRENCY=8`, `LIVE_ROUTE_COMPUTE_PROBE_TERRAIN=true`
- live cache TTLs: `LIVE_DATA_CACHE_TTL_S=120`, `LIVE_SCENARIO_CACHE_TTL_SECONDS=120`
- live HTTP/retry controls: `LIVE_HTTP_MAX_ATTEMPTS=6`, `LIVE_HTTP_RETRY_DEADLINE_MS=30000`, `LIVE_HTTP_RETRY_BACKOFF_BASE_MS=200`, `LIVE_HTTP_RETRY_BACKOFF_MAX_MS=2500`, `LIVE_HTTP_RETRY_JITTER_MS=150`, `LIVE_HTTP_RETRY_RESPECT_RETRY_AFTER=true`, `LIVE_HTTP_RETRYABLE_STATUS_CODES=429,500,502,503,504`
- scenario-source controls: `LIVE_SCENARIO_COEFFICIENT_URL`, `LIVE_SCENARIO_REQUIRE_URL_IN_STRICT=true`, `LIVE_SCENARIO_ALLOW_SIGNED_FALLBACK=false`, `LIVE_SCENARIO_ALLOWED_HOSTS`, `LIVE_SCENARIO_WEBTRIS_NEAREST_SITES=2`, `LIVE_SCENARIO_DFT_MAX_PAGES=2`, `LIVE_SCENARIO_DFT_NEAREST_LIMIT=48`, `LIVE_SCENARIO_DFT_MIN_STATION_COUNT=3`, `LIVE_SCENARIO_COEFFICIENT_MAX_AGE_MINUTES=4320`, `LIVE_SCENARIO_ALLOW_PARTIAL_SOURCES_STRICT=false`, `LIVE_SCENARIO_MIN_SOURCE_COUNT_STRICT=4`, `LIVE_SCENARIO_MIN_COVERAGE_OVERALL_STRICT=1.0`
- route compute ceilings: `ROUTE_COMPUTE_ATTEMPT_TIMEOUT_S=1200`, `ROUTE_COMPUTE_SINGLE_ATTEMPT_TIMEOUT_S=900`
- bounded OD probe knobs: `ROUTE_CONTEXT_PROBE_ENABLED=false`, `ROUTE_CONTEXT_PROBE_TIMEOUT_MS=2500`, `ROUTE_CONTEXT_PROBE_MAX_PATHS=2`, `ROUTE_CONTEXT_PROBE_MAX_STATE_BUDGET=15000`, `ROUTE_CONTEXT_PROBE_MAX_HOPS=320`
- route graph warmup and search knobs: `ROUTE_GRAPH_WARMUP_ON_STARTUP=1`, `ROUTE_GRAPH_WARMUP_FAILFAST=1`, `ROUTE_GRAPH_WARMUP_TIMEOUT_S=1200`, `ROUTE_GRAPH_FAST_STARTUP_ENABLED=true`, `ROUTE_GRAPH_FAST_STARTUP_LONG_CORRIDOR_BYPASS_KM=120`, `ROUTE_GRAPH_STATUS_CHECK_TIMEOUT_MS=1000`, `ROUTE_GRAPH_OD_FEASIBILITY_TIMEOUT_MS=30000`, `ROUTE_GRAPH_PRECHECK_TIMEOUT_FAIL_CLOSED=false`, `ROUTE_GRAPH_BINARY_CACHE_ENABLED=true`, `ROUTE_GRAPH_MAX_STATE_BUDGET=1200000`, `ROUTE_GRAPH_STATE_BUDGET_PER_HOP=1600`, `ROUTE_GRAPH_STATE_BUDGET_RETRY_MULTIPLIER=2.5`, `ROUTE_GRAPH_STATE_BUDGET_RETRY_CAP=8000000`, `ROUTE_GRAPH_SEARCH_INITIAL_TIMEOUT_MS=30000`, `ROUTE_GRAPH_SEARCH_RETRY_TIMEOUT_MS=120000`, `ROUTE_GRAPH_SEARCH_RESCUE_TIMEOUT_MS=150000`, `ROUTE_GRAPH_STATE_SPACE_RESCUE_ENABLED=true`, `ROUTE_GRAPH_STATE_SPACE_RESCUE_MODE=reduced`, `ROUTE_GRAPH_REDUCED_INITIAL_FOR_LONG_CORRIDOR=true`, `ROUTE_GRAPH_LONG_CORRIDOR_THRESHOLD_KM=150`, `ROUTE_GRAPH_LONG_CORRIDOR_MAX_PATHS=4`, `ROUTE_GRAPH_SKIP_INITIAL_SEARCH_LONG_CORRIDOR=true`, `ROUTE_GRAPH_SCENARIO_SEPARABILITY_FAIL=false`, `ROUTE_GRAPH_MIN_GIANT_COMPONENT_NODES=50000`, `ROUTE_GRAPH_MIN_GIANT_COMPONENT_RATIO=0.20`, `ROUTE_GRAPH_MAX_NEAREST_NODE_DISTANCE_M=10000`, `ROUTE_GRAPH_OD_CANDIDATE_LIMIT=2048`, `ROUTE_GRAPH_OD_CANDIDATE_MAX_RADIUS=12`, `ROUTE_GRAPH_MAX_HOPS=220`, `ROUTE_GRAPH_ADAPTIVE_HOPS_ENABLED=true`, `ROUTE_GRAPH_HOPS_PER_KM=18.0`, `ROUTE_GRAPH_HOPS_DETOUR_FACTOR=1.35`, `ROUTE_GRAPH_EDGE_LENGTH_ESTIMATE_M=75.0`, `ROUTE_GRAPH_HOPS_SAFETY_FACTOR=1.8`, `ROUTE_GRAPH_MAX_HOPS_CAP=15000`, `ROUTE_GRAPH_A_STAR_HEURISTIC_ENABLED=true`, `ROUTE_GRAPH_HEURISTIC_MAX_SPEED_KPH=220`, `ROUTE_GRAPH_SEARCH_APPLY_SCENARIO_EDGE_COSTS=false`
- strict terrain and frontend fallback knobs: `TERRAIN_DEM_FAIL_CLOSED_UK=true`, `TERRAIN_DEM_COVERAGE_MIN_UK=0.96`, `COMPUTE_ATTEMPT_TIMEOUT_MS=1200000`, `COMPUTE_ROUTE_FALLBACK_TIMEOUT_MS=900000`, `NEXT_PUBLIC_COMPUTE_ATTEMPT_TIMEOUT_MS=1200000`, `NEXT_PUBLIC_COMPUTE_ROUTE_FALLBACK_TIMEOUT_MS=900000`, `NEXT_PUBLIC_COMPUTE_DEGRADE_STEPS=12,6,3`, `NEXT_PUBLIC_ROUTE_GRAPH_WARMUP_BASELINE_MS=480000`
- dev live-call tracing knobs (development only; sensitive when enabled): `DEV_ROUTE_DEBUG_CONSOLE_ENABLED`, `DEV_ROUTE_DEBUG_INCLUDE_SENSITIVE`, `DEV_ROUTE_DEBUG_MAX_CALLS_PER_REQUEST`, `DEV_ROUTE_DEBUG_TRACE_TTL_SECONDS`, `DEV_ROUTE_DEBUG_MAX_REQUEST_TRACES`, `DEV_ROUTE_DEBUG_RETURN_RAW_PAYLOADS`, `DEV_ROUTE_DEBUG_MAX_RAW_BODY_CHARS`

## Route Compute Fallback Policy

Frontend route compute now runs bounded attempts with explicit degradation:

1. Stream Pareto (`/api/pareto/stream`)
2. JSON Pareto fallback (`/api/pareto`)
3. Single-route fallback (`/api/route`)

Default degrade policy: `12 -> 6 -> 3`.

Each fallback aborts the previous attempt before moving on, to avoid orphan backend work.
Strict business failures (`HTTP 4xx` from route/pareto endpoints) are terminal and intentionally halt additional fallback attempts.

Strict-live refresh is hybrid:

- Scenario coefficients are refreshed on each route attempt.
- Expensive live context feeds (DfT/WebTRIS/Traffic/Meteo) are short-cached by TTL instead of being force-cleared per request.

Route graph warmup lifecycle (strict mode):

- Backend starts graph warmup in the background on startup.
- If warmup is still running, strict route endpoints fail fast with `reason_code=routing_graph_warming_up`.
- If warmup exceeds timeout or errors, strict route endpoints fail fast with `reason_code=routing_graph_warmup_failed`.
- If the loaded graph fails connectivity quality gates, strict failures use `reason_code=routing_graph_fragmented`.
- OD-specific strict graph failures surface as:
  - `reason_code=routing_graph_coverage_gap`
  - `reason_code=routing_graph_disconnected_od`
- Check readiness at `GET /health/ready` and retry when `strict_route_ready=true`.
- `/health/ready` now includes `strict_live` diagnostics for scenario coefficients:
  - `ok`, `status`, `reason_code`, `message`
  - `as_of_utc`, `age_minutes`, `max_age_minutes`, `checked_at_utc`
- If graph is ready but strict live freshness is not, `recommended_action=refresh_live_sources`.
- Frontend compute is readiness-gated and remains disabled until `strict_route_ready=true`.

Diagnostics panel (`Route compute diagnostics`) now shows:

- active attempt/endpoint/stage
- backend heartbeat age
- attempt-level timeout and alternative count
- grouped trace entries by attempt
- per-attempt live API call table (URL, requested/success, status/error, cache/retry, headers/extra)

Terrain trace note: repeated same-tile route-cache hits are deduplicated per request in live-call rows; rely on summary counts/terrain diagnostics for full cache-hit volume.

Backend dev endpoint for this trace:

- `GET /debug/live-calls/{request_id}` (dev-gated)

## Latest Local Validation Snapshot

The most recent checked local evidence is written to `backend/out/model_assets/preflight_live_runtime.json` and companion summaries in `backend/out/`. This snapshot is useful for provenance and current-state inspection, not as a guarantee that every future run will match it.

- `backend/out/model_assets/preflight_live_runtime.json`
  - `checked_at_utc=2026-04-04T15:48:39Z`
  - `required_ok=true`
  - `required_failure_count=0`
  - `scenario_profiles.version=scenario_profiles_uk_v2_live`
  - `scenario_profiles.contexts=384`
  - `scenario_live_context.coverage.overall=1.0`
  - `scenario_live_context.coverage.webtris=1.0`
  - `scenario_live_context.coverage.traffic_england=1.0`
  - `scenario_live_context.coverage.dft=1.0`
  - `scenario_live_context.coverage.open_meteo=1.0`
  - `fuel_snapshot.as_of=2026-03-23T00:00:00Z`
  - `fuel_snapshot.signature_prefix=6092b11ca3f7`
  - `toll_tariffs.rule_count=220`
  - `toll_topology.segment_count=28`
  - `stochastic_regimes.regime_count=18`
  - `departure_profiles.region_count=11`
  - `bank_holidays.count=134`
  - `carbon_policy.price_per_kg=0.101`
  - `carbon_policy.scope_adjusted_emissions=1.121`
  - `osrm_engine_smoke.distance_m=189471.0`
  - `osrm_engine_smoke.duration_s=8794.2`
  - `ors_engine_smoke.distance_m=203868.1`
  - `ors_engine_smoke.duration_s=12280.8`
  - `ors_engine_smoke.engine_version=9.7.1`
  - `ors_engine_smoke.graph_date=2026-03-22T16:39:30Z`
  - `ors_engine_smoke.identity_status=graph_identity_verified`
- `backend/out/compare_r12_vs_r15_combo_summary.json`
  - `variant_count=4`
  - retained success rows `2`
  - regressions: `london_newcastle|C`, `london_newcastle|V0`
  - retained success rows: `london_newcastle|A`, `london_newcastle|B`
- `backend/out/focused_one_od_r4_vs_cap1600_diff.summary.json`
  - `variant_count=4`
  - after-state `success_rate=1.0` for `A`, `B`, `C`, and `V0`
  - after-state `route_evidence_ok_rate=1.0` for `A`, `B`, `C`, and `V0`
  - after-state `mean_frontier_count=2.0` for `A`, `B`, and `C`
  - after-state `mean_frontier_count=1.0` for `V0`
- `backend/out/corpus_ambiguity_refresh_summary.json`
  - `row_count=19`
  - `mean_ambiguity_index=0.239727`
  - `max_ambiguity_index=0.420932`
  - `mean_engine_disagreement_prior=0.413061`
  - `mean_hard_case_prior=0.419982`
  - `mean_od_ambiguity_confidence=0.899552`
  - `mean_od_ambiguity_source_count=2.526316`
  - `mean_od_ambiguity_source_support_strength=0.60263`
  - `accepted_count=0`
  - `bootstrap_prior_count=19`
  - `nonzero_ambiguity_prior_count=19`
  - `nonzero_engine_prior_count=19`
  - `nonzero_hard_case_prior_count=19`
  - `ambiguity_prior_source_mix.routing_graph_probe=19`
  - `ambiguity_prior_source_mix.repo_local_geometry_backfill=10`
  - `ambiguity_prior_source_mix.historical_results_bootstrap=19`
  - `od_ambiguity_source_mix.routing_graph_probe=59`
  - `od_ambiguity_source_mix.repo_local_geometry_backfill=40`
  - `od_ambiguity_source_mix.historical_results_bootstrap=144`

## Start Full Local Stack

From repo root:

```powershell
.\scripts\dev.ps1
```

`dev.ps1` now runs strict live preflight before backend/frontend launch. If a required live payload is stale/invalid, startup stops and writes:

- `backend/out/model_assets/preflight_live_runtime.json`

Expected services:

- OSRM: `http://localhost:5000`
- Backend API: `http://localhost:8000`
- Backend OpenAPI UI: `http://localhost:8000/docs`
- Backend readiness: `http://localhost:8000/health/ready`
- Frontend: `http://localhost:3000`

## Run Backend Only (Known-Good)

From repo root:

```powershell
Set-Location backend
uv sync --dev
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Preflight check before compute:

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health/ready" -Method Get
```

Do not run the backend from repo root with `uv run --project backend uvicorn ...` in this repo layout. Use `Set-Location backend` first so `app.main` module resolution stays stable.

## Run Frontend Only

From repo root:

```powershell
Set-Location frontend
pnpm install
pnpm dev
```

## Full Refresh / Rebuild (Long Path)

Use this when OSRM + backend assets + caches should be rebuilt end-to-end.

1. Stop active app terminals (`Ctrl+C`).
2. Clean compose/runtime caches:

```powershell
.\scripts\clean.ps1
```

3. Recreate stack:

```powershell
.\scripts\dev.ps1
```

4. Rebuild backend model assets:

```powershell
Set-Location backend
uv run python scripts/build_model_assets.py
uv run python scripts/publish_live_artifacts_uk.py
```

This publish step updates strict runtime tracked artifact targets:

- `backend/assets/uk/departure_profiles_uk.json`
- `backend/assets/uk/stochastic_regimes_uk.json`
- `backend/assets/uk/toll_topology_uk.json`
- `backend/assets/uk/toll_tariffs_uk.json`

5. Optional subsystem rebuilds:

```powershell
uv run python scripts/build_routing_graph_uk.py
uv run python scripts/build_terrain_tiles_uk.py --source-dem-glob "assets/uk/dem/*.tif"
```

## Runtime Output Locations

Under `backend/out/`:

- `model_assets/`
- `artifacts/{run_id}/`
- `manifests/{run_id}.json`
- `scenario_manifests/{run_id}.json`
- `provenance/{run_id}.jsonl`
- `test_runs/{timestamp}/` (safe test runner outputs)

## Safe Test Execution (Low Resource)

From repo root:

```powershell
.\scripts\run_backend_tests_safe.ps1 -MaxCores 1 -PriorityClass Idle -MaxWorkingSetMB 4096
```

Useful options:

- `-TestPattern test_tooling_scripts.py`
- `-PerFileTimeoutSec 900`
- `-NoOutputStallSec 240`
- `-IncludeCoverage`
- `-ExtraPytestArgs @("-k", "strict")`

Outputs:

- `backend/out/test_runs/<timestamp>/summary.csv`
- `backend/out/test_runs/<timestamp>/summary.json`
- `backend/out/test_runs/<timestamp>/failed_tests.txt`
- `backend/out/test_runs/<timestamp>/rerun_failed.ps1`

## Docs Serve and Validation

From repo root:

```powershell
.\scripts\serve_docs.ps1
python scripts/check_docs.py
```

Docs URL: `http://localhost:8088/`

## Stop Local Stack

1. Stop running terminals (`Ctrl+C`).
2. From repo root:

```powershell
docker compose down
```

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Backend APIs and Tooling](backend-api-tools.md)
- [Quality Gates and Benchmarks](quality-gates-and-benchmarks.md)
- [Model Assets and Data Sources](model-assets-and-data-sources.md)
