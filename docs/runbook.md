# Operator Runbook

This page is the shortest operator-facing reference for starting, checking, using, and stopping the current repository in its documented local configuration.

It is intentionally conservative. It describes the commands and artifact locations the repository already documents today. The maintained reviewer package is applied-evidence green for the current publishability/adoption verdict, while deployment-grade certification remains outside this runbook's scope.

## What This Runbook Covers

Use this page for:

- starting the local stack
- checking readiness before compute
- locating runtime artifacts and evaluation bundles
- handling common failure modes
- shutting the stack down cleanly

For deeper operational context, see [Run and Operations Guide](run-and-operations.md).

## Prerequisites

The documented local stack assumes:

- PowerShell on Windows
- Docker Desktop for OSRM, ORS, and compose workflows
- Python plus `uv` for the backend
- Node.js plus `pnpm` for the frontend

## Startup

From the repo root, the recommended one-command local workflow is:

```powershell
.\scripts\dev.ps1
```

That script is the current documented entry point for the full local stack. It:

- creates `.env` from `.env.example` if needed
- starts OSRM via Docker
- runs strict live preflight before backend startup
- starts the backend
- starts the frontend

Important:
- The current strict live preflight checks both OSRM and ORS engine smoke paths before backend startup.
- `.\scripts\dev.ps1` currently brings up OSRM itself, but it does not start ORS for you.
- If ORS is not already running, strict preflight can still fail even when OSRM is healthy.

If you want to run the backend by hand instead of using the full dev script:

```powershell
Set-Location backend
uv sync --dev
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

If you want the frontend by itself:

```powershell
Set-Location frontend
pnpm install
pnpm dev
```

## Readiness

Before attempting route compute, check backend readiness:

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health/ready" -Method Get
```

The current docs say readiness should show:

- `strict_route_ready=true` before route compute is expected to work
- `strict_live.ok=true` for the strict live-data path

If readiness is not good, the documented causes are typically:

- route graph warmup still in progress
- warmup timeout or warmup failure
- strict live source staleness or freshness failure
- ORS or OSRM engine smoke failure during strict live readiness
- graph fragmentation or OD-specific graph failure

## Core Commands

The following commands are the current documented operator entry points:

### Full local stack

```powershell
.\scripts\dev.ps1
```

### Backend only

```powershell
Set-Location backend
uv sync --dev
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Backend readiness check

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health/ready" -Method Get
```

### Backend live-call trace inspection

```powershell
GET /debug/live-calls/{request_id}
```

The trace endpoint is dev-gated and is documented in the operations guide as the place to inspect request-level live API calls.

### Safe low-resource test execution

```powershell
.\scripts\run_backend_tests_safe.ps1 -MaxCores 1 -PriorityClass Idle -MaxWorkingSetMB 4096
```

## Artifact Locations

The current docs and scripts place runtime outputs under `backend/out/`.

Useful locations include:

- `backend/out/model_assets/`
- `backend/out/artifacts/{run_id}/`
- `backend/out/manifests/{run_id}.json`
- `backend/out/scenario_manifests/{run_id}.json`
- `backend/out/provenance/{run_id}.json`
- `backend/out/test_runs/{timestamp}/`

Legacy-named evaluation bundles also appear under:

- `backend/out/thesis_campaigns/*`

The run-store and sample-manifest docs list representative artifact families such as:

- `metadata.json`
- `index.json`
- `index.md`
- `results.json`
- `results.csv`
- `routes.geojson`
- `dccs_candidates.jsonl`
- `dccs_summary.json`
- `strict_frontier.jsonl`
- `certificate_summary.json`
- `world_support_summary.json`
- `flip_radius_summary.json`
- `decision_region_summary.json`
- `certificate_witness.json`
- `certified_set_summary.json` when certified-set emission / normalized certified-set summary is available
- `route_fragility_map.json`
- `competitor_fragility_breakdown.json`
- `value_of_refresh.json`
- `sampled_world_manifest.json`
- `evidence_snapshot_manifest.json`
- `voi_action_trace.json`
- `voi_controller_state.jsonl`
- `voi_action_scores.csv`
- `voi_stop_certificate.json`
- `final_route_trace.json`
- `od_corpus.csv`
- `od_corpus.json`
- `od_corpus_summary.json`
- `thesis_summary.json`
- `thesis_summary_by_cohort.json`
- `thesis_metrics.json`
- `thesis_report.md`
- `methods_appendix.md`
- `evaluation_manifest.json`

For route-compute bundles, `index.json` is the machine-readable bundle index and `index.md` is the reviewer-readable summary of the same run folder. Legacy-named evaluation bundles may also carry the same pair when they were emitted or additively refreshed through the run-store path. Use those files first when you need the stable artifact list and artifact-presence status for one checked bundle; they do not imply committed PDF or SVG renders.

## Shutdown

To stop the local stack:

1. Stop the backend and frontend terminals with `Ctrl+C`.
2. If Docker services were started, bring them down from the repo root:

```powershell
docker compose down
```

If you are only stopping the backend, exit the `uvicorn` process with `Ctrl+C`.

## Common Failure Handling

The current docs identify these common failure modes:

- strict live runtime preflight failure
- missing backend dependencies in the local `uv` environment
- no supported live data for the selected mode
- route graph warmup still running
- strict live freshness checks failing
- terrain support being insufficient or unsupported

When this happens:

- re-run readiness
- inspect the latest `backend/out/model_assets/preflight_live_runtime.json`
- for scenario-profile freshness or provenance issues, inspect `checks[].details.resolved_source_locator` on the `scenario_profiles` check when present; it shows whether strict runtime is on the current repo-local scenario asset locator or a live URL-backed source, not a more specific winning repo-local file inside `repo_local_fresh`
- confirm both local routing engines are actually available, not just OSRM
- inspect the relevant `backend/out/thesis_campaigns/*` bundle
- check the live-call trace if the failure happened during compute

The documented expectation is fail-closed behavior, not silent fallback.

## What To Tell Reviewers

When writing or speaking about the system, keep the scope narrow:

- the runbook supports the documented local stack
- the documented outputs are under `backend/out/`
- readiness is explicit and inspectable
- failures are often a sign that the stack is correctly refusing unsafe work

Describe this as an applied-evidence green reviewer package, not as universal deployability or deployment-grade certification.

## Related Docs

- [Run and Operations Guide](run-and-operations.md)
- [Reviewer Quickstart](reviewer_quickstart.md)
- [Quality Gates and Benchmarks](quality-gates-and-benchmarks.md)
- [Sample Manifest and Outputs](sample-manifest.md)
- [Documentation Index](DOCS_INDEX.md)
