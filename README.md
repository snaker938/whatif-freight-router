# Carbon-Aware Multi-Objective Freight Router

This repository contains a local-first freight-routing stack with:

- self-hosted OSRM and ORS baseline engines
- a FastAPI backend with strict live/runtime gates and signed run artifacts
- a Next.js frontend with route tooling, diagnostics, tutorial flows, and devtools
- thesis-grade evaluation workflows for DCCS, REFC, and VOI-AD2R
- a separate hot-rerun benchmark for production-style cache/reuse validation

## Repo Layout

- `backend/`: FastAPI app, evaluation scripts, model assets, and runtime outputs
- `frontend/`: Next.js app, API proxy routes, map UI, tutorial system, and devtools
- `docs/`: authored documentation
- `notebooks/`: notebook policy only; the repo is maintained as a script-first workflow
- `osrm/`, `ors/`: local baseline-engine infrastructure and graph data
- `scripts/`: repo-level developer helpers and docs validation

## Prerequisites

- Docker Desktop
- PowerShell
- Python with `uv`
- Node.js with `pnpm`

## Recommended Local Development Flow

From repo root:

```powershell
.\scripts\dev.ps1
```

This is the preferred local startup path. It runs strict live preflight, boots the local routing stack, and starts backend/frontend services when needed.

Key URLs:

- Frontend: `http://localhost:3000`
- Backend OpenAPI docs: `http://localhost:8000/docs`
- Backend readiness: `http://localhost:8000/health/ready`
- OSRM: `http://localhost:5000`
- ORS health: `http://localhost:8082/ors/v2/health`

## Full Container Stack

Run the complete Dockerized stack:

```powershell
docker compose up --build
```

Do not run this at the same time as `.\scripts\dev.ps1`.

## Environment and Runtime Notes

- Copy `.env.example` to `.env` if you do not already have one.
- Strict startup runs `backend/scripts/preflight_live_runtime.py` before backend launch.
- Backend strict route readiness and strict live readiness both surface through `GET /health/ready`.
- Current compute fallback defaults in `.env.example` are intentionally long-running for thesis/evaluation workloads:
  - `ROUTE_COMPUTE_ATTEMPT_TIMEOUT_S=1200`
  - `COMPUTE_ATTEMPT_TIMEOUT_MS=1200000`
  - `COMPUTE_ROUTE_FALLBACK_TIMEOUT_MS=900000`
  - `NEXT_PUBLIC_COMPUTE_ATTEMPT_TIMEOUT_MS=1200000`
  - `NEXT_PUBLIC_COMPUTE_ROUTE_FALLBACK_TIMEOUT_MS=900000`
  - `LIVE_SCENARIO_ALLOW_PARTIAL_SOURCES_STRICT=false`
- Current live refresh and strict route-compute controls are driven by:
  - `LIVE_ROUTE_COMPUTE_REFRESH_MODE`
  - `LIVE_ROUTE_COMPUTE_REQUIRE_ALL_EXPECTED`
  - `LIVE_ROUTE_COMPUTE_FORCE_NO_CACHE_HEADERS`
  - `LIVE_ROUTE_COMPUTE_FORCE_UNCACHED`
  - `LIVE_ROUTE_COMPUTE_PREFETCH_TIMEOUT_MS`
  - `LIVE_ROUTE_COMPUTE_PREFETCH_MAX_CONCURRENCY`
  - `LIVE_ROUTE_COMPUTE_PROBE_TERRAIN`
- Current graph warmup and strict graph controls are driven by:
  - `ROUTE_GRAPH_WARMUP_ON_STARTUP`
  - `ROUTE_GRAPH_WARMUP_FAILFAST`
  - `ROUTE_GRAPH_WARMUP_TIMEOUT_S`
  - `ROUTE_GRAPH_STATUS_CHECK_TIMEOUT_MS`
  - `ROUTE_GRAPH_FAST_STARTUP_ENABLED`
  - `ROUTE_GRAPH_MIN_GIANT_COMPONENT_NODES`
  - `ROUTE_GRAPH_MIN_GIANT_COMPONENT_RATIO`
  - `ROUTE_GRAPH_MAX_NEAREST_NODE_DISTANCE_M`
- In strict runtime, settings validation forces strict live data on, requires expected live route-compute sources, requires strict graph readiness, and disables fast graph startup.

## Important Scripts

Repo-level:

- `.\scripts\dev.ps1`: preferred local startup
- `.\scripts\clean.ps1`: clean Docker/OSRM caches and force rebuild
- `.\scripts\run_backend_tests_safe.ps1`: low-resource backend test runner
- `.\scripts\serve_docs.ps1`: local markdown docs viewer
- `python scripts/check_docs.py`: docs path/link/endpoint/index consistency checks
- `.\scripts\demo_repro_run.ps1`: reproducibility demo path

Backend build and evaluation:

- `uv run --project backend python backend/scripts/build_model_assets.py`
- `uv run --project backend python backend/scripts/publish_live_artifacts_uk.py`
- `uv run --project backend python backend/scripts/preflight_live_runtime.py`
- `uv run --project backend python backend/scripts/run_headless_scenario.py`
- `uv run --project backend python backend/scripts/run_thesis_lane.py`
- `uv run --project backend python backend/scripts/run_thesis_evaluation.py`
- `uv run --project backend python backend/scripts/run_hot_rerun_benchmark.py`
- `uv run --project backend python backend/scripts/compose_thesis_suite_report.py`
- `uv run --project backend python backend/scripts/check_eta_concept_drift.py`

## Evaluation Workflows

- Use `backend/scripts/run_thesis_lane.py` for smaller lane-local checks and focused iteration.
- Use `backend/scripts/run_thesis_evaluation.py` for the broad cold thesis proof, focused REFC proof, focused VOI proof, and DCCS diagnostic probe. Evaluator runs also emit cohort-scaffolded metadata such as `evaluation_suite`, `cohort_scaffolding`, `thesis_summary_by_cohort.csv/json`, and `cohort_composition.json`.
- Use `backend/scripts/run_hot_rerun_benchmark.py` for the dedicated second-run reuse benchmark and hot-vs-cold comparison artifacts.
- Use `backend/scripts/compose_thesis_suite_report.py` to compose completed evaluation lanes into a final suite report with cohort summaries, cohort composition, suite-source provenance, and prior coverage summaries.

## Runtime Outputs

Current runtime and evaluation outputs are written under `backend/out/`:

- `backend/out/manifests/`
- `backend/out/scenario_manifests/`
- `backend/out/provenance/`
- `backend/out/artifacts/<run_id>/`
- `backend/out/model_assets/`
- `backend/out/test_runs/<timestamp>/`

Completed runs are typically inspected through signed manifests and the artifact API rather than by opening files directly:

- `GET /runs/{run_id}/manifest`
- `GET /runs/{run_id}/scenario-manifest`
- `GET /runs/{run_id}/signature`
- `GET /runs/{run_id}/scenario-signature`
- `GET /runs/{run_id}/artifacts`
- `GET /runs/{run_id}/artifacts/{artifact_name}`

Hot-rerun cache state can be restored separately through `POST /cache/hot-rerun/restore` when you are preparing a dedicated reuse benchmark.

Per-run artifact directories may include:

- core outputs: results.json, results.csv, metadata.json, routes.geojson, results_summary.csv
- DCCS outputs: dccs_candidates.jsonl, dccs_summary.json, refined_routes.jsonl, strict_frontier.jsonl
- REFC outputs: winner_summary.json, certificate_summary.json, route_fragility_map.json, competitor_fragility_breakdown.json, sampled_world_manifest.json, evidence_snapshot_manifest.json
- VOI outputs: value_of_refresh.json, voi_action_trace.json, voi_controller_state.jsonl, voi_action_scores.csv, voi_stop_certificate.json, final_route_trace.json
- evaluation outputs: thesis_results.*, thesis_summary.*, thesis_summary_by_cohort.*, thesis_metrics.json, thesis_plots.json, cohort_composition.json, evaluation_manifest.json
- report outputs: methods_appendix.md, thesis_report.md
- corpus/baseline helpers: od_corpus.*, ors_snapshot.json
- hot-rerun outputs when applicable: hot_rerun_vs_cold_comparison.json, hot_rerun_vs_cold_comparison.csv, hot_rerun_gate.json, hot_rerun_report.md

## Documentation

Start with [docs/DOCS_INDEX.md](docs/DOCS_INDEX.md).

High-value docs:

- [Run and Operations Guide](docs/run-and-operations.md)
- [Backend APIs and Tooling](docs/backend-api-tools.md)
- [API Cookbook](docs/api-cookbook.md)
- [Quality Gates and Benchmarks](docs/quality-gates-and-benchmarks.md)
- [Reproducibility Capsule](docs/reproducibility-capsule.md)
- [VOI Thesis Pipeline Spec](docs/voi-pipeline-spec.md)
- [Notebook Policy](notebooks/NOTEBOOKS_POLICY.md)

`docs/thesis-codebase-report.md` is maintained separately from the general docs set.
