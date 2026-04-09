# Tutorial Mode and Reporting

Last Updated: 2026-04-09
Applies To: frontend tutorial flows, reporting surfaces, and backend run artifact APIs

## Tutorial Mode Scope

Tutorial flow guides users through:

- route and Pareto generation
- scenario compare
- departure optimization
- duty chain and experiment workflows
- artifact inspection
- baseline comparison and live diagnostics
- tutorial-aware map interactions

The current tutorial shell has four operational states: `blocked`, `chooser`, `running`, and `completed`. It is desktop-gated, persists progress locally, and can resume or restart from saved state when an unfinished walkthrough exists.

## How Tutorial Mode Works

- The guided overlay is implemented in `frontend/app/components/TutorialOverlay.tsx` and driven by the step catalog under `frontend/app/lib/tutorial/*`.
- The overlay can lock the map, lock only a sidebar section, or stay free depending on the active step's lock scope.
- When a step needs a precise target and the target is not yet available, the overlay shows a positioning/loading state before it promotes the card into the proper location.
- Manual confirmation is still supported for steps that need explicit acknowledgement, such as map pin confirmation and other marked actions.
- The setup area can prefill canonical tutorial inputs, including the Newcastle-to-London example, duty-chain stops, and the tutorial experiment bundle.
- Locale changes are available during the tutorial, but they only affect labels and formatting. They do not change route math.

## Backend Reporting Endpoints Used By The UI

The run inspector proxies a strict allowlist of report and artifact paths. Current core report endpoints are:

- `GET /runs/{run_id}/manifest`
- `GET /runs/{run_id}/scenario-manifest`
- `GET /runs/{run_id}/provenance`
- `GET /runs/{run_id}/signature`
- `GET /runs/{run_id}/scenario-signature`
- `GET /runs/{run_id}/artifacts`

Current artifact endpoints exposed through the frontend proxy are:

- `GET /runs/{run_id}/artifacts/results.json`
- `GET /runs/{run_id}/artifacts/results.csv`
- `GET /runs/{run_id}/artifacts/metadata.json`
- `GET /runs/{run_id}/artifacts/routes.geojson`
- `GET /runs/{run_id}/artifacts/results_summary.csv`
- `GET /runs/{run_id}/artifacts/report.pdf`
- `GET /runs/{run_id}/artifacts/dccs_candidates.jsonl`
- `GET /runs/{run_id}/artifacts/dccs_summary.json`
- `GET /runs/{run_id}/artifacts/refined_routes.jsonl`
- `GET /runs/{run_id}/artifacts/strict_frontier.jsonl`
- `GET /runs/{run_id}/artifacts/winner_summary.json`
- `GET /runs/{run_id}/artifacts/certificate_summary.json`
- `GET /runs/{run_id}/artifacts/route_fragility_map.json`
- `GET /runs/{run_id}/artifacts/competitor_fragility_breakdown.json`
- `GET /runs/{run_id}/artifacts/value_of_refresh.json`
- `GET /runs/{run_id}/artifacts/sampled_world_manifest.json`
- `GET /runs/{run_id}/artifacts/voi_action_trace.json`
- `GET /runs/{run_id}/artifacts/voi_action_scores.csv`
- `GET /runs/{run_id}/artifacts/voi_stop_certificate.json`
- `GET /runs/{run_id}/artifacts/final_route_trace.json`
- `GET /runs/{run_id}/artifacts/od_corpus.csv`
- `GET /runs/{run_id}/artifacts/od_corpus_summary.json`
- `GET /runs/{run_id}/artifacts/ors_snapshot.json`
- `GET /runs/{run_id}/artifacts/thesis_results.csv`
- `GET /runs/{run_id}/artifacts/thesis_summary.csv`
- `GET /runs/{run_id}/artifacts/methods_appendix.md`
- `GET /runs/{run_id}/artifacts/thesis_report.md`

The `frontend/app/api/runs/[runId]/[...subpath]/route.ts` proxy is intentionally allowlisted, so reporting links stay predictable instead of exposing arbitrary backend paths.

## Current Reporting Panels

### Route Comparison

`frontend/app/components/RouteBaselineComparison.tsx` now reports a richer comparison than simple ETA alone. It shows:

- ETA improvement
- cost improvement
- CO2 improvement
- distance improvement
- `epic score`
- `epic tier`
- smart compute elapsed
- baseline fetch elapsed
- smart candidate count
- live-source coverage
- live calls observed

The comparison math is built from current route metrics and uses the existing baseline route as the reference. Positive percentages mean better when the metric is lower-is-better, which is why the panel labels the sign convention explicitly.

The map and summary panels currently support three baseline styles:

- OSRM baseline
- OpenRouteService baseline, including the proxy-backed variant
- academic reference selection

### Scenario Compare

`frontend/app/components/ScenarioComparison.tsx` compares no-sharing, partial-sharing, and full-sharing outcomes side by side. It also surfaces delta reason codes when a metric is missing, which makes it easier to explain when a comparison row is not fully populated.

### Experiment Reporting

`frontend/app/components/ExperimentManager.tsx` is the current save/replay surface for bundles. It supports catalog filtering by name, vehicle, scenario mode, and sort order, then lets the user load, open, edit metadata, replay, or delete a saved bundle.

### Oracle Quality Reporting

`frontend/app/components/OracleQualityDashboard.tsx` records source checks and reports:

- total checks
- source count
- stale threshold
- per-source pass rate
- schema failures
- signature failures
- stale count
- average latency
- last observed timestamp

The dashboard also exposes a CSV export at `GET /api/oracle/quality/dashboard.csv`.

### Departure And Duty Reporting

`frontend/app/components/DepartureOptimizerChart.tsx` reports each tested departure time with the chosen route, score, ETA, cost, and CO2, then lets the user apply one result back into the active request.

`frontend/app/components/DutyChainPlanner.tsx` reports per-leg outcomes plus total metrics across the full chain, including optional total energy when the backend returns it.

### Segment Reporting

`frontend/app/components/SegmentBreakdown.tsx` shows a per-segment table for the selected route, defaults to a 40-row preview, and can expand to the full set. It can also copy the visible rows as CSV.

## Compute And Debug Reporting

The compute trace overlay now captures the operational story of a run:

- request ID
- stage timing
- retry and fallback behavior
- live-call trace summary
- graph diagnostics
- scenario coverage gate information
- live refresh gate information
- slowest calls
- optional AI diagnostic bundle

The UI also has direct access to:

- `POST /api/route/baseline`
- `POST /api/route/baseline/ors`
- `GET /api/debug/live-calls/{requestId}`
- `GET /api/health/ready`
- `GET /api/metrics`

## Practical Operator Flow

1. trigger compute action in UI
2. capture returned `run_id`
3. inspect manifest, provenance, and signatures
4. inspect route compare panels, scenario deltas, and live-call traces
5. download artifact files needed for reporting or debugging

## Local Commands

Frontend:

```powershell
pnpm -C frontend dev
```

Backend headless scenario smoke:

```powershell
uv run --project backend python backend/scripts/run_headless_scenario.py --input-json docs/examples/sample_batch_request.json
```

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Frontend Dev Tools Coverage](frontend-dev-tools.md)
- [Sample Manifest and Outputs](sample-manifest.md)
- [Backend APIs and Tooling](backend-api-tools.md)

