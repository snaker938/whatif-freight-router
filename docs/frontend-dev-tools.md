# Frontend Dev Tools Coverage

Last Updated: 2026-04-09
Applies To: `frontend/app/components/devtools/*` and adjacent reporting panels

This page maps frontend Dev Tools panels to backend endpoints so backend features are reachable from the UI without leaving the app shell.

## Ops Diagnostics Panel

Current surface:

- `GET /health`
- `GET /health/ready`
- `GET /metrics`
- `GET /cache/stats`
- `DELETE /cache`

The panel now shows a health snapshot, cache statistics, and a metrics snapshot side by side. The ready-health payload is particularly useful because it includes `strict_route_ready`, route-graph status, and strict-live readiness fields that drive the disabled state of the main compute flow.

## Custom Vehicle Manager

Current surface:

- `GET /vehicles/custom`
- `POST /vehicles/custom`
- `PUT /vehicles/custom/{vehicle_id}`
- `DELETE /vehicles/custom/{vehicle_id}`

The UI keeps the create/edit payload as raw JSON so advanced users can tune the vehicle profile directly. The default example includes the current terrain parameters block, custom aliases, and the freight-class mapping fields that the backend expects.

## Batch Runner

Current surface:

- `POST /batch/pareto`
- `POST /batch/import/csv`

The batch runner now has two separate inputs:

- a JSON payload for direct OD pairs
- a CSV text area plus options JSON for bulk import runs

The built-in defaults currently target `rigid_hgv`, `no_sharing`, and `max_alternatives: 8`, which matches the current fast path used for demonstration and smoke testing.

## Run Inspector

Current surface:

- `GET /runs/{run_id}/manifest`
- `GET /runs/{run_id}/scenario-manifest`
- `GET /runs/{run_id}/provenance`
- `GET /runs/{run_id}/signature`
- `GET /runs/{run_id}/scenario-signature`
- `GET /runs/{run_id}/artifacts`
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

The inspector supports core-doc inspection, artifact listing, preview, and download. It is the primary in-app reporting surface for the thesis evaluation bundles and for the more specialized DCCS/VOI outputs that the backend emits.

## Signature Verifier

Current surface:

- `POST /verify/signature`

The component accepts either raw text or JSON payload input and can optionally include a secret for deterministic signature verification. That makes it useful both for QA and for checking exported report bundles.

## Adjacent Reporting Surfaces

These are not under `frontend/app/components/devtools/*`, but they are the other current UI panels that operators use to inspect or reproduce backend behavior:

- `ScenarioComparison` for no/partial/full sharing deltas and missing-metric reasons
- `OracleQualityDashboard` for feed freshness, schema validity, signature health, and CSV export
- `ExperimentManager` for saving, filtering, replaying, and deleting run bundles
- `DepartureOptimizerChart` for departure-time sweeps
- `DutyChainPlanner` for chained multi-stop runs
- `RouteBaselineComparison` for OSRM, OpenRouteService, and academic-baseline comparisons
- `MapView` for live-call traces, route badges, and map-level failure diagnostics

## Current Runtime Knobs

The current frontend/backend bridge honors these runtime values:

- `BACKEND_INTERNAL_URL`
- `NEXT_PUBLIC_BACKEND_URL`
- `COMPUTE_ATTEMPT_TIMEOUT_MS`
- `NEXT_PUBLIC_COMPUTE_ATTEMPT_TIMEOUT_MS`
- `COMPUTE_ROUTE_FALLBACK_TIMEOUT_MS`
- `NEXT_PUBLIC_COMPUTE_ROUTE_FALLBACK_TIMEOUT_MS`
- `NEXT_PUBLIC_COMPUTE_DEGRADE_STEPS`
- `NEXT_PUBLIC_ROUTE_GRAPH_WARMUP_BASELINE_MS`

Those values are used by the frontend fetch helper and by the compute/readiness UI, so they directly affect whether the main route button is enabled, how long fallback attempts wait, and how the warmup progress card is interpreted.

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Backend APIs and Tooling](backend-api-tools.md)
- [Tutorial Mode and Reporting](tutorial-and-reporting.md)
- [Frontend Accessibility and i18n](frontend-accessibility-i18n.md)

