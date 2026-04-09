# Frontend Dev Tools Coverage

Last Updated: 2026-04-09
Applies To: `frontend/app/components/devtools/*`, `frontend/app/api/*`, and adjacent operator panels in `frontend/app/components/*`

This page maps current frontend tooling surfaces to the backend or proxy routes they expose.

## Important Scope Note

The frontend mostly talks to backend endpoints through Next.js proxy routes under `frontend/app/api/*`.

Examples:

- frontend proxy: `/api/health/ready`
- backend target: `/health/ready`

Not every backend operator endpoint is surfaced through the frontend. In particular, there is no current frontend proxy or panel for `POST /cache/hot-rerun/restore`.

## Ops Diagnostics Panel

Current panel: `frontend/app/components/devtools/OpsDiagnosticsPanel.tsx`

Current coverage:

- backend health snapshot
- backend metrics snapshot
- cache stats
- cache clear

Observed proxy routes:

- `/api/health`
- `/api/health/ready`
- `/api/metrics`
- `/api/cache/stats`
- `/api/cache`

Not covered directly:

- hot-rerun cache restore
- arbitrary detached route/controller artifact inspection outside the frontend proxy allowlist

## Batch Runner

Current panel: `frontend/app/components/devtools/BatchRunner.tsx`

Current behavior:

- accepts raw JSON for `POST /batch/pareto`
- accepts CSV text plus JSON options for `POST /batch/import/csv`
- shows the returned `run_id`
- shows the raw JSON response payload

Observed proxy routes:

- `/api/batch/pareto`
- `/api/batch/import/csv`

## Run Inspector

Current panel: `frontend/app/components/devtools/RunInspector.tsx`

Current behavior:

- inspect manifest, scenario manifest, provenance, signature JSON, and scenario signature JSON
- list stable artifacts returned by the backend run-store endpoints
- preview and download only the bounded frontend proxy allowlist of artifact files
- enrich the active run with live `decision_package` state when the inspected run matches the current active route run
- group artifacts into `route-core`, `decision-package`, `support`, `dccs-refc`, `controller-voi`, `witness-refc`, `theorem-lane`, `evaluation`, and `other`
- surface per-group present/listed/pending counts so the browser can distinguish “persisted now”, “listed by lane/run state”, and “conditionally expected after fetch”

Current checked-in reporting anchors that can be inspected through this surface once the run id is known include:

- `thesis_eval_core120_green_repaired_20260404`
- `refc_focus_20260331_h2`
- `thesis_eval_20260331_r2_focused_voi`
- `hot_full_20260331_f2_cold`
- `hot_full_20260331_f2_hot`
- the tranche-local run ids under `backend/out/thesis_campaigns/`

Observed proxy path family:

- `/api/runs/[runId]/[...subpath]`

Practical proxy-backed preview/download targets now include:

- `results.json`
- `results.csv`
- `metadata.json`
- `routes.geojson`
- `results_summary.csv`
- `report.pdf`
- `dccs_candidates.jsonl`
- dccs_summary.json
- `refined_routes.jsonl`
- `strict_frontier.jsonl`
- winner_summary.json
- certificate_summary.json
- route_fragility_map.json
- competitor_fragility_breakdown.json
- `value_of_refresh.json`
- `sampled_world_manifest.json`
- `voi_action_scores.csv`
- thesis_report.md
- methods_appendix.md
- voi_action_trace.json
- voi_stop_certificate.json
- final_route_trace.json
- thesis_summary.csv
- `thesis_results.csv`
- `od_corpus.csv`
- `od_corpus_summary.json`
- `ors_snapshot.json`

This is intentionally narrower than the backend run-store allowlist. The frontend proxy currently exposes a bounded artifact subset, while the richer preference/support/certified-set/abstention/witness/controller/theorem-hook/lane-manifest view comes from the active route response's `decision_package`, not from arbitrary detached artifact retrieval.

Not currently proxy-backed for detached browsing:

- `decision_package.json`
- `preference_summary.json`
- `support_summary.json`
- `support_provenance.json`
- `certified_set.json`
- `certified_set_routes.jsonl`
- `abstention_summary.json`
- `witness_summary.json`
- `witness_routes.jsonl`
- `controller_summary.json`
- `controller_trace.jsonl`
- `voi_controller_state.jsonl`
- `voi_controller_trace_summary.json`
- `voi_replay_oracle_summary.json`
- `theorem_hook_map.json`
- `lane_manifest.json`
- `evaluation_manifest.json`

Those surfaces still exist in the backend run store. They are simply not part of the current frontend proxy allowlist, so the detached browser view and the live `decision_package` view must be documented separately.

## Signature Verifier

Current panel: `frontend/app/components/devtools/SignatureVerifier.tsx`

Current behavior:

- accepts JSON or string payloads
- accepts a required signature
- accepts an optional secret override
- displays the verification response from `/api/verify/signature`

## Adjacent Operator Panels

These are not under `components/devtools`, but they are part of the operator tooling surface and should be treated as related coverage:

- `frontend/app/components/OracleQualityDashboard.tsx`
- `frontend/app/components/ExperimentManager.tsx`
- `frontend/app/components/RouteCertificationPanel.tsx`
- `frontend/app/components/RouteBaselineComparison.tsx`
- `frontend/app/components/ScenarioComparison.tsx`

Oracle Quality Dashboard covers:

- `/api/oracle/quality/check`
- `/api/oracle/quality/dashboard`
- /api/oracle/quality/dashboard.csv

Experiment Manager covers:

- `/api/experiments`
- `/api/experiments/[experimentId]`
- `/api/experiments/[experimentId]/compare`

Route Certification Panel covers the route-result handoff into Run Inspector and exposes the active route’s certification, certified-set, support, world-support, world-fidelity, abstention, witness, controller, theorem-hook, and artifact-handoff state. The panel now treats `terminal_kind` plus the grouped artifact-family status as first-class UI elements, not as a thin certificate badge. The active Run Inspector view can surface the same run’s live `decision_package` summaries directly from route state; that live-route summary path is separate from the bounded `/api/runs/[runId]/[...subpath]` proxy used for detached artifact browsing.

RouteBaselineComparison covers the analyst-facing comparison surface for both OSRM and ORS reference routes and the academic comparator profile on the main page. The current UI shows ETA/cost/CO2/distance deltas, epic score and tier, compute-time deltas, candidate count, live-source coverage, improvement bars, and radar-chart comparisons.

Scenario Comparison covers scenario-level result comparison and signature inspection handoffs, again linking back into Run Inspector when a run id is available. The main page also exposes a current-profile vs academic-selection comparison panel and route overlay, so reporting docs should not imply the UI only compares against OSRM/ORS baselines.

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Backend APIs and Tooling](backend-api-tools.md)
- [Tutorial Mode and Reporting](tutorial-and-reporting.md)
- [Frontend Accessibility and i18n](frontend-accessibility-i18n.md)
