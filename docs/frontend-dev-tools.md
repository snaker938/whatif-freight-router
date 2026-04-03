# Frontend Dev Tools Coverage

Last Updated: 2026-04-03  
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
- detached run-artifact inspection

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

- inspect manifest
- inspect scenario manifest
- inspect provenance
- inspect signature JSON
- inspect scenario signature JSON
- list stable artifacts returned by the backend run-store endpoints
- preview and download only the current frontend proxy allowlist of artifact files
- render `decision_package` summaries inline when the inspected run matches the current active route run

Observed proxy path family:

- `/api/runs/[runId]/[...subpath]`

Practical proxy-backed preview/download targets now include:

- dccs_summary.json
- winner_summary.json
- certificate_summary.json
- route_fragility_map.json
- competitor_fragility_breakdown.json
- thesis_report.md
- methods_appendix.md
- value_of_refresh.json
- sampled_world_manifest.json
- voi_action_trace.json
- voi_stop_certificate.json
- final_route_trace.json
- thesis_summary.csv

This is intentionally narrower than the backend run-store allowlist. The frontend proxy currently exposes a bounded artifact subset, while the richer preference/support/certified-set/abstention/witness/controller/theorem-hook/lane-manifest view comes from the active route response's `decision_package`, not from arbitrary detached artifact retrieval.

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
- `frontend/app/components/ScenarioComparison.tsx`

Oracle Quality Dashboard covers:

- `/api/oracle/quality/check`
- `/api/oracle/quality/dashboard`
- /api/oracle/quality/dashboard.csv

Experiment Manager covers:

- `/api/experiments`
- `/api/experiments/[experimentId]`
- `/api/experiments/[experimentId]/compare`

Route Certification Panel covers the route-result handoff into Run Inspector and exposes the current certification summary, VOI stop summary, and `decision_package`-driven preference, support, certified-set, abstention, witness, controller, and lane-manifest inspection for the active run. The active Run Inspector view can also surface theorem-hook details from the same route response. That live-route summary path is separate from the bounded `/api/runs/[runId]/[...subpath]` proxy used for detached artifact browsing.

Scenario Comparison covers scenario-level result comparison and signature inspection handoffs, again linking back into Run Inspector when a run id is available.

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Backend APIs and Tooling](backend-api-tools.md)
- [Tutorial Mode and Reporting](tutorial-and-reporting.md)
- [Frontend Accessibility and i18n](frontend-accessibility-i18n.md)
