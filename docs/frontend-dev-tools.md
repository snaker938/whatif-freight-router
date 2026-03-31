# Frontend Dev Tools Coverage

Last Updated: 2026-03-31  
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
- list stable artifacts
- preview arbitrary artifact text
- download core docs and arbitrary artifacts

Observed proxy path family:

- `/api/runs/[runId]/[...subpath]`

Practical preview/download targets now include:

- evaluation_manifest.json
- thesis_summary.json
- thesis_report.md
- methods_appendix.md
- certificate_summary.json
- value_of_refresh.json
- hot_rerun_gate.json on dedicated hot-rerun benchmark runs
- hot_rerun_vs_cold_comparison.json on dedicated hot-rerun benchmark runs

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

Route Certification Panel covers the route-result handoff into Run Inspector and exposes the current certification summary and VOI stop summary for the active run.

Scenario Comparison covers scenario-level result comparison and signature inspection handoffs, again linking back into Run Inspector when a run id is available.

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Backend APIs and Tooling](backend-api-tools.md)
- [Tutorial Mode and Reporting](tutorial-and-reporting.md)
- [Frontend Accessibility and i18n](frontend-accessibility-i18n.md)
