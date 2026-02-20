# Frontend Dev Tools Coverage

Last Updated: 2026-02-19  
Applies To: `frontend/app/components/devtools/*`

This page maps frontend Dev Tools panels to backend endpoints so backend features are reachable from UI.

## Ops Diagnostics Panel

- `GET /health`
- `GET /metrics`
- `GET /cache/stats`
- `DELETE /cache`

## Custom Vehicle Manager

- `GET /vehicles/custom`
- `POST /vehicles/custom`
- `PUT /vehicles/custom/{vehicle_id}`
- `DELETE /vehicles/custom/{vehicle_id}`

## Batch Runner

- `POST /batch/pareto`
- `POST /batch/import/csv`

## Run Inspector

- `GET /runs/{run_id}/manifest`
- `GET /runs/{run_id}/scenario-manifest`
- `GET /runs/{run_id}/provenance`
- `GET /runs/{run_id}/signature`
- `GET /runs/{run_id}/scenario-signature`
- `GET /runs/{run_id}/artifacts`
- `GET /runs/{run_id}/artifacts/*`

## Signature Verifier

- `POST /verify/signature`

## Related Docs

- [Documentation Index](README.md)
- [Backend APIs and Tooling](backend-api-tools.md)
- [Tutorial Mode and Reporting](tutorial-and-reporting.md)
- [Frontend Accessibility and i18n](frontend-accessibility-i18n.md)
