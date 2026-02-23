# Tutorial Mode and Reporting

Last Updated: 2026-02-23  
Applies To: frontend tutorial flows and backend run artifact APIs

## Tutorial Mode Scope

Tutorial flow guides users through:

- route and pareto generation
- scenario compare
- departure optimization
- duty chain and experiment workflows
- artifact inspection

## Backend Reporting Endpoints Used by UI

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

## Practical Operator Flow

1. trigger compute action in UI
2. capture returned `run_id`
3. inspect manifest + provenance + signatures
4. download artifact files needed for reporting/debug

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

