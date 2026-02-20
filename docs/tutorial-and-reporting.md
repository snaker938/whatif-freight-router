# Tutorial Mode and Reporting

Last Updated: 2026-02-19  
Applies To: `frontend/app/components/TutorialOverlay.tsx`, run artifacts APIs

## Tutorial Mode

The frontend tutorial is chaptered and stateful (`tutorial_v3_*` keys). It walks through:

- setup and map interaction
- route/pareto compute
- scenario compare and departure optimization
- duty chain and experiments
- run artifact inspection

Tutorial behavior is desktop-first and designed to keep users inside core workflows.

## Reporting and Run Artifacts

Successful compute flows can produce run artifacts retrievable from backend:

- `GET /runs/{run_id}/artifacts`
- `GET /runs/{run_id}/artifacts/report.pdf`
- `GET /runs/{run_id}/manifest`
- `GET /runs/{run_id}/provenance`

## Practical Flow

1. Execute compute from the sidebar.
2. Capture returned `run_id`.
3. Inspect provenance/signature.
4. Download `report.pdf` and result files.

## Commands

From repo root:

```powershell
pnpm -C frontend dev
```

From `backend/` (optional validation):

```powershell
uv run python scripts/run_headless_scenario.py --input-json ../docs/examples/sample_batch_request.json
```

## Related Docs

- [Documentation Index](README.md)
- [Frontend Dev Tools Coverage](frontend-dev-tools.md)
- [Sample Manifest and Outputs](sample-manifest.md)
- [API Cookbook](api-cookbook.md)
