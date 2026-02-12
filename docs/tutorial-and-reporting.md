# Tutorial Mode and PDF Reporting

## Interactive Tutorial Mode

The frontend includes a guided tutorial overlay to onboard first-time users.

- First-run behavior:
  - Automatically opens on first load if `localStorage["tutorial_v1_seen"]` is not `"1"`.
- Dismissal behavior:
  - Finishing or skipping sets `tutorial_v1_seen=1`.
- Manual replay:
  - Click `Start tutorial` in the `Setup` card.

Tutorial steps cover:

1. Setting start/destination pins
2. Computing Pareto routes
3. Comparing No/Partial/Full scenarios
4. Running departure optimization
5. Reviewing artifacts and reports

## PDF Run Report Artifact

Every successful `POST /batch/pareto` run generates:

- `results.json`
- `results.csv`
- `metadata.json`
- `routes.geojson`
- `results_summary.csv`
- `report.pdf`

Retrieve artifacts via:

- `GET /runs/{run_id}/artifacts` (lists available artifacts)
- `GET /runs/{run_id}/artifacts/report.pdf` (download PDF directly)

`report.pdf` includes run metadata, summary metrics, top route rows, and manifest/signature references.

## Regenerate a Report from Existing Artifacts

From `backend/`:

```powershell
uv run python scripts/generate_run_report.py --run-id <run_id> --out-dir out
```

Optional overrides:

- `--manifest <path>`
- `--results <path>`
- `--metadata <path>`
- `--output <path>`
