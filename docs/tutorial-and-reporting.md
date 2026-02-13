# Tutorial Mode and PDF Reporting

## Interactive Tutorial Mode (v2)

The frontend now uses a strict, chaptered guided tutorial that walks through all major frontend workflows.

- Tutorial style:
  - strict progression (required checklist actions must be completed)
  - chaptered flow with detailed "what changed" and "how results changed" guidance
  - spotlight + directional callout with auto-scroll to the active control
- Scope:
  - map pin lifecycle and popup actions
  - setup, advanced parameters, preferences, routes, selected-route explainability
  - scenario compare, departure optimization, timeline playback
  - duty chain, oracle quality, and experiments lifecycle
- Optional fields:
  - each optional step requires either interaction or explicit "use default" confirmation
- Desktop guard:
  - full guided mode is desktop-only in this wave (`>=1100px` viewport)

### Progress Persistence

Tutorial state is persisted in browser local storage:

- `tutorial_v2_progress`
  - stores `stepIndex`, completed actions, optional decisions, and update timestamp
- `tutorial_v2_completed`
  - `"1"` when the tutorial is completed

Behavior:

1. If unfinished progress exists, opening tutorial shows Resume/Restart.
2. Close keeps progress.
3. Finish marks completion and clears saved progress.
4. Setup card still includes a manual restart entry point.

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

## QA Checklist (Tutorial)

1. Start tutorial from Setup.
2. Confirm chapter/step counters render.
3. Verify Next is disabled until required actions are complete.
4. Close tutorial, reopen, and verify Resume works.
5. Restart and verify state resets to chapter 1.
6. Verify spotlight + callout follows active control.
7. Confirm desktop-only notice appears on narrow viewports.
