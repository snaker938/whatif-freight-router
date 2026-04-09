# Reviewer Quickstart

This page gives a minimal, conservative path to reproduce one representative thesis artifact bundle from the current repository.

It does not claim that all gates are green. It only documents the commands used to reproduce the current checked artifact flow.

## What This Reproduces

The quickest end-to-end check is the thesis evaluation path that emits a representative bundle under `backend/out/`.

Expected outputs from a successful run include a bundle with:

- `methods_appendix.md`
- `thesis_report.md`
- the run-level JSON summary and manifest files written by the evaluator

## Commands

From the repo root:

```powershell
Set-Location backend
uv sync --dev
uv run python scripts/preflight_live_runtime.py
uv run python scripts/run_thesis_lane.py --manage-local-backend
```

If you want a narrower slice instead of a full lane run, inspect the lane runner help after preflight:

```powershell
Set-Location backend
uv run python scripts/run_thesis_lane.py --help
```

## What To Check

After the run, inspect the newest thesis lane report under `backend/out/` and confirm that the bundle contains:

- a thesis report
- a methods appendix
- the mode-specific artifact files for the selected lane

If the run stops early, the most likely causes are:

- strict live runtime preflight failure
- missing backend dependencies in the local `uv` environment
- no supported live data for the selected mode

## Notes

- The exact artifact names depend on the lane and selected mode.
- This quickstart is intentionally narrow and does not attempt to restate the full claim matrix.
- For operational details, see [Run and Operations Guide](run-and-operations.md).
- For the larger thesis narrative, see [Thesis-Grade Codebase Report](thesis-codebase-report.md).
