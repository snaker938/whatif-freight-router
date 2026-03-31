# Tutorial Mode and Reporting

Last Updated: 2026-03-31  
Applies To: frontend tutorial flows, devtools reporting surfaces, run artifacts, and signature inspection

This page describes the current frontend tutorial/reporting flow and the backend artifact surfaces it depends on.

## Tutorial Scope

The guided flow covers:

- route and Pareto generation
- baseline comparison
- scenario compare
- departure optimization
- duty chain planning
- oracle quality recording
- experiment save/load/replay flows
- run artifact inspection and signature verification

The tutorial overlay is implemented separately from generated thesis reports. It is UI guidance, not thesis proof.

## Frontend Reporting Surfaces

Current reporting-adjacent UI surfaces include:

- Run Inspector
- Signature Verifier
- Ops Diagnostics
- Oracle Quality Dashboard
- Experiment Manager
- Route Certification Panel
- Scenario Comparison
- route readiness and compute trace overlays on the main page

## Backend Reporting Endpoints Used By UI

- `GET /runs/{run_id}/manifest`
- `GET /runs/{run_id}/scenario-manifest`
- `GET /runs/{run_id}/provenance`
- `GET /runs/{run_id}/signature`
- `GET /runs/{run_id}/scenario-signature`
- `GET /runs/{run_id}/artifacts`
- `GET /runs/{run_id}/artifacts/{artifact_name}`
- `POST /verify/signature`
- `GET /oracle/quality/dashboard`
- `GET /oracle/quality/dashboard.csv`
- `POST /oracle/quality/check`

## Run Inspector

The current Run Inspector can:

- inspect manifest, scenario manifest, provenance, signature, and scenario signature
- list the stable public artifact set for a run
- preview artifact text inline
- download core docs and arbitrary artifacts by name
- open directly from route certification and scenario comparison flows when a run id is present

This is no longer a PDF-centered flow. Common inspection targets now include:

- evaluation_manifest.json
- thesis_summary.json
- thesis_report.md
- methods_appendix.md
- value_of_refresh.json
- certificate_summary.json
- hot_rerun_gate.json on dedicated hot-rerun benchmark runs
- hot_rerun_vs_cold_comparison.json on dedicated hot-rerun benchmark runs

## Signature Verification

The Signature Verifier accepts:

- JSON payloads
- string payloads
- a required signature
- an optional secret override

It returns the backend verification response so operators can compare `valid` and `expected_signature`.

## Practical Operator Flow

1. trigger a compute action in the UI
2. capture the returned `run_id`
3. open Run Inspector and load manifest, provenance, and signatures
4. inspect or download the exact artifacts needed for debugging or reporting
5. use Signature Verifier when you need detached verification of a manifest payload
6. use Route Certification Panel or Scenario Comparison when you need a direct handoff into run-level inspection

## Local Commands

Frontend:

```powershell
pnpm -C frontend dev
```

Backend headless scenario smoke:

```powershell
uv run --project backend python backend/scripts/run_headless_scenario.py --input-json docs/examples/sample_batch_request.json
```

Evaluation/report outputs:

```powershell
uv run --project backend python backend/scripts/run_thesis_evaluation.py
uv run --project backend python backend/scripts/run_hot_rerun_benchmark.py
uv run --project backend python backend/scripts/compose_thesis_suite_report.py
```

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Frontend Dev Tools Coverage](frontend-dev-tools.md)
- [Sample Manifest and Outputs](sample-manifest.md)
- [Backend APIs and Tooling](backend-api-tools.md)
