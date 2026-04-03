# Documentation Index

Last Updated: 2026-03-31  
Applies To: current repo state across `backend/`, `frontend/`, `scripts/`, and thesis/production evaluation workflows

This is the main authored-docs index for the project. Generated outputs under `backend/out/artifacts/<run_id>/` are runtime artifacts, not replacements for these authored docs.

## Start Here

- [Run and Operations Guide](run-and-operations.md)
- [Backend APIs and Tooling](backend-api-tools.md)
- [API Cookbook](api-cookbook.md)
- [Quality Gates and Benchmarks](quality-gates-and-benchmarks.md)
- [Reproducibility Capsule](reproducibility-capsule.md)
- [Sample Manifest and Outputs](sample-manifest.md)
- [Notebook Policy](../notebooks/NOTEBOOKS_POLICY.md)

## Backend and Runtime

- [Backend APIs and Tooling](backend-api-tools.md)
- [Strict Error Contract Reference](strict-errors-reference.md)
- [Model Assets and Data Sources](model-assets-and-data-sources.md)
- [Run and Operations Guide](run-and-operations.md)
- [Quality Gates and Benchmarks](quality-gates-and-benchmarks.md)
- [Performance Profiling Notes](performance-profiling-notes.md)
- [ETA Concept Drift Checks](eta-concept-drift.md)
- [CO2e Validation Notes](co2e-validation.md)

## Thesis / Evaluation / Reporting

- [VOI Thesis Pipeline Spec](voi-pipeline-spec.md)
- [Quality Gates and Benchmarks](quality-gates-and-benchmarks.md)
- [Reproducibility Capsule](reproducibility-capsule.md)
- [Tutorial Mode and Reporting](tutorial-and-reporting.md)
- [Sample Manifest and Outputs](sample-manifest.md)
- [Thesis-Grade Codebase Report](thesis-codebase-report.md)

Generated evaluation families described by these docs; the evaluator also emits cohort-scaffolded metadata (`evaluation_suite`, `cohort_scaffolding`) and cohort summary/composition artifacts:

- broad cold thesis proof
- focused REFC proof
- focused VOI proof
- DCCS diagnostic probe
- hot-rerun cold source and hot-rerun proof
- composed suite outputs with cohort summaries, cohort composition, and suite-source provenance

## Frontend

- [Frontend Dev Tools Coverage](frontend-dev-tools.md)
- [Frontend Accessibility and i18n](frontend-accessibility-i18n.md)
- [Map Overlays and Tooltips](map-overlays-tooltips.md)
- [Tutorial Mode and Reporting](tutorial-and-reporting.md)

## Modeling / Math / Scenario Notes

- [Dissertation Math Overview](dissertation-math-overview.md)
- [Expanded Math Appendix](math-appendix.md)
- [Graph Theory Notes](appendix-graph-theory-notes.md)
- [Synthetic Incidents and Weather](synthetic-incidents-weather.md)
- [VOI Thesis Pipeline Spec](voi-pipeline-spec.md)
- [CO2e Validation Notes](co2e-validation.md)

## Examples and Operator References

- [API Cookbook](api-cookbook.md)
- [Sample Manifest and Outputs](sample-manifest.md)
- [Backend APIs and Tooling](backend-api-tools.md)
- [Run and Operations Guide](run-and-operations.md)

## Maintenance

Tracked docs checks from repo root:

```powershell
python scripts/check_docs.py
python scripts/check_docs.py --check-links
python scripts/check_docs.py --check-orphans
python scripts/check_docs.py --check-paths
python scripts/check_docs.py --check-endpoints
```

Local docs viewer:

```powershell
.\scripts\serve_docs.ps1
```

Strict runtime preflight before backend startup or deploy checks:

```powershell
uv run --project backend python backend/scripts/preflight_live_runtime.py
```

## Notes

- `docs/thesis-codebase-report.md` is intentionally maintained separately from the general docs refresh cycle.
- Generated reports such as thesis_report.md, methods_appendix.md, hot_rerun_report.md, or hot_rerun_gate.json under `backend/out/artifacts/<run_id>/` should be cited as runtime artifacts, not edited as authored docs.
- Signed run metadata and artifacts are consumed through `GET /runs/{run_id}/manifest`, `GET /runs/{run_id}/signature`, `GET /runs/{run_id}/artifacts`, and `GET /runs/{run_id}/artifacts/{artifact_name}`.
- Dedicated hot-rerun reuse proofs are separate from cold thesis proofs and can restore cache state through `POST /cache/hot-rerun/restore` before benchmarking.

## Related Docs

- [Run and Operations Guide](run-and-operations.md)
- [Quality Gates and Benchmarks](quality-gates-and-benchmarks.md)
- [Reproducibility Capsule](reproducibility-capsule.md)
- [Sample Manifest and Outputs](sample-manifest.md)
