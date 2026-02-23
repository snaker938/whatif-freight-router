# Documentation Index

Last Updated: 2026-02-23  
Applies To: `backend/` + `frontend/` (strict v2 runtime)

This is the source-of-truth index for all project documentation.  
Start here, then follow topic links.

## Getting Started

- [Run and Operations Guide](run-and-operations.md)
- [API Cookbook](api-cookbook.md)
- [Backend APIs and Tooling](backend-api-tools.md)
- [Strict Error Contract Reference](strict-errors-reference.md)
- [Model Assets and Data Sources](model-assets-and-data-sources.md)

## Backend Core

- [Backend APIs and Tooling](backend-api-tools.md)
- [Strict Error Contract Reference](strict-errors-reference.md)
- [Model Assets and Data Sources](model-assets-and-data-sources.md)
- [Quality Gates and Benchmarks](quality-gates-and-benchmarks.md)
- [Performance Profiling Notes](performance-profiling-notes.md)
- [Reproducibility Capsule](reproducibility-capsule.md)
- [CO2e Validation Notes](co2e-validation.md)
- [ETA Concept Drift Checks](eta-concept-drift.md)
- [Sample Manifest and Outputs](sample-manifest.md)

## Frontend

- [Tutorial Mode and Reporting](tutorial-and-reporting.md)
- [Frontend Dev Tools Coverage](frontend-dev-tools.md)
- [Frontend Accessibility and i18n](frontend-accessibility-i18n.md)
- [Map Overlays and Tooltips](map-overlays-tooltips.md)

## Routing and Modeling

- [Synthetic Incidents and Weather](synthetic-incidents-weather.md)
- [Sample Manifest and Outputs](sample-manifest.md)
- [CO2e Validation Notes](co2e-validation.md)
- [Dissertation Math Overview](dissertation-math-overview.md)
- [Expanded Math Appendix](math-appendix.md)
- [Graph Theory Notes](appendix-graph-theory-notes.md)

## Quality, Performance, Reproducibility

- [Quality Gates and Benchmarks](quality-gates-and-benchmarks.md)
- [Performance Profiling Notes](performance-profiling-notes.md)
- [Reproducibility Capsule](reproducibility-capsule.md)
- [ETA Concept Drift Checks](eta-concept-drift.md)

## Docs Maintenance

- Validate docs consistency and backend endpoint parity:

```powershell
python scripts/check_docs.py
```

- Run individual checks:

```powershell
python scripts/check_docs.py --check-links
python scripts/check_docs.py --check-orphans
python scripts/check_docs.py --check-paths
python scripts/check_docs.py --check-endpoints
```

## Viewing and Running Docs

Project docs are markdown files under `docs/`. There is no separate docs build pipeline required.

- FastAPI interactive API docs:
  - start backend (`.\scripts\dev.ps1` from repo root)
  - open `http://localhost:8000/docs`
- Markdown docs (local web view):
  - run `.\scripts\serve_docs.ps1`
  - open `http://localhost:8088/`
- Markdown docs (IDE preview):
  - open any `docs/*.md` in VS Code
  - run `Markdown: Open Preview` / `Markdown: Open Preview to the Side`

## Related Docs

- [Run and Operations Guide](run-and-operations.md)
- [Backend APIs and Tooling](backend-api-tools.md)
- [Strict Error Contract Reference](strict-errors-reference.md)
- [Quality Gates and Benchmarks](quality-gates-and-benchmarks.md)

