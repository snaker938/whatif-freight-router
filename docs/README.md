# Documentation Index

Last Updated: 2026-02-19  
Applies To: `backend/` + `frontend/` (strict v2 runtime)

This is the source-of-truth index for all project documentation.  
Start here, then follow topic links.

## Getting Started

- [API Cookbook](api-cookbook.md)
- [Backend APIs and Tooling](backend-api-tools.md)
- [Strict Error Contract Reference](strict-errors-reference.md)
- [Model Assets and Data Sources](model-assets-and-data-sources.md)

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

- Run docs checks:

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

## Related Docs

- [Backend APIs and Tooling](backend-api-tools.md)
- [Strict Error Contract Reference](strict-errors-reference.md)
- [Quality Gates and Benchmarks](quality-gates-and-benchmarks.md)
