# CO2e Validation Notes

Last Updated: 2026-02-19  
Applies To: emissions and carbon-cost calculations in route outputs

## Purpose

This document tracks sanity checks for emissions calculations and carbon-price application in route and segment breakdowns.

## Validation Focus

- emissions are non-negative
- carbon cost equals emissions times active carbon-price policy/override
- decomposition identity holds with carbon component included

## Recommended Check Path

From `backend/`:

```powershell
uv run python -m pytest backend/tests/test_metrics.py backend/tests/test_segment_breakdown.py
```

## Related Docs

- [Documentation Index](README.md)
- [Backend APIs and Tooling](backend-api-tools.md)
- [Expanded Math Appendix](math-appendix.md)
- [Quality Gates and Benchmarks](quality-gates-and-benchmarks.md)
