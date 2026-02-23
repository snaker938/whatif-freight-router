# CO2e Validation Notes

Last Updated: 2026-02-23  
Applies To: emissions and carbon-cost calculations in backend route outputs

## Purpose

Capture the current backend validation focus for emissions and carbon pricing semantics.

## Validation Focus

- route and segment emissions are non-negative
- cost decomposition keeps carbon cost consistent with configured carbon policy
- output metrics stay coherent across:
  - route-level `metrics`
  - segment breakdown rows
  - compare-mode deltas where available

## Runtime Inputs that Affect CO2e

- `emissions_context` (fuel type, euro class, ambient temperature)
- `cost_toggles.carbon_price_per_kg`
- carbon live source / schedule under strict policy (`LIVE_CARBON_SCHEDULE_URL`)
- scenario multipliers (`scenario_summary.emissions_multiplier`)
- terrain and weather effects that indirectly alter energy/fuel usage

## Recommended Validation Commands

From repo root:

```powershell
uv run --project backend pytest backend/tests/test_metrics.py backend/tests/test_segment_breakdown.py
```

Targeted scenario/emissions checks:

```powershell
uv run --project backend pytest backend/tests/test_emissions_context.py backend/tests/test_emissions_models.py
```

## Strict Failure Expectations

When strict carbon inputs are unavailable, route-producing endpoints should emit canonical reason codes:

- `carbon_policy_unavailable`
- `carbon_intensity_unavailable`

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Backend APIs and Tooling](backend-api-tools.md)
- [Expanded Math Appendix](math-appendix.md)
- [Strict Error Contract Reference](strict-errors-reference.md)

