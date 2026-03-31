# CO2e Validation Notes

Last Updated: 2026-03-31  
Applies To: emissions and carbon-cost calculations in backend route outputs and evaluation summaries

## Purpose

Capture the current backend validation focus for emissions and carbon pricing semantics.

## Validation Focus

- route and segment emissions remain non-negative
- monetary decomposition keeps carbon cost consistent with configured policy inputs
- route-level `metrics.emissions_kg` stays coherent with segment breakdown totals
- scenario and environment multipliers remain visible in route summaries
- evaluation outputs stay internally consistent with route-level emissions and cost data
- when carbon inputs are missing, the backend emits canonical strict reason codes rather than partial silent values

## Runtime Inputs That Affect CO2e

- `emissions_context`
- `cost_toggles.carbon_price_per_kg`
- `backend/assets/uk/carbon_price_schedule_uk.json`
- `backend/assets/uk/fuel_prices_uk.json`
- scenario multipliers through `scenario_summary`
- terrain and weather effects that alter energy or fuel use
- EV-style energy fields such as `energy_kwh` where applicable

## What To Compare

When validating emissions behavior, inspect:

- route-level `metrics.emissions_kg`
- route-level monetary totals that include carbon cost
- `segment_breakdown` emissions and monetary rows
- scenario-driven multiplier effects
- evaluation summary consistency in `thesis_results.*` and `thesis_summary.*`
- any route-level `scenario_summary` values that explain the applied environmental and policy context

## Recommended Validation Commands

From repo root:

```powershell
uv run --project backend pytest backend/tests/test_metrics.py backend/tests/test_segment_breakdown.py
uv run --project backend pytest backend/tests/test_emissions_context.py backend/tests/test_emissions_models.py
uv run --project backend pytest backend/tests/test_evaluation_metrics.py
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
