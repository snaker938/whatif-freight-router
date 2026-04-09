# CO2e Validation Notes

Last Updated: 2026-04-09
Applies To: emissions and carbon-cost calculations in backend route outputs

## Purpose

Capture the current backend validation focus for emissions and carbon-pricing semantics, using the latest local fuel, carbon, and thesis-evaluation evidence currently present in the repo.

## Current Asset-Backed Evidence

### Fuel snapshot

`backend/assets/uk/fuel_prices_uk.json` currently records:

- as-of `2026-03-23T00:00:00Z`
- refreshed `2026-03-30T00:20:29Z`
- provider contract version `fuel-live-v1`
- diesel `1.6688 GBP/L`
- petrol `1.4416 GBP/L`
- LNG `1.015 GBP/L`
- grid electricity `0.248 GBP/kWh`

### Carbon policy snapshot

The latest strict preflight records:

- carbon price `0.101 GBP/kg`
- scope-adjusted emissions factor `1.121`

`backend/assets/uk/carbon_price_schedule_uk.json` currently records for 2026:

- central `0.101 GBP/kg`
- high `0.11918 GBP/kg`
- low `0.08282 GBP/kg`

### Fuel-surface geometry

`backend/out/model_assets/fuel_consumption_surface_uk_compiled.json` currently exposes:

- 4 vehicle classes
- 4 load-factor levels
- 5 speed levels
- 5 grade levels
- 5 ambient-temperature levels

That 4x4x5x5x5 surface is the main structured input behind route-level fuel and emissions estimates.

## Validation Focus

- route and segment emissions remain non-negative
- carbon cost stays consistent with the configured carbon policy
- output metrics remain coherent across route-level metrics, segment breakdown rows, and compare-mode deltas
- scenario, terrain, weather, and departure multipliers do not break emissions accounting
- fuel-surface lookups remain consistent with vehicle profile, load, speed, grade, and ambient-temperature assumptions

## Runtime Inputs That Affect CO2e

- `emissions_context` such as fuel type, Euro class, and ambient temperature
- `cost_toggles.carbon_price_per_kg`
- carbon live source / schedule under strict policy via `LIVE_CARBON_SCHEDULE_URL`
- scenario multipliers, especially `scenario_summary.emissions_multiplier`
- terrain uplift and weather effects that alter speed or energy use
- route distance, stop-and-go pressure, and stochastic route-state changes

## Latest Thesis-Lane CO2e Signal

The latest checked thesis campaign bundle records the following mean emissions delta versus `V0`:

- `A`: `-1.4795 kg`
- `B`: `-1.4795 kg`
- `C`: `-1.4795 kg`

That does not prove a universal external win rate, but it does confirm that the currently checked thesis lane preserves measurable emissions improvements relative to the matched-budget legacy comparator.

## Recommended Validation Commands

From repo root:

```powershell
uv run --project backend pytest backend/tests/test_metrics.py backend/tests/test_segment_breakdown.py
```

Targeted emissions checks:

```powershell
uv run --project backend pytest backend/tests/test_emissions_context.py backend/tests/test_emissions_models.py
```

Useful companion checks:

```powershell
uv run --project backend pytest backend/tests/test_cost_model.py backend/tests/test_weather_adapter.py
```

## Strict Failure Expectations

When strict carbon inputs are unavailable, route-producing endpoints should emit canonical reason codes:

- `carbon_policy_unavailable`
- `carbon_intensity_unavailable`

Fuel-source failures should remain explicit as fuel-source or auth failures rather than silently backfilling carbon cost from stale assumptions.

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Backend APIs and Tooling](backend-api-tools.md)
- [Expanded Math Appendix](math-appendix.md)
- [Strict Error Contract Reference](strict-errors-reference.md)
