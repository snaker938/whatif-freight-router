# CO2e Validation Reference Examples

This note validates the current v0 emissions formula against two explicit
reference calculations.

## Formula under test

Current backend function: `route_emissions_kg(...)` in
`backend/app/objectives_emissions.py`.

Per segment:

- moving emissions:
  - `mass_tonnes * distance_km * emission_factor_kg_per_tkm * speed_factor(speed_kmh)`
- low-speed idle add-on:
  - if `speed_kmh < 5`, add `(duration_s / 3600) * idle_emissions_kg_per_hour`

## Vehicle assumptions used

Using existing `rigid_hgv` profile:

- `mass_tonnes = 12.0`
- `emission_factor_kg_per_tkm = 0.10`
- `idle_emissions_kg_per_hour = 3.5`
- `eco_speed_kmh = 55.0` (default in `speed_factor`)

## Example A: 10 km at 40 km/h

Inputs:

- distance = 10,000 m
- duration = 900 s
- speed = 40 km/h

Manual calculation:

- `speed_factor = 1 + 0.18 * ((55 - 40) / 55)^2 = 1.013388...`
- moving emissions:
  - `12 * 10 * 0.10 * 1.013388... = 12.160661... kg`
- idle add-on:
  - none (`40 >= 5`)

Expected total:

- `12.160661... kg CO2e`

## Example B: 1 km at 2 km/h (idle included)

Inputs:

- distance = 1,000 m
- duration = 1,800 s
- speed = 2 km/h

Manual calculation:

- `speed_factor = 1 + 0.18 * ((55 - 2) / 55)^2 = 1.167207...`
- moving emissions:
  - `12 * 1 * 0.10 * 1.167207... = 1.400648... kg`
- idle add-on:
  - `(1800 / 3600) * 3.5 = 1.75 kg`

Expected total:

- `3.150648... kg CO2e`

## Test coverage

These examples are encoded as tests in:

- `backend/tests/test_emissions_models.py`

The tests assert `route_emissions_kg(...)` matches the above expected values.

## Assumptions and limitations

- This validates implementation consistency, not real-world calibration.
- `speed_factor` is intentionally heuristic in v0 and may differ from external
  standards (DEFRA/HBEFA).
- Weather, gradient, traffic state, and load variation are not yet modeled.
