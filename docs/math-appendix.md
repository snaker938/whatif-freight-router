# Expanded Math Appendix

Last Updated: 2026-04-09
Applies To: backend objective decomposition and uncertainty summaries

## Objective Decomposition

At route and segment levels, monetary decomposition follows:

`total_monetary = time_cost + fuel_cost + toll_cost + carbon_cost`

Current asset-backed terms are:

- `fuel_cost` is read from `uk_fuel_surface_v1` over the `vehicle_class`, `load_factor`, `speed_kmh`, `grade_pct`, and `ambient_temp_c` axes.
- `toll_cost` uses the empirical UK toll topology and tariff set, with the current confidence calibration `uk-toll-confidence-v2-empirical`.
- `carbon_cost` is priced against the current preflight carbon schedule at `price_per_kg=0.101`.

## Uncertainty Invariants

For each modeled objective family:

- `q50 <= q90 <= q95 <= cvar95`

The current stochastic calibration uses empirical regime fits rather than fixed-form shocks. The fitted `backend/assets/uk/stochastic_regimes_uk.json` asset currently exposes 18 regimes, 2,832 posterior context keys, 12 hour-slot coverage, 9 corridor coverage, and a `quantile_mapping_v1` transform family over `traffic`, `incident`, `weather`, `price`, and `eco`.

At the regime level, the current blend weights are:

- duration mix: `0.56 * traffic + 0.29 * incident + 0.15 * weather`
- monetary mix: `0.72 * price + 0.28 * eco`
- emissions mix: `0.84 * traffic + 0.16 * weather`

The live scenario profile fit uses a weighted L1 context similarity with weights `road_mix_distance=0.04`, `vehicle_penalty=0.10`, `day_penalty=0.12`, `road_penalty=0.16`, `weather_penalty=0.12`, `hour_distance=0.12`, and `geo_distance=0.34`, capped at `1.25`.

## Calibration Notes

- `scenario_profiles_uk_v2_live` is fit with an empirical live feature transform that learns the traffic, incident, weather, duration, fuel, emissions, and stochastic-sigma multipliers from the observed scenario corpus.
- `gaussian_5x5_uk_v3_calibrated` uses a 5 x 5 residual correlation structure and quantile mappings at `z = [-2, -1, 0, 1, 2]` with `q = [0.025, 0.16, 0.50, 0.84, 0.975]`.
- `uk-toll-confidence-v2-empirical` is a compact logit model with coefficients `intercept=0.074138`, `class_signal=0.07386`, and `segment_signal=0.107248`, followed by five reliability bins calibrated to `0.1`, `0.3`, `0.5455`, `0.7`, and `0.9`.

## Dominance Semantics

Strict Pareto filtering is applied over generated candidates, then downstream selection picks based on configured method.

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Dissertation Math Overview](dissertation-math-overview.md)
- [Graph Theory Notes](appendix-graph-theory-notes.md)
- [Strict Error Contract Reference](strict-errors-reference.md)

