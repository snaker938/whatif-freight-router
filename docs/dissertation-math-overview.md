# Dissertation Math Overview

Last Updated: 2026-04-09
Applies To: route scoring, Pareto selection, and robust utility summaries

## Core Objective Family

Route candidates are evaluated across at least:

- duration
- monetary cost
- emissions

Risk-aware summaries include utility quantiles and CVaR where uncertainty is active.

## Pareto Concept

The strict frontier contains candidates not dominated on configured objectives.

## Utility and Risk

Utility fields in route outputs:

- `utility_mean`
- `utility_q95`
- `utility_cvar95`
- `robust_score`

## Current Calibration Snapshot

The current thesis-facing calibration assets are empirical rather than synthetic:

- `scenario_profiles_uk_v2_live` is calibrated with `empirical_live_fit` over 384 contextual profiles and a 192-context holdout slice, using `temporal_forward_plus_corridor_block` with 6 hour slots and 16 corridors in the holdout partition.
- That scenario fit currently reports `mode_separation_mean=0.169764`, zero holdout MAPE on duration/monetary/emissions, `full_identity_share=0.291667`, and an even split between observed and projected mode contexts at `0.5` each.
- `gaussian_5x5_uk_v3_calibrated` is calibrated with `v4-uk-residual-fit` over 50,000 holdout rows, with holdout coverage `1.0`, PIT mean `0.5149096244101729`, CRPS mean `0.47377558811984366`, and duration MAPE `0.14058663646867195`.
- The fuel-consumption surface used in objective decomposition is `uk_fuel_surface_v1` with a 4 x 4 x 5 x 5 x 5 axis grid, and the current toll-confidence calibration is `uk-toll-confidence-v2-empirical` with five calibrated reliability bins.

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Expanded Math Appendix](math-appendix.md)
- [Graph Theory Notes](appendix-graph-theory-notes.md)
- [CO2e Validation Notes](co2e-validation.md)

