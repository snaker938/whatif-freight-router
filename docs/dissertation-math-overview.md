# Dissertation Math Overview

Last Updated: 2026-02-19  
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

## Related Docs

- [Documentation Index](README.md)
- [Expanded Math Appendix](math-appendix.md)
- [Graph Theory Notes](appendix-graph-theory-notes.md)
- [CO2e Validation Notes](co2e-validation.md)
