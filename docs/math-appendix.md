# Expanded Math Appendix

Last Updated: 2026-02-19  
Applies To: backend objective decomposition and uncertainty summaries

## Objective Decomposition

At route and segment levels, monetary decomposition follows:

`total_monetary = time_cost + fuel_cost + toll_cost + carbon_cost`

## Uncertainty Invariants

For each modeled objective family:

- `q50 <= q90 <= q95 <= cvar95`

## Dominance Semantics

Strict Pareto filtering is applied over generated candidates, then downstream selection picks based on configured method.

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Dissertation Math Overview](dissertation-math-overview.md)
- [Graph Theory Notes](appendix-graph-theory-notes.md)
- [Strict Error Contract Reference](strict-errors-reference.md)

