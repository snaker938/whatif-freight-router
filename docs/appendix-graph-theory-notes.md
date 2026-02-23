# Graph Theory Notes

Last Updated: 2026-02-19  
Applies To: candidate generation and Pareto frontier selection

## Current Practical Model

Routing uses generated candidate path families and strict frontier filtering, with deterministic candidate budgets.

## Why Frontier Quality Depends on Candidate Breadth

Pareto quality is bounded by candidate-space coverage. Better path enumeration increases meaningful frontier diversity.

## Diagnostics to Watch

- `candidate_count_raw`
- `candidate_count_deduped`
- `epsilon_feasible_count`
- `pareto_count`
- `strict_frontier_applied`

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Dissertation Math Overview](dissertation-math-overview.md)
- [Expanded Math Appendix](math-appendix.md)
- [Quality Gates and Benchmarks](quality-gates-and-benchmarks.md)

