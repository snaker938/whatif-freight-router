# Graph Theory Notes

Last Updated: 2026-03-31  
Applies To: graph-led candidate generation, DCCS selection, refinement, and strict frontier construction

## Current Practical Model

The current thesis-facing framing is:

`K_raw -> R -> F`

Where:

- `K_raw` is the raw graph candidate set
- `R` is the refined route set
- `F` is the strict frontier

## Why Frontier Quality Depends On Candidate Breadth

Frontier quality is bounded by coverage in `K_raw` and by which candidates survive refinement into `R`.

That is why the project tracks more than winner identity:

- DCCS frontier recall at budget
- decision-critical yield
- nontrivial frontier rate
- frontier diversity and frontier count
- collapse-prone representative rows

## Diagnostics To Watch

Current useful diagnostics include:

- `dccs_dc_yield`
- `dccs_frontier_recall_at_budget`
- `candidate_count_raw`
- `refined_count`
- `frontier_count`
- `nontrivial_frontier_rate`
- `mean_frontier_diversity_index`
- collapse cases where rescue fired but the observed frontier still collapsed to a single member

Current thesis outputs track these same ideas in evaluation summaries via `mean_dccs_frontier_recall_at_budget`, `mean_dccs_dc_yield`, `nontrivial_frontier_rate`, `mean_frontier_count`, and `mean_frontier_diversity_index`.

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Dissertation Math Overview](dissertation-math-overview.md)
- [Expanded Math Appendix](math-appendix.md)
- [Quality Gates and Benchmarks](quality-gates-and-benchmarks.md)
