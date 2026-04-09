# Graph Theory Notes

Last Updated: 2026-04-09
Applies To: candidate generation and Pareto frontier selection

## Current Practical Model

Routing uses generated candidate path families and strict frontier filtering, with deterministic candidate budgets.

The current UK routing graph coverage report passes on a graph with `16,782,614` nodes, `17,271,476` edges, `graph_size_mb=4123.27`, and `worst_fixture_nearest_node_m=2545.053`, so the frontier machinery is being exercised against a very large but fully validated graph asset.

## Validation Snapshot

The current graph validation stack is centered on `backend/scripts/validate_graph_coverage.py` and the persisted report `backend/out/model_assets/routing_graph_coverage_report.json`.

- Bounding box: `lat 49.75..61.1`, `lon -8.75..2.25`
- Coverage status: `coverage_passed=true`
- Fixture proximity: the worst fixture still lands within 2.55 km of a graph node
- Validation purpose: ensure route generation and strict frontier selection are operating on a graph that can support the thesis evaluation fixtures without silent coverage drift

## Why Frontier Quality Depends on Candidate Breadth

Pareto quality is bounded by candidate-space coverage. Better path enumeration increases meaningful frontier diversity.

## Diagnostics to Watch

- `candidate_count_raw`
- `candidate_count_deduped`
- `epsilon_feasible_count`
- `pareto_count`
- `strict_frontier_applied`
- `coverage_passed`
- `worst_fixture_nearest_node_m`
- `graph_size_mb`
- `backend/out/model_assets/routing_graph_coverage_report.json`

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Dissertation Math Overview](dissertation-math-overview.md)
- [Expanded Math Appendix](math-appendix.md)
- [Quality Gates and Benchmarks](quality-gates-and-benchmarks.md)

