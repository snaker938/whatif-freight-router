# Expanded Math Appendix

Last Updated: 2026-03-31  
Applies To: backend objective decomposition, dominance semantics, uncertainty summaries, and certificate-related invariants

## Objective Decomposition

At route and segment levels, the monetary decomposition is:

`total_monetary = time_cost + fuel_cost + toll_cost + carbon_cost`

This decomposition should remain consistent between route-level metrics and `segment_breakdown` rows.

Current route responses can also carry `vehicle_profile_id`, `vehicle_profile_version`, `scenario_summary`, `weather_summary`, `terrain_summary`, and `incident_events`, so the decomposition remains tied to the active scenario context.

## Dominance Semantics

- strict Pareto filtering is applied over refined candidates
- the strict frontier is the only frontier used for certification and VOI decisions
- baseline comparisons and presentation summaries may exist outside the strict frontier, but they should not redefine it

## Uncertainty and Robustness

When uncertainty is active, route outputs may expose:

- utility means
- quantiles
- robust scores
- uncertainty metadata and sample summaries

These are descriptive summaries, not replacements for certificate semantics.

## Certificate Invariants

For a fixed selector and sampled-world set:

- `0 <= C(r) <= 1`
- certification requires `C(r*) >= certificate_threshold`
- uncertified results remain explicitly uncertified
- fragility maps and value-of-refresh summaries are conditioned on the same sampled-world frame used by the certificate calculation

The current certificate summary contract exposes the same frame explicitly through `RouteCertificationSummary.active_families`, `top_fragility_families`, `top_competitor_route_id`, `top_value_of_refresh_family`, and `ambiguity_context`.

## VOI Budget Invariants

- search and evidence budgets are counted in explicit units
- stop certificates must record used budgets and stop reason
- the controller should not imply certification when it stopped without reaching threshold

The current stop-summary contract records `final_route_id`, `iteration_count`, `search_budget_used`, `evidence_budget_used`, `stop_reason`, `best_rejected_action`, and `best_rejected_q` so the stopping rationale stays auditable.

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Dissertation Math Overview](dissertation-math-overview.md)
- [Graph Theory Notes](appendix-graph-theory-notes.md)
- [Strict Error Contract Reference](strict-errors-reference.md)
