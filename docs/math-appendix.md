# Expanded Math Appendix

Last Updated: 2026-04-03  
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

## Support and Audit Semantics

The maintained certification layer distinguishes between empirical world sampling and support-quality state:

- `ProbabilisticWorldBundle` tracks active families, world-kind weights, family-state weights, proxy fraction, stress fraction, refreshed fraction, targeting fraction, and effective world count
- `AuditWorldBundle` and `ProxyAuditRecord` summarize proxy-heavy, fallback-heavy, or low-coverage evidence families and apply an explicit audit correction penalty
- `WorldSupportState` aggregates ambiguity support, source entropy/count, provenance coverage, bundle support mass, and audit correction into a support-strength summary and recommended fidelity mode

Support strength is not itself a certificate. It is a separate signal that can justify downgrade, abstention, or additional audit work even when empirical winner frequencies look strong.

## Certificate Invariants

For the current maintained certification layer:

- `0 <= C(r) <= 1`
- empirical certification still requires `C(r*) >= certificate_threshold`
- winner-confidence state should satisfy `0 <= lower_bound <= point_estimate <= upper_bound <= 1`
- pairwise-gap state should satisfy `min_gap <= mean_gap <= max_gap`
- flip radius should remain nonnegative
- certified-set state should only mark a route certified when its lower bound clears the threshold
- uncertified results remain explicitly uncertified
- fragility maps and value-of-refresh summaries are conditioned on the same sampled-world frame used by the certificate calculation

The maintained code now names this layer explicitly through `WinnerConfidenceState`, `PairwiseGapState`, `FlipRadiusState`, `DecisionRegionState`, `CertifiedSetState`, `CertificationState`, `CertificateWitness`, and `AbstentionRecord`.

These are maintained support/certification state objects. They should not be conflated with the smaller public summary contracts carried by `RouteCertificationSummary` and `VoiStopSummary`.

## VOI Budget Invariants

- search and evidence budgets are counted in explicit units
- stop certificates must record used budgets and stop reason
- the controller should not imply certification when it stopped without reaching threshold

The current stop-summary contract records `final_route_id`, `iteration_count`, `search_budget_used`, `evidence_budget_used`, `stop_reason`, `best_rejected_action`, and `best_rejected_q` so the stopping rationale stays auditable.

That stop-summary surface is intentionally smaller than the maintained certification-state layer. It records controller-stop facts without claiming to carry every witness, abstention, or challenger-state object.

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Dissertation Math Overview](dissertation-math-overview.md)
- [Graph Theory Notes](appendix-graph-theory-notes.md)
- [Strict Error Contract Reference](strict-errors-reference.md)
