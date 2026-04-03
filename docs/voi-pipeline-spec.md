# VOI Thesis Pipeline Spec

Last Updated: 2026-04-03
Applies To: `backend/app/main.py`, `backend/app/models.py`, `backend/app/run_store.py`, `backend/app/decision_critical.py`, `backend/app/evidence_certification.py`, `backend/app/voi_controller.py`, and the evaluation scripts

This document defines the current thesis-facing routing pipeline for the DCCS, REFC, and VOI-AD2R workstream.

## 1. Pipeline Modes

The pipeline supports five public modes:

- `legacy`: existing staged routing flow with no DCCS, no REFC, and no VOI loop
- `dccs`: DCCS triage before refinement, then normal route build and frontier selection
- `dccs_refc`: DCCS plus strict-frontier certification
- `voi`: full thesis pipeline, including DCCS, REFC, and the VOI-AD2R controller
- `tri_source`: current public default route mode; single-leg OD requests dispatch through the VOI-backed thesis runtime while preserving `pipeline_mode="tri_source"` on the public response and persisted route artifacts

The effective public mode is carried on route, Pareto, and batch requests through `pipeline_mode`.

For the current route seam:

- single-leg requests in `tri_source` execute on the internal `voi` runtime path
- requests with waypoints still fall back to `legacy`, and that fallback is surfaced in the route response, manifest warnings, and route artifacts

## 2. Candidate Lifecycle

The pipeline uses the following named sets:

- `K_raw`: raw graph candidates after graph-led exploration and deduplication/prefiltering
- `R`: refined routes
- `F`: strict Pareto frontier extracted from `R`
- `r*`: the current winner under the fixed selector

The high-level lifecycle is:

`preflight -> K_raw -> DCCS -> R -> F -> REFC -> VOI-AD2R -> stop certificate`

The graph-theoretic interpretation used in the current docs is `K_raw -> R -> F`.

## 3. Module Interfaces

### DCCS

`backend/app/decision_critical.py` provides the DCCS configuration and candidate-ledger logic.

Auditable DCCS fields include:

- candidate id
- graph path and proxy objectives
- mechanism descriptor
- overlap, stretch, objective gap, and mechanism gap
- predicted refine cost
- flip probability
- selected/skipped decision
- observed refine outcome when refinement completes

### REFC

`backend/app/evidence_certification.py` provides:

- evidence-family activation
- deterministic world sampling
- certificate computation
- fragility attribution
- ambiguity-aware targeted stress packs

REFC operates on the strict frontier `F` and populates both thesis-facing outputs and controller-facing summaries.

### VOI-AD2R

`backend/app/voi_controller.py` provides:

- action menu construction
- action scoring
- controller state evolution
- stop certificate construction

Core public action kinds remain:

- refine top-1 DCCS candidate
- refine top-k DCCS candidates
- refresh top-1 value-of-refresh evidence family
- increase stochastic samples
- stop

The controller can also bridge sparse empirical refresh evidence by using controller-side ranking metadata and `raw_refresh_gain_fallback` when empirical value-of-refresh is zero.

## 4. Current Response Contract

### `RouteCertificationSummary`

Current fields include:

- `route_id`
- `certificate`
- `certified`
- `threshold`
- `active_families`
- `top_fragility_families`
- `top_competitor_route_id`
- `top_value_of_refresh_family`
- `ambiguity_context`

### `VoiStopSummary`

Current fields include:

- `final_route_id`
- `certificate`
- `certified`
- `iteration_count`
- `search_budget_used`
- `evidence_budget_used`
- `stop_reason`
- `best_rejected_action`
- `best_rejected_q`
- `search_completeness_score`
- `search_completeness_gap`
- `credible_search_uncertainty`

Current stop reasons include controller outcomes such as:

- `no_action_worth_it`
- `search_incomplete_no_action_worth_it`
- `iteration_cap_reached`
- `error_missing_action_hooks`

### `RouteResponse`

Current thesis-relevant route response fields include:

- `selected`
- `candidates`
- `run_id`
- `pipeline_mode`
- `manifest_endpoint`
- `artifacts_endpoint`
- `provenance_endpoint`
- `decision_package` (optional in the public model and populated by the landed `/route` seam; compatibility handling should remain additive)
- `selected_certificate`
- `voi_stop_summary`

### `DecisionPackage`

Current `decision_package` payloads summarize the public thesis decision state without removing the legacy route fields. The top-level summaries include:

- `preference_summary`
- `support_summary`
- `certified_set_summary`
- `abstention_summary`
- `witness_summary`
- `controller_summary`
- `theorem_hook_summary`
- `lane_manifest`

The current runtime assembles this package from the existing route, certification, VOI, and artifact facts already produced during the request. It is a public summary seam, not a second independent decision engine.

`preference_summary` is the current landed preference bridge and remains summary-only selector/runtime metadata rather than a public preference query/update surface.

## 5. Stable Artifact Contract

The stable public run-store allowlist includes:

- dccs_candidates.jsonl
- dccs_summary.json
- refined_routes.jsonl
- strict_frontier.jsonl
- winner_summary.json
- certificate_summary.json
- decision_package.json
- preference_summary.json
- support_summary.json
- support_trace.jsonl
- support_provenance.json
- certified_set.json
- certified_set_routes.jsonl
- abstention_summary.json
- witness_summary.json
- witness_routes.jsonl
- controller_summary.json
- controller_trace.jsonl
- theorem_hook_map.json
- lane_manifest.json
- route_fragility_map.json
- competitor_fragility_breakdown.json
- value_of_refresh.json
- sampled_world_manifest.json
- evidence_snapshot_manifest.json
- voi_action_trace.json
- voi_controller_state.jsonl
- voi_action_scores.csv
- voi_stop_certificate.json
- final_route_trace.json
- thesis_results.csv
- thesis_results.json
- thesis_summary.csv
- thesis_summary.json
- thesis_metrics.json
- thesis_plots.json
- methods_appendix.md
- thesis_report.md
- evaluation_manifest.json

Focused and broad evaluation runs also emit OD corpus and baseline helper artifacts such as od_corpus.* and ors_snapshot.json.

The public allowlist is intentionally narrow: code should only rely on the named artifact family outputs above, plus signed manifests and provenance, rather than on ad hoc intermediate files.

For the landed route seam, `decision_package.json`, `preference_summary.json`, `support_summary.json`, `support_trace.jsonl`, `support_provenance.json`, `certified_set.json`, and `certified_set_routes.jsonl` are normal public outputs. The abstention, witness, controller, theorem-hook, and lane-manifest artifacts are also public and remain additive contract members when the corresponding summary is populated.

The run-store schema registry now has explicit schema-version coverage for both the VOI trace artifacts and the decision-package family, including:

- `decision_package.json`
- `preference_summary.json`
- `support_summary.json`
- `support_trace.jsonl`
- `support_provenance.json`
- `certified_set.json`
- `certified_set_routes.jsonl`
- `abstention_summary.json`
- `witness_summary.json`
- `witness_routes.jsonl`
- `controller_summary.json`
- `controller_trace.jsonl`
- `theorem_hook_map.json`
- `lane_manifest.json`

- `voi_action_trace.json`
- `voi_stop_certificate.json`
- `final_route_trace.json`

These filenames are unchanged. The schema-version entries exist so additive controller-trace fields can evolve without inventing a new artifact family.

Composed suite runs additionally produce:

- thesis_summary_by_cohort.csv
- thesis_summary_by_cohort.json
- suite_sources.json
- prior_coverage_summary.json
- cohort_composition.json

Hot-rerun benchmarking is separate and adds:

- hot_rerun_vs_cold_comparison.json
- hot_rerun_vs_cold_comparison.csv
- hot_rerun_gate.json
- hot_rerun_report.md

These hot-rerun outputs are production/reuse evidence, not additional thesis-lane route-quality proof.

## 6. Value-of-Refresh Semantics

The thesis-facing value_of_refresh.json remains the conservative empirical summary.

Current controller-facing refresh metadata can also include:

- `controller_ranking_basis`
- `controller_ranking`
- `top_refresh_family_controller`
- `top_refresh_gain_controller`
- `controller_refresh_fallback_activated`
- `controller_empirical_vs_raw_refresh_disagreement`

evidence_snapshot_manifest.json and sampled-world outputs carry the upstream context required to interpret these values.

## 7. VOI Trace Value Contract

The VOI trace contract is additive. The existing top-level action fields remain valid, and richer replay/value fields may appear on each action row without removing the old ones.

The additive fields are:

- `trace_metadata`
- `action_menu_value_estimates`
- `chosen_action_value_record`

`trace_metadata` is run-time control context for the row. It is intended to capture the controller frame in which the menu was ranked, such as:

- trace source and trace version
- iteration index
- selected and winning route ids
- current certificate and certificate margin
- remaining and used search/evidence budget
- controller uncertainty flags

`action_menu_value_estimates` is the predicted menu snapshot for that iteration. Each entry is an additive value record derived from one action candidate and can include:

- action id, kind, target, and reason
- predicted certificate, margin, and frontier deltas
- weighted per-term value contributions
- total predicted value and total cost
- `base_q_score`, `ranked_q_score`, and any additive score-adjustment accounting

`chosen_action_value_record` is the replay-oriented record for the selected action. It uses the same predicted estimate shape as the menu entries and may also carry realization fields when the action outcome becomes known.

The intended behavior is:

- predicted-only: when the trace row is recorded before action execution, `chosen_action_value_record.realization` is absent or null and the record captures only the controller's forecast
- predicted-plus-realized: when the same action row is updated after execution, `chosen_action_value_record.realization` carries realized certificate, runner-up-gap, frontier, route-change, and evidence-uncertainty outcomes while the original predicted estimate remains intact

`voi_action_trace.json` is the primary action-row carrier for these fields.

`voi_stop_certificate.json` and `final_route_trace.json` inline the same action trace, so the same additive row semantics apply there as well.

## 8. Budgets and Reproducibility

Current request-side knobs are:

- `pipeline_mode`
- `pipeline_seed`
- `search_budget`
- `evidence_budget`
- `cert_world_count`
- `certificate_threshold`
- `tau_stop`

The intended contract is:

- DCCS uses explicit search-budget units
- REFC uses explicit world-count and evidence-family state
- VOI uses explicit search and evidence budgets with deterministic action scoring

For a fixed request, seed chain, evidence state, and model-asset state, the pipeline should be replayable through signed manifests, provenance logs, and per-run artifacts.

## 9. Strict and Uncertified Semantics

- REFC and VOI operate on the strict frontier only
- `C(r)` is the fraction of sampled worlds in which route `r` wins under the fixed selector
- `r*` is certified only if `C(r*) >= certificate_threshold`
- uncertified outcomes must remain explicit
- budget exhaustion must remain explicit
- the controller should stop honestly when no admissible action clears the threshold

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Quality Gates and Benchmarks](quality-gates-and-benchmarks.md)
- [Reproducibility Capsule](reproducibility-capsule.md)
- [Graph Theory Notes](appendix-graph-theory-notes.md)
