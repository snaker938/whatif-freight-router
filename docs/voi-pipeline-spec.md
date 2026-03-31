# VOI Thesis Pipeline Spec

Last Updated: 2026-03-31  
Applies To: `backend/app/decision_critical.py`, `backend/app/evidence_certification.py`, `backend/app/voi_controller.py`, `backend/app/models.py`, and the evaluation scripts

This document defines the current thesis-facing routing pipeline for the DCCS, REFC, and VOI-AD2R workstream.

## 1. Pipeline Modes

The pipeline supports four modes:

- `legacy`: existing staged routing flow with no DCCS, no REFC, and no VOI loop
- `dccs`: DCCS triage before refinement, then normal route build and frontier selection
- `dccs_refc`: DCCS plus strict-frontier certification
- `voi`: full thesis pipeline, including DCCS, REFC, and the VOI-AD2R controller

The effective mode is carried on route, Pareto, and batch requests through `pipeline_mode`.

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

- `run_id`
- `pipeline_mode`
- `manifest_endpoint`
- `artifacts_endpoint`
- `provenance_endpoint`
- `selected_certificate`
- `voi_stop_summary`

## 5. Stable Artifact Contract

The stable public run-store allowlist includes:

- dccs_candidates.jsonl
- dccs_summary.json
- refined_routes.jsonl
- strict_frontier.jsonl
- winner_summary.json
- certificate_summary.json
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

## 7. Budgets and Reproducibility

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

## 8. Strict and Uncertified Semantics

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
