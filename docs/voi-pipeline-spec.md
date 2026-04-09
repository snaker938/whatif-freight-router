# VOI Thesis Pipeline Spec

This document defines the thesis-facing routing pipeline for the DCCS, REFC, and VOI-AD2R workstream. It is intentionally aligned with the current backend contracts in `backend/app/decision_critical.py`, `backend/app/evidence_certification.py`, `backend/app/voi_controller.py`, and the request/response models in `backend/app/models.py`.

## 1. Pipeline Modes

The pipeline supports four modes:

* `legacy`: existing staged routing flow with no DCCS, no REFC, and no VOI loop.
* `dccs`: apply DCCS triage before refinement, then continue with the normal route build and frontier selection.
* `dccs_refc`: DCCS plus strict-frontier certification.
* `voi`: full thesis pipeline, including DCCS, REFC, and the VOI-AD2R controller.

The effective mode is carried on route requests through `RouteRequest.pipeline_mode`, `ParetoRequest.pipeline_mode`, and the batch request variants. The runtime resolves an explicit request override against the deployment default, and the thesis modes are selected explicitly rather than being the unconditional user-facing default.
In thesis-suite runs, those modes are mapped as `V0 -> legacy`, `A -> dccs`, `B -> dccs_refc`, and `C -> voi`.

## 2. Candidate Lifecycle

The pipeline uses four named sets:

* `K_raw`: raw graph candidates after graph-led exploration and deduplication/prefiltering.
* `R`: OSRM-refined routes.
* `F`: strict Pareto frontier extracted from `R`.
* `r*`: the current winner under the fixed selector.

The lifecycle is:

`preflight and OD feasibility -> K_raw -> DCCS -> R -> F -> REFC -> VOI-AD2R -> stop certificate`

`K_raw`, `R`, `F`, and `r*` must each be serialisable as run artifacts so they can be replayed and cited.

## 3. Module Interfaces

### DCCS

`backend/app/decision_critical.py` provides:

* `DCCSConfig`
* `DCCSCandidateRecord`
* `DCCSResult`
* `stable_candidate_id(...)`
* `build_candidate_record(...)`
* `build_candidate_ledger(...)`
* `select_candidates(...)`
* `record_refine_outcome(...)`

Each candidate record carries the auditable triage fields needed for thesis analysis: graph path, proxy objectives, mechanism descriptor, confidence, overlap, stretch, objective gap, mechanism gap, predicted refine cost, flip probability, score terms, selected/skipped decision, and observed refine cost once refinement completes.
The current DCCS record contract also includes candidate-envelope bounds, envelope provenance or source, support mass or support status, known dominance metadata, and explicit safe-elimination provenance fields such as `safe_eliminated`, `necessary_dominated`, `dominated_by_route_id`, `dominance_margin`, and `safe_elimination_reason`.
Those fields support an auditable candidate-level selection workflow; they are not being presented here as theorem-level guarantees.

### REFC

`backend/app/evidence_certification.py` provides:

* `EVIDENCE_FAMILIES`
* `EVIDENCE_STATES`
* `EvidenceProvenance`
* `WorldSample`
* `CertificateConfig`
* `CertificateResult`
* `FragilityResult`
* `active_evidence_families(...)`
* `sample_world_manifest(...)`
* `dependency_tensor(...)`
* `compute_certificate(...)`
* `compute_fragility_maps(...)`

REFC operates only on the strict frontier `F` and computes certificate values, fragility attribution, and value-of-refresh rankings over deterministic sampled worlds.

### VOI-AD2R

`backend/app/voi_controller.py` provides:

* `VOIConfig`
* `VOIAction`
* `VOIControllerState`
* `VOIActionHooks`
* `VOIStopCertificate`
* `build_action_menu(...)`
* `score_action(...)`
* `run_controller(...)`

The controller uses a fixed auditable action set: refine top-1 DCCS candidate, refine top-k DCCS candidates, refresh the top-1 value-of-refresh evidence family, increase stochastic samples for the near-tie set, and stop.

## 4. Artifact Contract

Artifacts are stored per run under the existing run store conventions. The current evaluator treats `strict_frontier.jsonl` as the strict frontier artifact and validates `voi_controller_state.jsonl` alongside the other VOI traces.

The current checked thesis example bundle is:

`backend/out/thesis_campaigns/dominance_cluster5_cardiff_bath_corr12p5_r2/tranche_001/artifacts/dominance_cluster5_cardiff_bath_corr12p5_r2_t001`

Within that bundle, the route-level artifact set includes:

* metadata.json
* dccs_candidates.jsonl
* dccs_summary.json
* strict_frontier.jsonl
* final_route_trace.json
* certificate_summary.json
* route_fragility_map.json
* competitor_fragility_breakdown.json
* value_of_refresh.json
* sampled_world_manifest.json
* voi_controller_state.jsonl
* voi_action_trace.json
* voi_stop_certificate.json

When the controller runs in refinement mode, the evaluator also accepts before/after snapshot artifacts:

* initial_certificate_summary.json
* initial_route_fragility_map.json
* initial_competitor_fragility_breakdown.json
* initial_value_of_refresh.json
* initial_sampled_world_manifest.json

The thesis-suite evaluation outputs in the same checked bundle include:

* thesis_results.csv
* thesis_results.json
* thesis_summary.csv
* thesis_summary.json
* thesis_summary_by_cohort.csv
* thesis_summary_by_cohort.json
* od_corpus.csv
* od_corpus.json
* od_corpus_summary.json
* cohort_composition.json
* thesis_plots.json
* methods_appendix.md
* thesis_report.md
* evaluation_manifest.json

Recommended schemas:

* dccs_candidates.jsonl: one record per candidate, with candidate id, graph path, objective proxy, gaps, overlap, stretch, predicted refine cost, flip probability, decision, and observed refine-cost fields when available.
* dccs_summary.json: mode, search budget, transition reason, candidate counts, selected/skipped counts, DC-yield, challenger hit rate, and frontier gain per refinement.
* certificate_summary.json: winner route id, certificate map, threshold, certified flag, selected route id, selector config, and world manifest summary.
* sampled_world_manifest.json: seed, world count, active families, state catalog, and the exact sampled worlds.
* voi_action_trace.json: per-iteration feasible actions, chosen action, scores, budgets remaining, and stop diagnostics.
* voi_controller_state.jsonl: per-iteration controller state snapshots, including the selected action, scores, and budget bookkeeping.
* voi_stop_certificate.json: final winner id, objective vector, frontier size, certificate value, certified flag, budgets used/remaining, stop reason, best rejected action, and ambiguity summary.
* final_route_trace.json: end-to-end route trace, including stage timings, DCCS decisions, REFC certificate summary, VOI action trace, and artifact pointers.

The route response models expose `run_id`, `pipeline_mode`, artifact endpoints, `selected_certificate`, and `voi_stop_summary` so a request can be tied back to the run store.

## 5. Config Knobs and Budgets

The request-side knobs already present in `backend/app/models.py` are:

* `pipeline_mode`
* `pipeline_seed`
* `search_budget`
* `evidence_budget`
* `cert_world_count`
* `certificate_threshold`
* `tau_stop`

Baseline default behavior should remain conservative:

* `search_budget`: small bounded integer budget for OSRM refinements.
* `evidence_budget`: small bounded integer budget for evidence refresh or resampling actions.
* `cert_world_count`: bounded sample size for world generation.
* `certificate_threshold`: the certification cutoff for `C(r*)`.
* `tau_stop`: minimum expected value required for VOI actions.

DCCS and VOI use these budgets in budget units, not in abstract percentages, so every action is auditable and replayable.

## 6. Determinism And Reproducibility

The pipeline must be deterministic for a fixed request, seed, evidence snapshot, and candidate set.

The seed contract is:

* `pipeline_seed` is the run-level seed.
* DCCS, REFC, and VOI derive their internal tie-break and sampling streams from that seed.
* REFC world sampling is deterministic for the same active evidence families and seed.
* VOI action selection is deterministic because action scoring is deterministic.
* Any stochastic uncertainty model should receive the same seed chain that is recorded in the run manifest.

The run manifest and signed artifact chain should include:

* the request payload or request hash
* the resolved pipeline mode
* the run seed
* the evidence snapshot identifiers
* the budget configuration
* the selector configuration
* the backend build/version metadata

If live evidence can drift, evaluation runs should use snapshot mode so that certificates and stop decisions can be replayed exactly.

## 7. Strict And Uncertified Semantics

The pipeline uses strict-frontier semantics for REFC and VOI:

* `F` must be strict, meaning no presentation-layer backfill should contaminate the frontier used for certification.
* `C(r)` is the fraction of sampled worlds in which route `r` wins under the fixed selector.
* `r*` is certified only when `C(r*) >= certificate_threshold`.
* If the threshold is not met, the result must be explicitly marked `uncertified`.
* If budgets are exhausted before certification, the result remains `uncertified` and the stop certificate must say so.
* If no action clears the VOI stop threshold, the controller must stop explicitly with `no_action_worth_it`.

The implementation should never imply certainty when the pipeline has only produced an uncertified frontier or an incomplete evidence state.

## 8. Minimal Thesis Outputs

Each route run should be able to produce a compact thesis bundle:

* candidate ledger for DCCS
* strict frontier and winner summary
* certificate and fragility summaries
* VOI action trace and stop certificate
* final route trace linking all of the above

That bundle is the citable unit for thesis figures, ablations, and replay checks.

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Thesis-Grade Codebase Report](thesis-codebase-report.md)
- [Model Assets and Data Sources](model-assets-and-data-sources.md)
