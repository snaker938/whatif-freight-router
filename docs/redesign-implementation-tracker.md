# Redesign Implementation Tracker

This tracker maps the current redesign contract to concrete files, artifacts, tests, and evaluator lanes that exist in the repo today.

It is intentionally conservative:

- it records source presence, not blanket completion;
- it distinguishes documented/scaffolded surfaces from runtime-verified gates;
- it treats the current redesign as a hardening of the existing `legacy -> dccs -> dccs_refc -> voi` substrate, not a replacement of it;
- it keeps `legacy` as the baseline-only comparator while the default runtime now resolves to `dccs_refc`, with waypoint fallback preserved in `backend/app/main.py`.

## Evidence Legend

- `source-present`: the module or surface exists in the current tree.
- `test-present`: a focused test file exists in the current tree.
- `doc-present`: a doc/report surface already describes the contract.
- `scaffold-only`: the surface is explicit, but the repo still treats it as structure or contract rather than theorem-level proof.
- `open`: the tracker still needs a follow-on packet, additional implementation, or runtime verification.
- `not-verified-this-turn`: source inspection found the surface, but no Python or full suite was run in this turn.

## Tracker

| Major area | Concrete files / modules | Target artifacts / outputs | Target tests | Target evaluator lanes | Current evidence status | Next gap / note |
| --- | --- | --- | --- | --- | --- | --- |
| Core identity and runtime backbone | `backend/app/settings.py`, `backend/app/main.py`, `backend/app/models.py`, `backend/app/abstention.py`, `backend/app/certification_models.py`, `frontend/app/page.tsx`, `frontend/app/components/RouteCertificationPanel.tsx`, `frontend/app/components/DecisionStateSummary.tsx`, `README.md`, `docs/voi-pipeline-spec.md`, `docs/thesis-codebase-report.md` | Top-level `DecisionPackage`, typed abstention response, stage trace fields, default-mode routing, public mode labels | `backend/tests/test_route_pipeline_mode_default.py`, `backend/tests/test_route_terminal_semantics.py`, `backend/tests/test_route_response_refc_smoke.py`, `backend/tests/test_abstention_reason_classifier.py` | Broad cold proof, focused REFC proof, focused VOI proof, route-response smoke | `source-present`, `test-present`, `doc-present`, `not-verified-this-turn` | Default is now `dccs_refc`; explicit `legacy` override remains. No equal-status user-facing mode switch was found in the inspected backend/frontend surfaces. |
| DCCS / search-deficiency / candidate criticality | `backend/app/decision_critical.py`, `backend/app/candidate_bounds.py`, `backend/app/candidate_criticality.py`, `backend/app/main.py`, `docs/voi-pipeline-spec.md`, `docs/thesis-codebase-report.md`, `docs/claim_matrix.md`, `docs/theorem_map.md` | `dccs_candidates.jsonl`, `dccs_summary.json`, candidate envelopes, safe-elimination provenance, criticality ranking traces | `backend/tests/test_dccs.py` | DCCS diagnostic probe, broad cold proof, focused REFC proof | `source-present`, `test-present`, `doc-present`, `scaffold-only` | The code and docs expose envelope/provenance fields, but the repo still frames safe elimination and candidate criticality as audited implementation, not a fully proven theorem package. |
| Preference subsystem | `backend/app/preference_state.py`, `backend/app/preference_queries.py`, `backend/app/preference_update.py`, `backend/app/models.py`, `backend/app/main.py`, `frontend/app/components/RouteCertificationPanel.tsx`, `frontend/app/components/DecisionStateSummary.tsx`, `docs/voi-pipeline-spec.md` | `preference_state.json`, `preference_query_trace.json`, compatible-set summary, query history, shrinkage trace, irrelevance flags | `backend/tests/test_preference_surface.py` | Preference proof lane, focused VOI proof, preference-sensitive scenarios | `source-present`, `test-present`, `open` | The query/state machinery exists in source, but the repo still needs the remaining claim-gate evidence for all preference performance thresholds. |
| Multi-fidelity / support-aware evidence | `backend/app/fidelity_model.py`, `backend/app/audit_correction.py`, `backend/app/support_model.py`, `backend/app/world_policies.py`, `backend/app/uncertainty_model.py`, `backend/app/scenario.py`, `backend/app/main.py`, `backend/app/models.py`, `docs/voi-pipeline-spec.md`, `docs/quality-gates-and-benchmarks.md` | Proxy vs audit state, correction metadata, propensity metadata, support flags, world bundles, calibration summaries | `backend/tests/test_support_fidelity_world_models.py`, `backend/tests/test_uncertainty_model_unit.py`, `backend/tests/test_risk_model.py`, `backend/tests/test_scenario_resolution.py` | Proxy-audit calibration lane, public transfer lane, support-fragile scenarios | `source-present`, `test-present`, `doc-present`, `open` | The split is explicit in code, but the current repo evidence does not yet prove the full leakage-safe correction and held-out superiority gates. |
| REFC / selective certification / fragility | `backend/app/evidence_certification.py`, `backend/app/confidence_sequences.py`, `backend/app/pairwise_gap_model.py`, `backend/app/flip_radius.py`, `backend/app/decision_region.py`, `backend/app/certificate_witness.py`, `backend/app/certified_set.py`, `backend/app/run_store.py`, `backend/app/main.py`, `backend/app/models.py` | `certificate_summary.json`, `route_fragility_map.json`, `competitor_fragility_breakdown.json`, `value_of_refresh.json`, `sampled_world_manifest.json`, `evidence_snapshot_manifest.json`, `flip_radius_summary.json`, `world_support_summary.json`, `decision_region_summary.json`, `certificate_witness.json`, `certified_set_summary.json` | `backend/tests/test_refc.py`, `backend/tests/test_refc_artifact_contract.py`, `backend/tests/test_route_response_refc_smoke.py`, `backend/tests/test_route_terminal_semantics.py` | Focused REFC proof, optional-stopping coverage lane, perturbation / flip-radius lane | `source-present`, `test-present`, `scaffold-only`, `not-verified-this-turn` | The repo now carries the requested state objects and artifact names, but the contract is still documented as implementation evidence rather than a completed theorem suite. |
| VOI / tri-source controller | `backend/app/voi_controller.py`, `backend/app/replay_oracle.py`, `backend/app/fidelity_model.py`, `backend/app/evidence_certification.py`, `backend/app/main.py`, `backend/app/run_store.py`, `backend/app/models.py`, `docs/voi-pipeline-spec.md` | `voi_action_trace.json`, `voi_controller_state.jsonl`, `voi_action_scores.csv`, `voi_stop_certificate.json`, replay-oracle summaries | `backend/tests/test_voi_controller.py`, `backend/tests/test_oracle_quality_and_experiment_consumers.py`, `backend/tests/test_route_cache_live_trace_state.py` | Focused VOI proof, hot-rerun cold-source proof, hot-rerun proof | `source-present`, `test-present`, `scaffold-only` | The controller exists and the replay-oracle path is explicit, but the repo still needs runtime evidence for the requested certification-aligned gain gates. |
| API and frontend semantics | `backend/app/models.py`, `backend/app/main.py`, `frontend/app/page.tsx`, `frontend/app/components/RouteCertificationPanel.tsx`, `frontend/app/components/DecisionStateSummary.tsx`, `frontend/app/lib/types.ts`, `frontend/app/components/*` | `DecisionPackage` response shape, singleton/set/abstain distinction, route card, evidence panel, controller trace surfaces | `backend/tests/test_decision_package_summary_payload.py`, `backend/tests/test_route_terminal_semantics.py`, `backend/tests/test_route_response_refc_smoke.py`, `backend/tests/test_app_smoke_all.py` | Frontend proof dashboards, route-response smoke | `source-present`, `test-present`, `not-verified-this-turn` | The frontend already renders the redesigned response shape, but the UI remains a viewer for backend state rather than a full proof dashboard for every required gate. |
| Artifact provenance, cache semantics, and reuse honesty | `backend/app/run_store.py`, `backend/app/route_cache.py`, `backend/app/live_call_trace.py`, `backend/app/experiment_store.py`, `backend/app/oracle_quality_store.py`, `backend/app/settings.py`, `docs/reproducibility-capsule.md`, `docs/run-and-operations.md`, `docs/sample-manifest.md` | Versioned artifacts, stable cache keys, reuse metadata, signed manifests, run bundle indexes, cold-vs-hot provenance | `backend/tests/test_run_store_artifacts.py`, `backend/tests/test_route_cache_live_trace_state.py`, `backend/tests/test_route_cache.py`, `backend/tests/test_live_call_trace_rollup.py` | Hot-rerun cold-source proof, hot-rerun proof, broad cold proof | `source-present`, `test-present`, `doc-present`, `open` | The artifact contract is explicit and partially test-backed, but the repo still has many unreconciled claim surfaces that should remain labeled as source evidence only until rerun. |
| Evaluation lanes, cohorts, and benchmark contracts | `backend/tests/test_thesis_evaluation_runner.py`, `backend/tests/test_thesis_lane_script.py`, `backend/scripts/*`, `docs/quality-gates-and-benchmarks.md`, `docs/DOCS_INDEX.md`, `docs/voi-pipeline-spec.md` | Broad cold proof, focused REFC proof, focused VOI proof, DCCS diagnostic probe, hot-rerun lanes, proxy-audit calibration lane, perturbation lane, transfer lane | `backend/tests/test_thesis_evaluation_runner.py`, `backend/tests/test_thesis_lane_script.py`, `backend/tests/test_scripts_smoke_all.py`, `backend/tests/test_scripts_test_matrix.py`, `backend/tests/test_scripts_quality_extended.py` | All named evaluation lanes and cohorts | `source-present`, `test-present`, `doc-present`, `open` | The lane inventory is present in source, but the tracker should not label the hard gates green without a fresh evaluation run. |
| Documentation and claim discipline | `docs/thesis-codebase-report.md`, `docs/voi-pipeline-spec.md`, `docs/claim_matrix.md`, `docs/theorem_map.md`, `docs/quality-gates-and-benchmarks.md`, `README.md`, `docs/DOCS_INDEX.md` | Claim matrix, theorem map, implementation tracker, reproducibility notes, limitations, benchmark summaries | `backend/tests/test_app_test_matrix.py`, `backend/tests/test_dev_preflight.py`, `backend/tests/test_scripts_test_matrix.py` | Documentation-facing proof surfaces | `source-present`, `doc-present`, `scaffold-only`, `open` | The docs now distinguish scaffold-only claims from empirical ones, but the report set still needs a later pass if any wording drifts away from current code reality. |

## File-level notes

### Backend runtime and response contract

- `backend/app/settings.py` currently defaults `route_pipeline_default_mode` to `dccs_refc`.
- `backend/app/main.py` still honors explicit `legacy` request overrides and keeps waypoint requests on the legacy path.
- `backend/app/abstention.py` centralizes typed-abstention construction through `build_abstention_record(...)`.
- `backend/app/models.py` exposes the top-level `DecisionPackage` / `RouteResponse` shape and the explicit terminal-type fields.

### Redesign module inventory

The following files now exist as explicit redesign-facing modules:

- `backend/app/candidate_bounds.py`
- `backend/app/candidate_criticality.py`
- `backend/app/preference_state.py`
- `backend/app/preference_queries.py`
- `backend/app/preference_update.py`
- `backend/app/fidelity_model.py`
- `backend/app/audit_correction.py`
- `backend/app/world_policies.py`
- `backend/app/support_model.py`
- `backend/app/confidence_sequences.py`
- `backend/app/pairwise_gap_model.py`
- `backend/app/flip_radius.py`
- `backend/app/certification_models.py`
- `backend/app/decision_region.py`
- `backend/app/certificate_witness.py`
- `backend/app/certified_set.py`
- `backend/app/replay_oracle.py`

### Current doc/test evidence

- `docs/voi-pipeline-spec.md` still describes the `V0 -> legacy`, `A -> dccs`, `B -> dccs_refc`, `C -> voi` ablation ladder.
- `docs/claim_matrix.md` and `docs/theorem_map.md` both label the new response and artifact surfaces as `scaffold-only` where the repo is still documenting structure rather than theorem-level proof.
- The backend test tree already contains focused tests for default-mode resolution, abstention semantics, REFC artifacts, preference surfaces, support/fidelity models, route cache/live trace state, and VOI controller behavior.

## Use of This Tracker

Treat this file as a working map for the remaining redesign packets:

- if a row says `scaffold-only` or `open`, it still needs additional implementation, evaluation, or wording cleanup;
- if a row says `source-present` but `not-verified-this-turn`, the code exists but this packet did not run the runtime check;
- if a row names a concrete test file, that is the minimum regression surface to consult before expanding the packet;
- if a row names a lane, that lane is the smallest evaluator slice that should be used when a verification packet is opened.
