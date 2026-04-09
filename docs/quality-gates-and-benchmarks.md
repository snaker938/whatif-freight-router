# Quality Gates and Benchmarks

Last Updated: 2026-04-09
Applies To: backend tests, thesis evaluation lanes, composed suite reporting, and hot-rerun reuse benchmarks

This page defines the current gate sequence used locally for backend quality, thesis proof, and production-style hot-rerun validation.

## Core Gate Ladder

Use the smallest valid rung first:

1. docs and authored-doc consistency checks
2. targeted backend tests
3. lane-local evaluation checks
4. full thesis evaluation suite
5. dedicated hot-rerun benchmark
6. final composed suite reporting

## Authoritative Commands

From repo root:

```powershell
python scripts/check_docs.py
.\scripts\run_backend_tests_safe.ps1 -MaxCores 1 -PriorityClass Idle
uv run --project backend python backend/scripts/run_thesis_lane.py --help
uv run --project backend python backend/scripts/run_thesis_evaluation.py
uv run --project backend python backend/scripts/run_thesis_campaign.py --help
uv run --project backend python backend/scripts/run_thesis_hard_story_suite.py --help
uv run --project backend python backend/scripts/expand_thesis_broad_corpus.py --help
uv run --project backend python backend/scripts/run_hot_rerun_benchmark.py
uv run --project backend python backend/scripts/compare_thesis_runs.py --help
uv run --project backend python backend/scripts/compose_thesis_sharded_report.py --help
uv run --project backend python backend/scripts/compose_thesis_suite_report.py
```

`backend/scripts/run_thesis_evaluation.py` also accepts `--toll-cost-per-km <rate>` when a thesis lane needs an
explicit configured toll rate instead of dropping toll-sensitive rows on unresolved tariff matches.

## Test and Validation Lanes

### Authored Docs Checks

- `python scripts/check_docs.py`
- validates links, local file paths, docs index coverage, and backend endpoint parity

### Backend Regression and Strict Runtime Tests

- use `.\scripts\run_backend_tests_safe.ps1` on constrained machines
- use `uv run --project backend pytest backend/tests` only when a full suite is appropriate
- strict failures, baseline identity, and artifact semantics should be treated as hard regressions
- the public `tri_source` route seam is part of that hard-regression surface; the smallest focused backend check starts with `backend/tests/test_api_streaming.py` for public `tri_source` mode preservation, internal `voi` execution, `decision_package` response population, and emitted decision-package artifacts, with `backend/tests/test_route_baseline_api.py` remaining the focused waypoint-fallback check for `tri_source -> legacy`
- the frontend certificate/artifact inspection seam is part of the same contract surface; the smallest non-Python check is `pnpm --dir frontend exec tsc --noEmit`, followed by a focused browser smoke that the certification panel and Run Inspector still group guaranteed vs conditional artifacts from the existing run-store listing
- the thesis runner now writes `thesis_cohort_scaffolding_v2` and `thesis_metric_family_scaffolding_v1` into the run payloads it owns; those fields are instrumentation for later threshold tuning, not proof that the corresponding gates were cleared in this pass
- `thesis_metric_family_scaffolding_v1` now distinguishes fully wired families from evaluator-only metadata states: `preference` is `metadata_wired` at `cohort_metadata_only`, and `multi_fidelity_support` is `partially_wired` at `shared_summary_metrics`; neither label is a claim that the corresponding proof thresholds are green
- the thin smoke path for thesis artifacts starts with `backend/scripts/run_thesis_evaluation.py`, which emits `evaluation_manifest.json`, `thesis_metrics.json`, `thesis_plots.json`, `thesis_summary_by_cohort.*`, and `cohort_composition.json` alongside the lane metadata

## Thesis Evaluation Lanes

`backend/scripts/run_thesis_evaluation.py` is the main runner for thesis-facing proof lanes.

For a longer but still bounded harder-story packet, `backend/scripts/run_thesis_hard_story_suite.py` runs the
checked-in `hard_mixed_24` and `longcorr_hard_32` corpora serially and then composes their completed lane outputs into
a single harder-suite bundle. That suite is a harder narrative slice, not a replacement for the sequential widening
gate.

The latest completed broader evidence is split across:

- the repaired 120-OD broad-cold bundle at `backend/out/thesis_eval_core120_green_repaired/artifacts/thesis_eval_core120_green_repaired_20260404/`
- the current 2026-04-06 campaign-ledger bundles under `backend/out/thesis_campaigns/`
- the current 1200-row expanded corpus and its 10 shards under `backend/data/eval/` and `backend/out/thesis_corpus/`

Treat those as separate surfaces. The 1200-row corpus is the latest corpus asset, not the latest completed result run.

## Sequential OD Campaigns

`backend/scripts/run_thesis_campaign.py` wraps the thesis runner for publication-style widening loops where previously green ODs must stay green as new ODs are admitted.

- bootstrap a known-green regression seed with `--bootstrap-csv`
- widen by only `--new-od-batch-size` unseen ODs per tranche
- rerun the full green regression set on every tranche
- gate promotion by per-OD target-variant results instead of only run-level averages
- fail the tranche if an expected regression/new OD disappears from evaluator output instead of silently treating it as preserved
- require the core thesis artifacts (`thesis_results.csv`, `thesis_summary.csv`, and `evaluation_manifest.json`) before a tranche counts as publication-evidence green
- persist a campaign ledger, tranche-local filtered corpora, and per-OD status summaries with gate config, evaluated-vs-expected OD inventory, and artifact-evidence status
- forward normal thesis-run arguments through `--evaluation-args`
- the widening overlay can now require `--require-dominance-win` and `--require-time-preserving-win`; those checks are applied per OD and per target variant, not just from run-level averages
- the widening overlay can also require `--require-proof-grade-readiness`; when enabled, the tranche stays blocked unless the top-level run metadata shows `strict_route_ready=true`, `route_graph.ready_mode="full"`, `route_graph_full_hydration_observed=true`, `degraded_evaluation_observed=false`, and `strict_full_search_proof_eligible=true`
- weighted and balanced wins alone are not an honest widening-green claim when dominance, time-preserving wins, or proof-grade readiness are still red

Under tight RAM caps, the publication-honest way to run this loop is to build a tranche-local graph
with `backend/scripts/build_route_graph_subset.py --corridor-km <km>` and then pass that asset into
the in-process evaluator through `--route-graph-asset-path` on
`backend/scripts/run_thesis_campaign.py` or `ROUTE_GRAPH_ASSET_PATH` for direct evaluator runs. The
subset helper now keeps the union of per-OD corridor buffers and stages kept nodes to disk, so a
small tranche does not inherit one large corpus-wide bbox just because the OD family spans a broad
rectangle.

When you want the campaign itself to stage that subset, use
`--route-graph-subset-corridor-km <km>` so the staged-asset cache key and the corridor width stay in
sync. The in-process staged-subset path is only valid for a single non-resume tranche; otherwise the
shared app instance would keep the first loaded subset resident across later widening steps.

When that low-RAM path remains in fast-startup mode, the summary artifacts must be read as degraded-evaluation evidence rather than full-hydration proof. `evaluation_manifest.json` and `metadata.json` now expose `route_graph_readiness_class`, `route_graph_full_hydration_observed`, `degraded_evaluation_observed`, `degraded_reason_codes_observed`, `precheck_gate_actions_observed`, `route_fallback_observed`, and `strict_full_search_proof_eligible` for that distinction. Benign precheck `"ok"` values are not counted as degradation by that aggregation, but any real non-`"ok"` precheck signal or any observed route fallback still keeps the run out of proof-grade status. `route_graph_readiness_class="fast_startup_metadata_ready"` or `degraded_evaluation_observed=true` is not a publishable full-search proof claim.

Use this path when the question is “did the next OD stay green without regressing the earlier ones?” rather than “what are the final broad-suite averages?”

For replay-ledger semantics, the campaign keeps previously green ODs in the replay set on every widening tranche. A tranche is only green when:

- all replayed green ODs stay green under the configured overlay gates
- all newly admitted ODs pass the same overlay gates
- no expected OD disappears from the evaluator output
- the required artifact bundle exists
- proof-grade readiness is present when `--require-proof-grade-readiness` is enabled

Current locally checked campaign states worth citing explicitly:

- `hard_mixed24_corr12p5_t4_inproc_r4`: green first tranche, `4` promoted ODs, `publication_evidence_ok=true`, `strict_full_search_proof_eligible=true`
- `dominance_cluster5_cardiff_bath_corr12p5_r2`: blocked tranche, `4` promoted ODs plus `cardiff_bath` red, blocked solely because proof-grade readiness stayed false
- `longcorr_cardiff4_corr12p5_r1`: red tranche, `4` red ODs with `routing_graph_disconnected_od` driving the failure surface
- `publishable_seq_fast_bootstrap_v2`: regression-start ledger against the 1200-row expanded corpus, red on its first replay tranche

### Broad Cold Thesis Proof

- scope: broad suite
- role: cold thesis proof
- ablations: `V0`, `A`, `B`, `C`
- expected outputs include thesis_results.*, thesis_summary.*, thesis_summary_by_cohort.*, thesis_metrics.json, thesis_plots.json, cohort_composition.json, methods_appendix.md, thesis_report.md, and evaluation_manifest.json
- evaluator metadata also records `evaluation_suite`, `thesis_cohort_scaffolding_v2`, and `thesis_metric_family_scaffolding_v1`
- empirical threshold clearance is now evidenced from the repaired 2026-04-04 120-OD broad-cold bundle at `backend/out/thesis_eval_core120_green_repaired/artifacts/thesis_eval_core120_green_repaired_20260404/`
- that repaired bundle covers `120` ODs / `480` evaluated rows with `0` failures, but it is a repair composition anchored by `repair_manifest.json`, not a fresh single-run runner bundle
- the older generated runner bundle remains the canonical single-run anchor when a discussion needs `evaluation_manifest.json` or the full thesis-report family
- the larger `backend/data/eval/uk_od_corpus_thesis_broad_expanded_1200.csv` corpus and its 10 shards are current broad-corpus assets, but they should not be cited as completed proof output until a composed or complete result bundle exists

### Focused REFC Proof

- scope: focused REFC lane
- goal: concentrated certification, fragility, and refine-cost evidence
- expected outputs include the core thesis outputs plus REFC artifacts such as certificate_summary.json, route_fragility_map.json, competitor_fragility_breakdown.json, sampled_world_manifest.json, thesis_summary_by_cohort.*, and cohort_composition.json
- evaluator metadata also records `evaluation_suite`, `thesis_cohort_scaffolding_v2`, and `thesis_metric_family_scaffolding_v1`
- this lane only counts as full graph-search proof when the same top-level summary records `route_graph_full_hydration_observed=true` and `strict_full_search_proof_eligible=true`; otherwise it remains a bounded degraded-evaluation slice
- this lane is wired for artifact production, but the empirical proof bar is still pending

### Focused VOI Proof

- scope: focused VOI lane
- goal: controller engagement, waste, lift, and refine-cost evidence
- expected outputs include VOI artifacts such as value_of_refresh.json, voi_action_trace.json, voi_action_scores.csv, voi_controller_trace_summary.json, voi_replay_oracle_summary.json, voi_stop_certificate.json, voi_controller_state.jsonl, and the public decision-package family when the route seam is exercised: decision_package.json, preference_summary.json, support_summary.json, support_trace.jsonl, support_provenance.json, certified_set.json, certified_set_routes.jsonl, abstention_summary.json, witness_summary.json, witness_routes.jsonl, controller_summary.json, controller_trace.jsonl, theorem_hook_map.json, lane_manifest.json, thesis_summary_by_cohort.*, and cohort_composition.json
- evaluator metadata also records `evaluation_suite`, `thesis_cohort_scaffolding_v2`, and `thesis_metric_family_scaffolding_v1`
- this lane is instrumented end to end, but its claimed gains remain to be validated by a later empirical pass

### Preference Proof

- scope: focused preference lane
- goal: keep preference-sensitive cohort behavior and stop-logic traces explicit in evaluator metadata
- expected outputs reuse the core thesis bundle, with `evaluation_suite.focus = "preference"` and `thesis_metric_family_scaffolding_v1` recording `preference` as `metadata_wired` at `cohort_metadata_only`
- this lane currently relies on shared cohort/report outputs rather than a dedicated preference-only artifact family

### Optional-Stopping Coverage

- scope: focused optional-stopping lane
- goal: keep certificate stopping and stop-hint coverage runs explicit in manifests and reports
- expected outputs reuse the core thesis bundle, with `evaluation_suite.focus = "optional_stopping"` and no separate optional-stopping artifact family beyond the shared thesis outputs

### Proxy-Audit Calibration

- scope: focused proxy-audit lane
- goal: keep proxy-vs-audit calibration runs explicit in evaluator metadata without overstating the current metric separation
- expected outputs reuse the core thesis bundle, with `evaluation_suite.focus = "proxy_audit"` and `multi_fidelity_support` remaining `partially_wired` through shared summary metrics rather than a dedicated calibration artifact family

### Perturbation / Flip-Radius

- scope: focused perturbation lane
- goal: keep fragility and perturbation-sensitive runs explicit in manifests and reports
- expected outputs reuse the core thesis bundle plus the existing fragility/certification artifacts already written by the certification-heavy paths

### Public Transfer

- scope: transfer lane
- goal: distinguish transfer-facing evaluation runs from the broad cold and focused internal proof families
- expected outputs reuse the core thesis bundle, with `evaluation_suite.scope = "transfer"` and `focus = "public_transfer"`

### Synthetic Ground-Truth

- scope: synthetic lane
- goal: distinguish synthetic evaluation runs from the public-transfer and broad-corpus families
- expected outputs reuse the core thesis bundle, with `evaluation_suite.scope = "synthetic"` and `focus = "synthetic_ground_truth"`

### DCCS Diagnostic Probe

- scope: diagnostic probe lane
- goal: inspect candidate richness, rescue behavior, and collapse-prone rows
- expected outputs emphasize the DCCS export contract now present in code: `dccs_summary.json`, `dccs_candidates.jsonl`, and `strict_frontier.jsonl`, plus `refined_routes.jsonl`, the route-facing certification outputs that feed the public decision package on the landed route seam, `thesis_summary_by_cohort.*`, and `cohort_composition.json`
- `dccs_summary.json` carries the exported `control_state` block, and `dccs_candidates.jsonl` / `strict_frontier.jsonl` carry the candidate-row DCCS vocabulary for `safe_elimination_reason`, `dominance_margin`, `dominating_candidate_ids`, `dominated_candidate_ids`, `search_deficiency_score`, `hidden_challenger_score`, `anti_collapse_quota`, and `long_corridor_search_completeness`
- evaluator metadata also records `evaluation_suite`, `thesis_cohort_scaffolding_v2`, and `thesis_metric_family_scaffolding_v1`
- the DCCS metric family is wired in the runner; the export surfaces above are instrumented for the next validation pass, but empirical recall and rescue thresholds remain unverified in this pass

### Metric Family Wiring Notes

- `preference` is `metadata_wired` at `cohort_metadata_only`; the exported evaluator surfaces are `preference_sensitive`, `summary_by_cohort_rows`, and `cohort_composition`, and standalone preference-proof metrics are not yet computed
- `multi_fidelity_support` is `partially_wired` at `shared_summary_metrics`; the exported evaluator surfaces are `mean_support_richness`, `mean_supported_ambiguity_alignment`, `mean_refc_world_reuse_rate`, and `mean_refc_stress_world_fraction`, and standalone proxy-audit calibration metrics are not yet separated
- `dccs`, `refc`, `selective_certification`, `voi`, `route_quality`, `runtime_reuse`, `support`, and `evaluation_size` remain runner-level `wired` families

## Current Result Anchors

- repaired broad-cold headline bundle: `backend/out/thesis_eval_core120_green_repaired/artifacts/thesis_eval_core120_green_repaired_20260404/`
  This is the latest measured broad-cold bundle in the repo: `120` ODs, `480` evaluated rows, `0` failures, and per-variant `success_rate=1.0`.
  The current headline deltas versus `V0` are `weighted_win_rate_v0=0.541667` / `balanced_win_rate_v0=0.583333` for `A`, `0.55` / `0.591667` for `B`, and `0.633333` / `0.65` for `C`.
  The current broad-cold certificate / controller anchors are `mean_certificate=0.818356` for `B`, `mean_certificate=0.855271` for `C`, `mean_voi_realized_certificate_lift=0.168752` for `C`, and `voi_controller_engagement_rate=0.666667` for `C`.
- focused REFC anchor: `backend/out/artifacts/refc_focus_20260331_h2/`
  Variant `C` currently reports `mean_certificate=0.870634`, `mean_certificate_margin=0.054134`, `mean_voi_realized_certificate_lift=0.165028`, `mean_voi_action_count=1.4`, `voi_controller_engagement_rate=0.9`, and `mean_runtime_ms=8520.1158`.
- focused VOI anchor: `backend/out/artifacts/thesis_eval_20260331_r2_focused_voi/`
  Variant `C` currently reports `mean_certificate=0.861459`, `mean_certificate_margin=0.04596`, `mean_voi_realized_certificate_lift=0.097638`, `mean_voi_action_count=1.3`, `voi_controller_engagement_rate=0.8`, and `mean_runtime_ms=11374.6391`.
- hot-rerun anchor pair: `backend/out/artifacts/hot_full_20260331_f2_cold/` and `backend/out/artifacts/hot_full_20260331_f2_hot/`
  `mean_runtime_ms` drops from `18903.988263 -> 1629.783632` for `V0`, `22722.158474 -> 2537.876947` for `A`, `6407.758 -> 2247.957474` for `B`, and `10213.524474 -> 2122.009263` for `C`.
  `mean_refc_world_reuse_rate` rises from `0.0` in the cold source to `1.0` in the hot rerun for `B` and `C`.
- harder-story / campaign anchors:
  `backend/out/thesis_campaigns/hard_mixed24_corr12p5_t4_inproc_r4/tranche_001/artifacts/hard_mixed24_corr12p5_t4_inproc_r4_t001/` is the current fully green harder-story proof-grade tranche with `4` ODs, `success_rate=1.0` for `V0/A/B/C`, `mean_certificate=0.934153` for `B`, and `route_graph_full_hydration_observed=true`.
  `backend/out/thesis_campaigns/longcorr_cardiff_newcastle4_corr12p5_r1/tranche_001/artifacts/longcorr_cardiff_newcastle4_corr12p5_r1_t001/` is the current long-corridor partial-green anchor with `success_rate=0.75` for every variant and `strict_full_search_proof_eligible=false`.
  `backend/out/thesis_campaigns/dominance_cluster5_cardiff_bath_corr12p5_r2/tranche_001/artifacts/dominance_cluster5_cardiff_bath_corr12p5_r2_t001/` is the current dominance-cluster partial-green anchor with `success_rate=0.8` for every variant and `strict_full_search_proof_eligible=false`.
  `backend/out/thesis_campaigns/longcorr_cardiff4_corr12p5_r1/tranche_001/artifacts/longcorr_cardiff4_corr12p5_r1_t001/` is the current checked-in all-red negative-control campaign bundle.
- exhaustive metric inventory anchor:
  `backend/out/thesis_campaigns/hard_mixed24_corr12p5_t4_inproc_r4/tranche_001/artifacts/hard_mixed24_corr12p5_t4_inproc_r4_t001/thesis_metrics.json` currently writes `328` per-variant summary-row metrics, `331` per-cohort summary metrics, `4` `run_validity` fields, `3` startup/warmup fields, and `10` metric-family labels.
- staged but incomplete broadening:
  `backend/data/eval/uk_od_corpus_thesis_broad_expanded_1200.csv` and `backend/data/eval/thesis_shards_1200/*.csv` are current corpus assets, but `backend/out/thesis_1200_s01_repo_local/artifacts/thesis_1200_s01_repo_local/` only contains `repo_asset_preflight.json` and `baseline_smoke_summary.json`; do not treat that directory as a completed proof lane.

## Hot-Rerun Production / Reuse Benchmark

`backend/scripts/run_hot_rerun_benchmark.py` produces a dedicated second-run proof. It is separate from the cold thesis suite.

The benchmark is only meaningful when the comparison artifact is read as a paired cold/hot run, not as an isolated hot rerun.

Expected outputs include:

- a cold source run
- a hot rerun using restored cache state
- hot_rerun_vs_cold_comparison.json
- hot_rerun_vs_cold_comparison.csv
- hot_rerun_gate.json
- hot_rerun_report.md

Hot-rerun reporting focuses on:

- route cache hit rate
- option-build cache hit rate
- option-build reuse rate
- REFC world reuse rate
- runtime-ratio improvement versus the paired cold source
- the same cold/hot pair should be used for both the JSON comparison and the human report

## Composed Final Suite Reporting

`backend/scripts/compose_thesis_suite_report.py` composes completed evaluation lanes into a final report set.

Expected composed outputs include:

- thesis_results.csv
- thesis_summary.csv
- thesis_summary.json
- thesis_summary_by_cohort.csv
- thesis_summary_by_cohort.json
- cohort_composition.json
- evaluation_manifest.json, which is the composition anchor for run provenance
- methods_appendix.md
- thesis_report.md
- suite_sources.json
- prior_coverage_summary.json
- evaluator metadata should retain `evaluation_suite`, `thesis_cohort_scaffolding_v2`, and `thesis_metric_family_scaffolding_v1` in the run payloads and composed outputs

The composed suite is where provenance-visible summary views should live; it should not be used to rewrite the cold thesis proof or the hot-rerun proof.

## Sharded Composition And Run Comparison

`backend/scripts/compose_thesis_sharded_report.py` composes completed shard `thesis_results.csv` files into a single
report bundle. Its distinguishing provenance surface is `shard_sources.json`, not `suite_sources.json`.

Use it when:

- the corpus has been expanded and split into shard CSVs
- each shard already has a completed `thesis_results.csv`
- the final report must preserve per-shard provenance and replay/rung evidence instead of pretending the result came from a single runner invocation

`backend/scripts/compare_thesis_runs.py` is the current row-diff utility for before/after analysis. It emits an OD and
variant diff CSV and, when both summary CSVs are available, an optional summary JSON that compares route-quality,
certificate/frontier, win-rate, margin, and runtime families across runs.

## Current Gate Philosophy

- cold thesis proof and hot-rerun proof are separate claims and should stay separate
- broad, focused, transfer, synthetic, probe, and hot-rerun roles should remain explicit in metadata and output naming
- `thesis_cohort_scaffolding_v2` and `thesis_metric_family_scaffolding_v1` are instrumentation surfaces; do not read them as empirically cleared gates
- public route-contract changes should be documented and regression-tested together; `pipeline_mode="tri_source"`, optional `decision_package`, and legacy compatibility fields are one contract surface
- frontend inspection changes should be documented and regression-tested together with that route contract; artifact-family grouping in the browser must stay derived from existing artifact names and `decision_package`, not hidden backend-only fields
- authored docs are not proof artifacts; generated run outputs are not authored docs
- do not treat partial or interrupted artifacts as successful evidence

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Run and Operations Guide](run-and-operations.md)
- [Reproducibility Capsule](reproducibility-capsule.md)
- [VOI Thesis Pipeline Spec](voi-pipeline-spec.md)
