# Reproducibility Capsule

Last Updated: 2026-04-09
Applies To: deterministic evaluation, signed run artifacts, cold thesis proof, hot-rerun reuse proof, and archived public route-seam replay bundles

This page summarizes how to reproduce and archive the current proof workflows without mixing cold and hot claims.

This capsule is primarily evaluator/reporting proof guidance. Direct `/route` seam runs are adjacent runtime artifacts, but when they are archived for public route-seam replay they should preserve the additive decision-package artifact family when present. The route-facing replay and diagnostics surfaces are instrumented in code, but this capsule treats them as honesty surfaces rather than as fresh empirical proof in this pass.

## One-Command Demo

From repo root:

```powershell
.\scripts\demo_repro_run.ps1
```

## Manual Repro Path

From repo root:

```powershell
uv run --project backend python backend/scripts/build_model_assets.py
uv run --project backend python backend/scripts/preflight_live_runtime.py
uv run --project backend python backend/scripts/expand_thesis_broad_corpus.py --help
uv run --project backend python backend/scripts/run_thesis_evaluation.py
uv run --project backend python backend/scripts/run_thesis_campaign.py --help
uv run --project backend python backend/scripts/run_thesis_hard_story_suite.py --help
uv run --project backend python backend/scripts/run_hot_rerun_benchmark.py
uv run --project backend python backend/scripts/compare_thesis_runs.py --help
uv run --project backend python backend/scripts/compose_thesis_sharded_report.py --help
uv run --project backend python backend/scripts/compose_thesis_suite_report.py
```

## Reproducibility Controls

- fixed code revision
- fixed request payloads and corpus rows
- explicit `pipeline_mode` and budget settings
- fixed seed chains (`pipeline_seed`, stochastic seeds, evaluation seeds)
- strict runtime policy made explicit in settings and manifests
- signed manifests plus provenance logs

## Current Archive Anchors

When you need the latest checked-in evidence rather than only the oldest canonical runner bundles, archive these families explicitly:

- repaired broad-cold headline evidence: `backend/out/thesis_eval_core120_green_repaired/artifacts/thesis_eval_core120_green_repaired_20260404/`
  This is the current measured broad-cold headline bundle (`120` ODs / `480` rows / `0` failures), but it is a repair composition anchored by `repair_manifest.json`, not a fresh single-run thesis-runner bundle.
- focused REFC / VOI anchors: `backend/out/artifacts/refc_focus_20260331_h2/` and `backend/out/artifacts/thesis_eval_20260331_r2_focused_voi/`
- hot-rerun proof pair: `backend/out/artifacts/hot_full_20260331_f2_cold/` and `backend/out/artifacts/hot_full_20260331_f2_hot/`
- harder-story and widening anchors under `backend/out/thesis_campaigns/`, especially `hard_mixed24_corr12p5_t4_inproc_r4_t001`, `longcorr_cardiff_newcastle4_corr12p5_r1_t001`, `dominance_cluster5_cardiff_bath_corr12p5_r2_t001`, and `longcorr_cardiff4_corr12p5_r1_t001`
- single-OD tuning packet under `backend/out/single_od_london_newcastle_*`
  Preserve the lease snapshots and evaluation manifests as part of the tuning evidence because the split-process memory-limit conclusions are encoded in those files, not only in the run names.
- staged but incomplete broadening assets:
  `backend/data/eval/uk_od_corpus_thesis_broad_expanded_1200.csv`,
  `backend/data/eval/thesis_shards_1200/*.csv`,
  and `backend/out/thesis_1200_s01_repo_local/`
  The current `thesis_1200_s01_repo_local` artifact family contains only preflight plus baseline smoke evidence, so archive it as staging/probe evidence, not as a completed proof bundle.

## Cold Thesis Proof Bundle

Treat the cold thesis proof as its own bundle. Archive:

- `backend/out/manifests/<run_id>.json`
- `backend/out/scenario_manifests/<run_id>.json`
- `backend/out/provenance/<run_id>.jsonl`
- `backend/out/artifacts/<run_id>/metadata.json`
- `backend/out/artifacts/<run_id>/results.json`
- `backend/out/artifacts/<run_id>/thesis_summary_by_cohort.csv`
- `backend/out/artifacts/<run_id>/thesis_summary_by_cohort.json`
- `backend/out/artifacts/<run_id>/thesis_results.csv`
- `backend/out/artifacts/<run_id>/thesis_summary.csv`
- `backend/out/artifacts/<run_id>/thesis_summary.json`
- `backend/out/artifacts/<run_id>/thesis_metrics.json`
- `backend/out/artifacts/<run_id>/thesis_plots.json`
- `backend/out/artifacts/<run_id>/methods_appendix.md`
- `backend/out/artifacts/<run_id>/thesis_report.md`
- `backend/out/artifacts/<run_id>/evaluation_manifest.json`
- evaluator payloads should also retain `evaluation_suite`, `thesis_cohort_scaffolding_v2`, and `thesis_metric_family_scaffolding_v1` in their JSON metadata; the cohort scaffold should carry the evaluator-defined taxonomy including `collapse_prone`, `osrm_brittle`, `ors_brittle`, `refresh_sensitive`, `time_preserving_conflict`, `low_ambiguity_fast_path`, `preference_sensitive`, `support_fragile`, `audit_heavy`, and `proxy_friendly`
- `evaluation_suite` should preserve the explicit role family when present: `broad_cold_proof`, `focused_refc_proof`, `focused_voi_proof`, `preference_proof`, `optional_stopping_coverage`, `proxy_audit_calibration`, `perturbation_flip_radius`, `public_transfer`, `synthetic_ground_truth`, `dccs_diagnostic_probe`, `hot_rerun_cold_source`, or `hot_rerun`
- `thesis_metric_family_scaffolding_v1` should preserve that `preference` is `metadata_wired` at `cohort_metadata_only` and `multi_fidelity_support` is `partially_wired` at `shared_summary_metrics`; those labels are honesty metadata, not empirical gate clearance

If a run is part of a sequential OD widening campaign, preserve the campaign ledger beside the run-local artifacts. At minimum that ledger should retain:

- `campaign_state.json`
- `campaign_result.json`
- `campaign_report.md`
- each tranche-local `tranche_<n>/tranche_od_corpus.csv`
- each tranche-local `tranche_<n>/per_od_status.json`

Those campaign outputs are the reproducibility anchor for regression carry-forward claims because
they record which prior green ODs were retested before a new OD was promoted, which new ODs were
admitted in each tranche, and why the campaign stopped (`campaign_complete`,
`max_tranches_reached`, `stop_on_red_tranche`, or another explicit stop reason).
The tranche ledger should also preserve the gate configuration used for promotion, the
evaluated-vs-expected OD inventory for each tranche, and the core artifact-evidence status so a
missing `thesis_results.csv`, `thesis_summary.csv`, or `evaluation_manifest.json` cannot be
mistaken for a publishable green tranche.

If a campaign or evaluator run used a low-RAM tranche-local graph asset, archive that subset beside
the tranche ledger as part of the proof surface. At minimum preserve:

- the exact subset graph JSON built by `backend/scripts/build_route_graph_subset.py`
- its sibling `.meta.json` report, including `corridor_km`, `filter_mode`, and `corridor_union_bbox`
- the `--route-graph-asset-path` argument or `ROUTE_GRAPH_ASSET_PATH` value that selected it

That is the only honest way to replay a strict low-RAM tranche later without silently falling back
to the full UK graph.

If a direct `/route` seam run is being archived alongside this proof surface for public replay, preserve the route-seam artifacts when present rather than assuming every evaluator run emits them. The additive route-seam family is:

- `backend/out/artifacts/<run_id>/decision_package.json`
- `backend/out/artifacts/<run_id>/preference_summary.json`
- `backend/out/artifacts/<run_id>/support_summary.json`
- `backend/out/artifacts/<run_id>/support_provenance.json`
- `backend/out/artifacts/<run_id>/support_trace.jsonl`
- `backend/out/artifacts/<run_id>/certified_set.json`
- `backend/out/artifacts/<run_id>/certified_set_routes.jsonl`
- any populated `abstention_summary.json`, `witness_summary.json`, `witness_routes.jsonl`, `controller_summary.json`, `controller_trace.jsonl`, `voi_controller_trace_summary.json`, `voi_replay_oracle_summary.json`, `theorem_hook_map.json`, and `lane_manifest.json`
- `backend/out/artifacts/<run_id>/final_route_trace.json` as the route-trace anchor when present; it is the sibling trace artifact that carries `artifact_pointers` to the emitted decision/support/controller family

## Hot-Rerun Production / Reuse Proof Bundle

Treat the hot-rerun proof as a separate bundle. Archive:

- paired cold source run id
- paired hot rerun run id
- `backend/out/artifacts/<hot_run_id>/hot_rerun_vs_cold_comparison.json`
- `backend/out/artifacts/<hot_run_id>/hot_rerun_vs_cold_comparison.csv`
- `backend/out/artifacts/<hot_run_id>/hot_rerun_gate.json`
- `backend/out/artifacts/<hot_run_id>/hot_rerun_report.md`
- paired manifests and provenance for both the cold source and hot run

The hot bundle should be interpreted alongside the paired cold source bundle, not as a replacement for the cold thesis proof bundle. Hot-rerun payloads should preserve the same evaluator scaffold names (`thesis_cohort_scaffolding_v2` and `thesis_metric_family_scaffolding_v1`) so the hot-vs-cold comparison remains traceable without re-parsing filenames.

## Composed Final Suite Bundle

When composing multiple completed lanes, archive:

- `backend/out/artifacts/<run_id>/thesis_summary_by_cohort.csv`
- `backend/out/artifacts/<run_id>/thesis_summary_by_cohort.json`
- `backend/out/artifacts/<run_id>/cohort_composition.json`
- `backend/out/artifacts/<run_id>/suite_sources.json`
- `backend/out/artifacts/<run_id>/prior_coverage_summary.json`
- `backend/out/artifacts/<run_id>/evaluation_manifest.json`
- composed suite payloads should also retain `evaluation_suite`, `thesis_cohort_scaffolding_v2`, and `thesis_metric_family_scaffolding_v1`

## Sharded Composition Bundle

When composing completed shard result CSVs with `backend/scripts/compose_thesis_sharded_report.py`, archive:

- `backend/out/artifacts/<run_id>/thesis_results.csv`
- `backend/out/artifacts/<run_id>/thesis_summary.csv`
- `backend/out/artifacts/<run_id>/thesis_summary.json`
- `backend/out/artifacts/<run_id>/thesis_summary_by_cohort.csv`
- `backend/out/artifacts/<run_id>/thesis_summary_by_cohort.json`
- `backend/out/artifacts/<run_id>/cohort_composition.json`
- `backend/out/artifacts/<run_id>/prior_coverage_summary.json`
- `backend/out/artifacts/<run_id>/shard_sources.json`
- `backend/out/artifacts/<run_id>/evaluation_manifest.json`
- `backend/out/artifacts/<run_id>/methods_appendix.md`
- `backend/out/artifacts/<run_id>/thesis_report.md`

This is distinct from the older suite composer: sharded composition should preserve per-shard provenance through
`shard_sources.json` rather than flattening the merge into `suite_sources.json`.

## Comparing Two Runs

For the current row-diff path, use `backend/scripts/compare_thesis_runs.py` with:

- `--before-results-csv`
- `--after-results-csv`
- `--out-csv`
- optional `--before-summary-csv`, `--after-summary-csv`, and `--summary-json`

The comparison CSV/JSON should be archived when the user is making a before/after claim about:

- selected-route deltas such as distance, duration, money, and emissions
- certificate and frontier metrics such as certificate, margin, runner-up gap, frontier count, hypervolume, diversity, and entropy
- win-rate and comparator-margin changes
- runtime and algorithm-runtime deltas

Before attributing differences to algorithmic changes, verify:

1. same payload family or same OD corpus rows
2. same scenario and uncertainty settings
3. same `pipeline_mode`
4. same budget and threshold settings
5. same model asset state
6. same strict runtime policy
7. same baseline availability and identity
8. no interrupted or partial artifact was substituted as proof

## Provenance Expectations

- manifests are signed with `HMAC-SHA256`
- provenance logs live under `backend/out/provenance/`
- evaluation manifests should clearly distinguish broad, focused, transfer, synthetic, probe, hot-rerun cold source, and hot-rerun proof roles
- evaluator payloads should keep `evaluation_suite`, `thesis_cohort_scaffolding_v2`, and `thesis_metric_family_scaffolding_v1` explicit so downstream readers can recover lane and cohort context without parsing filenames; the named cohort taxonomy remains the evaluator-defined set including `collapse_prone`, `osrm_brittle`, `ors_brittle`, `refresh_sensitive`, `time_preserving_conflict`, `low_ambiguity_fast_path`, `preference_sensitive`, `support_fragile`, `audit_heavy`, and `proxy_friendly`
- evaluator metric-family metadata should keep the mixed wiring state explicit: `preference` is metadata-only through cohort/report surfaces, and `multi_fidelity_support` is partially wired through shared summary metrics
- composed suite outputs should record `suite_sources.json`, while sharded-report outputs should record `shard_sources.json`, so stale-versus-fresh provenance stays explicit
- signed manifests should carry `run_id`, `created_at`, the payload body, and a nested signature block
- archived public route-seam replay bundles should keep any populated decision-package artifacts with the same manifest/provenance discipline as the rest of the run

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Quality Gates and Benchmarks](quality-gates-and-benchmarks.md)
- [Sample Manifest and Outputs](sample-manifest.md)
- [Performance Profiling Notes](performance-profiling-notes.md)
