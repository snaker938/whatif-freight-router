# Reproducibility Capsule

Last Updated: 2026-04-03
Applies To: deterministic evaluation, signed run artifacts, cold thesis proof, hot-rerun reuse proof, and archived public route-seam replay bundles

This page summarizes how to reproduce and archive the current proof workflows without mixing cold and hot claims.

This capsule is primarily evaluator/reporting proof guidance. Direct `/route` seam runs are adjacent runtime artifacts, but when they are archived for public route-seam replay they should preserve the additive decision-package artifact family when present.

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
uv run --project backend python backend/scripts/run_thesis_evaluation.py
uv run --project backend python backend/scripts/run_hot_rerun_benchmark.py
uv run --project backend python backend/scripts/compose_thesis_suite_report.py
```

## Reproducibility Controls

- fixed code revision
- fixed request payloads and corpus rows
- explicit `pipeline_mode` and budget settings
- fixed seed chains (`pipeline_seed`, stochastic seeds, evaluation seeds)
- strict runtime policy made explicit in settings and manifests
- signed manifests plus provenance logs

## Cold Thesis Proof Bundle

Treat the cold thesis proof as its own bundle. Archive:

- `backend/out/manifests/<run_id>.json`
- `backend/out/scenario_manifests/<run_id>.json`
- `backend/out/provenance/<run_id>.jsonl`
- `backend/out/artifacts/<run_id>/metadata.json`
- `backend/out/artifacts/<run_id>/results.json`
- `backend/out/artifacts/<run_id>/thesis_results.csv`
- `backend/out/artifacts/<run_id>/thesis_summary.csv`
- `backend/out/artifacts/<run_id>/thesis_summary.json`
- `backend/out/artifacts/<run_id>/thesis_summary_by_cohort.csv`
- `backend/out/artifacts/<run_id>/thesis_summary_by_cohort.json`
- `backend/out/artifacts/<run_id>/thesis_metrics.json`
- `backend/out/artifacts/<run_id>/thesis_plots.json`
- `backend/out/artifacts/<run_id>/methods_appendix.md`
- `backend/out/artifacts/<run_id>/thesis_report.md`
- `backend/out/artifacts/<run_id>/evaluation_manifest.json`
- evaluator payloads should also retain `evaluation_suite` and `cohort_scaffolding` in their JSON metadata

If a direct `/route` seam run is being archived alongside this proof surface for public replay, preserve the route-seam artifacts when present rather than assuming every evaluator run emits them. The additive route-seam family is:

- `backend/out/artifacts/<run_id>/decision_package.json`
- `backend/out/artifacts/<run_id>/preference_summary.json`
- `backend/out/artifacts/<run_id>/support_summary.json`
- `backend/out/artifacts/<run_id>/support_provenance.json`
- `backend/out/artifacts/<run_id>/support_trace.jsonl`
- `backend/out/artifacts/<run_id>/certified_set.json`
- `backend/out/artifacts/<run_id>/certified_set_routes.jsonl`
- any populated `abstention_summary.json`, `witness_summary.json`, `witness_routes.jsonl`, `controller_summary.json`, `controller_trace.jsonl`, `theorem_hook_map.json`, and `lane_manifest.json`

## Hot-Rerun Production / Reuse Proof Bundle

Treat the hot-rerun proof as a separate bundle. Archive:

- paired cold source run id
- paired hot rerun run id
- `backend/out/artifacts/<hot_run_id>/hot_rerun_vs_cold_comparison.json`
- `backend/out/artifacts/<hot_run_id>/hot_rerun_vs_cold_comparison.csv`
- `backend/out/artifacts/<hot_run_id>/hot_rerun_gate.json`
- `backend/out/artifacts/<hot_run_id>/hot_rerun_report.md`
- paired manifests and provenance for both the cold source and hot run

The hot bundle should be interpreted alongside the paired cold source bundle, not as a replacement for the cold thesis proof bundle.

## Composed Final Suite Bundle

When composing multiple completed lanes, archive:

- `backend/out/artifacts/<run_id>/thesis_summary_by_cohort.csv`
- `backend/out/artifacts/<run_id>/thesis_summary_by_cohort.json`
- `backend/out/artifacts/<run_id>/cohort_composition.json`
- `backend/out/artifacts/<run_id>/suite_sources.json`
- `backend/out/artifacts/<run_id>/prior_coverage_summary.json`
- `backend/out/artifacts/<run_id>/evaluation_manifest.json`
- composed suite payloads should also retain `evaluation_suite` and `cohort_scaffolding`

## Comparing Two Runs

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
- evaluation manifests should clearly distinguish broad, focused, probe, hot-rerun cold source, and hot-rerun proof roles
- evaluator payloads should keep `evaluation_suite` and `cohort_scaffolding` explicit so downstream readers can recover lane and cohort context without parsing filenames
- composed suite outputs should record suite_sources.json so stale-versus-fresh provenance stays explicit
- signed manifests should carry `run_id`, `created_at`, the payload body, and a nested signature block
- archived public route-seam replay bundles should keep any populated decision-package artifacts with the same manifest/provenance discipline as the rest of the run

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Quality Gates and Benchmarks](quality-gates-and-benchmarks.md)
- [Sample Manifest and Outputs](sample-manifest.md)
- [Performance Profiling Notes](performance-profiling-notes.md)
