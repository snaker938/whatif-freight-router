# Performance Profiling Notes

Last Updated: 2026-03-31  
Applies To: backend runtime budgets, route-graph warmup, thesis runtime artifacts, and hot-rerun reuse profiling

This page summarizes the current profiling entry points and the runtime evidence surfaces emitted by the evaluation stack.

## Primary Profiling Entry Points

From repo root:

```powershell
uv run --project backend python backend/scripts/run_thesis_evaluation.py
uv run --project backend python backend/scripts/run_hot_rerun_benchmark.py
```

Related helpers:

- `backend/scripts/run_thesis_lane.py`
- `backend/scripts/preflight_live_runtime.py`
- `backend/scripts/check_eta_concept_drift.py`

## Runtime Metrics Now Emitted

The current evaluation stack emits runtime summary fields such as:

- `mean_runtime_ms`
- `mean_algorithm_runtime_ms`
- `mean_runtime_ratio_vs_osrm`
- `mean_runtime_ratio_vs_ors`
- `mean_algorithm_runtime_ratio_vs_osrm`
- `mean_algorithm_runtime_ratio_vs_ors`
- `mean_stage_option_build_ms`
- `mean_stage_dccs_ms`
- `mean_stage_refc_ms`
- `mean_stage_voi_ms`
- `mean_route_cache_hit_rate`
- `mean_option_build_cache_hit_rate`
- `mean_option_build_reuse_rate`
- `mean_refc_world_reuse_rate`
- `mean_route_graph_warmup_elapsed_ms`

These metrics are surfaced in the thesis summary CSV/JSON outputs, thesis_metrics.json, thesis_plots.json, and the hot-rerun comparison artifacts.

For cold thesis evaluation, `mean_algorithm_runtime_ms` and the per-stage timings describe the one-pass cost of the configured pipeline.
For hot-rerun reuse proof, the comparison artifacts are the right place to read cache hit rates, reuse rates, and runtime-ratio improvement against the paired cold source.

## Route-Graph Warmup

Warmup is now part of the strict readiness story, not a hidden startup detail.

Operational notes:

- warmup state is surfaced through `GET /health/ready`
- strict route endpoints fail fast while warmup is incomplete
- the broad evaluation artifacts record `mean_route_graph_warmup_elapsed_ms`

When profiling startup regressions, inspect:

- `backend/out/model_assets/preflight_live_runtime.json`
- `GET /health/ready`
- `backend/out/artifacts/<run_id>/thesis_summary.csv`

## Hot-Rerun Profiling

Use `backend/scripts/run_hot_rerun_benchmark.py` when the question is reuse efficiency rather than cold quality.

The hot-rerun benchmark writes:

- hot_rerun_vs_cold_comparison.json
- hot_rerun_vs_cold_comparison.csv
- hot_rerun_gate.json
- hot_rerun_report.md

The comparison JSON/CSV should be read together: the JSON carries structured per-variant comparison records, while the CSV is the machine-friendly flat summary.

These outputs are where route-cache hits, option-build reuse, REFC world reuse, and runtime-ratio improvements should be read from.

## Low-Risk Local Profiling Strategy

On constrained machines:

1. run one benchmark or evaluation script at a time
2. avoid overlapping full pytest with evaluation runs
3. use `.\scripts\run_backend_tests_safe.ps1` before expensive reruns
4. use lane-local reruns before full-suite reruns
5. use the hot-rerun benchmark only after a successful cold path is available

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Quality Gates and Benchmarks](quality-gates-and-benchmarks.md)
- [Model Assets and Data Sources](model-assets-and-data-sources.md)
- [Reproducibility Capsule](reproducibility-capsule.md)
