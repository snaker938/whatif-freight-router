# Performance Profiling Notes

Last Updated: 2026-04-09  
Applies To: backend runtime budgets, route-graph warmup, thesis runtime artifacts, and hot-rerun reuse profiling

This page summarizes the current profiling entry points and the runtime evidence surfaces emitted by the evaluation stack.

## Primary Profiling Entry Points

From repo root:

```powershell
uv run --project backend python backend/scripts/run_thesis_evaluation.py
uv run --project backend python backend/scripts/run_hot_rerun_benchmark.py
```

For profiling runs that must keep toll-sensitive thesis rows alive under a configured pricing model, add
`--toll-cost-per-km <rate>` to the thesis evaluation command.

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

## Current Measured Runtime Anchors

- repaired broad-cold headline bundle: `backend/out/thesis_eval_core120_green_repaired/artifacts/thesis_eval_core120_green_repaired_20260404/`
  `mean_runtime_ms` is currently `59456.350583` for `V0`, `71719.256567` for `A`, `49724.427308` for `B`, and `49589.171025` for `C`.
  `mean_algorithm_runtime_ms` is `15267.6395` for `V0`, `27530.545483` for `A`, `5535.716225` for `B`, and `5400.459942` for `C`.
- hot-rerun pair:
  `backend/out/artifacts/hot_full_20260331_f2_cold/` versus `backend/out/artifacts/hot_full_20260331_f2_hot/`
  currently shows `mean_runtime_ms` reductions of `18903.988263 -> 1629.783632` for `V0`, `22722.158474 -> 2537.876947` for `A`, `6407.758 -> 2247.957474` for `B`, and `10213.524474 -> 2122.009263` for `C`.
  `mean_refc_world_reuse_rate` rises from `0.0` in the cold source to `1.0` in the hot rerun for `B` and `C`.
- harder-story runtime anchor:
  `backend/out/thesis_campaigns/hard_mixed24_corr12p5_t4_inproc_r4/tranche_001/artifacts/hard_mixed24_corr12p5_t4_inproc_r4_t001/`
  currently reports `mean_runtime_ms=8726.4415` for `V0`, `5823.45775` for `A`, `7182.2645` for `B`, and `2886.40825` for `C`.

## April 5 Split-Process Tuning Packet

The current single-OD tuning evidence lives under `backend/out/single_od_london_newcastle_*` and should be read as a profiling packet, not as headline thesis proof.

- the tuning path converged on a split-process topology driven by `backend/scripts/start_backend_logged.ps1` and `backend/scripts/run_with_job_memory_limit.ps1`
- the unstable multi-variant backend-cap sweeps `r16` / `r17` / `r18` all recorded backend `limit_exceeded` exits at backend caps `2368`, `2400`, and `2432` MB respectively
- the later isolated single-variant proof-grade runs `r30` / `r31` / `r32` stabilized with backend `2304` MB, evaluator `1024` MB, and `strict_full_search_proof_eligible=true`
- the latest isolated runtime anchors are `mean_runtime_ms=8792.812` for `A` (`r30`), `9388.677` for `B` (`r31`), and `16455.836` for `C` (`r32`)
- the latest isolated controller anchor is `voi_controller_engagement_rate=1.0` on `r32` for variant `C`

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
