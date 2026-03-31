# Quality Gates and Benchmarks

Last Updated: 2026-03-31  
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
uv run --project backend python backend/scripts/run_hot_rerun_benchmark.py
uv run --project backend python backend/scripts/compose_thesis_suite_report.py
```

## Test and Validation Lanes

### Authored Docs Checks

- `python scripts/check_docs.py`
- validates links, local file paths, docs index coverage, and backend endpoint parity

### Backend Regression and Strict Runtime Tests

- use `.\scripts\run_backend_tests_safe.ps1` on constrained machines
- use `uv run --project backend pytest backend/tests` only when a full suite is appropriate
- strict failures, baseline identity, and artifact semantics should be treated as hard regressions

## Thesis Evaluation Lanes

`backend/scripts/run_thesis_evaluation.py` is the main runner for thesis-facing proof lanes.

### Broad Cold Thesis Proof

- scope: broad suite
- role: cold thesis proof
- ablations: `V0`, `A`, `B`, `C`
- expected outputs include thesis_results.*, thesis_summary.*, thesis_metrics.json, thesis_plots.json, methods_appendix.md, thesis_report.md, and evaluation_manifest.json

### Focused REFC Proof

- scope: focused REFC lane
- goal: concentrated certification, fragility, and refine-cost evidence
- expected outputs include the core thesis outputs plus REFC artifacts such as certificate_summary.json, route_fragility_map.json, competitor_fragility_breakdown.json, and sampled_world_manifest.json

### Focused VOI Proof

- scope: focused VOI lane
- goal: controller engagement, waste, lift, and refine-cost evidence
- expected outputs include VOI artifacts such as value_of_refresh.json, voi_action_trace.json, voi_action_scores.csv, voi_stop_certificate.json, and voi_controller_state.jsonl

### DCCS Diagnostic Probe

- scope: diagnostic probe lane
- goal: inspect candidate richness, rescue behavior, and collapse-prone rows
- expected outputs emphasize dccs_candidates.jsonl, dccs_summary.json, refined_routes.jsonl, and strict_frontier.jsonl

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
- methods_appendix.md
- thesis_report.md
- suite_sources.json
- prior_coverage_summary.json
- cohort_composition.json
- evaluation_manifest.json

The composed suite is where provenance-visible summary views should live; it should not be used to rewrite the cold thesis proof or the hot-rerun proof.

## Current Gate Philosophy

- cold thesis proof and hot-rerun proof are separate claims and should stay separate
- broad, focused, probe, and hot-rerun roles should remain explicit in metadata and output naming
- authored docs are not proof artifacts; generated run outputs are not authored docs
- do not treat partial or interrupted artifacts as successful evidence

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Run and Operations Guide](run-and-operations.md)
- [Reproducibility Capsule](reproducibility-capsule.md)
- [VOI Thesis Pipeline Spec](voi-pipeline-spec.md)
