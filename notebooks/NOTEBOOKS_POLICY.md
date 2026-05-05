# Notebook Directory

This repository is intentionally maintained as a notebook-free, script-first workflow.

## Policy

- Do not introduce `.ipynb` notebooks for routine development, evaluation, or reporting.
- Prefer checked-in scripts under `backend/scripts/` or repo-level helpers under `scripts/`.
- Treat generated artifacts under `backend/out/` as outputs of scripts, not hand-maintained analysis notebooks.

## Script Replacements For Notebook Work

Use these entry points instead of ad hoc notebook analysis:

- `backend/scripts/run_headless_scenario.py`: backend scenario smoke and headless replay
- `backend/scripts/run_thesis_lane.py`: smaller lane-local thesis/evaluation reruns
- `backend/scripts/run_thesis_evaluation.py`: broad cold proof, focused REFC, focused VOI, and DCCS probe
- `backend/scripts/run_hot_rerun_benchmark.py`: dedicated second-run hot-rerun reuse benchmark
- `backend/scripts/compose_thesis_suite_report.py`: compose completed lanes into a final suite report
- `backend/scripts/check_eta_concept_drift.py`: offline ETA drift analysis
- `scripts/check_docs.py`: authored-doc consistency checks

## Output Conventions

Evaluation and reporting outputs are written under `backend/out/`:

- `backend/out/artifacts/<run_id>/`
- `backend/out/manifests/<run_id>.json`
- `backend/out/scenario_manifests/<run_id>.json`
- `backend/out/provenance/<run_id>.jsonl`

Typical per-run artifact families include:

- route outputs: `results.json`, `results.csv`, `metadata.json`, `routes.geojson`
- DCCS/REFC/VOI outputs: `dccs_summary.json`, `certificate_summary.json`, `value_of_refresh.json`, `voi_action_trace.json`
- thesis outputs: `thesis_results.*`, `thesis_summary.*`, `thesis_metrics.json`, `thesis_plots.json`, `evaluation_manifest.json`
- report outputs: `methods_appendix.md`, `thesis_report.md`
- hot-rerun outputs when applicable: `hot_rerun_vs_cold_comparison.json`, `hot_rerun_gate.json`, `hot_rerun_report.md`
- composed suite outputs when applicable: `thesis_summary_by_cohort.csv`, `thesis_summary_by_cohort.json`, `suite_sources.json`, `prior_coverage_summary.json`, `cohort_composition.json`

## Authoritative References

- Use [API Cookbook](../docs/api-cookbook.md) for reproducible CLI examples.
- Use [Run and Operations Guide](../docs/run-and-operations.md) for startup, preflight, rebuild, and test workflows.
- Use [Quality Gates and Benchmarks](../docs/quality-gates-and-benchmarks.md) for the current evaluation ladder.
- Use [Reproducibility Capsule](../docs/reproducibility-capsule.md) for cold-proof versus hot-rerun proof handling.
- Use [Documentation Index](../docs/DOCS_INDEX.md) as the top-level map.
