# Documentation Index

Last Updated: 2026-04-09  
Applies To: current repo state across `backend/`, `frontend/`, `scripts/`, and thesis/production evaluation workflows

This is the main authored-docs index for the project. Generated outputs under `backend/out/artifacts/<run_id>/` are runtime artifacts, not replacements for these authored docs.

## Start Here

- [Run and Operations Guide](run-and-operations.md)
- [Backend APIs and Tooling](backend-api-tools.md)
- [API Cookbook](api-cookbook.md)
- [Quality Gates and Benchmarks](quality-gates-and-benchmarks.md)
- [Reproducibility Capsule](reproducibility-capsule.md)
- [Sample Manifest and Outputs](sample-manifest.md)
- [Notebook Policy](../notebooks/NOTEBOOKS_POLICY.md)

## Backend and Runtime

- [Backend APIs and Tooling](backend-api-tools.md)
- [Strict Error Contract Reference](strict-errors-reference.md)
- [Model Assets and Data Sources](model-assets-and-data-sources.md)
- [Run and Operations Guide](run-and-operations.md)
- [Quality Gates and Benchmarks](quality-gates-and-benchmarks.md)
- [Performance Profiling Notes](performance-profiling-notes.md)
- [ETA Concept Drift Checks](eta-concept-drift.md)
- [CO2e Validation Notes](co2e-validation.md)

## Thesis / Evaluation / Reporting

- [VOI Thesis Pipeline Spec](voi-pipeline-spec.md)
- [Quality Gates and Benchmarks](quality-gates-and-benchmarks.md)
- [Reproducibility Capsule](reproducibility-capsule.md)
- [Tutorial Mode and Reporting](tutorial-and-reporting.md)
- [Sample Manifest and Outputs](sample-manifest.md)
- [Thesis-Grade Codebase Report](thesis-codebase-report.md)

Generated evaluation families described by these docs; the evaluator also emits cohort-scaffolded metadata (`evaluation_suite`, `cohort_scaffolding`) and metric-family scaffolding metadata (`metric_family_scaffolding`) in the run manifests and summary tables:

- `cohort_scaffolding_version = thesis_cohort_scaffolding_v2`
- `metric_family_scaffolding_version = thesis_metric_family_scaffolding_v1`

- broad cold thesis proof
- expanded broad-corpus generation, including the 120-row and 1200-row corpora plus the 10-shard 1200 split corpus
- sequential OD campaign with regression carry-forward
- campaign ledgers under `backend/out/thesis_campaigns/<campaign>/`
- focused REFC proof
- focused VOI proof
- DCCS diagnostic probe
- harder-story packet runs built from `hard_mixed_24` and `longcorr_hard_32`
- hot-rerun cold-source proof
- hot rerun proof
- composed suite outputs with cohort summaries, cohort composition, and suite-source provenance
- sharded report composition outputs with `shard_sources.json`
- run-to-run thesis comparison outputs from `backend/scripts/compare_thesis_runs.py`

Named cohort labels used by the evaluator include `collapse_prone`, `osrm_brittle`, `ors_brittle`, `refresh_sensitive`, `time_preserving_conflict`, `low_ambiguity_fast_path`, `preference_sensitive`, `support_fragile`, `audit_heavy`, and `proxy_friendly`.

Current checked-in result anchors used across the authored docs are:

- repaired broad-cold headline bundle: `backend/out/thesis_eval_core120_green_repaired/artifacts/thesis_eval_core120_green_repaired_20260404/`
- focused REFC and VOI bundles: `backend/out/artifacts/refc_focus_20260331_h2/` and `backend/out/artifacts/thesis_eval_20260331_r2_focused_voi/`
- hot-rerun pair: `backend/out/artifacts/hot_full_20260331_f2_cold/` and `backend/out/artifacts/hot_full_20260331_f2_hot/`
- harder-story / widening campaign bundles under `backend/out/thesis_campaigns/`
- staged but incomplete broadening assets under `backend/data/eval/uk_od_corpus_thesis_broad_expanded_1200.csv`, `backend/data/eval/thesis_shards_1200/`, and `backend/out/thesis_1200_s01_repo_local/`

## Frontend

- [Frontend Dev Tools Coverage](frontend-dev-tools.md)
- [Frontend Accessibility and i18n](frontend-accessibility-i18n.md)
- [Map Overlays and Tooltips](map-overlays-tooltips.md)
- [Tutorial Mode and Reporting](tutorial-and-reporting.md)

## Modeling / Math / Scenario Notes

- [Dissertation Math Overview](dissertation-math-overview.md)
- [Expanded Math Appendix](math-appendix.md)
- [Graph Theory Notes](appendix-graph-theory-notes.md)
- [Synthetic Incidents and Weather](synthetic-incidents-weather.md)
- [VOI Thesis Pipeline Spec](voi-pipeline-spec.md)
- [CO2e Validation Notes](co2e-validation.md)

## Examples and Operator References

- [API Cookbook](api-cookbook.md)
- [Sample Manifest and Outputs](sample-manifest.md)
- [Backend APIs and Tooling](backend-api-tools.md)
- [Run and Operations Guide](run-and-operations.md)

## Maintenance

Tracked docs checks from repo root:

```powershell
python scripts/check_docs.py
python scripts/check_docs.py --check-links
python scripts/check_docs.py --check-orphans
python scripts/check_docs.py --check-paths
python scripts/check_docs.py --check-endpoints
```

Local docs viewer:

```powershell
.\scripts\serve_docs.ps1
```

Strict runtime preflight before backend startup or deploy checks:

```powershell
uv run --project backend python backend/scripts/preflight_live_runtime.py
```

## Notes

- `docs/thesis-codebase-report.md` is intentionally maintained separately from the general docs refresh cycle.
- Generated reports such as thesis_report.md, methods_appendix.md, hot_rerun_report.md, or hot_rerun_gate.json under `backend/out/artifacts/<run_id>/` should be cited as runtime artifacts, not edited as authored docs.
- The latest broad corpus asset is currently `backend/data/eval/uk_od_corpus_thesis_broad_expanded_1200.csv`, but the 1200-row evaluation itself is not yet a completed thesis-result bundle.
- Signed run metadata and artifacts are consumed through `GET /runs/{run_id}/manifest`, `GET /runs/{run_id}/signature`, `GET /runs/{run_id}/artifacts`, and `GET /runs/{run_id}/artifacts/{artifact_name}`.
- Dedicated hot-rerun reuse proofs are separate from cold thesis proofs and can restore cache state through `POST /cache/hot-rerun/restore` before benchmarking.

## Related Docs

- [Run and Operations Guide](run-and-operations.md)
- [Quality Gates and Benchmarks](quality-gates-and-benchmarks.md)
- [Reproducibility Capsule](reproducibility-capsule.md)
- [Sample Manifest and Outputs](sample-manifest.md)
