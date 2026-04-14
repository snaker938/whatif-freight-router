# Evaluation Card

This page is the evaluator-facing reference for the current thesis-style backend lanes, cohort labels, and checked artifacts.

It is intentionally conservative. It explains what the evaluator covers, where to look for current evidence, and what is still only partially evidenced or explicitly scaffolded.

## Purpose

The evaluator in this repository is not a single monolithic benchmark. It is a set of named lanes and evidence bundles that support different parts of the thesis pipeline:

- DCCS diagnostics
- REFC certification and fragility
- VOI controller behavior
- replay and hot-rerun checks
- preference, support, and calibration surfaces
- transfer and synthetic sanity checks

Use this page as a map, not as a claim of completion.

## Lane Definitions

The current evaluator vocabulary is now explicit in `backend/scripts/run_thesis_evaluation.py`. The lane names below are present suite roles in source. That does not mean the corresponding redesign gates are closed; it only means the evaluator role registry is explicit and machine-readable.

| Lane label used in docs | Status in inspected evaluator source | Evidence basis | Publication-safe wording |
| --- | --- | --- | --- |
| `broad cold proof` | present | `run_thesis_evaluation.py` role `broad_cold_proof` | explicit suite role |
| `focused REFC proof` | present | `run_thesis_evaluation.py` role `focused_refc_proof` | explicit suite role |
| `focused VOI proof` | present | `run_thesis_evaluation.py` role `focused_voi_proof` | explicit suite role |
| `DCCS diagnostic probe` | present | `run_thesis_evaluation.py` role `dccs_diagnostic_probe` | explicit suite role |
| `hot-rerun cold-source proof` | present as a docs synonym | `run_thesis_evaluation.py` role `hot_rerun_cold_source` | explicit suite role; keep the code label visible |
| `hot-rerun proof` | present as a checked reviewer companion bundle | `run_thesis_evaluation.py` role `hot_rerun`, plus checked reviewer companion bundle `out/headline_exports/current_checked/full_suite_curated_latest_20260411_hot_rerun_hot/` | explicit suite role with a checked local runtime-observability/reuse bundle; keep the hot-rerun gate evidence visible |
| `preference proof` | present | `run_thesis_evaluation.py` role `preference_proof`, plus checked full-suite root `out/headline_exports/current_checked/full_suite_curated_latest_20260411/` | explicit suite role; the checked full-suite root now carries the seed-repeat and claim-discipline closure surfaces via `headline_seed_claims_summary.*`, `sample_size_gate_summary.*`, and `out/headline_exports/current_checked/full_suite_curated_latest_20260411/publishability_verdict.json` |
| `optional-stopping coverage` | present | `run_thesis_evaluation.py` role `optional_stopping_coverage`, plus checked full-suite root `out/headline_exports/current_checked/full_suite_curated_latest_20260411/` | explicit suite role; the checked full-suite root now carries direct row-count sample-size rows for the maintained `G11.54` contract, with `evaluation_requirement_observed_count` sourced from the emitted observed counts rather than a synthesized lower bound |
| `proxy-audit calibration` | present | `run_thesis_evaluation.py` role `proxy_audit_calibration`, plus checked full-suite root `out/headline_exports/current_checked/full_suite_curated_latest_20260411/` | explicit suite role with a `3` bias regime x `3` audit-budget level x `2` support-condition grid; the checked full-suite root now records the maintained row-count requirement for `G11.56` directly in `sample_size_gate_summary.*`, alongside the publishability verdict and failure-atlas closure surfaces |
| `perturbation / flip-radius` | present | `run_thesis_evaluation.py` role `perturbation_flip_radius` | explicit suite role; gate evidence still needs concrete checked bundles |
| `threshold sensitivity` | present | `run_thesis_evaluation.py` role `threshold_sensitivity`, plus checked reviewer companion bundle `out/headline_exports/current_checked/full_suite_curated_latest_20260411_threshold_sensitivity/` | explicit suite role for one-factor-at-a-time sweeps over certificate threshold, fast-path threshold, and certified-set cap; the checked companion bundle now cites `out/headline_exports/current_checked/full_suite_curated_latest_20260411_threshold_sensitivity/lane_metadata.json`, `out/headline_exports/current_checked/full_suite_curated_latest_20260411_threshold_sensitivity/threshold_sensitivity_summary.csv`, `out/headline_exports/current_checked/full_suite_curated_latest_20260411_threshold_sensitivity/threshold_sensitivity_summary.json`, `out/headline_exports/current_checked/full_suite_curated_latest_20260411_threshold_sensitivity/threshold_sensitivity_report.md`, and the `threshold_sensitivity_vs_variant` plot family as concrete local artifacts |
| `public transfer` | present | `run_thesis_evaluation.py` role `public_transfer`, plus checked reviewer companion bundle `out/headline_exports/current_checked/full_suite_curated_latest_20260411_public_transfer/` | explicit suite role; the checked companion bundle now cites both corridor-family and weather-regime transfer slices via `thesis_summary_by_transfer_slice.*`, `thesis_summary_by_weather_regime_transfer_slice.*`, and the paired `leave_one_corridor_family_out_transfer_vs_variant` / `leave_one_weather_regime_out_transfer_vs_variant` plot families; this is concrete evaluator coverage, not a claim that the transfer-size gates are green |
| `synthetic ground-truth` | present | `run_thesis_evaluation.py` role `synthetic_ground_truth`, plus checked full-suite root `out/headline_exports/current_checked/full_suite_curated_latest_20260411/` | explicit suite role; the checked full-suite root now treats `G11.53` as a real sample-size row with explicit requirement metadata rather than a stale false-green aggregate |

These lane names are the right vocabulary for the evaluator story because they are explicit suite roles in the current source. Treat them as implemented evaluator roles whose checked closure evidence now lives either in the lane-specific reviewer companion bundles or in the checked full-suite root for publishability, sample-size, failure-atlas, and seed-claim reporting.

## Additive Checked Lane Metadata

In addition to the source-registered suite roles above, the current checked campaign bundle and the full-suite regeneration path now carry metadata-backed lane surfaces for reviewer-facing checks. These are named checked-lane surfaces for reviewers; they are not claims that `backend/scripts/run_thesis_evaluation.py` already exposes new suite roles with the same names.

| Lane label used in docs | Status in checked artifacts | Evidence basis | Publication-safe wording |
| --- | --- | --- | --- |
| `baseline fairness audit` | present as checked metadata lane; the older campaign slice failed, and the current full-suite bundle now reports `fairness_failure_count = 0` | `out/headline_exports/current_checked/full_suite_curated_latest_20260411/universal_baseline_audit.csv` plus `out/headline_exports/current_checked/full_suite_curated_latest_20260411/universal_baseline_audit.json`, with `out/headline_exports/current_checked/full_suite_curated_latest_20260411/osrm_baseline_identity_manifest.json` and `out/headline_exports/current_checked/full_suite_curated_latest_20260411/ors_baseline_identity_manifest.json` copied into the same checked bundle root | explicit checked audit of matched OD, departure, vehicle, restrictions, and route-feasibility context for the current headline OSRM/ORS comparison slice; the fairness slice is now clear in the checked bundle, `P14.32` has manifest attachments in the bundle, and `P9.5` is supported by the preserved graph date, graph digest, image/config identity, and source graph metadata in those manifests, though the broader comparator/report claim discipline still needs to remain conservative |
| `failure atlas` | present as generated full-suite checked-lane surface with a cited checked reviewer companion bundle | `out/headline_exports/current_checked/full_suite_curated_latest_20260411/failure_atlas_lane_metadata.json` plus `out/headline_exports/current_checked/full_suite_curated_latest_20260411/failure_atlas.json` and `out/headline_exports/current_checked/full_suite_curated_latest_20260411/failure_atlas.md` | explicit named full-suite lane surface collecting `wrong_singleton`, `support_downgrade`, and `abstention` exemplars across focused lanes; the current checked lane metadata records `lane_status = present_complete`, required-kind counts of `wrong_singleton = 82`, `support_downgrade = 82`, and `abstention = 40`, so the atlas family is complete and green |

## Cohort Semantics

The current evaluator and report surfaces reference the following cohorts or cohort-like slices:

- `collapse_prone`
- `osrm_brittle`
- `ors_brittle`
- `refresh_sensitive`
- `time_preserving_conflict`
- `low_ambiguity_fast_path`
- `preference_sensitive`
- `support_fragile`
- `audit_heavy`
- `proxy_friendly`

These cohort labels are useful because they keep the evidence organized around failure-sensitive cases rather than only around aggregate averages. `support_fragile` is a derived support-richness slice, not a mutually exclusive cohort class, and it can overlap with other cohort labels when a row is both support-fragile and another labeled case under the current evaluator thresholding.

In the current evaluator source, these cohort summaries are backed by an explicit cohort registry. Rows can enter the registry either by raw `corpus_group`/`corpus_kind` labels or by derived heuristics for slices such as `support_fragile`, `collapse_prone`, `refresh_sensitive`, `time_preserving_conflict`, `low_ambiguity_fast_path`, and `controller_stress`. Not every checked bundle populates every cohort.

What is currently evidenced:

- the docs and report explicitly name these cohorts as part of the evaluation story
- the quality-gates page records current local artifacts and thesis bundles
- the claim matrix and theorem map distinguish scaffold-only surfaces from empirical ones

What is not yet evidenced as a blanket claim:

- that every cohort meets every headline threshold
- that every lane has the same maturity or the same completeness
- that cohort behavior generalizes outside the checked bundles

## What The Evaluator Currently Shows

The strongest current evidence is recorded in the following places:

- `docs/quality-gates-and-benchmarks.md`
  - latest local validation snapshot
  - CI lane definitions
  - minimum acceptance gates
  - quality thresholds
- `docs/thesis-codebase-report.md`
  - thesis-bundle evidence
  - limitations and scope notes
  - what the report does not overclaim
- `docs/sample-manifest.md`
  - run outputs and artifact bundle shape
  - manifest and report file names
- `docs/reproducibility-capsule.md`
  - repro controls and artifact provenance anchors
- `docs/claim_matrix.md`
  - current surfaces marked `scaffold-only`, `empirical`, or `theorem-backed`
- `docs/theorem_map.md`
  - theorem slots that are still open

The evaluator is therefore best understood as a set of evidence surfaces with different maturity levels, not as a single all-green scorecard.

## What Is Evidenced Now

The current repo supports the following evaluator-facing facts:

- strict preflight and readiness checks are present and produce checked local artifacts
- some thesis evaluation bundles exist with report and summary outputs, especially focused-VOI and campaign snapshots cited elsewhere in the docs
- the inspected evaluator source explicitly defines suite roles for broad cold, focused REFC, focused VOI, DCCS diagnostic probe, hot-rerun cold source, hot rerun, preference proof, optional-stopping coverage, proxy-audit calibration, perturbation / flip-radius, public transfer, and synthetic ground truth
- current run-store artifacts include DCCS, REFC, selective-certification, VOI, support, and preference summaries
- the checked focused-VOI bundle exposes selective-certification metrics `certificate_selectivity_rate`, `certificate_selectivity_denominator`, and `broad_hard_case_certificate_selectivity_rate` in `thesis_summary.*`, `thesis_plots.json` (`hard_case_transfer_vs_variant`), and `thesis_report.md`
- the current checked full-suite bundle now carries `out/headline_exports/current_checked/full_suite_curated_latest_20260411/universal_baseline_audit.csv` and `.json`, plus the copied `out/headline_exports/current_checked/full_suite_curated_latest_20260411/osrm_baseline_identity_manifest.json` and `out/headline_exports/current_checked/full_suite_curated_latest_20260411/ors_baseline_identity_manifest.json` attachment files for the headline OSRM/ORS comparison slice; the older campaign-backed `backend/out/thesis_campaigns/dominance_cluster5_cardiff_bath_corr12p5_r2/tranche_001/artifacts/dominance_cluster5_cardiff_bath_corr12p5_r2_t001/baseline_fairness_audit.json` remains a historical contrast and is not the current proof surface
- current thesis-evaluation source registers lane purpose, row-count target wiring, cohort/support-bin composition references, and any configured headline seed-repeat plan, but the checked bundles in this slice should not be read as proving a per-run lane-metadata artifact unless one is cited explicitly
- the checked public-transfer reviewer companion at `out/headline_exports/current_checked/full_suite_curated_latest_20260411_public_transfer/` now cites both transfer-generalization surfaces explicitly: the corridor-family slice through `thesis_summary_by_transfer_slice.csv` / `.json` and `thesis_plots.json` family `leave_one_corridor_family_out_transfer_vs_variant`, plus the weather-regime slice through `thesis_summary_by_weather_regime_transfer_slice.csv` / `.json` and `thesis_plots.json` family `leave_one_weather_regime_out_transfer_vs_variant`
- the checked hot-rerun reviewer companion at `out/headline_exports/current_checked/full_suite_curated_latest_20260411_hot_rerun_hot/` now cites the runtime-observability and reuse family explicitly: `out/headline_exports/current_checked/full_suite_curated_latest_20260411_hot_rerun_hot/hot_rerun_gate.json`, `out/headline_exports/current_checked/full_suite_curated_latest_20260411_hot_rerun_hot/hot_rerun_vs_cold_comparison.json`, `out/headline_exports/current_checked/full_suite_curated_latest_20260411_hot_rerun_hot/thesis_metrics.json`, `out/headline_exports/current_checked/full_suite_curated_latest_20260411_hot_rerun_hot/thesis_plots.json`, and `out/headline_exports/current_checked/full_suite_curated_latest_20260411_hot_rerun_hot/thesis_report.md` are available as a checked local bundle, and the bundle now records `controller_reuse_reporting` for variant `C` with the pair-benchmark hot-rerun gate all green on its own
- the checked threshold-sensitivity reviewer companion at `out/headline_exports/current_checked/full_suite_curated_latest_20260411_threshold_sensitivity/` now cites `out/headline_exports/current_checked/full_suite_curated_latest_20260411_threshold_sensitivity/lane_metadata.json`, `out/headline_exports/current_checked/full_suite_curated_latest_20260411_threshold_sensitivity/threshold_sensitivity_summary.csv`, `out/headline_exports/current_checked/full_suite_curated_latest_20260411_threshold_sensitivity/threshold_sensitivity_summary.json`, `out/headline_exports/current_checked/full_suite_curated_latest_20260411_threshold_sensitivity/threshold_sensitivity_report.md`, and the `threshold_sensitivity_vs_variant` plot family as concrete local artifacts
- the repo now also carries `backend/scripts/run_full_latest_suite.py` as the publishability-facing regeneration entrypoint, and the latest checked full-suite assessment bundle (`full_suite_curated_latest_20260411`) now reports `publishable_on_current_evidence = false`, `adoption_claim_supported = false`, `sample_size_failure_count = 0`, `fairness_failure_count = 0`, `optional_stopping_gate_failure_count = 0`, `perturbation_gate_failure_count = 0`, and `hot_rerun_all_green = true`, with remaining blockers `dccs_hard_gates_not_all_green`, `refine_cost_forecast_gates_not_all_green`, and `voi_hard_gates_not_all_green`
- the checked full-suite reviewer companion at `out/headline_exports/current_checked/full_suite_curated_latest_20260411/` now also carries `out/headline_exports/current_checked/full_suite_curated_latest_20260411/failure_atlas_lane_metadata.json`, whose current payload records `lane_status = present_complete`, `required_kind_counts = { wrong_singleton: 82, support_downgrade: 82, abstention: 40 }`, and root-cause family counts that now include support-downgrade coverage; the atlas family is complete and green
- when a headline lane is actually rerun across multiple configured seeds, the runner emits dedicated seed-summary, BCa-bootstrap CI, Holm-adjusted claim-summary, and report-table artifacts that aggregate the repeated runs, report between-seed spread, and flag sign disagreement or claim-narrowing conditions instead of only recording the seed plan
- those repeated-run artifacts also include a reviewer-facing summary table derived from the structured claim summary, but only when the repeated runs themselves are present
- the docs deliberately separate scaffold-only surfaces from empirical ones

For maintained docs consumption of those repeated-run outputs:

- use the BCa-bootstrap CI artifact for point estimate, paired delta, and interval reading
- use the Holm-adjusted claim-summary artifact for adjusted p-value reading
- if a CI crosses zero or the claim summary flags claim narrowing, treat the comparison as inconclusive or narrowed rather than positive; report-table exports remain downstream presentation surfaces only
- read point estimate and paired delta together with the emitted repeated-run sample-size context: `seed count` plus `paired rows / seed`; repeated seeds are not pooled into one larger paired sample
- read the 95% BCa CI together with the emitted bootstrap method and resample count; the current runner uses BCa bootstrapping with `10,000` resamples when repeated-run artifacts are present
- read effect size as the between-seed standardized effect size for the headline metric; if effect size is blank, that means between-seed spread was zero rather than an automatically null or gate-closing result

Current artifact examples referenced by the docs:

- `backend/out/model_assets/preflight_live_runtime.json`
- `backend/out/model_assets/routing_graph_coverage_report.json`
- `backend/out/thesis_campaigns/*/campaign_report.md`
- `backend/out/thesis_campaigns/*/thesis_summary.json`
- `backend/out/thesis_campaigns/*/thesis_metrics.json`
- `backend/out/thesis_campaigns/*/methods_appendix.md`
- `backend/out/thesis_campaigns/*/evaluation_manifest.json`
- `backend/out/artifacts/thesis_eval_20260331_r2_focused_voi/index.json` and `backend/out/artifacts/thesis_eval_20260331_r2_focused_voi/index.md` as additively backfilled bundle-level inspection entrypoints for artifact presence/status only; they do not create new headline surfaces or imply committed PDF/SVG renders

## What Is Not Yet Green

Do not treat this repository slice as fully green on every published gate.

The current docs still acknowledge:

- limitations in scoped or universal claims
- support-sensitive failures that can still fail closed
- bounded local evidence rather than universal generalization
- theorem-map rows that still distinguish theorem-grade maturity from checked empirical gate closure
- the latest checked full-suite assessment bundle now cites the newly explicit suite roles in its publishability-facing summaries, and the repaired root bundle keeps those role records synchronized with the current red checked verdict in `out/headline_exports/current_checked/full_suite_curated_latest_20260411/publishability_verdict.json`
- the latest checked full-suite assessment bundle is not publishable on current evidence and not adoption-ready: headline adoption checks are green and evaluation-size requirements are met, but the suite verdict still carries `dccs_hard_gates_not_all_green`, `refine_cost_forecast_gates_not_all_green`, and `voi_hard_gates_not_all_green`. The separate pair-benchmark hot-rerun bundle remains green on its own, and the repaired suite verdict now agrees that hot rerun is no longer a blocker
- the current checked bundle now closes the maintained sample-size `G11.53`, `G11.54`, and `G11.56` rows, and `sample_size_gate_summary.*` records the gate-driving observed counts directly instead of relying on a reviewer-safe lower-bound fallback
- the reviewer package now also exports runtime-observability, runtime-action-observability, runtime-stage-quantiles, and preference-burden surfaces for the checked broad-cold and focused-VOI bundles, including `figure.latest_checked_campaign.runtime_distribution_vs_variant`, `figure.latest_checked_campaign.runtime_breakdown_vs_variant`, `figure.latest_checked_campaign.runtime_stage_quantiles_vs_stage`, `table.latest_checked_campaign.runtime_observability_summary`, `table.latest_checked_campaign.runtime_action_observability_summary`, `table.latest_checked_campaign.runtime_stage_quantiles`, `table.focused_voi.preference_burden_summary`, and `table.focused_voi.preference_burden_by_cohort`; the checked bundle surfaces zero preference query counts on the available rows, the runtime-action table exposes budget-used and budget-utilization proxies plus VOI action-family counts and action-family budget shares, and the runtime tables now ship paired CSV and JSON source companions for consistency, the runtime-action table already exposes exact fast-path precision/recall and denominators, the hot-rerun bundle's `thesis_metrics.json` reports `artifact_generation_ms` at lane scope, and the checked hot-rerun `thesis_summary.*` / `thesis_metrics.json` surfaces now carry populated peak RSS/VMS summary fields for `A`, `B`, and `C`, so `P14.47`, `P14.49`, and `P14.50` are supportable on current checked evidence
- headline lane seed-repeat plans may now be configured and emitted in metadata, but a configured plan is not the same thing as completed 3-seed evidence
- repeated seed-summary, BCa-bootstrap CI, Holm-adjusted claim-summary, and report-table artifacts only count as stronger evidence when the repeated runs themselves are present; otherwise the repository remains at plan-level scaffolding
- bootstrap-ready reviewer summaries and report-table exports improve claim-discipline reporting, but they do not by themselves satisfy BCa-sample-size, multiple-testing, or headline-table requirements
- multi-seed, bootstrap-CI, Holm-adjusted, calibration-ECE, witness-sparsity, and failure-atlas-completeness claims remain open unless separately evidenced

That means this page should be used to orient reviewers, not to imply a completed proof package.

## Where Reviewers Should Look

For current evidence, reviewers should start with:

1. `docs/quality-gates-and-benchmarks.md` for the latest local validation and gate vocabulary
2. `docs/thesis-codebase-report.md` for the thesis narrative, limitations, and checked bundle references
3. `docs/sample-manifest.md` for artifact bundle names and retrieval flow
4. `docs/reviewer_quickstart.md` for the current checked reviewer bundle and focused-VOI source-surface workflow
5. `docs/reproducibility-capsule.md` for provenance and repro anchors
6. `docs/claim_matrix.md` and `docs/theorem_map.md` for claim status and proof gaps

For backend scripts, the evaluator-facing entry points are currently:

- `backend/scripts/preflight_live_runtime.py`
- `backend/scripts/score_model_quality.py`
- `backend/scripts/benchmark_model_v2.py`
- `backend/scripts/benchmark_batch_pareto.py`
- `backend/scripts/validate_graph_coverage.py`
- `backend/scripts/run_thesis_evaluation.py`
- `backend/scripts/run_thesis_lane.py`
- `backend/scripts/run_full_latest_suite.py`

## Suggested Review Framing

When reviewing evaluator output, use these questions:

- Which lane produced the artifact?
- Which cohort slice does the row or bundle belong to?
- Is the evidence a checked artifact, a scaffold-only surface, or a theorem-backed claim?
- Does the doc point to a concrete file in `backend/out/` or only to a conceptual surface?
- Are the claims scoped to the observed bundle, or do they drift toward universal language?

## Bottom Line

The evaluator is already rich enough to support thesis-style review, but not every lane or cohort is fully mature.

The safe publication reading is:

- the lanes are explicit
- the cohorts are explicit
- the artifact paths are explicit
- the claim boundaries are still conservative

That is the right level of honesty for the current repository state.

## Related Docs

- [Quality Gates and Benchmarks](quality-gates-and-benchmarks.md)
- [Reviewer Quickstart](reviewer_quickstart.md)
- [Claim Matrix](claim_matrix.md)
- [Theorem Map](theorem_map.md)
