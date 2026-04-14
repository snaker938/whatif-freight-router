# Quality Gates and Benchmarks

Last Updated: 2026-04-11
Applies To: `backend/scripts/preflight_live_runtime.py`, `backend/scripts/score_model_quality.py`, `backend/scripts/benchmark_model_v2.py`, `backend/scripts/benchmark_batch_pareto.py`, `backend/scripts/validate_graph_coverage.py`, `backend/scripts/run_full_latest_suite.py`, thesis evaluation artifacts under `backend/out/thesis_campaigns/*`, and CI lanes in [.github/workflows/backend-ci.yml](../.github/workflows/backend-ci.yml)

This page defines operational backend gates used locally and in CI, and records the latest local evidence currently present in the repo. It does not by itself certify any redesign-complete `G11.*` gate or any publishability `P14.*` item as green.

## Core Gate Sequence

From `backend/`:

```powershell
uv run python scripts/build_model_assets.py
uv run python scripts/preflight_live_runtime.py
uv run python scripts/score_model_quality.py
uv run python scripts/benchmark_model_v2.py --iterations 8 --p95-gate-ms 2000
```

Targeted subsystem scoring examples:

```powershell
uv run python scripts/score_model_quality.py --subsystem fuel_price
uv run python scripts/score_model_quality.py --subsystem scenario_profile
uv run python scripts/score_model_quality.py --subsystem stochastic_sampling
uv run python scripts/score_model_quality.py --subsystem toll_classification
```

Batch and thesis-oriented benchmark helpers:

```powershell
uv run python scripts/benchmark_batch_pareto.py --mode inprocess-fake --pair-count 100 --seed 20260212
uv run python scripts/run_thesis_evaluation.py --corpus-csv data/eval/uk_od_corpus_thesis_broad.csv
uv run python scripts/run_full_latest_suite.py
uv run python scripts/validate_graph_coverage.py
```

## Latest Local Validation

### Strict preflight evidence

`backend/out/model_assets/preflight_live_runtime.json` records a successful strict run at `2026-04-11T03:06:27Z` with:

- `required_ok: true`
- `required_failure_count: 0`
- scenario profile version `scenario_profiles_uk_v2_live`
- scenario live-context coverage `1.0` for WebTRIS, Traffic England, DfT, Open-Meteo, and overall coverage
- 384 scenario contexts recorded by preflight
- 220 toll tariff rules and 28 toll-topology segments
- 18 stochastic regimes and 11 departure-profile regions
- 134 bank holidays
- carbon policy `0.101 GBP/kg` with scope-adjusted emissions factor `1.121`
- OSRM smoke route `189471.0 m / 8794.2 s`
- ORS smoke route `203868.1 m / 12280.8 s`, engine version `9.7.1`, graph identity `graph_identity_verified`

### Routing-graph evidence

`backend/out/model_assets/routing_graph_coverage_report.json` currently records:

- 16,782,614 nodes
- 17,271,476 edges
- graph size about `4123.27 MB`
- worst fixture nearest-node gap `2545.053 m`
- bounding box `lat 49.75..61.1`, `lon -8.75..2.25`
- `coverage_passed: true`

### Latest thesis-lane benchmark evidence

The newest checked thesis campaign bundle is `backend/out/thesis_campaigns/dominance_cluster5_cardiff_bath_corr12p5_r2`.

This thesis section is descriptive only. The snapshots below are checked local bundles, not gate-closing proof runs. Treat every `G11.*` and `P14.*` item as open unless this page cites the exact artifact path, measured value, required threshold, and required sample size for that item.

Campaign-level validity:

- 20 evaluation rows and 4 summary rows
- `scenario_profile_unavailable_rate: 0.0`
- `strict_live_readiness_pass_rate: 1.0`
- `evaluation_rerun_success_rate: 0.8`
- backend ready wait `161.38 ms`
- route-graph warmup elapsed `10000.0 ms`

Baseline smoke from `backend/out/thesis_campaigns/dominance_cluster5_cardiff_bath_corr12p5_r2/tranche_001/artifacts/dominance_cluster5_cardiff_bath_corr12p5_r2_t001/thesis_metrics.json`:

- OSRM `140.54 ms`, `189.471 km`, `13306.17 s`
- local ORS `170.91 ms`, `203.868 km`, `18581.61 s`

Variant summary from `backend/out/thesis_campaigns/dominance_cluster5_cardiff_bath_corr12p5_r2/tranche_001/artifacts/dominance_cluster5_cardiff_bath_corr12p5_r2_t001/thesis_summary.json`:

| Variant | Mean runtime ms | Mean algorithm ms | Mean hypervolume | Safe reading |
| --- | --- | --- | --- | --- |
| `V0` | `9176.741` | `8506.491` | `1954475824.173189` | local matched-budget legacy snapshot |
| `A` | `5315.91825` | `4645.66825` | `2234025023.091551` | single-bundle runtime is lower than `V0`; not a passed runtime gate |
| `B` | `5787.824` | `5117.574` | `2234025023.091551` | single-bundle certificate snapshot `0.950546`; not calibration or publishability proof |
| `C` | `2794.579` | `2124.329` | `2234025023.091551` | single-bundle fastest runtime among these four variants; not a seed-robust headline result |

Across `A`, `B`, and `C`, the current bundle records the same mean objective deltas versus `V0`:

- weighted-margin gain `1.5225`
- balanced-gain delta `0.015621`
- duration gain `1227.32 s`
- monetary gain `2.47`
- emissions gain `-1.4795 kg`

These thesis runtimes are not the same thing as the `backend/scripts/benchmark_model_v2.py` p95 gate. They include thesis-run orchestration, baseline acquisition, certification, and startup overhead.

In the inspected evaluator source this turn, explicit suite roles are `broad_cold_proof`, `focused_refc_proof`, `focused_voi_proof`, `dccs_diagnostic_probe`, `hot_rerun_cold_source`, `hot_rerun`, `preference_proof`, `optional_stopping_coverage`, `proxy_audit_calibration`, `perturbation_flip_radius`, `threshold_sensitivity`, `public_transfer`, and `synthetic_ground_truth`. The `proxy_audit_calibration` role now also spells out its source-level `3` bias regime x `3` audit-budget level x `2` support-condition cell structure. Checked local reviewer companions now also exist for `threshold_sensitivity` and `public_transfer` under `out/headline_exports/current_checked/full_suite_curated_latest_20260411_threshold_sensitivity/` and `out/headline_exports/current_checked/full_suite_curated_latest_20260411_public_transfer/`. The threshold companion cites `out/headline_exports/current_checked/full_suite_curated_latest_20260411_threshold_sensitivity/lane_metadata.json`, `out/headline_exports/current_checked/full_suite_curated_latest_20260411_threshold_sensitivity/threshold_sensitivity_summary.csv`, `out/headline_exports/current_checked/full_suite_curated_latest_20260411_threshold_sensitivity/threshold_sensitivity_summary.json`, `out/headline_exports/current_checked/full_suite_curated_latest_20260411_threshold_sensitivity/threshold_sensitivity_report.md`, and the `threshold_sensitivity_vs_variant` plot family. The public-transfer companion cites both `out/headline_exports/current_checked/full_suite_curated_latest_20260411_public_transfer/thesis_summary_by_transfer_slice.*` and `out/headline_exports/current_checked/full_suite_curated_latest_20260411_public_transfer/thesis_summary_by_weather_regime_transfer_slice.*` together with the paired transfer plot families. Treat those checked companions as evidence that the evaluator lanes and maintained surfaces exist; do not treat them as green `G11.*` or `P14.*` proof unless this page also cites the required sample sizes and thresholds.

The current checked `proxy_audit_calibration` bundle is also explicit about what it is not: the `C` row is proxy-only, with `mean_audit_world_count = 0.0`, `mean_audited_route_pair_count = 0.0`, `proxy_only_fraction = 1.0`, `weak_overlap_detected_rate = 0.0`, `positivity_ok_rate = 0.0`, and `proxy_audit_calibration_in_support_ece = 0.054062`. Read that as support for the calibration/surface machinery, including the `P5.1` calibration-plot surface and the `P14.11`, `P14.12`, `P14.15`, and `P14.16` plot/metric rows, but not as a closed audit lane. `P14.13` and `P14.14` remain open because the checked `C` row does not satisfy the required synthetic and real in-support ECE thresholds. The audit-specific `P5.2` and `P5.4` rows remain open as well because the checked bundle does not contain a positive audited-overlap heatmap or a deliberately adversarial low-overlap audited slice.

Fresh publishability-facing regeneration is now handled by `backend/scripts/run_full_latest_suite.py`. The latest checked full-suite assessment bundle (`full_suite_curated_latest_20260411`) already carries lane-publishability summaries, universal-baseline audit summaries, sample-size gate summaries, headline seed-claim summaries, `out/headline_exports/current_checked/full_suite_curated_latest_20260411/failure_atlas_lane_metadata.json`, failure-atlas files, a publishability-verdict JSON artifact, and a publishability-assessment Markdown report. The local reviewer companion copy of that bundle now lives under `out/headline_exports/current_checked/full_suite_curated_latest_20260411/`, where `out/headline_exports/current_checked/full_suite_curated_latest_20260411/failure_atlas_lane_metadata.json` currently records `lane_status = present_complete`, `required_kind_counts = { wrong_singleton: 82, support_downgrade: 82, abstention: 40 }`, `required_kind_presence = { wrong_singleton: true, support_downgrade: true, abstention: true }`, `counts_by_kind = { wrong_singleton: 0, support_downgrade: 82, abstention: 0, certified_set_violation: 0, route_failure: 0 }`, `counts_by_support_status = { unsupported: 82 }`, and `root_cause_family_counts = { support_failure: 82, hidden_challenger: 0, proxy_bias: 0, preference_ambiguity: 0, budget_cut: 0, other: 0 }`. The current checked suite is not green on the checked verdict surfaces: `publishable_on_current_evidence = false`, `adoption_claim_supported = false`, `hot_rerun_all_green = true`, `sample_size_failure_count = 0`, `fairness_failure_count = 0`, `optional_stopping_gate_failure_count = 0`, and `perturbation_gate_failure_count = 0`. The remaining publishability blockers are `dccs_hard_gates_not_all_green`, `refine_cost_forecast_gates_not_all_green`, and `voi_hard_gates_not_all_green`. The failure-atlas family is complete in the checked artifacts, but the bundle is not publishable on current evidence.

If a cited lane-registration surface shows a configured headline seed-repeat plan, read that as scaffolding only. It does not satisfy `P14.7-P14.10` unless separate repeated runs and their aggregated outputs are actually present.

When repeated headline runs are actually executed, the runner now writes dedicated seed-summary artifacts, BCa-bootstrap CI summaries, Holm-adjusted claim summaries, reviewer-facing repeated-seed summaries, and report-table exports for those repeated seeds. Those artifacts improve reporting readiness by exposing between-seed spread, confidence-interval crossings, multiple-comparison adjustments, and claim-narrowing warnings, but they still do not make a gate green unless the underlying repeated runs hit the required row counts and thresholds.

- Use the BCa-bootstrap CI summary as the maintained surface for point estimate, paired delta, and interval reading.
- Use the Holm-adjusted claim summary as the maintained surface for multiple-comparison adjusted p-value reading.
- If a CI crosses zero or the claim summary flags claim narrowing, read the comparison as inconclusive or narrowed rather than positive. Report-table exports are downstream presentation surfaces and do not by themselves make a gate green.
- Read point estimate and paired delta together with the emitted repeated-run sample-size context: `seed count` plus `paired rows / seed`. Repeated seeds are not pooled into one larger paired sample.
- Read the 95% BCa CI together with the emitted bootstrap method and resample count; the current runner uses BCa bootstrapping with `10,000` resamples when repeated-run artifacts are present.
- Read effect size as the between-seed standardized effect size for the headline metric. If effect size is blank, that means between-seed spread was zero, not that the comparison is automatically null or gate-closing.

For the local reviewer slice, repeated-run evidence is still not directly consumable from the focused bundle alone. The reviewer-indexed focused bundle at `backend/out/artifacts/thesis_eval_20260331_r2_focused_voi/` records `20` requested OD rows and `80` result rows in `evaluation_manifest.json`, and the latest checked campaign bundle remains a single checked tranche with `20` evaluation rows. Those are evaluator row counts, not repeated-seed sample sizes. The newer checked full-suite assessment, however, now does consume repeated headline runs and reports `headline_seed_failure_count = 0`. Even with that stronger seed evidence and a green sample-size summary, the suite is still not publishable on current evidence because the checked verdict carries `dccs_hard_gates_not_all_green`, `refine_cost_forecast_gates_not_all_green`, and `voi_hard_gates_not_all_green`. So the right reading is narrower than before: repeated-seed evidence now exists at full-suite level, but it does not override the remaining hard-gate blockers.

The maintained baseline-fairness reviewer lane in the latest checked campaign remains failed. Do not treat that campaign-backed OSRM/ORS reviewer slice as a clean adoption or publication green-light. Separately, the newer checked full-suite assessment now reports `fairness_failure_count = 0`, `sample_size_failure_count = 0`, `headline_claim_narrowing_count = 0`, `optional_stopping_gate_failure_count = 0`, `perturbation_gate_failure_count = 0`, and `hot_rerun_all_green = true`, with `publishability_blockers = [dccs_hard_gates_not_all_green, refine_cost_forecast_gates_not_all_green, voi_hard_gates_not_all_green]`. So the current checked suite no longer carries fairness, sample-size, optional-stopping, perturbation, claim-narrowing, or hot-rerun failures, but it still carries DCCS, refine-cost, and VOI blockers. The copied OSRM/ORS baseline identity manifests in that full-suite bundle preserve graph date, graph digest, image/config identity, and source graph metadata, which is the checked packaging evidence behind `P9.5` and the manifest-attachment portion of `P14.32`.

The root reviewer index now also names `table.focused_voi.preference_burden_summary` and `table.focused_voi.preference_burden_by_cohort` as maintained source surfaces backed by `thesis_summary.*`, `thesis_summary_by_cohort.*`, and `evaluation_manifest.json` from that same focused bundle. Those surfaces expose the maintained preference-burden fields `median_preference_query_count`, `p90_preference_query_count`, `max_preference_query_count`, and `preference_certification_success_rate`. For the current checked local slice, read them as descriptive focused-bundle surfaces only: they may be `0.0` or null on the available rows, and they do not by themselves close `P14.17-P14.20`.

That same focused bundle also carries selective-certification metrics as maintained single-bundle source surfaces. In `thesis_summary.*`, the current checked bundle emits `certificate_selectivity_rate`, `certificate_selectivity_denominator`, and `broad_hard_case_certificate_selectivity_rate`; in `thesis_plots.json`, the `hard_case_transfer_vs_variant` surface repeats `broad_hard_case_certificate_selectivity_rate`; and in `thesis_report.md`, the per-variant summary lines surface `certificate_selectivity_rate` directly. Treat these as descriptive checked-bundle outputs for the current slice rather than as repeated-seed publishability proof.

Current checked runtime and report surfaces now publish per-stage runtime quantiles, p90 RSS/VMS and max RSS/VMS memory surfaces, exact fast-path precision/recall with denominators, action-family budget shares, and populated peak RSS/VMS summary fields in the checked hot-rerun `thesis_summary.*` / `thesis_metrics.json` companion. The checked full-suite companion also now carries `out/headline_exports/current_checked/full_suite_curated_latest_20260411/lane_artifact_generation_summary.json` and `out/headline_exports/current_checked/full_suite_curated_latest_20260411/lane_artifact_generation_summary.md`, backed by the copied lane `thesis_metrics.json` files, so per-lane artifact-generation time is reviewer-visible without recomputing the bundles. Those surfaces support `P14.46-P14.50` on current checked evidence. Separately, `out/headline_exports/current_checked/full_suite_curated_latest_20260411_hot_rerun_hot/hot_rerun_gate.json` is green for the pair-benchmark hot-rerun companion on its own, and the repaired suite-level verdict now agrees that hot rerun is not the reason the suite remains red; the remaining blockers are DCCS, refine-cost forecasting, and VOI.

## CI Lanes (Authoritative)

Workflow: [.github/workflows/backend-ci.yml](../.github/workflows/backend-ci.yml)

### `fast-lane`

- `STRICT_RUNTIME_TEST_BYPASS=1`
- deterministic fixture-first smoke/regression subset
- validates day-to-day behavior with short runtime

### `strict-live-lane`

- `STRICT_RUNTIME_TEST_BYPASS=0`
- signed fallback disabled for key feeds
- validates strict reason-code parity, strict data-path behavior, and fail-closed subsystem behavior

## Minimum Acceptance Gates

- subsystem quality scores pass configured thresholds
- dropped routes do not exceed configured cap, with the strict target effectively `0`
- `backend/scripts/benchmark_model_v2.py` keeps flat and hilly `p95_ms` under the configured gate, which defaults to `2000 ms`
- strict reason-code behavior remains stable for missing, stale, invalid, or unsupported model data
- graph coverage remains inside the UK asset guardrails

## Quality Thresholds

`backend/scripts/score_model_quality.py` currently enforces a score threshold of `95` for:

- `risk_aversion`
- `dominance`
- `scenario_profile`
- `departure_time`
- `stochastic_sampling`
- `terrain_profile`
- `toll_classification`
- `fuel_price`
- `carbon_price`
- `toll_cost`

When `STRICT_LIVE_DATA_REQUIRED=true`, the scorer also expects raw evidence to exist for:

- `backend/data/raw/uk/scenario_live_observed.jsonl`
- `backend/data/raw/uk/stochastic_residuals_raw.csv`
- `backend/data/raw/uk/dft_counts_raw.csv`
- `backend/data/raw/uk/fuel_prices_raw.json`
- `backend/data/raw/uk/carbon_intensity_hourly_raw.json`
- `backend/data/raw/uk/toll_classification/`
- `backend/data/raw/uk/toll_pricing/`
- `backend/data/raw/uk/toll_tariffs_operator_truth.json`

## Benchmark Defaults

`backend/scripts/benchmark_model_v2.py` currently profiles `build_option()` using:

- fixture corpus `backend/tests/fixtures/uk_routes`
- `8` iterations by default
- `rigid_hgv`
- `ScenarioMode.NO_SHARING`
- `use_tolls=false`
- `toll_cost_per_km=0.2`
- `carbon_price_per_kg=0.12`
- stochastic config `enabled=true`, `seed=42`, `sigma=0.08`, `samples=32`
- emissions context `diesel`, `euro6`, `ambient_temp_c=12`
- departure time `2026-02-18T08:30:00Z`
- one flat-terrain pass and one hilly-terrain pass
- p95 success gate `2000 ms`

## Subsystem Expectations

### Fuel

- energy/emissions quantiles remain ordered
- empirical snapshot ingestion stays fresh enough for strict mode
- signature/auth failures map cleanly to canonical fuel reason codes

### Scenario

- sharing modes remain monotone in the expected direction
- holdout separation and MAPE metrics remain inside the current empirical-fit envelope
- `scenario_summary` stays populated on modeled routes
- missing or invalid scenario assets still fail with canonical strict reason codes

### Stochastic

- posterior regime probabilities and `quantile_mapping_v1` transforms remain present
- regime coverage stays grounded in the 18-regime UK corpus
- clipping, coverage, and calibrated factor scales remain bounded

### Vehicle Profiles

- built-in and custom profiles remain schema-compatible
- strict unknown/invalid mapping stays on `vehicle_profile_unavailable` or `vehicle_profile_invalid`

### Terrain

- UK fail-closed coverage stays above the configured cutoff
- unsupported-region behavior stays explicit
- missing DEM assets still map to canonical terrain failures

## Recommended Test Execution Modes

Full local backend suite:

```powershell
uv run --project backend pytest backend/tests
```

Low-resource sequential execution:

```powershell
.\scripts\run_backend_tests_safe.ps1 -MaxCores 1 -PriorityClass Idle
```

## Docs Drift Check

From repo root:

```powershell
python scripts/check_docs.py
```

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Evaluation Card](evaluation_card.md)
- [Performance Profiling Notes](performance-profiling-notes.md)
- [Model Assets and Data Sources](model-assets-and-data-sources.md)
- [Strict Error Contract Reference](strict-errors-reference.md)


