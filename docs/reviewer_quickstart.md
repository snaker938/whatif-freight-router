# Reviewer Quickstart

This page gives a conservative reviewer path for the current checked publication slice.

It does not claim that all gates are green. It documents the current checked bundle, the source files behind the currently indexed headline tables and figure sources, the checked full-suite verdict companion bundle, the checked threshold-sensitivity, optional-stopping, perturbation, public-transfer, and hot-rerun lane companions, and the commands used to inspect, rerun, and export that slice.

## Current Checked Reviewer Bundle

The current checked local bundle used in this quickstart is:

- `backend/out/artifacts/thesis_eval_20260331_r2_focused_voi/`
- `out/headline_exports/current_checked/full_suite_curated_latest_20260411/`
- `out/headline_exports/current_checked/full_suite_curated_latest_20260411_threshold_sensitivity/`
- `out/headline_exports/current_checked/full_suite_curated_latest_20260411_optional_stopping_coverage/`
- `out/headline_exports/current_checked/full_suite_curated_latest_20260411_perturbation_flip_radius/`
- `out/headline_exports/current_checked/full_suite_curated_latest_20260411_public_transfer/`
- `out/headline_exports/current_checked/full_suite_curated_latest_20260411_hot_rerun_hot/`

There is still no checked local `backend/out/artifacts/full_latest_suite_*` bundle in this workspace. The maintained reviewer package now carries a checked companion copy of the latest full-suite assessment bundle (`full_suite_curated_latest_20260411`) under `out/headline_exports/current_checked/`. That companion now reports `publishable_on_current_evidence = false`, `adoption_claim_supported = false`, `hot_rerun_all_green = true`, `sample_size_failure_count = 0`, `optional_stopping_gate_failure_count = 0`, `perturbation_gate_failure_count = 0`, and `publishability_blockers = ["dccs_hard_gates_not_all_green", "refine_cost_forecast_gates_not_all_green", "voi_hard_gates_not_all_green"]`.

The reviewer-facing artifact map for this slice lives at:

- [`paper_artifact_index.json`](../paper_artifact_index.json)

Scope note:

- This quickstart reproduces the maintained checked reviewer package: the focused-VOI checked bundle, the latest checked campaign-backed source surfaces indexed in `paper_artifact_index.json`, a checked local companion copy of the latest full-suite publishability/adoption verdict bundle under `out/headline_exports/current_checked/full_suite_curated_latest_20260411/`, and checked local companion copies of the threshold-sensitivity, optional-stopping, perturbation, public-transfer, and hot-rerun lane bundles under `out/headline_exports/current_checked/`.
- The full-suite companion now mirrors the checked suite-root verdict exactly: optional-stopping and perturbation are surfaced and green on the copied proof surfaces, while publishability/adoption remain blocked by the current hot-rerun, DCCS, refine-cost, and VOI families.

## End-To-End Thesis Lane Reproduction

From the repo root:

```powershell
Set-Location backend
uv sync --dev
uv run python scripts/preflight_live_runtime.py
uv run python scripts/run_thesis_lane.py --manage-local-backend
```

If you want a narrower slice instead of a full lane run, inspect the lane runner help after preflight:

```powershell
Set-Location backend
uv run python scripts/run_thesis_lane.py --help
```

## Full Latest Suite Regeneration

From the repo root:

```powershell
Set-Location backend
uv sync --dev
uv run python scripts/preflight_live_runtime.py
uv run python scripts/run_full_latest_suite.py
```

If the suite completes, it writes a fresh `backend/out/artifacts/full_latest_suite_*` bundle whose `index.json` points at:

- a lane-publishability summary pair in CSV and JSON form
- a universal-baseline audit pair in CSV and JSON form
- a sample-size gate summary pair in CSV and JSON form
- a headline seed-claim summary pair in CSV and JSON form
- a failure-atlas lane-metadata JSON artifact
- a failure atlas in JSON plus Markdown form
- a publishability-verdict JSON artifact
- a publishability-assessment Markdown report

Treat this as regeneration guidance for local reruns. The maintained reviewer package indexed in `paper_artifact_index.json` still points at the checked focused-VOI bundle and the latest checked campaign slice, while the newer checked full-suite assessment now exposes green optional-stopping and perturbation proofs but still records publishability/adoption blockers from the hot-rerun, DCCS, refine-cost, and VOI families. A successful local rerun of the quickstart package should reproduce that same checked suite-level verdict, not a weaker surrogate.

## Checked Full-Suite Verdict Companion

The maintained reviewer package now carries a checked local companion copy of the latest full-suite publishability/adoption verdict bundle at:

- `out/headline_exports/current_checked/full_suite_curated_latest_20260411/`

The export helper refreshes that directory from the checked source bundle at `C:\app\out\artifacts\full_suite_curated_latest_20260411` when it is available on the current machine. The copied bundle preserves the current checked suite-root verdict exactly: `publishable_on_current_evidence = false`, `adoption_claim_supported = false`, `hot_rerun_all_green = true`, zero optional-stopping/perturbation gate failures, and the current blocker list `["dccs_hard_gates_not_all_green", "refine_cost_forecast_gates_not_all_green", "voi_hard_gates_not_all_green"]`.

From the repo root:

```powershell
$suiteBundle = ".\out\headline_exports\current_checked\full_suite_curated_latest_20260411"
Get-ChildItem $suiteBundle
Get-Content ".\out\headline_exports\current_checked\full_suite_curated_latest_20260411\publishability_verdict.json" -Raw |
  ConvertFrom-Json |
  Select-Object publishable_on_current_evidence, adoption_claim_supported, hot_rerun_all_green, fairness_failure_count, sample_size_failure_count, optional_stopping_gate_failure_count, perturbation_gate_failure_count, publishability_blockers
Get-Content ".\out\headline_exports\current_checked\full_suite_curated_latest_20260411\failure_atlas_lane_metadata.json" -Raw |
  ConvertFrom-Json |
  Select-Object lane_status, required_kind_counts, root_cause_family_counts
Import-Csv ".\out\headline_exports\current_checked\full_suite_curated_latest_20260411\lane_publishability_summary.csv" |
  Select-Object -First 12
Get-Content ".\out\headline_exports\current_checked\full_suite_curated_latest_20260411\lane_artifact_generation_summary.json" -Raw |
  ConvertFrom-Json |
  Select-Object -ExpandProperty rows
Get-Content ".\out\headline_exports\current_checked\full_suite_curated_latest_20260411\publishability_assessment.md" -TotalCount 40
```

## Checked Threshold Sensitivity Lane

The maintained reviewer package now carries a checked local companion copy of the threshold-sensitivity lane bundle at:

- `out/headline_exports/current_checked/full_suite_curated_latest_20260411_threshold_sensitivity/`

The export helper refreshes that directory from the checked source bundle at `C:\app\out\artifacts\full_suite_curated_latest_20260411_threshold_sensitivity` when it is available on the current machine. The copied bundle is evidence that the `threshold_sensitivity` evaluator lane exists and emits its maintained sweep artifacts; it is not a claim that the related `P14.40-P14.43` or `G11.*` gates are green.

From the repo root:

```powershell
$thresholdBundle = ".\out\headline_exports\current_checked\full_suite_curated_latest_20260411_threshold_sensitivity"
Get-ChildItem $thresholdBundle
Get-Content ".\out\headline_exports\current_checked\full_suite_curated_latest_20260411_threshold_sensitivity\lane_metadata.json" -Raw |
  ConvertFrom-Json |
  Select-Object evaluation_suite, why_this_lane_exists, threshold_sensitivity_reporting
Import-Csv ".\out\headline_exports\current_checked\full_suite_curated_latest_20260411_threshold_sensitivity\threshold_sensitivity_summary.csv" |
  Select-Object -First 12
Get-Content ".\out\headline_exports\current_checked\full_suite_curated_latest_20260411_threshold_sensitivity\threshold_sensitivity_report.md" -TotalCount 60
```

This checked lane metadata records:

- role `threshold_sensitivity`
- one-factor-at-a-time sweeps for certificate threshold, low-ambiguity fast-path threshold, and certified-set cap
- the checked summary/report surfaces `out/headline_exports/current_checked/full_suite_curated_latest_20260411_threshold_sensitivity/threshold_sensitivity_summary.csv`, `out/headline_exports/current_checked/full_suite_curated_latest_20260411_threshold_sensitivity/threshold_sensitivity_summary.json`, and `out/headline_exports/current_checked/full_suite_curated_latest_20260411_threshold_sensitivity/threshold_sensitivity_report.md`
- the checked plot family `threshold_sensitivity_vs_variant`

## Checked Public Transfer Lane

The maintained reviewer package now carries a checked local companion copy of the public-transfer lane bundle at:

- `out/headline_exports/current_checked/full_suite_curated_latest_20260411_public_transfer/`

The export helper refreshes that directory from the checked source bundle at `C:\app\out\artifacts\full_suite_curated_latest_20260411_public_transfer` when it is available on the current machine. The copied bundle is evidence that the `public_transfer` evaluator lane emits both the corridor-family and weather-regime holdout surfaces; it is not a claim that the related transfer-size or publishability gates are green.

From the repo root:

```powershell
$transferBundle = ".\out\headline_exports\current_checked\full_suite_curated_latest_20260411_public_transfer"
Get-ChildItem $transferBundle
Get-Content ".\out\headline_exports\current_checked\full_suite_curated_latest_20260411_public_transfer\lane_metadata.json" -Raw |
  ConvertFrom-Json |
  Select-Object evaluation_suite, why_this_lane_exists, observed_sample_size, transfer_slice_reporting
Import-Csv ".\out\headline_exports\current_checked\full_suite_curated_latest_20260411_public_transfer\thesis_summary_by_transfer_slice.csv" |
  Select-Object -First 12
Import-Csv ".\out\headline_exports\current_checked\full_suite_curated_latest_20260411_public_transfer\thesis_summary_by_weather_regime_transfer_slice.csv" |
  Select-Object -First 12
```

This checked lane metadata records:

- role `public_transfer`
- the leave-one-corridor-family-out summary surfaces `out/headline_exports/current_checked/full_suite_curated_latest_20260411_public_transfer/thesis_summary_by_transfer_slice.csv` / `.json`
- the leave-one-weather-regime-out summary surfaces `out/headline_exports/current_checked/full_suite_curated_latest_20260411_public_transfer/thesis_summary_by_weather_regime_transfer_slice.csv` / `.json`
- the paired plot families inside `thesis_plots.json`
- current observed sample size `row_count = 52`, which now clears the `G11.52` minimum of `50`

## Checked Optional-Stopping Coverage Lane

The maintained reviewer package now carries a checked local companion copy of the optional-stopping coverage lane bundle at:

- `out/headline_exports/current_checked/full_suite_curated_latest_20260411_optional_stopping_coverage/`

The export helper refreshes that directory from the checked source bundle at `C:\app\out\artifacts\full_suite_curated_latest_20260411_optional_stopping_coverage` when it is available on the current machine. The lane companion exposes the checked optional-stopping evaluator artifacts, while the canonical gate readings for `G11.17-G11.19` and `G11.54` live in the full-suite companion `lane_publishability_summary.*` and `sample_size_gate_summary.*` surfaces.

From the repo root:

```powershell
$suiteBundle = ".\out\headline_exports\current_checked\full_suite_curated_latest_20260411"
$optionalStoppingBundle = ".\out\headline_exports\current_checked\full_suite_curated_latest_20260411_optional_stopping_coverage"
Get-ChildItem $optionalStoppingBundle
Get-Content "$optionalStoppingBundle\lane_metadata.json" -Raw |
  ConvertFrom-Json |
  Select-Object evaluation_suite, observed_sample_size, evaluation_size_requirement
Import-Csv "$suiteBundle\lane_publishability_summary.csv" |
  Where-Object { $_.lane_role -eq 'optional_stopping_coverage' } |
  Select-Object variant_id, pipeline_mode, optional_stopping_method_recorded_rate, optional_stopping_delta_recorded_rate, optional_stopping_validity_tested_rate, optional_stopping_validity_violation_rate, optional_stopping_guaranteed_coverage_floor, optional_stopping_required_coverage_floor
Import-Csv "$suiteBundle\sample_size_gate_summary.csv" |
  Where-Object { $_.lane_role -eq 'optional_stopping_coverage' } |
  Select-Object evaluation_requirement_id, evaluation_requirement_observed_count, evaluation_requirement_observed_count_source, evaluation_requirement_total_minimum, evaluation_requirement_met
```

This checked lane records:

- role `optional_stopping_coverage`
- the maintained `G11.54` lane-size contract as `samples >= 30000`
- the canonical full-suite proof metrics for recorded CS method/delta, direct interval-validity checks, zero validity violations, and the guaranteed coverage floor implied by the recorded `delta`

## Checked Perturbation Flip-Radius Lane

The maintained reviewer package now carries a checked local companion copy of the perturbation / flip-radius lane bundle at:

- `out/headline_exports/current_checked/full_suite_curated_latest_20260411_perturbation_flip_radius/`

The export helper refreshes that directory from the checked source bundle at `C:\app\out\artifacts\full_suite_curated_latest_20260411_perturbation_flip_radius` when it is available on the current machine. The lane companion exposes the checked perturbation evaluator artifacts, while the canonical gate readings for `G11.20`, `G11.21`, and `G11.55` live in the full-suite companion `lane_publishability_summary.*` and `sample_size_gate_summary.*` surfaces.

From the repo root:

```powershell
$suiteBundle = ".\out\headline_exports\current_checked\full_suite_curated_latest_20260411"
$perturbationBundle = ".\out\headline_exports\current_checked\full_suite_curated_latest_20260411_perturbation_flip_radius"
Get-ChildItem $perturbationBundle
Get-Content "$perturbationBundle\lane_metadata.json" -Raw |
  ConvertFrom-Json |
  Select-Object evaluation_suite, observed_sample_size, evaluation_size_requirement
Import-Csv "$suiteBundle\lane_publishability_summary.csv" |
  Where-Object { $_.lane_role -eq 'perturbation_flip_radius' } |
  Select-Object variant_id, pipeline_mode, real_lane_flip_radius_violation_rate, exact_synthetic_flip_radius_violation_rate, perturbation_exact_synthetic_world_count, perturbation_minimum_flip_budget_min, perturbation_world_kind_counts_json
Import-Csv "$suiteBundle\sample_size_gate_summary.csv" |
  Where-Object { $_.lane_role -eq 'perturbation_flip_radius' } |
  Select-Object evaluation_requirement_id, evaluation_requirement_observed_real_count, evaluation_requirement_real_minimum, evaluation_requirement_observed_exact_synthetic_count, evaluation_requirement_exact_synthetic_minimum, evaluation_requirement_met
```

This checked lane records:

- role `perturbation_flip_radius`
- the maintained compound `G11.55` lane-size contract as `>= 30` real rows and `>= 500` exact synthetic worlds
- the canonical full-suite proof metrics for exact-synthetic and real-lane flip-radius violation rates plus the minimum recorded flip budget on the checked valid slice

## Focused-VOI Headline Table And Figure Commands

The commands below do not invent new report assets. They materialize the four focused-VOI headline source surfaces from the current checked bundle:

- variant summary table source
- cohort summary table source
- certificate-vs-variant figure source
- runtime-vs-variant figure source

Use the additional sections below for the other focused-VOI table surfaces and the latest checked campaign-backed table and figure sources indexed in `paper_artifact_index.json`.

Note:
- This quickstart now gives one documented command block for every currently indexed headline table or figure source surface.
- Thesis-like `index.json` and `index.md` bundle indexes are emitted for newly written or additively refreshed thesis-like bundles through the run-store path. The current checked focused bundle at `backend/out/artifacts/thesis_eval_20260331_r2_focused_voi/` now carries additively backfilled copies of those files, while other older checked bundles may still legitimately lack them if they predate that behavior and have not been refreshed.
- Presence or absence of those bundle-index files does not invalidate the underlying checked CSV, JSON, or Markdown artifacts. Source-inspection commands stay source-first, while the export helper in `Headline SVG And PDF Export Commands` materializes checked SVG, print-ready HTML, and PDF renders for the indexed headline surfaces.

From the repo root:

```powershell
$bundle = "backend/out/artifacts/thesis_eval_20260331_r2_focused_voi"
$reviewerOut = Join-Path $env:TEMP "whatif_reviewer_exports"
$variantTableOut = Join-Path $reviewerOut ("variant_table_" + [guid]::NewGuid().ToString() + ".csv")
$cohortTableOut = Join-Path $reviewerOut ("cohort_table_" + [guid]::NewGuid().ToString() + ".csv")
$certificateFigureOut = Join-Path $reviewerOut ("certificate_figure_" + [guid]::NewGuid().ToString() + ".json")
$runtimeFigureOut = Join-Path $reviewerOut ("runtime_figure_" + [guid]::NewGuid().ToString() + ".json")
New-Item -ItemType Directory -Force -Path $reviewerOut | Out-Null

Import-Csv "$bundle/thesis_summary.csv" |
  Select-Object variant_id,pipeline_mode,weighted_win_rate_best_baseline,mean_runtime_ms,mean_certificate |
  Tee-Object -Variable variantTable |
  Format-Table -AutoSize

$variantTable | Export-Csv $variantTableOut -NoTypeInformation

Import-Csv "$bundle/thesis_summary_by_cohort.csv" |
  Select-Object variant_id,cohort,weighted_win_rate_best_baseline,mean_runtime_ms,mean_certificate |
  Tee-Object -Variable cohortTable |
  Format-Table -AutoSize

$cohortTable | Export-Csv $cohortTableOut -NoTypeInformation

$plots = Get-Content "$bundle/thesis_plots.json" -Raw | ConvertFrom-Json
$plots.certificate_vs_variant |
  Tee-Object -Variable certificateFigure |
  Format-Table -AutoSize

$certificateFigure | ConvertTo-Json -Depth 6 | Set-Content $certificateFigureOut

$plots.runtime_vs_variant |
  Tee-Object -Variable runtimeFigure |
  Format-Table -AutoSize

$runtimeFigure | ConvertTo-Json -Depth 6 | Set-Content $runtimeFigureOut

Write-Host "Variant table source written to $variantTableOut"
Write-Host "Cohort table source written to $cohortTableOut"
Write-Host "Certificate figure source written to $certificateFigureOut"
Write-Host "Runtime figure source written to $runtimeFigureOut"
```

These commands produce:

- a CSV source file for the variant summary table in the runtime-created `$reviewerOut` directory
- a CSV source file for the cohort summary table in the runtime-created `$reviewerOut` directory
- a JSON source file for the certificate-vs-variant figure in the runtime-created `$reviewerOut` directory
- a JSON source file for the runtime-vs-variant figure in the runtime-created `$reviewerOut` directory

The current checked figure surface remains plot-ready JSON as the truth anchor for this command block. Use `Headline SVG And PDF Export Commands` below when you want checked SVG, print-ready HTML, and PDF renders for the indexed headline surfaces.

## Focused-VOI Additional Table Commands

This block materializes the remaining focused-VOI table surfaces indexed in `paper_artifact_index.json`:

- aggregate variant evidence table source
- preference-burden summary table source
- preference-burden by-cohort table source
- cohort/support-bin composition table source

From the repo root:

```powershell
$bundle = "backend/out/artifacts/thesis_eval_20260331_r2_focused_voi"
$reviewerOut = Join-Path $env:TEMP "whatif_reviewer_exports"
$aggregateTableOut = Join-Path $reviewerOut ("aggregate_variant_evidence_" + [guid]::NewGuid().ToString() + ".json")
$preferenceSummaryOut = Join-Path $reviewerOut ("preference_burden_summary_" + [guid]::NewGuid().ToString() + ".csv")
$preferenceByCohortOut = Join-Path $reviewerOut ("preference_burden_by_cohort_" + [guid]::NewGuid().ToString() + ".csv")
$compositionSourceOut = Join-Path $reviewerOut ("cohort_support_composition_" + [guid]::NewGuid().ToString() + ".json")
New-Item -ItemType Directory -Force -Path $reviewerOut | Out-Null

$aggregateSource = [pscustomobject]@{
  thesis_summary = Get-Content "$bundle/thesis_summary.json" -Raw | ConvertFrom-Json
  thesis_metrics = Get-Content "$bundle/thesis_metrics.json" -Raw | ConvertFrom-Json
  evaluation_manifest = Get-Content "$bundle/evaluation_manifest.json" -Raw | ConvertFrom-Json
}
$aggregateSource | ConvertTo-Json -Depth 8 | Set-Content $aggregateTableOut

Import-Csv "$bundle/thesis_summary.csv" |
  Select-Object variant_id,pipeline_mode,median_preference_query_count,p90_preference_query_count,max_preference_query_count,preference_certification_success_rate |
  Tee-Object -Variable preferenceSummary |
  Format-Table -AutoSize

$preferenceSummary | Export-Csv $preferenceSummaryOut -NoTypeInformation

Import-Csv "$bundle/thesis_summary_by_cohort.csv" |
  Select-Object variant_id,cohort,median_preference_query_count,p90_preference_query_count,max_preference_query_count,preference_certification_success_rate |
  Tee-Object -Variable preferenceByCohort |
  Format-Table -AutoSize

$preferenceByCohort | Export-Csv $preferenceByCohortOut -NoTypeInformation

Copy-Item "$bundle/cohort_composition.json" $compositionSourceOut

Write-Host "Aggregate variant evidence source written to $aggregateTableOut"
Write-Host "Preference-burden summary source written to $preferenceSummaryOut"
Write-Host "Preference-burden by-cohort source written to $preferenceByCohortOut"
Write-Host "Cohort/support-bin composition source written to $compositionSourceOut"
```

These commands produce:

- a JSON source bundle for the focused-VOI aggregate variant evidence table in the runtime-created `$reviewerOut` directory
- a CSV source file for the focused-VOI preference-burden summary table in the runtime-created `$reviewerOut` directory
- a CSV source file for the focused-VOI preference-burden by-cohort table in the runtime-created `$reviewerOut` directory
- a JSON source file for the focused-VOI cohort/support-bin composition table in the runtime-created `$reviewerOut` directory

## Latest Checked Campaign Table And Figure Commands

This block materializes the current campaign-backed headline source surfaces indexed in `paper_artifact_index.json`:

- summary and metrics table source
- gain-versus-V0 figure source
- run-validity figure source
- baseline-smoke figure source
- certificate-margin-versus-variant figure source
- ambiguity-alignment-versus-variant figure source
- cohort-composition figure source
- performance-versus-variant figure source
- runtime-distribution figure source
- runtime-breakdown figure source
- controller-refresh-split-versus-variant figure source
- hard-case-transfer-versus-variant figure source
- win-rate-versus-variant figure source
- startup-and-warmup figure source
- cohort/support-bin composition table source
- runtime-observability summary table source
- runtime-action observability table source

From the repo root:

```powershell
$campaign = "backend/out/thesis_campaigns/dominance_cluster5_cardiff_bath_corr12p5_r2"
$tranche = Join-Path $campaign "tranche_001/artifacts/dominance_cluster5_cardiff_bath_corr12p5_r2_t001"
$reviewerOut = Join-Path $env:TEMP "whatif_reviewer_campaign_exports"
$runTag = [guid]::NewGuid().ToString()
$campaignTableDir = Join-Path $reviewerOut ("campaign_summary_and_metrics_" + $runTag)
$campaignSharedDir = Join-Path $reviewerOut ("campaign_shared_context_" + $runTag)
$campaignCompositionDir = Join-Path $reviewerOut ("campaign_composition_" + $runTag)
New-Item -ItemType Directory -Force -Path $campaignTableDir | Out-Null
New-Item -ItemType Directory -Force -Path $campaignSharedDir | Out-Null
New-Item -ItemType Directory -Force -Path $campaignCompositionDir | Out-Null

Copy-Item "$campaign/campaign_report.md" (Join-Path $campaignTableDir "campaign_report.md")
Copy-Item "$campaign/campaign_result.json" (Join-Path $campaignTableDir "campaign_result.json")
Copy-Item "$tranche/thesis_summary.csv" (Join-Path $campaignTableDir "thesis_summary.csv")
Copy-Item "$tranche/thesis_summary.json" (Join-Path $campaignTableDir "thesis_summary.json")
Copy-Item "$tranche/evaluation_manifest.json" (Join-Path $campaignTableDir "evaluation_manifest.json")
Copy-Item "$tranche/osrm_baseline_identity_manifest.json" (Join-Path $campaignTableDir "out/headline_exports/current_checked/full_suite_curated_latest_20260411/osrm_baseline_identity_manifest.json")
Copy-Item "$tranche/ors_baseline_identity_manifest.json" (Join-Path $campaignTableDir "out/headline_exports/current_checked/full_suite_curated_latest_20260411/ors_baseline_identity_manifest.json")
Copy-Item "$tranche/baseline_fairness_audit.json" (Join-Path $campaignTableDir "baseline_fairness_audit.json")
Copy-Item "$tranche/baseline_fairness_lane_metadata.json" (Join-Path $campaignTableDir "baseline_fairness_lane_metadata.json")
Copy-Item "$campaign/campaign_result.json" (Join-Path $campaignSharedDir "campaign_result.json")
Copy-Item "$tranche/evaluation_manifest.json" (Join-Path $campaignSharedDir "evaluation_manifest.json")
Copy-Item "$tranche/osrm_baseline_identity_manifest.json" (Join-Path $campaignSharedDir "out/headline_exports/current_checked/full_suite_curated_latest_20260411/osrm_baseline_identity_manifest.json")
Copy-Item "$tranche/ors_baseline_identity_manifest.json" (Join-Path $campaignSharedDir "out/headline_exports/current_checked/full_suite_curated_latest_20260411/ors_baseline_identity_manifest.json")
Copy-Item "$tranche/baseline_fairness_audit.json" (Join-Path $campaignSharedDir "baseline_fairness_audit.json")
Copy-Item "$tranche/baseline_fairness_lane_metadata.json" (Join-Path $campaignSharedDir "baseline_fairness_lane_metadata.json")
Copy-Item "$tranche/cohort_composition.json" (Join-Path $campaignCompositionDir "cohort_composition.json")
Copy-Item "$tranche/results.json" (Join-Path $campaignCompositionDir "results.json")
Copy-Item "$tranche/evaluation_manifest.json" (Join-Path $campaignCompositionDir "evaluation_manifest.json")

$plots = Get-Content "$tranche/thesis_plots.json" -Raw | ConvertFrom-Json
$plotSelectors = @(
  'gain_vs_v0',
  'run_validity',
  'baseline_smoke',
  'certificate_margin_vs_variant',
  'ambiguity_alignment_vs_variant',
  'cohort_composition',
  'performance_vs_variant',
  'runtime_distribution_vs_variant',
  'controller_refresh_split_vs_variant',
  'hard_case_transfer_vs_variant',
  'win_rate_vs_variant',
  'startup_and_warmup'
)

foreach ($selector in $plotSelectors) {
  $plotOut = Join-Path $reviewerOut ("campaign_" + $selector + "_" + $runTag + ".json")
  $plots.$selector | ConvertTo-Json -Depth 8 | Set-Content $plotOut
  Write-Host "$selector figure source written to $plotOut"
}

Write-Host "Campaign summary-and-metrics source bundle written to $campaignTableDir"
Write-Host "Campaign shared context written to $campaignSharedDir"
Write-Host "Campaign cohort/support-bin composition source bundle written to $campaignCompositionDir"
```

These commands produce:

- a directory containing the current checked campaign summary-and-metrics table source files
- one JSON source file per indexed campaign-backed figure selector in the runtime-created reviewer export directory
- a shared context directory with copied campaign-result, evaluation-manifest, OSRM/ORS baseline-identity-manifest, baseline-fairness-audit, and baseline-fairness-lane-metadata JSON files for the campaign-backed figure family
- a composition-source directory with copied cohort_composition, results, and evaluation-manifest JSON files for the explicit latest-checked-campaign cohort/support-bin composition table
- the copied `thesis_summary.csv/json` files in the staged full-suite broad-cold proof bundle also back the checked runtime-observability and runtime-action observability table surfaces, while the copied `thesis_results.json` file in that same bundle backs the checked runtime-stage-quantiles table and figure surfaces

Fairness note:
- The copied `backend/out/thesis_campaigns/dominance_cluster5_cardiff_bath_corr12p5_r2/tranche_001/artifacts/dominance_cluster5_cardiff_bath_corr12p5_r2_t001/baseline_fairness_audit.json` is conservative for the current checked slice. It records matched OD and departure context, but it does not claim a 100% fairness pass rate because the preserved OSRM baseline uses `car.lua` while the preserved ORS baseline uses `driving-hgv`, and the four `cardiff_bath` comparison rows fail `route_evidence_ok` with `routing_graph_disconnected_od`.
- The copied `backend/out/thesis_campaigns/dominance_cluster5_cardiff_bath_corr12p5_r2/tranche_001/artifacts/dominance_cluster5_cardiff_bath_corr12p5_r2_t001/baseline_fairness_lane_metadata.json` makes that fairness review a named checked lane surface for the maintained headline baseline-comparison slice. It does not soften the failed result.

## Headline SVG And PDF Export Commands

This helper turns the currently indexed headline tables and figures plus the explicit focused-VOI, latest-checked-campaign cohort/support-bin composition, runtime-observability, runtime-action-observability, runtime-stage-quantiles, and preference-burden tables into checked reviewer-facing exports without manual editing. It preserves the source-first CSV/JSON surfaces above, writes additive rendered outputs plus co-packaged `*.source.csv` or `*.source.json` companions and co-packaged `*.provenance.json` companions under `out/headline_exports/current_checked/`, and stages checked local companion copies of the latest full-suite publishability/adoption verdict bundle, the broad-cold runtime-observability bundle, the optional-stopping lane bundle, and the perturbation lane bundle under `out/headline_exports/current_checked/`.

Requirements:

- run from the repo root
- local `node`
- local Microsoft Edge or Google Chrome so the helper can print the generated HTML layouts to PDF

From the repo root:

```powershell
node .\scripts\export_headline_surfaces.mjs
Get-ChildItem .\out\headline_exports\current_checked
```

These commands produce one `*.svg`, one `*.print.html`, one `*.pdf`, one co-packaged `*.source.csv` or `*.source.json` file, and one co-packaged `*.provenance.json` file for every headline table or figure surface currently listed under `paper_artifact_index.json` `headline_surfaces`, plus the explicit focused-VOI and latest-checked-campaign cohort/support-bin composition tables and the runtime-observability / runtime-action / runtime-stage-quantiles / preference-burden reviewer tables. They also refresh the checked local full-suite verdict, threshold-sensitivity, optional-stopping, perturbation, public-transfer, and broad-cold runtime-observability companion directories under `out/headline_exports/current_checked/`, including the full-suite `lane_artifact_generation_summary.*` files and the copied lane `thesis_metrics.json` payloads they summarize.

They also refresh the corresponding `paper_artifact_index.json` headline entries so `rendered_outputs`, `packaged_source_companions`, `packaged_provenance_companions`, `export_formats_available`, and `quickstart_reference` point at the checked export files and this command block, while the reviewer-package companion entries for the checked full-suite verdict, broad-cold runtime-observability, threshold-sensitivity, optional-stopping, perturbation, public-transfer, and hot-rerun bundles point at the staged local copies.

## Checked Hot Rerun Benchmark Companion

The checked hot-rerun companion bundle at `out/headline_exports/current_checked/full_suite_curated_latest_20260411_hot_rerun_hot/` is the maintained reviewer surface for runtime-observability and reuse evidence. It is staged from the repo-local pair-benchmark bundle `backend/out/artifacts/full_suite_curated_latest_20260411_hot_rerun_pair_hot/`, and it carries the copied `out/headline_exports/current_checked/full_suite_curated_latest_20260411_hot_rerun_hot/hot_rerun_gate.json`, `out/headline_exports/current_checked/full_suite_curated_latest_20260411_hot_rerun_hot/hot_rerun_vs_cold_comparison.json`, `out/headline_exports/current_checked/full_suite_curated_latest_20260411_hot_rerun_hot/thesis_metrics.json`, `out/headline_exports/current_checked/full_suite_curated_latest_20260411_hot_rerun_hot/thesis_plots.json`, and `out/headline_exports/current_checked/full_suite_curated_latest_20260411_hot_rerun_hot/thesis_report.md` surfaces together with the bundle index and lane metadata. Read it separately from the full-suite verdict companion: the pair-benchmark hot-rerun bundle is green on its own gate JSON, and the repaired checked full-suite verdict now agrees that hot rerun is no longer a blocker.

From the repo root:

```powershell
node .\scripts\export_headline_surfaces.mjs
Get-ChildItem .\out\headline_exports\current_checked\full_suite_curated_latest_20260411_hot_rerun_hot
```

What to check in that companion:

- `out/headline_exports/current_checked/full_suite_curated_latest_20260411_hot_rerun_hot/hot_rerun_gate.json` now reports `all_green = true`, `controller_reuse_reporting = [{"metric": "mean_controller_reuse_rate", "variant_id": "C", "cold_value": 0.157895, "hot_value": 0.210526, "delta": 0.052631, "cold_source_metric": "mean_voi_dccs_cache_hit_rate", "hot_source_metric": "mean_voi_dccs_cache_hit_rate"}]`, `mean_refc_world_reuse_rate = 1.0` for `B` and `C`, `hot_cold_parity_rate = 1.0` for `B` and `C`, and zero LCB / semantic-drift values across the gate-scoped variants
- `out/headline_exports/current_checked/full_suite_curated_latest_20260411_hot_rerun_hot/hot_rerun_vs_cold_comparison.json` carries the cold-versus-hot delta rows and cache snapshots needed for runtime-observability review
- `thesis_metrics.json` carries the row-level runtime and resource metrics, including per-stage runtime fields, `process_rss_mb`, `baseline_runtime_share`, and the stage-level reuse/runtime measures used by the current hot-rerun reporting; the checked `thesis_summary.*` / `thesis_metrics.json` summary rows now also carry populated `mean_peak_process_rss_mb`, `mean_peak_process_rss_p90_mb`, `max_peak_process_rss_mb`, `mean_peak_process_vms_mb`, `mean_peak_process_vms_p90_mb`, and `max_peak_process_vms_mb` fields for `A`, `B`, and `C`
- `thesis_plots.json` and `thesis_report.md` are the reviewer-facing presentation surfaces for the same checked bundle

This companion is evidence-bearing and green on its own pair-benchmark gate JSON. It makes the runtime-observability and reuse proof explicit, but it does not override the separate full-suite verdict companion.

## What To Check

After the run or bundle inspection, confirm that:

- the checked bundle contains the evaluator manifest, variant summary CSV/JSON pair, cohort summary CSV/JSON pair, metrics JSON, plots JSON, methods appendix, and thesis report
- `paper_artifact_index.json` points to the same checked bundle paths
- the exported reviewer source files or directories in the runtime-created temp directories match the indexed current bundle sources referenced in `paper_artifact_index.json`
- each checked headline export in `out/headline_exports/current_checked/` has a sibling `*.source.csv` or `*.source.json` companion
- each checked headline export in `out/headline_exports/current_checked/` has a sibling `*.provenance.json` companion with `git_commit_hash`, `environment_lockfile_hash`, container identity when available, and `policy_hashes`
- each checked focused-VOI and latest-checked-campaign composition table export contains both `composition_family = cohort` rows and `composition_family = support_bin` rows with explicit row counts
- `out/headline_exports/current_checked/full_suite_curated_latest_20260411/` contains the checked copied `index.json`, `out/headline_exports/current_checked/full_suite_curated_latest_20260411/publishability_verdict.json`, `out/headline_exports/current_checked/full_suite_curated_latest_20260411/publishability_assessment.md`, `lane_publishability_summary.csv/json`, `sample_size_gate_summary.csv/json`, `headline_seed_claims_summary.csv/json`, `out/headline_exports/current_checked/full_suite_curated_latest_20260411/failure_atlas_lane_metadata.json`, `out/headline_exports/current_checked/full_suite_curated_latest_20260411/failure_atlas.json` / `.md`, and `universal_baseline_audit.csv/json` files; `sample_size_gate_summary.*` now records the maintained row-count requirement, direct `evaluation_requirement_observed_count`, `evaluation_requirement_observed_count_source`, and `evaluation_requirement_total_minimum`
- `out/headline_exports/current_checked/full_suite_curated_latest_20260411/` also contains the checked copied `out/headline_exports/current_checked/full_suite_curated_latest_20260411/osrm_baseline_identity_manifest.json` and `out/headline_exports/current_checked/full_suite_curated_latest_20260411/ors_baseline_identity_manifest.json` attachments for the headline baseline-comparison surface; those manifests preserve graph date, graph digest, image/config identity, and source graph metadata for the preserved OSRM/ORS baseline engines
- `out/headline_exports/current_checked/full_suite_curated_latest_20260411_threshold_sensitivity/` contains the checked copied `out/headline_exports/current_checked/full_suite_curated_latest_20260411_threshold_sensitivity/lane_metadata.json`, `out/headline_exports/current_checked/full_suite_curated_latest_20260411_threshold_sensitivity/threshold_sensitivity_summary.csv` / `.json`, `out/headline_exports/current_checked/full_suite_curated_latest_20260411_threshold_sensitivity/threshold_sensitivity_report.md`, and `thesis_plots.json` files
- `out/headline_exports/current_checked/full_suite_curated_latest_20260411_optional_stopping_coverage/` contains the checked copied `out/headline_exports/current_checked/full_suite_curated_latest_20260411_optional_stopping_coverage/lane_metadata.json`, `results.json`, `thesis_results.json`, `thesis_summary.csv` / `.json`, and `thesis_plots.json` files, while the canonical `G11.17-G11.19` and `G11.54` gate rows remain in `out/headline_exports/current_checked/full_suite_curated_latest_20260411/lane_publishability_summary.csv` / `.json` and `out/headline_exports/current_checked/full_suite_curated_latest_20260411/sample_size_gate_summary.csv` / `.json`
- `out/headline_exports/current_checked/full_suite_curated_latest_20260411_perturbation_flip_radius/` contains the checked copied `out/headline_exports/current_checked/full_suite_curated_latest_20260411_perturbation_flip_radius/lane_metadata.json`, `results.json`, `thesis_results.json`, `thesis_summary.csv` / `.json`, and `thesis_plots.json` files, while the canonical `G11.20`, `G11.21`, and `G11.55` gate rows remain in `out/headline_exports/current_checked/full_suite_curated_latest_20260411/lane_publishability_summary.csv` / `.json` and `out/headline_exports/current_checked/full_suite_curated_latest_20260411/sample_size_gate_summary.csv` / `.json`
- `out/headline_exports/current_checked/full_suite_curated_latest_20260411_public_transfer/` contains the checked copied `out/headline_exports/current_checked/full_suite_curated_latest_20260411_public_transfer/lane_metadata.json`, `out/headline_exports/current_checked/full_suite_curated_latest_20260411_public_transfer/thesis_summary_by_transfer_slice.csv` / `.json`, `out/headline_exports/current_checked/full_suite_curated_latest_20260411_public_transfer/thesis_summary_by_weather_regime_transfer_slice.csv` / `.json`, and `thesis_plots.json` files
- `out/headline_exports/current_checked/full_suite_curated_latest_20260411_hot_rerun_hot/` contains the checked copied `out/headline_exports/current_checked/full_suite_curated_latest_20260411_hot_rerun_hot/lane_metadata.json`, `out/headline_exports/current_checked/full_suite_curated_latest_20260411_hot_rerun_hot/hot_rerun_gate.json`, `out/headline_exports/current_checked/full_suite_curated_latest_20260411_hot_rerun_hot/hot_rerun_vs_cold_comparison.json`, `out/headline_exports/current_checked/full_suite_curated_latest_20260411_hot_rerun_hot/thesis_metrics.json`, `out/headline_exports/current_checked/full_suite_curated_latest_20260411_hot_rerun_hot/thesis_plots.json`, and `out/headline_exports/current_checked/full_suite_curated_latest_20260411_hot_rerun_hot/thesis_report.md` files

If the run stops early, the most likely causes are:

- strict live runtime preflight failure
- missing backend dependencies in the local `uv` environment
- no supported live data for the selected mode

## Notes

- The exact artifact names depend on the lane and selected mode when you rerun the evaluator.
- This quickstart is intentionally narrow and does not attempt to restate the full claim matrix.
- The current reviewer slice stays source-artifact first: CSV and JSON files remain the checked truth anchors for the table and figure surfaces listed in `paper_artifact_index.json`, while `scripts/export_headline_surfaces.mjs` generates additive SVG, print-ready HTML, PDF, co-packaged `*.source.csv` / `*.source.json` companions, and co-packaged `*.provenance.json` companions for the current headline slice together with the explicit focused-VOI and latest-checked-campaign cohort/support-bin composition tables and the checked full-suite verdict, threshold-sensitivity, optional-stopping, perturbation, public-transfer, and hot-rerun companion bundles.
- The latest checked campaign composition artifact currently preserves cohort counts directly in `cohort_composition.json`; the export helper derives support-bin counts from checked `results.json` `support_richness` rows using the maintained weak/mid/strong thresholds when those counts are not already saved in the campaign composition file.
- When a headline source artifact already carries `artifact_provenance.headline_identity`, the exporter copies that identity into the headline provenance companion. Older checked campaign-backed headline surfaces that predate embedded headline identity are reconstructed from the checked `evaluation_manifest.json` plus sibling `metadata.json` together with the current workspace git, lockfile, and container context.
- The latest checked campaign-backed source anchors in `paper_artifact_index.json` currently resolve to `backend/out/thesis_campaigns/dominance_cluster5_cardiff_bath_corr12p5_r2/`, and the campaign command block above materializes their table and figure source surfaces into a runtime-created temp directory without manual editing.
- Current truth anchors for the live runtime and UI are [backend/README.md](../backend/README.md), [frontend/README.md](../frontend/README.md), and [docs/api-cookbook.md](api-cookbook.md).
- For operational details, see [Run and Operations Guide](run-and-operations.md).
- For the larger thesis narrative, see [Thesis-Grade Codebase Report](thesis-codebase-report.md).

## Related Docs

- [Paper Artifact Index](../paper_artifact_index.json)
- [Sample Manifest and Outputs](sample-manifest.md)
- [Reproducibility Capsule](reproducibility-capsule.md)
- [Run and Operations Guide](run-and-operations.md)
- [Evaluation Card](evaluation_card.md)
- [Thesis-Grade Codebase Report](thesis-codebase-report.md)
