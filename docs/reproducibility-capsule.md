# Reproducibility Capsule

Last Updated: 2026-04-11
Applies To: deterministic benchmark workflows, thesis-evaluation bundles, and artifact provenance under `backend/out/*`

## One-Command Repro Demo

From repo root:

```powershell
.\scripts\demo_repro_run.ps1
```

The script currently writes a capsule JSON to `backend/out/capsule/repro_capsule_<timestamp>.json` and runs:

```powershell
uv run python scripts/benchmark_batch_pareto.py `
  --mode inprocess-fake `
  --pair-count 100 `
  --seed 20260212 `
  --max-alternatives 3 `
  --output <capsule-path>
```

As of `2026-04-09`, there is no checked `backend/out/capsule` directory in the repo, so the latest reproducibility evidence lives in manifests, preflight artifacts, and thesis bundles rather than in a pre-generated capsule export.

## Manual Repro Path

From `backend/`:

```powershell
uv run python scripts/build_model_assets.py
uv run python scripts/preflight_live_runtime.py
uv run python scripts/benchmark_batch_pareto.py `
  --mode inprocess-fake `
  --pair-count 100 `
  --seed 20260212 `
  --max-alternatives 3 `
  --output out/capsule/repro_capsule_manual.json
```

For thesis-lane reproduction, use the same OD corpus, budgets, and strict-evidence settings recorded in `backend/out/thesis_campaigns/dominance_cluster5_cardiff_bath_corr12p5_r2/tranche_001/artifacts/dominance_cluster5_cardiff_bath_corr12p5_r2_t001/methods_appendix.md`.

## Current Local Reproducibility Anchors

### Model-asset provenance

`backend/out/model_assets/manifest.json` currently records:

- version `model-v2-uk`
- generated/as-of `2026-04-10T15:18:10.176657Z`
- source policy `repo_local_fresh`
- 19 built assets
- manifest signature `580e01e2da3350bf83182cc7900a82c54a0424a31146bc7e58911bbe3fd444ac`

`backend/out/model_assets/refresh_manifest.json` fixes the repo-local input set used for that build, including hashes and as-of timestamps for:

- `backend/data/raw/uk/dft_counts_raw.csv`
- `backend/data/raw/uk/stochastic_residuals_raw.csv`
- `backend/data/raw/uk/scenario_live_observed.jsonl`
- `backend/data/raw/uk/scenario_mode_outcomes_observed.jsonl`
- `backend/data/raw/uk/fuel_prices_raw.json`
- `backend/data/raw/uk/carbon_intensity_hourly_raw.json`
- `backend/data/raw/uk/toll_tariffs_operator_truth.json`
- toll classification/pricing corpora

`backend/out/model_assets/live_publish_summary.json` records the latest successful publish handoff at `2026-04-10T15:18:18Z`, including:

- scenario signature prefix `e2499fbc342d`
- fuel signature prefix `39e656b5ca83`

### Strict-readiness anchors

The latest checked local strict preflight is `backend/out/model_assets/preflight_live_runtime.json` at `2026-04-12T13:33:57Z`, with `required_ok: true` and `0` required failures.

The checked passing slice includes:

- `scenario_profiles` passing with `source=repo_local:scenario_profiles_uk.json`, `resolved_source_locator=backend/assets/uk/scenario_profiles_uk.json`, `calibration_basis=empirical_live_fit`, and `contexts=192`
- `scenario_live_context` passing with `as_of_utc=2026-04-10T13:53:23Z` and source coverage `overall=1.0`

That same preflight surface records the resolved scenario-profile locator used by strict runtime selection. Use it to verify whether strict runtime is currently on the repo-local scenario asset locator `backend/assets/uk/scenario_profiles_uk.json` or a live URL-backed source. It does not currently expose which repo-local file won inside `repo_local_fresh`.

The latest checked thesis-bundle repo preflight is `backend/out/thesis_campaigns/dominance_cluster5_cardiff_bath_corr12p5_r2/tranche_001/artifacts/dominance_cluster5_cardiff_bath_corr12p5_r2_t001/repo_asset_preflight.json` at `2026-04-06T09:36:17Z`, also with `required_ok: true`.

So the current checked local asset/publish slice and the current checked local strict-readiness result are now aligned on a passing repo-local preflight. Treat that as a local strict-runtime readiness anchor only.

The current checked publishability/adoption verdict does not come from those local readiness anchors. It comes from the newer checked full-suite assessment bundle `out/headline_exports/current_checked/full_suite_curated_latest_20260411`, whose suite-level verdict JSON now reports `publishable_on_current_evidence=true`, `adoption_claim_supported=true`, `sample_size_failure_count=0`, `fairness_failure_count=0`, `hot_rerun_all_green=true`, and `publishability_blockers=[]`.

That distinction matters for the maintained row status:

- `P14.35` is not closed by the local preflight/manifests alone; any closure argument still has to pass through the checked full-suite bundle and its documented one-command reproduction surfaces.
- `P14.39` is not closed by the local repro anchors alone; the checked reviewer package and current suite-level publishability/adoption verdict are still insufficient to claim closure today.

### Thesis-lane parameter anchors

The current methods appendix for the newest checked thesis campaign records:

- variants `V0=legacy`, `A=dccs`, `B=dccs_refc`, `C=voi`
- matched search budget `4`
- evidence budget `2`
- certificate world count `64`
- certificate threshold `0.8`
- stop threshold `0.02`
- baseline refinement policy `corridor_uniform`
- secondary baseline policy `local_service`
- backend readiness timeout `1800.0 s`
- backend readiness poll `5.0 s`
- max alternatives `8`
- strict evidence policy `no_synthetic_no_proxy_no_fallback`
- in-process backend `True`

The same campaign also records a fixed route-graph subset asset with:

- mode `explicit_subset_asset`
- corridor width `12.5 km`
- `1,515,878` nodes kept
- `1,568,264` edges kept

## Reproducibility Controls

- fixed seed where the runner supports it
- fixed OD corpus or fixture set
- fixed strict/runtime flags, including `STRICT_RUNTIME_TEST_BYPASS`
- fixed model-asset snapshot
- fixed comparator policy where thesis runs use `local_service` or another declared baseline mode
- fixed route-graph asset or subset asset when the thesis lane stages one explicitly
- fixed search/evidence budgets when comparing `V0`, `A`, `B`, and `C`

## Reproducibility Artifacts To Archive

For ordinary route or batch runs:

- `backend/out/manifests/{run_id}.json`
- `backend/out/scenario_manifests/{run_id}.json`
- `backend/out/artifacts/{run_id}/metadata.json`
- `backend/out/artifacts/{run_id}/results.json`
- `backend/out/provenance/{run_id}.json`

For asset reproducibility:

- `backend/out/model_assets/manifest.json`
- `backend/out/model_assets/refresh_manifest.json`
- `backend/out/model_assets/live_publish_summary.json`
- `backend/out/model_assets/preflight_live_runtime.json`

For thesis-grade evaluation:

- `backend/out/thesis_campaigns/dominance_cluster5_cardiff_bath_corr12p5_r2/campaign_result.json`
- `backend/out/thesis_campaigns/dominance_cluster5_cardiff_bath_corr12p5_r2/campaign_report.md`
- `backend/out/thesis_campaigns/dominance_cluster5_cardiff_bath_corr12p5_r2/tranche_001/artifacts/dominance_cluster5_cardiff_bath_corr12p5_r2_t001/evaluation_manifest.json`
- `backend/out/thesis_campaigns/dominance_cluster5_cardiff_bath_corr12p5_r2/tranche_001/artifacts/dominance_cluster5_cardiff_bath_corr12p5_r2_t001/repo_asset_preflight.json`
- `backend/out/thesis_campaigns/dominance_cluster5_cardiff_bath_corr12p5_r2/tranche_001/artifacts/dominance_cluster5_cardiff_bath_corr12p5_r2_t001/thesis_summary.json`
- `backend/out/thesis_campaigns/dominance_cluster5_cardiff_bath_corr12p5_r2/tranche_001/artifacts/dominance_cluster5_cardiff_bath_corr12p5_r2_t001/thesis_metrics.json`
- `backend/out/thesis_campaigns/dominance_cluster5_cardiff_bath_corr12p5_r2/tranche_001/artifacts/dominance_cluster5_cardiff_bath_corr12p5_r2_t001/thesis_plots.json` (plot-ready source behind the campaign-backed figure entries indexed in `paper_artifact_index.json`)
- `backend/out/thesis_campaigns/dominance_cluster5_cardiff_bath_corr12p5_r2/tranche_001/artifacts/dominance_cluster5_cardiff_bath_corr12p5_r2_t001/methods_appendix.md`
- `paper_artifact_index.json`

For the latest checked suite-level verdict carried in the maintained local reviewer companion bundle:

- `out/headline_exports/current_checked/full_suite_curated_latest_20260411/index.md`
- `out/headline_exports/current_checked/full_suite_curated_latest_20260411/publishability_verdict.json`
- `out/headline_exports/current_checked/full_suite_curated_latest_20260411/publishability_assessment.md`

Use that checked local companion bundle when you need the current publishability/adoption judgment. When present on the current machine, it is exported from `C:\app\out\artifacts\full_suite_curated_latest_20260411\`. The local reviewer package now carries that checked local copy alongside the checked local table/figure source surfaces and the local campaign-backed evidence family; those scopes are related, but they are not identical.

Recommended metadata bundle:

1. git commit SHA
2. `.env` hash or sanitized config snapshot
3. model-asset manifest signature
4. benchmark or thesis command line
5. seed, budget, and comparator-policy settings

For the current reviewer-facing publication slice, `paper_artifact_index.json` is the authoritative map from indexed headline table/figure surfaces to the checked local source artifacts. At present that map is source-artifact first: it records CSV/JSON evidence paths and does not claim that every indexed surface already has a committed PDF/SVG export.

The checked focused-VOI bundle at `backend/out/artifacts/thesis_eval_20260331_r2_focused_voi/` now also carries additively backfilled `index.json` and `index.md` as bundle-level inspection entrypoints for artifact presence and export-status checks only. Treat those files as current inspection aids for this checked local bundle, not as proof that every older thesis-like bundle originally emitted them; older pre-refresh bundles may still legitimately lack those index files.

That reviewer index now also names `table.focused_voi.preference_burden_summary` and `table.focused_voi.preference_burden_by_cohort` as maintained reviewer surfaces, but the staged `out/headline_exports/current_checked/table.focused_voi.preference_burden_*` source/provenance sidecars still cite the older single-seed `backend/out/artifacts/thesis_eval_20260331_r2_focused_voi/` bundle. The separate repeated-seed proof for `P14.17-P14.20` is `out/artifacts/full_suite_curated_latest_20260411_focused_voi_proof/` plus the `_seed20260421` and `_seed20260522` companions, where `thesis_summary.*`, `thesis_summary_by_cohort.*`, and `evaluation_manifest.json` expose `median_preference_query_count`, `p90_preference_query_count`, `max_preference_query_count`, and `preference_certification_success_rate`. For the current checked local slice, the aggregate repeated-seed burden rows keep `median_preference_query_count = 0.0` with `p90_preference_query_count <= 1.0`, and the cohort split keeps `preference_certification_success_rate` visible alongside the burden counts. This remains a family-specific preference-burden closure surface rather than a claim that the full suite is publishable.

## Comparing Two Runs

Use these checks before attributing differences to model changes:

1. Same route/request payloads or the same OD corpus.
2. Same scenario mode and stochastic settings.
3. Same strict bypass mode and live-source policy.
4. Same asset versions, manifest signature, and refresh-manifest input hashes.
5. Same comparator policy and route-graph subset policy for thesis lanes.
6. No stale, missing, or degraded readiness warnings in provenance or preflight.

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Quality Gates and Benchmarks](quality-gates-and-benchmarks.md)
- [Paper Artifact Index](../paper_artifact_index.json)
- [Reviewer Quickstart](reviewer_quickstart.md)
- [Sample Manifest and Outputs](sample-manifest.md)
- [Performance Profiling Notes](performance-profiling-notes.md)
