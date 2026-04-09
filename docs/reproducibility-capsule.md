# Reproducibility Capsule

Last Updated: 2026-04-09
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
- generated/as-of `2026-03-21T13:09:12.262992Z`
- source policy `repo_local_fresh`
- 19 built assets
- manifest signature `87270329a4e941f63c991fc3edf23298ecdaad803a74e2ab5388a16db4690d0a`

`backend/out/model_assets/refresh_manifest.json` fixes the repo-local input set used for that build, including hashes and as-of timestamps for:

- `backend/data/raw/uk/dft_counts_raw.csv`
- `backend/data/raw/uk/stochastic_residuals_raw.csv`
- `backend/data/raw/uk/scenario_live_observed.jsonl`
- `backend/data/raw/uk/scenario_mode_outcomes_observed.jsonl`
- `backend/data/raw/uk/fuel_prices_raw.json`
- `backend/data/raw/uk/carbon_intensity_hourly_raw.json`
- `backend/data/raw/uk/toll_tariffs_operator_truth.json`
- toll classification/pricing corpora

`backend/out/model_assets/live_publish_summary.json` records the publish handoff at `2026-03-21T13:09:18Z`, including:

- scenario signature prefix `dbca97d56394`
- fuel signature prefix `02ec87074710`

### Strict-readiness anchors

The latest checked local strict preflight is `backend/out/model_assets/preflight_live_runtime.json` at `2026-04-04T15:48:39Z`, with `required_ok: true` and zero required failures.

The latest checked thesis-bundle repo preflight is `backend/out/thesis_campaigns/dominance_cluster5_cardiff_bath_corr12p5_r2/tranche_001/artifacts/dominance_cluster5_cardiff_bath_corr12p5_r2_t001/repo_asset_preflight.json` at `2026-04-06T09:36:17Z`, also with `required_ok: true`.

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
- `backend/out/provenance/{run_id}.jsonl`

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
- `backend/out/thesis_campaigns/dominance_cluster5_cardiff_bath_corr12p5_r2/tranche_001/artifacts/dominance_cluster5_cardiff_bath_corr12p5_r2_t001/methods_appendix.md`

Recommended metadata bundle:

1. git commit SHA
2. `.env` hash or sanitized config snapshot
3. model-asset manifest signature
4. benchmark or thesis command line
5. seed, budget, and comparator-policy settings

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
- [Sample Manifest and Outputs](sample-manifest.md)
- [Performance Profiling Notes](performance-profiling-notes.md)
