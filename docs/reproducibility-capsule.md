# Reproducibility Capsule

Last Updated: 2026-02-23  
Applies To: deterministic benchmark and artifact workflows in `backend/out/*`

## One-Command Repro Demo

From repo root:

```powershell
.\scripts\demo_repro_run.ps1
```

## Manual Repro Path

From `backend/`:

```powershell
uv run python scripts/build_model_assets.py
uv run python scripts/benchmark_batch_pareto.py --mode inprocess-fake --pair-count 100 --seed 20260212
```

## Reproducibility Controls

- fixed seed (`--seed`)
- fixed OD corpus / fixture set
- fixed code revision and config
- fixed strict policy behavior (`STRICT_RUNTIME_TEST_BYPASS` value explicit)
- fixed model assets snapshot

## Reproducibility Artifacts to Archive

- `backend/out/manifests/{run_id}.json`
- `backend/out/scenario_manifests/{run_id}.json`
- `backend/out/artifacts/{run_id}/metadata.json`
- `backend/out/artifacts/{run_id}/results.json`
- `backend/out/provenance/{run_id}.jsonl`

Recommended metadata bundle:

1. git commit SHA
2. `.env` hash (not raw secrets)
3. model asset directory hash snapshot
4. benchmark command line and seed

## Comparing Two Runs

Use these checks before attributing differences to model changes:

1. same route/request payloads
2. same scenario mode and stochastic parameters
3. same strict bypass mode (`STRICT_RUNTIME_TEST_BYPASS`)
4. same asset versions in manifests
5. no stale/missing live-source warnings in provenance

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Quality Gates and Benchmarks](quality-gates-and-benchmarks.md)
- [Sample Manifest and Outputs](sample-manifest.md)
- [Performance Profiling Notes](performance-profiling-notes.md)

