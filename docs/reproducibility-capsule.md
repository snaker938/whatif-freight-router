# Reproducibility Capsule

Last Updated: 2026-02-19  
Applies To: deterministic benchmark and artifact workflows

## One-Command Demo

From repo root:

```powershell
.\scripts\demo_repro_run.ps1
```

## Manual Equivalent

From `backend/`:

```powershell
uv run python scripts/benchmark_batch_pareto.py --mode inprocess-fake --pair-count 100 --seed 20260212
```

## Reproducibility Controls

- fixed seed
- fixed OD corpus
- fixed model assets manifest version
- unchanged settings and code revision

## Related Docs

- [Documentation Index](README.md)
- [Quality Gates and Benchmarks](quality-gates-and-benchmarks.md)
- [Sample Manifest and Outputs](sample-manifest.md)
- [ETA Concept Drift Checks](eta-concept-drift.md)
