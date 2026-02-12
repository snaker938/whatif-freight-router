Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$backendDir = Join-Path $repoRoot "backend"
$capsuleDir = Join-Path $backendDir "out/capsule"

New-Item -ItemType Directory -Path $capsuleDir -Force | Out-Null

$timestamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$outputPath = Join-Path $capsuleDir ("repro_capsule_" + $timestamp + ".json")

Push-Location $backendDir
try {
  uv run python scripts/benchmark_batch_pareto.py `
    --mode inprocess-fake `
    --pair-count 100 `
    --seed 20260212 `
    --max-alternatives 3 `
    --output $outputPath
} finally {
  Pop-Location
}

Write-Host "Reproducibility capsule written to: $outputPath"
