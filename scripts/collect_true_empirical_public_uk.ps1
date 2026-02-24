[CmdletBinding()]
param(
  [string]$RepoRoot = (Join-Path $PSScriptRoot ".."),
  [int]$MaxRetries = 3,
  [int]$RetryBackoffSec = 8,
  [switch]$SkipScenarioBatch,
  [switch]$SkipFullModelBuild
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Assert-PathExists {
  param(
    [string]$Path,
    [string]$Label
  )
  if (-not (Test-Path $Path)) {
    throw "$Label missing: $Path"
  }
}

function Invoke-Step {
  param(
    [string]$Name,
    [scriptblock]$Action,
    [int]$Retries = 3,
    [int]$BackoffSec = 8
  )

  $attempt = 0
  $lastError = $null
  $started = Get-Date
  while ($attempt -lt [Math]::Max(1, $Retries)) {
    $attempt++
    Write-Host ("[{0}] attempt {1}/{2}" -f $Name, $attempt, $Retries)
    try {
      & $Action
      $duration = [Math]::Round(((Get-Date) - $started).TotalSeconds, 2)
      Write-Host ("[{0}] PASS ({1}s)" -f $Name, $duration)
      return @{
        name = $Name
        status = "passed"
        attempts = $attempt
        duration_s = $duration
      }
    } catch {
      $lastError = $_
      Write-Host ("[{0}] FAIL attempt {1}: {2}" -f $Name, $attempt, $_.Exception.Message)
      if ($attempt -lt [Math]::Max(1, $Retries)) {
        Start-Sleep -Seconds ([Math]::Max(1, $BackoffSec) * $attempt)
      }
    }
  }
  $durationFinal = [Math]::Round(((Get-Date) - $started).TotalSeconds, 2)
  throw ("[{0}] failed after {1} attempts in {2}s: {3}" -f $Name, $attempt, $durationFinal, $lastError.Exception.Message)
}

function Run-UvPython {
  param([string[]]$Args)
  & uv run --project backend python @Args
  if ($LASTEXITCODE -ne 0) {
    throw "uv run failed with exit code $LASTEXITCODE for args: $($Args -join ' ')"
  }
}

$resolvedRepoRoot = (Resolve-Path $RepoRoot).Path
Set-Location $resolvedRepoRoot

$summaryOut = Join-Path $resolvedRepoRoot "backend/out/model_assets/collect_true_empirical_public_uk.summary.json"
$summaryDir = Split-Path -Parent $summaryOut
New-Item -ItemType Directory -Path $summaryDir -Force | Out-Null

$results = New-Object System.Collections.Generic.List[object]
$suiteStarted = Get-Date

Write-Host "Repo root: $resolvedRepoRoot"
Write-Host "Max retries: $MaxRetries | Backoff: $RetryBackoffSec"
Write-Host ""

if (-not $SkipScenarioBatch) {
  $results.Add((Invoke-Step -Name "scenario_batch" -Retries $MaxRetries -BackoffSec $RetryBackoffSec -Action {
        Run-UvPython -Args @(
          "backend/scripts/fetch_scenario_live_uk.py",
          "--batch",
          "--output", "backend/out/model_assets/scenario_live_batch_summary.json",
          "--output-jsonl", "backend/data/raw/uk/scenario_live_observed.jsonl",
          "--day-kinds", "weekday,weekend",
          "--hour-slots", "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23",
          "--workers", "12",
          "--allow-partial-sources"
        )
        Assert-PathExists -Path "backend/data/raw/uk/scenario_live_observed.jsonl" -Label "Scenario corpus"
      }))
}

$results.Add((Invoke-Step -Name "collect_dft_raw" -Retries $MaxRetries -BackoffSec $RetryBackoffSec -Action {
      Run-UvPython -Args @(
        "backend/scripts/collect_dft_raw_counts_uk.py",
        "--output", "backend/data/raw/uk/dft_counts_raw.csv",
        "--years", "2023,2024,2025,2026",
        "--max-pages-per-query", "50",
        "--target-min-rows", "200000"
      )
      Assert-PathExists -Path "backend/data/raw/uk/dft_counts_raw.csv" -Label "DfT raw counts"
    }))

$results.Add((Invoke-Step -Name "collect_stochastic_raw" -Retries $MaxRetries -BackoffSec $RetryBackoffSec -Action {
      Run-UvPython -Args @(
        "backend/scripts/collect_stochastic_residuals_raw_uk.py",
        "--scenario-jsonl", "backend/data/raw/uk/scenario_live_observed.jsonl",
        "--dft-raw", "backend/data/raw/uk/dft_counts_raw.csv",
        "--output", "backend/data/raw/uk/stochastic_residuals_raw.csv",
        "--target-min-rows", "50000"
      )
      Assert-PathExists -Path "backend/data/raw/uk/stochastic_residuals_raw.csv" -Label "Stochastic residual raw"
    }))

$results.Add((Invoke-Step -Name "collect_fuel_raw" -Retries $MaxRetries -BackoffSec $RetryBackoffSec -Action {
      Run-UvPython -Args @(
        "backend/scripts/collect_fuel_history_raw_uk.py",
        "--output", "backend/data/raw/uk/fuel_prices_raw.json",
        "--min-days", "1095"
      )
      Assert-PathExists -Path "backend/data/raw/uk/fuel_prices_raw.json" -Label "Fuel raw"
    }))

$results.Add((Invoke-Step -Name "collect_carbon_raw" -Retries $MaxRetries -BackoffSec $RetryBackoffSec -Action {
      Run-UvPython -Args @(
        "backend/scripts/collect_carbon_intensity_raw_uk.py",
        "--output", "backend/data/raw/uk/carbon_intensity_hourly_raw.json",
        "--regions", "uk_default,london,south_east,midlands,scotland,wales,north_west,north_east"
      )
      Assert-PathExists -Path "backend/data/raw/uk/carbon_intensity_hourly_raw.json" -Label "Carbon raw"
    }))

$results.Add((Invoke-Step -Name "collect_toll_raw" -Retries $MaxRetries -BackoffSec $RetryBackoffSec -Action {
      Run-UvPython -Args @(
        "backend/scripts/collect_toll_truth_raw_uk.py",
        "--classification-out", "backend/data/raw/uk/toll_classification",
        "--pricing-out", "backend/data/raw/uk/toll_pricing",
        "--tariffs-out", "backend/data/raw/uk/toll_tariffs_operator_truth.json",
        "--classification-target", "220",
        "--pricing-target", "100",
        "--min-tariff-rules", "220"
      )
      Assert-PathExists -Path "backend/data/raw/uk/toll_tariffs_operator_truth.json" -Label "Toll tariffs raw"
      Assert-PathExists -Path "backend/data/raw/uk/toll_classification" -Label "Toll classification raw dir"
      Assert-PathExists -Path "backend/data/raw/uk/toll_pricing" -Label "Toll pricing raw dir"
    }))

$results.Add((Invoke-Step -Name "collect_mode_outcomes_proxy" -Retries $MaxRetries -BackoffSec $RetryBackoffSec -Action {
      Run-UvPython -Args @(
        "backend/scripts/collect_scenario_mode_outcomes_proxy_uk.py",
        "--scenario-jsonl", "backend/data/raw/uk/scenario_live_observed.jsonl",
        "--output", "backend/data/raw/uk/scenario_mode_outcomes_observed.jsonl"
      )
      Assert-PathExists -Path "backend/data/raw/uk/scenario_mode_outcomes_observed.jsonl" -Label "Scenario mode outcomes"
    }))

$results.Add((Invoke-Step -Name "build_departure_counts" -Retries $MaxRetries -BackoffSec $RetryBackoffSec -Action {
      Run-UvPython -Args @(
        "backend/scripts/fetch_dft_counts_uk.py",
        "--raw-csv", "backend/data/raw/uk/dft_counts_raw.csv",
        "--output", "backend/assets/uk/departure_counts_empirical.csv",
        "--min-rows", "20000",
        "--min-unique-regions", "10",
        "--min-unique-road-buckets", "5",
        "--min-unique-hours", "24"
      )
      Assert-PathExists -Path "backend/assets/uk/departure_counts_empirical.csv" -Label "Departure empirical asset"
    }))

$results.Add((Invoke-Step -Name "build_stochastic_residuals" -Retries $MaxRetries -BackoffSec $RetryBackoffSec -Action {
      Run-UvPython -Args @(
        "backend/scripts/fetch_stochastic_residuals_uk.py",
        "--raw-csv", "backend/data/raw/uk/stochastic_residuals_raw.csv",
        "--output", "backend/assets/uk/stochastic_residuals_empirical.csv",
        "--target-rows", "50000",
        "--min-unique-regimes", "6",
        "--min-unique-road-buckets", "5",
        "--min-unique-weather-profiles", "4",
        "--min-unique-vehicle-types", "3",
        "--min-unique-local-slots", "12",
        "--min-unique-corridors", "8"
      )
      Assert-PathExists -Path "backend/assets/uk/stochastic_residuals_empirical.csv" -Label "Stochastic empirical asset"
    }))

$results.Add((Invoke-Step -Name "build_fuel_and_carbon_assets" -Retries $MaxRetries -BackoffSec $RetryBackoffSec -Action {
      Run-UvPython -Args @(
        "backend/scripts/fetch_fuel_history_uk.py",
        "--source", "backend/data/raw/uk/fuel_prices_raw.json",
        "--output", "backend/assets/uk/fuel_prices_uk.json",
        "--min-days", "1095"
      )
      Run-UvPython -Args @(
        "backend/scripts/fetch_carbon_intensity_uk.py",
        "--intensity-source", "backend/data/raw/uk/carbon_intensity_hourly_raw.json",
        "--intensity-output", "backend/assets/uk/carbon_intensity_hourly_uk.json",
        "--schedule", "backend/assets/uk/carbon_price_schedule_uk.json"
      )
      Assert-PathExists -Path "backend/assets/uk/fuel_prices_uk.json" -Label "Fuel asset"
      Assert-PathExists -Path "backend/assets/uk/carbon_intensity_hourly_uk.json" -Label "Carbon asset"
    }))

$results.Add((Invoke-Step -Name "build_toll_truth_assets" -Retries $MaxRetries -BackoffSec $RetryBackoffSec -Action {
      Run-UvPython -Args @(
        "backend/scripts/fetch_toll_truth_uk.py",
        "--classification-source", "backend/data/raw/uk/toll_classification",
        "--pricing-source", "backend/data/raw/uk/toll_pricing",
        "--classification-out", "backend/tests/fixtures/toll_classification",
        "--pricing-out", "backend/tests/fixtures/toll_pricing",
        "--classification-target", "220",
        "--pricing-target", "100",
        "--calibration-out", "backend/assets/uk/toll_confidence_calibration_uk.json"
      )
      Assert-PathExists -Path "backend/assets/uk/toll_confidence_calibration_uk.json" -Label "Toll calibration asset"
    }))

$results.Add((Invoke-Step -Name "build_scenario_profiles" -Retries $MaxRetries -BackoffSec $RetryBackoffSec -Action {
      Run-UvPython -Args @(
        "backend/scripts/build_scenario_profiles_uk.py",
        "--raw-jsonl", "backend/data/raw/uk/scenario_live_observed.jsonl",
        "--observed-modes-jsonl", "backend/data/raw/uk/scenario_mode_outcomes_observed.jsonl",
        "--min-contexts", "24",
        "--min-observed-mode-row-share", "0.0",
        "--max-projection-dominant-context-share", "1.0",
        "--output", "backend/assets/uk/scenario_profiles_uk.json"
      )
      Assert-PathExists -Path "backend/assets/uk/scenario_profiles_uk.json" -Label "Scenario profile asset"
    }))

if (-not $SkipFullModelBuild) {
  $results.Add((Invoke-Step -Name "build_model_assets_full" -Retries $MaxRetries -BackoffSec $RetryBackoffSec -Action {
        Run-UvPython -Args @(
          "backend/scripts/build_model_assets.py",
          "--force-rebuild-topology",
          "--force-rebuild-graph",
          "--force-rebuild-terrain"
        )
        Assert-PathExists -Path "backend/out/model_assets" -Label "Compiled model asset dir"
      }))

  $results.Add((Invoke-Step -Name "publish_live_artifacts" -Retries $MaxRetries -BackoffSec $RetryBackoffSec -Action {
        Run-UvPython -Args @(
          "backend/scripts/publish_live_artifacts_uk.py"
        )
        Assert-PathExists -Path "backend/assets/uk/departure_profiles_uk.json" -Label "Published departure profile asset"
        Assert-PathExists -Path "backend/assets/uk/stochastic_regimes_uk.json" -Label "Published stochastic regime asset"
        Assert-PathExists -Path "backend/assets/uk/toll_topology_uk.json" -Label "Published toll topology asset"
        Assert-PathExists -Path "backend/assets/uk/toll_tariffs_uk.json" -Label "Published toll tariffs asset"
      }))
}

$suiteFinished = Get-Date
$summary = @{
  run_started_utc = $suiteStarted.ToUniversalTime().ToString("o")
  run_finished_utc = $suiteFinished.ToUniversalTime().ToString("o")
  duration_s = [Math]::Round(($suiteFinished - $suiteStarted).TotalSeconds, 2)
  repo_root = $resolvedRepoRoot
  steps = $results
}

$summary | ConvertTo-Json -Depth 8 | Set-Content -Path $summaryOut -Encoding UTF8
Write-Host ""
Write-Host "All steps complete."
Write-Host "Summary: $summaryOut"
