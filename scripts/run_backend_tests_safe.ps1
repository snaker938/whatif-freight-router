[CmdletBinding()]
param(
  [string]$RepoRoot = (Join-Path $PSScriptRoot ".."),
  [string]$PythonExe = "",
  [string]$TestsDir = "backend/tests",
  [string]$TestPattern = "test_*.py",
  [int]$PerFileTimeoutSec = 900,
  [int]$NoOutputStallSec = 240,
  [int]$PollIntervalSec = 5,
  [int]$HeartbeatSec = 30,
  [int]$MaxCores = 2,
  [int]$MaxWorkingSetMB = 4096,
  [ValidateSet("Idle", "BelowNormal", "Normal")] [string]$PriorityClass = "BelowNormal",
  [string]$LogRoot = "backend/out/test_runs",
  [switch]$IncludeCoverage,
  [string[]]$ExtraPytestArgs = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Stop-ProcessTree {
  param([int]$ProcessId)
  try {
    & taskkill /PID $ProcessId /T /F 1>$null 2>$null
  } catch {
    # Best effort cleanup.
  }
}

function Get-FileSizeBytes {
  param([string]$Path)
  if (Test-Path $Path) {
    return (Get-Item $Path).Length
  }
  return 0L
}

function Get-LogicalCoreMask {
  param([int]$CoreCount)
  $mask = [uint64]0
  for ($i = 0; $i -lt $CoreCount; $i++) {
    $mask = $mask -bor ([uint64]1 -shl $i)
  }
  return [intptr]$mask
}

function Resolve-AbsolutePath {
  param(
    [string]$BaseDir,
    [string]$MaybeRelative
  )
  if ([IO.Path]::IsPathRooted($MaybeRelative)) {
    return (Resolve-Path $MaybeRelative).Path
  }
  return (Resolve-Path (Join-Path $BaseDir $MaybeRelative)).Path
}

$resolvedRepoRoot = (Resolve-Path $RepoRoot).Path
$backendDir = Join-Path $resolvedRepoRoot "backend"

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
  $PythonExe = Join-Path $backendDir ".venv/Scripts/python.exe"
}

if (-not (Test-Path $PythonExe)) {
  throw "Python executable not found: $PythonExe"
}

$testsDirPath = Resolve-AbsolutePath -BaseDir $resolvedRepoRoot -MaybeRelative $TestsDir
if (-not (Test-Path $testsDirPath)) {
  throw "Tests directory not found: $testsDirPath"
}

$logRootPath = if ([IO.Path]::IsPathRooted($LogRoot)) {
  $LogRoot
} else {
  Join-Path $resolvedRepoRoot $LogRoot
}

$runStamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$runDir = Join-Path $logRootPath $runStamp
$logsDir = Join-Path $runDir "logs"
New-Item -ItemType Directory -Path $logsDir -Force | Out-Null

$threadVars = @(
  "OMP_NUM_THREADS",
  "OPENBLAS_NUM_THREADS",
  "MKL_NUM_THREADS",
  "NUMEXPR_NUM_THREADS",
  "VECLIB_MAXIMUM_THREADS",
  "BLIS_NUM_THREADS"
)
foreach ($varName in $threadVars) {
  Set-Item -Path "Env:$varName" -Value "1"
}

$allTests = @(Get-ChildItem -Path $testsDirPath -File -Filter $TestPattern | Sort-Object Name)
$allTestsCount = $allTests.Count
if ($allTests.Count -eq 0) {
  throw "No test files found in $testsDirPath with pattern '$TestPattern'."
}

$logicalCores = [Environment]::ProcessorCount
if ($MaxCores -lt 1) { $MaxCores = 1 }
if ($MaxCores -gt $logicalCores) { $MaxCores = $logicalCores }
$coreMask = Get-LogicalCoreMask -CoreCount $MaxCores

$commonArgs = @("-m", "pytest", "-vv", "--disable-warnings", "--color=no")
if ($IncludeCoverage) {
  $commonArgs += @("--cov=backend/app", "--cov=backend/scripts", "--cov-report=term-missing")
}
if (@($ExtraPytestArgs).Count -gt 0) {
  $commonArgs += $ExtraPytestArgs
}

$results = New-Object System.Collections.Generic.List[object]
$suiteStart = Get-Date

Write-Host "Repo root: $resolvedRepoRoot"
Write-Host "Python:    $PythonExe"
Write-Host "Tests dir: $testsDirPath"
Write-Host "Run dir:   $runDir"
Write-Host "Files:     $allTestsCount"
Write-Host "Throttle:  priority=$PriorityClass, max_cores=$MaxCores, max_ram_mb=$MaxWorkingSetMB"
Write-Host ""

$idx = 0
foreach ($testFile in $allTests) {
  $idx++
  $safeBase = [IO.Path]::GetFileNameWithoutExtension($testFile.Name)
  $stdoutLog = Join-Path $logsDir ("{0:000}_{1}.stdout.log" -f $idx, $safeBase)
  $stderrLog = Join-Path $logsDir ("{0:000}_{1}.stderr.log" -f $idx, $safeBase)

  $args = @()
  $args += $commonArgs
  $args += $testFile.FullName

  Write-Host ("[{0}/{1}] START {2}" -f $idx, $allTestsCount, $testFile.Name)

  $start = Get-Date
  $status = "unknown"
  $exitCode = $null
  $reason = ""
  $peakRamMB = 0.0

  $proc = Start-Process `
    -FilePath $PythonExe `
    -ArgumentList $args `
    -WorkingDirectory $backendDir `
    -NoNewWindow `
    -PassThru `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog

  Start-Sleep -Milliseconds 200
  try {
    if (-not $proc.HasExited) {
      $proc.PriorityClass = $PriorityClass
      $proc.ProcessorAffinity = $coreMask
    }
  } catch {
    # Best effort only; continue even if the OS denies these settings.
  }

  $lastOutBytes = Get-FileSizeBytes -Path $stdoutLog
  $lastErrBytes = Get-FileSizeBytes -Path $stderrLog
  $lastOutputAt = Get-Date
  $lastCpu = 0.0
  $nextHeartbeat = (Get-Date).AddSeconds($HeartbeatSec)

  while (-not $proc.HasExited) {
    Start-Sleep -Seconds $PollIntervalSec

    $now = Get-Date
    $elapsedSec = ($now - $start).TotalSeconds
    $outBytes = Get-FileSizeBytes -Path $stdoutLog
    $errBytes = Get-FileSizeBytes -Path $stderrLog
    $processInfo = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue

    $cpuNow = 0.0
    $ramMB = 0.0
    if ($null -ne $processInfo) {
      $cpuNow = [double]$processInfo.CPU
      $ramMB = [Math]::Round(($processInfo.WorkingSet64 / 1MB), 1)
      if ($ramMB -gt $peakRamMB) {
        $peakRamMB = $ramMB
      }
    }

    if (($outBytes -ne $lastOutBytes) -or ($errBytes -ne $lastErrBytes)) {
      $lastOutputAt = $now
      $lastOutBytes = $outBytes
      $lastErrBytes = $errBytes
    }

    $idleSec = ($now - $lastOutputAt).TotalSeconds
    $cpuDelta = $cpuNow - $lastCpu
    $lastCpu = $cpuNow

    if ($now -ge $nextHeartbeat) {
      Write-Host ("[{0}] RUN {1}s | idle {2}s | cpu {3}s (+{4}s) | ram {5}MB" -f `
          $testFile.Name, `
          [int]$elapsedSec, `
          [int]$idleSec, `
          [Math]::Round($cpuNow, 1), `
          [Math]::Round($cpuDelta, 2), `
          $ramMB)
      $nextHeartbeat = $now.AddSeconds($HeartbeatSec)
    }

    if (($MaxWorkingSetMB -gt 0) -and ($ramMB -gt $MaxWorkingSetMB)) {
      $status = "memory_limit_exceeded"
      $reason = "Working set ${ramMB}MB > ${MaxWorkingSetMB}MB"
      Stop-ProcessTree -ProcessId $proc.Id
      break
    }

    if ($elapsedSec -ge $PerFileTimeoutSec) {
      $status = "timeout"
      $reason = "Elapsed ${elapsedSec}s exceeded ${PerFileTimeoutSec}s"
      Stop-ProcessTree -ProcessId $proc.Id
      break
    }

    if ($idleSec -ge $NoOutputStallSec) {
      if ($cpuDelta -lt 0.1) {
        $status = "stalled_no_output_idle"
        $reason = "No output for ${idleSec}s and CPU mostly idle"
      } else {
        $status = "stalled_no_output_busy"
        $reason = "No output for ${idleSec}s while CPU still active"
      }
      Stop-ProcessTree -ProcessId $proc.Id
      break
    }
  }

  if ($proc.HasExited) {
    $exitCode = $proc.ExitCode
    if ($status -eq "unknown") {
      if ($exitCode -eq 0) {
        $status = "passed"
      } else {
        $status = "failed"
      }
    }
  } else {
    # Ensure process is fully gone if we terminated it for guard conditions.
    Stop-ProcessTree -ProcessId $proc.Id
    $proc.Refresh()
    $exitCode = $proc.ExitCode
    if ($status -eq "unknown") {
      $status = "terminated"
    }
  }

  $durationSec = [Math]::Round(((Get-Date) - $start).TotalSeconds, 2)
  $outBytesFinal = Get-FileSizeBytes -Path $stdoutLog
  $errBytesFinal = Get-FileSizeBytes -Path $stderrLog

  $record = [PSCustomObject]@{
    test_file       = $testFile.Name
    status          = $status
    exit_code       = $exitCode
    duration_sec    = $durationSec
    peak_ram_mb     = $peakRamMB
    stdout_log      = $stdoutLog
    stderr_log      = $stderrLog
    stdout_bytes    = $outBytesFinal
    stderr_bytes    = $errBytesFinal
    reason          = $reason
  }
  $results.Add($record) | Out-Null

  if ($status -eq "passed") {
    Write-Host ("[{0}/{1}] PASS {2} ({3}s)" -f $idx, $allTestsCount, $testFile.Name, $durationSec) -ForegroundColor Green
  } else {
    Write-Host ("[{0}/{1}] {2} {3} ({4}s)" -f $idx, $allTestsCount, $status.ToUpperInvariant(), $testFile.Name, $durationSec) -ForegroundColor Yellow
    if (-not [string]::IsNullOrWhiteSpace($reason)) {
      Write-Host ("           reason: {0}" -f $reason) -ForegroundColor DarkYellow
    }
  }
}

$suiteDurationSec = [Math]::Round(((Get-Date) - $suiteStart).TotalSeconds, 2)

$summaryCsv = Join-Path $runDir "summary.csv"
$summaryJson = Join-Path $runDir "summary.json"
$failedList = Join-Path $runDir "failed_tests.txt"
$rerunScript = Join-Path $runDir "rerun_failed.ps1"

$results | Export-Csv -Path $summaryCsv -NoTypeInformation -Encoding UTF8
($results | ConvertTo-Json -Depth 5) | Set-Content -Path $summaryJson -Encoding UTF8

$failed = @($results | Where-Object { $_.status -ne "passed" })
$failed | ForEach-Object { $_.test_file } | Set-Content -Path $failedList -Encoding UTF8

$rerunLines = @(
  "Set-StrictMode -Version Latest",
  '$ErrorActionPreference = "Stop"',
  ('$repoRoot = "{0}"' -f $resolvedRepoRoot),
  ('Set-Location (Join-Path $repoRoot "backend")'),
  ('$py = "{0}"' -f $PythonExe)
)
if ($failed.Count -eq 0) {
  $rerunLines += 'Write-Host "No failed tests to rerun."'
} else {
  foreach ($f in $failed) {
    $testPath = Join-Path $testsDirPath $f.test_file
    $rerunLines += ('& $py -m pytest -vv --disable-warnings --color=no "{0}"' -f $testPath)
  }
}
$rerunLines | Set-Content -Path $rerunScript -Encoding UTF8

$passedCount = @($results | Where-Object { $_.status -eq "passed" }).Count
$failedCount = $failed.Count

Write-Host ""
Write-Host "Run complete in ${suiteDurationSec}s"
Write-Host ("Passed: {0} | Non-passed: {1} | Total: {2}" -f $passedCount, $failedCount, $results.Count)
Write-Host "Summary CSV:  $summaryCsv"
Write-Host "Summary JSON: $summaryJson"
Write-Host "Failed list:  $failedList"
Write-Host "Rerun script: $rerunScript"

if ($failedCount -gt 0) {
  Write-Host ""
  Write-Host "Non-passing tests:" -ForegroundColor Yellow
  $failed | ForEach-Object {
    Write-Host ("- {0} [{1}]" -f $_.test_file, $_.status) -ForegroundColor Yellow
  }
}
