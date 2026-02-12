Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

if (-not (Test-Path ".env")) {
  if (Test-Path ".env.example") {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example"
  } else {
    throw "Missing .env and .env.example. Cannot continue."
  }
}

Write-Host "Starting OSRM (Docker)..."
docker compose up -d osrm | Out-Null

$osrmContainerId = (docker compose ps -q osrm).Trim()
if (-not $osrmContainerId) {
  throw "Could not resolve OSRM container ID. Check 'docker compose ps'."
}

$probeUrl = "http://localhost:5000/"
$timeoutAt = (Get-Date).AddMinutes(45)
Write-Host "Waiting for OSRM to become ready (first run can take a while)..."

while ($true) {
  $osrmState = docker inspect --format "{{.State.Status}} {{.State.ExitCode}}" $osrmContainerId 2>$null
  if ($osrmState -match "^exited ") {
    throw "OSRM container exited unexpectedly ($osrmState). Check logs with: docker compose logs --tail=200 osrm"
  }

  try {
    $resp = Invoke-WebRequest -UseBasicParsing -SkipHttpErrorCheck -Uri $probeUrl -TimeoutSec 5
    if ($resp.StatusCode -gt 0) {
      break
    }
  } catch {
    # Keep waiting while OSRM starts.
  }

  if ((Get-Date) -ge $timeoutAt) {
    throw "Timed out waiting for OSRM. Check logs with: docker compose logs -f osrm"
  }

  Start-Sleep -Seconds 5
}

function Test-PortListening {
  param(
    [Parameter(Mandatory = $true)][int]$Port
  )

  try {
    $null = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop
    return $true
  } catch {
    return $false
  }
}

$backendCmd = @'
$env:OSRM_BASE_URL = "http://localhost:5000"
$env:OUT_DIR = "out"
if (-not (Test-Path ".venv")) { uv sync --dev }
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
'@

$frontendCmd = @'
$env:BACKEND_INTERNAL_URL = "http://localhost:8000"
if (-not (Test-Path "node_modules")) { pnpm install }
pnpm dev
'@

if (Test-PortListening -Port 8000) {
  Write-Host "Port 8000 already in use. Skipping backend launch."
} else {
  $backendEncoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($backendCmd))
  Start-Process pwsh -WorkingDirectory (Join-Path $repoRoot "backend") -ArgumentList @(
    "-NoExit",
    "-EncodedCommand",
    $backendEncoded
  ) | Out-Null
}

if (Test-PortListening -Port 3000) {
  Write-Host "Port 3000 already in use. Skipping frontend launch."
} else {
  $frontendEncoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($frontendCmd))
  Start-Process pwsh -WorkingDirectory (Join-Path $repoRoot "frontend") -ArgumentList @(
    "-NoExit",
    "-EncodedCommand",
    $frontendEncoded
  ) | Out-Null
}

Write-Host ""
Write-Host "Startup complete:"
Write-Host "  Frontend: http://localhost:3000"
Write-Host "  Backend:  http://localhost:8000/docs"
Write-Host "  OSRM:     http://localhost:5000"
