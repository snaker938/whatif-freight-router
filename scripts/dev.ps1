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

function Test-DockerEngineReady {
  try {
    docker info 1>$null 2>$null
    return ($LASTEXITCODE -eq 0)
  } catch {
    return $false
  }
}

function Start-DockerDesktopIfAvailable {
  if (-not $IsWindows) {
    return $false
  }

  $candidates = @(
    (Join-Path $env:ProgramFiles "Docker\Docker\Docker Desktop.exe"),
    (Join-Path $env:LocalAppData "Docker\Docker Desktop.exe")
  ) | Where-Object { $_ -and (Test-Path $_) }

  $exe = $candidates | Select-Object -First 1
  if (-not $exe) {
    return $false
  }

  try {
    Start-Process -FilePath $exe | Out-Null
    return $true
  } catch {
    return $false
  }
}

function Ensure-DockerReady {
  if (Test-DockerEngineReady) {
    return
  }

  Write-Host "Docker engine not ready. Trying to start Docker Desktop..."
  $attemptedStart = Start-DockerDesktopIfAvailable

  $timeoutAt = (Get-Date).AddMinutes(2)
  while ((Get-Date) -lt $timeoutAt) {
    if (Test-DockerEngineReady) {
      Write-Host "Docker engine is ready."
      return
    }
    Start-Sleep -Seconds 3
  }

  if ($attemptedStart) {
    throw @"
Docker Desktop was started but the engine did not become ready in time.
Open Docker Desktop and wait until it reports 'Engine running', then rerun:
  pwsh ./scripts/dev.ps1
"@
  }

  throw @"
Docker engine is not available.
Start Docker Desktop (or Docker Engine service), then rerun:
  pwsh ./scripts/dev.ps1
"@
}

Ensure-DockerReady

Write-Host "Starting OSRM (Docker)..."
docker compose up -d osrm | Out-Null
if ($LASTEXITCODE -ne 0) {
  throw "Failed to start OSRM with docker compose. Check Docker and run: docker compose logs --tail=200 osrm"
}

$osrmContainerId = $null
$resolveContainerTimeout = (Get-Date).AddSeconds(20)
while ((Get-Date) -lt $resolveContainerTimeout) {
  $rawContainerId = docker compose ps -q osrm 2>$null

  if ($LASTEXITCODE -eq 0) {
    $firstLine = $rawContainerId | Select-Object -First 1
    if ($null -ne $firstLine) {
      $candidate = "$firstLine"
      if (-not [string]::IsNullOrWhiteSpace($candidate)) {
        $osrmContainerId = $candidate.Trim()
        break
      }
    }
  }

  Start-Sleep -Seconds 1
}

if ([string]::IsNullOrWhiteSpace($osrmContainerId)) {
  throw "Could not resolve OSRM container ID after startup. Check 'docker compose ps' and logs with: docker compose logs --tail=200 osrm"
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
