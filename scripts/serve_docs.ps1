[CmdletBinding()]
param(
  [int]$Port = 8088,
  [switch]$OpenBrowser
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$docsDir = Join-Path $repoRoot "docs"

if (-not (Test-Path $docsDir)) {
  throw "Docs directory not found: $docsDir"
}

$pythonExe = $null
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if ($null -ne $pythonCmd) {
  $pythonExe = $pythonCmd.Source
}

if ([string]::IsNullOrWhiteSpace($pythonExe)) {
  $venvPython = Join-Path $repoRoot "backend/.venv/Scripts/python.exe"
  if (Test-Path $venvPython) {
    $pythonExe = $venvPython
  }
}

if ([string]::IsNullOrWhiteSpace($pythonExe)) {
  throw "Python executable not found. Install Python or create backend/.venv."
}

$url = "http://localhost:$Port/"
Write-Host "Serving docs from: $docsDir"
Write-Host "URL: $url"
Write-Host "Press Ctrl+C to stop."

if ($OpenBrowser) {
  Start-Process $url | Out-Null
}

& $pythonExe -m http.server $Port --directory $docsDir
