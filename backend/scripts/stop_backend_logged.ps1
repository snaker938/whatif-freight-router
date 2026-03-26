param(
    [int]$Port = 8000
)

$backendRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pidFile = Join-Path $backendRoot "out\backend_server.pid"

function Get-ProcessInfoById {
    param(
        [int]$ProcessId
    )

    if ($ProcessId -le 0) {
        return $null
    }

    try {
        $proc = Get-CimInstance Win32_Process -Filter ("ProcessId = {0}" -f $ProcessId) -ErrorAction Stop
        return [pscustomobject]@{
            ProcessId   = [int]$proc.ProcessId
            Name        = [string]$proc.Name
            CommandLine = [string]$proc.CommandLine
        }
    } catch {
        return $null
    }
}

function Stop-BackendProcess {
    param(
        [int]$ProcessId
    )

    if ($ProcessId -le 0) {
        return
    }

    Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
    try {
        Wait-Process -Id $ProcessId -Timeout 15 -ErrorAction Stop
    } catch {
    }
}

function Get-BackendListenerProcess {
    param(
        [int]$TargetPort
    )

    $listener = Get-NetTCPConnection -LocalPort $TargetPort -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($null -eq $listener) {
        return $null
    }

    try {
        $proc = Get-CimInstance Win32_Process -Filter ("ProcessId = {0}" -f [int]$listener.OwningProcess) -ErrorAction Stop
        return [pscustomobject]@{
            ProcessId   = [int]$proc.ProcessId
            Name        = [string]$proc.Name
            CommandLine = [string]$proc.CommandLine
        }
    } catch {
        return [pscustomobject]@{
            ProcessId   = [int]$listener.OwningProcess
            Name        = ""
            CommandLine = ""
        }
    }
}

function Get-BackendProcesses {
    $rows = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^python(3)?\.exe$' -and $_.CommandLine -match 'uvicorn\s+app\.main:app' }
    if ($null -eq $rows) {
        return @()
    }
    return @($rows | ForEach-Object {
        [pscustomobject]@{
            ProcessId   = [int]$_.ProcessId
            Name        = [string]$_.Name
            CommandLine = [string]$_.CommandLine
        }
    })
}

function Test-IsBackendProcess {
    param(
        [object]$ProcessInfo
    )

    if ($null -eq $ProcessInfo) {
        return $false
    }
    $commandLine = [string]$ProcessInfo.CommandLine
    return $commandLine.Contains("uvicorn") -and $commandLine.Contains("app.main:app")
}

function Wait-BackendPortClear {
    param(
        [int]$TargetPort,
        [int]$TimeoutSeconds = 15
    )

    $deadline = (Get-Date).AddSeconds([Math]::Max(1, $TimeoutSeconds))
    do {
        $listener = Get-BackendListenerProcess -TargetPort $TargetPort
        if ($null -eq $listener) {
            return $true
        }
        Start-Sleep -Milliseconds 250
    } while ((Get-Date) -lt $deadline)
    return $false
}

$stopped = $false

if (Test-Path $pidFile) {
    $serverPid = Get-Content $pidFile | Select-Object -First 1
    if ($serverPid) {
        $serverInfo = Get-ProcessInfoById -ProcessId ([int]$serverPid)
        if (Test-IsBackendProcess -ProcessInfo $serverInfo) {
            Stop-BackendProcess -ProcessId ([int]$serverPid)
            $stopped = $true
        }
    }
}

$listenerProcess = Get-BackendListenerProcess -TargetPort $Port
if (($null -ne $listenerProcess) -and (Test-IsBackendProcess -ProcessInfo $listenerProcess)) {
    Stop-BackendProcess -ProcessId ([int]$listenerProcess.ProcessId)
    $stopped = $true
}

foreach ($backendProcess in (Get-BackendProcesses)) {
    Stop-BackendProcess -ProcessId ([int]$backendProcess.ProcessId)
    $stopped = $true
}

if ($stopped) {
    [void](Wait-BackendPortClear -TargetPort $Port -TimeoutSeconds 15)
}

if (Test-Path $pidFile) {
    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
}

if ($stopped) {
    Write-Output "stopped"
    exit 0
}

Write-Output "no_backend_process"
