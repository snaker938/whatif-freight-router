param(
    [int]$Port = 8000
)

$backendRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$outDir = Join-Path $backendRoot "out"
$primaryPidFile = Join-Path $outDir ("backend_server_{0}.pid" -f $Port)
$pidFiles = @($primaryPidFile)
if ($Port -eq 8000) {
    $pidFiles += Join-Path $outDir "backend_server.pid"
}

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
        Where-Object {
            ($_.Name -match '^python(3)?\.exe$' -and $_.CommandLine -match 'uvicorn\s+app\.main:app') -or
            ($_.Name -match '^powershell(\.exe)?$|^pwsh(\.exe)?$' -and $_.CommandLine -match 'run_with_job_memory_limit\.ps1' -and $_.CommandLine -match 'uvicorn' -and $_.CommandLine -match 'app\.main:app')
        }
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
    return (
        $commandLine.Contains("uvicorn") -and $commandLine.Contains("app.main:app")
    ) -or (
        $commandLine.Contains("run_with_job_memory_limit.ps1") -and
        $commandLine.Contains("uvicorn") -and
        $commandLine.Contains("app.main:app")
    )
}

function Test-BackendProcessMatchesPort {
    param(
        [object]$ProcessInfo,
        [int]$TargetPort
    )

    if (-not (Test-IsBackendProcess -ProcessInfo $ProcessInfo)) {
        return $false
    }

    $commandLine = [string]$ProcessInfo.CommandLine
    $escapedPort = [regex]::Escape([string]$TargetPort)
    return [regex]::IsMatch($commandLine, "--port(?:'|`"|,|\s)+$escapedPort\b")
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

foreach ($pidFile in $pidFiles) {
    if (Test-Path $pidFile) {
        $serverPid = Get-Content $pidFile | Select-Object -First 1
        if ($serverPid) {
            $serverInfo = Get-ProcessInfoById -ProcessId ([int]$serverPid)
            if (Test-BackendProcessMatchesPort -ProcessInfo $serverInfo -TargetPort $Port) {
                Stop-BackendProcess -ProcessId ([int]$serverPid)
                $stopped = $true
            }
        }
    }
}

$listenerProcess = Get-BackendListenerProcess -TargetPort $Port
if (($null -ne $listenerProcess) -and (Test-IsBackendProcess -ProcessInfo $listenerProcess)) {
    Stop-BackendProcess -ProcessId ([int]$listenerProcess.ProcessId)
    $stopped = $true
}

if ($stopped) {
    [void](Wait-BackendPortClear -TargetPort $Port -TimeoutSeconds 15)
}

foreach ($pidFile in $pidFiles) {
    if (Test-Path $pidFile) {
        Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
    }
}

if ($stopped) {
    Write-Output "stopped"
    exit 0
}

Write-Output "no_backend_process"
