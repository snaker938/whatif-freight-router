param(
    [int]$Port = 8000
)

$backendRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $backendRoot ".venv\Scripts\python.exe"
$outDir = Join-Path $backendRoot "out"
$stdout = Join-Path $outDir "backend_stdout.log"
$stderr = Join-Path $outDir "backend_stderr.log"
$pidFile = Join-Path $outDir "backend_server.pid"

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

if (-not (Test-Path $python)) {
    Write-Error ("Backend virtualenv python not found: {0}" -f $python)
    exit 1
}

New-Item -ItemType Directory -Force -Path $outDir | Out-Null

if (Test-Path $pidFile) {
    $existingPid = Get-Content $pidFile | Select-Object -First 1
    if ($existingPid) {
        $existingInfo = Get-ProcessInfoById -ProcessId ([int]$existingPid)
        if (Test-IsBackendProcess -ProcessInfo $existingInfo) {
            Stop-BackendProcess -ProcessId ([int]$existingPid)
        }
    }
    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
}

foreach ($backendProcess in (Get-BackendProcesses)) {
    Stop-BackendProcess -ProcessId ([int]$backendProcess.ProcessId)
}

$listenerProcess = Get-BackendListenerProcess -TargetPort $Port
if ($null -ne $listenerProcess) {
    if (Test-IsBackendProcess -ProcessInfo $listenerProcess) {
        Stop-BackendProcess -ProcessId ([int]$listenerProcess.ProcessId)
        if (-not (Wait-BackendPortClear -TargetPort $Port -TimeoutSeconds 15)) {
            Write-Error ("Timed out waiting for backend port {0} to clear." -f $Port)
            exit 1
        }
    } else {
        Write-Error (
            "Port {0} is already in use by pid={1} name={2}; refusing to start backend." -f
            $Port, [int]$listenerProcess.ProcessId, ([string]$listenerProcess.Name)
        )
        exit 1
    }
}

foreach ($path in @($stdout, $stderr, $pidFile)) {
    if (Test-Path $path) {
        Remove-Item $path -Force -ErrorAction SilentlyContinue
    }
}

$proc = Start-Process `
    -FilePath $python `
    -ArgumentList @("-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "$Port") `
    -WorkingDirectory $backendRoot `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -PassThru `
    -WindowStyle Hidden

Start-Sleep -Milliseconds 800
$proc.Refresh()
if ($proc.HasExited) {
    $stderrTail = ""
    if (Test-Path $stderr) {
        $stderrTail = ((Get-Content $stderr -Tail 40 -ErrorAction SilentlyContinue) -join [Environment]::NewLine).Trim()
    }
    $exitCode = [int]$proc.ExitCode
    $message = "Backend exited during startup (exit_code=$exitCode)."
    if ($stderrTail) {
        $message = "$message`n$stderrTail"
    }
    Write-Error $message
    exit ([Math]::Max(1, $exitCode))
}

Set-Content -Path $pidFile -Value $proc.Id -Encoding ascii
Write-Output $proc.Id
