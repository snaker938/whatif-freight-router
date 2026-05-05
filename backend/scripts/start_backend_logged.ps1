param(
    [int]$Port = 8000,
    [int]$MemoryLimitMB = 0,
    [string]$LeaseManifestPath = "",
    [string]$LeaseTopology = "split_process"
)

$backendRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $backendRoot ".venv\Scripts\python.exe"
$memoryWrapper = Join-Path $backendRoot "scripts\run_with_job_memory_limit.ps1"
$outDir = Join-Path $backendRoot "out"
$stdout = Join-Path $outDir "backend_stdout.log"
$stderr = Join-Path $outDir "backend_stderr.log"
$primaryPidFile = Join-Path $outDir ("backend_server_{0}.pid" -f $Port)
$pidFiles = @($primaryPidFile)
if ($Port -eq 8000) {
    $pidFiles += Join-Path $outDir "backend_server.pid"
}
$resolvedLeaseManifestPath = if ([string]::IsNullOrWhiteSpace($LeaseManifestPath)) {
    ""
} elseif ([System.IO.Path]::IsPathRooted($LeaseManifestPath)) {
    [System.IO.Path]::GetFullPath($LeaseManifestPath)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $backendRoot $LeaseManifestPath))
}

function Write-BackendLeaseManifest {
    param($Payload)

    if ([string]::IsNullOrWhiteSpace($resolvedLeaseManifestPath)) {
        return
    }

    $manifestDirectory = Split-Path -Parent $resolvedLeaseManifestPath
    if (-not [string]::IsNullOrWhiteSpace($manifestDirectory)) {
        New-Item -ItemType Directory -Force -Path $manifestDirectory | Out-Null
    }

    $tempPath = "$resolvedLeaseManifestPath.tmp"
    $json = $Payload | ConvertTo-Json -Depth 8
    $utf8NoBom = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::WriteAllText($tempPath, $json, $utf8NoBom)
    Move-Item -LiteralPath $tempPath -Destination $resolvedLeaseManifestPath -Force
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

if (-not (Test-Path $python)) {
    Write-Error ("Backend virtualenv python not found: {0}" -f $python)
    exit 1
}
if (($MemoryLimitMB -gt 0) -and (-not (Test-Path $memoryWrapper))) {
    Write-Error ("Backend memory-limit wrapper not found: {0}" -f $memoryWrapper)
    exit 1
}

New-Item -ItemType Directory -Force -Path $outDir | Out-Null

foreach ($pidFile in $pidFiles) {
    if (Test-Path $pidFile) {
        $existingPid = Get-Content $pidFile | Select-Object -First 1
        if ($existingPid) {
            $existingInfo = Get-ProcessInfoById -ProcessId ([int]$existingPid)
            if (Test-BackendProcessMatchesPort -ProcessInfo $existingInfo -TargetPort $Port) {
                Stop-BackendProcess -ProcessId ([int]$existingPid)
            }
        }
        Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
    }
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

foreach ($path in @($stdout, $stderr) + $pidFiles) {
    if (Test-Path $path) {
        Remove-Item $path -Force -ErrorAction SilentlyContinue
    }
}

if ($MemoryLimitMB -gt 0) {
    $launcher = if (Get-Command pwsh -ErrorAction SilentlyContinue) { "pwsh" } else { "powershell.exe" }
    $commandParts = @(
        "&", ('"' + $memoryWrapper + '"'),
        "-MemoryLimitMB", "$MemoryLimitMB",
        "-WorkingDirectory", ('"' + $backendRoot + '"'),
        "-FilePath", ('"' + $python + '"'),
        "-ArgumentList", "@('-m','uvicorn','app.main:app','--host','127.0.0.1','--port','{0}')" -f $Port
    )
    if (-not [string]::IsNullOrWhiteSpace($resolvedLeaseManifestPath)) {
        $commandParts += @(
            "-LeaseManifestPath", ('"' + $resolvedLeaseManifestPath + '"'),
            "-LeaseRole", '"backend"',
            "-LeaseTopology", ('"' + $LeaseTopology + '"')
        )
    }
    $command = $commandParts -join " "
    $proc = Start-Process `
        -FilePath $launcher `
        -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $command) `
        -WorkingDirectory $backendRoot `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -PassThru `
        -WindowStyle Hidden
} else {
    $proc = Start-Process `
        -FilePath $python `
        -ArgumentList @("-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "$Port") `
        -WorkingDirectory $backendRoot `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -PassThru `
        -WindowStyle Hidden
    if (-not [string]::IsNullOrWhiteSpace($resolvedLeaseManifestPath)) {
        Write-BackendLeaseManifest -Payload ([ordered]@{
            schema_version = 1
            role = "backend"
            topology = [string]$LeaseTopology
            started_at_utc = (Get-Date).ToUniversalTime().ToString("o")
            updated_at_utc = (Get-Date).ToUniversalTime().ToString("o")
            status = "running"
            watchdog_triggered = $false
            memory_limit_mb = [int]$MemoryLimitMB
            working_directory = $backendRoot
            resolved_file_path = $python
            argument_list = @("-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "$Port")
            command_line = ('"{0}" -m uvicorn app.main:app --host 127.0.0.1 --port {1}' -f $python, $Port)
            root_process_pid = [int]$proc.Id
            primary_tracked_process_id = [int]$proc.Id
            tracked_child_pids = @()
            tracked_pids = @([int]$proc.Id)
            aggregate_working_set_mb = 0.0
        })
    }
}

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

Set-Content -Path $primaryPidFile -Value $proc.Id -Encoding ascii
if ($Port -eq 8000) {
    Set-Content -Path (Join-Path $outDir "backend_server.pid") -Value $proc.Id -Encoding ascii
}
Write-Output $proc.Id
