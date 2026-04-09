param(
    [Parameter(Mandatory = $true)]
    [ValidateRange(1, [int]::MaxValue)]
    [int]$MemoryLimitMB,

    [Parameter(Mandatory = $true)]
    [string]$FilePath,

    [string[]]$ArgumentList = @(),

    [string]$WorkingDirectory = (Get-Location).Path,

    [string]$LeaseManifestPath = "",

    [string]$LeaseRole = "process",

    [string]$LeaseTopology = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$resolvedWorkingDirectory = (Resolve-Path -LiteralPath $WorkingDirectory).Path
$candidateFilePath = if ([System.IO.Path]::IsPathRooted($FilePath)) {
    $FilePath
}
else {
    [System.IO.Path]::Combine($resolvedWorkingDirectory, $FilePath)
}
$resolvedFilePath = (Resolve-Path -LiteralPath $candidateFilePath).Path
$resolvedLeaseManifestPath = ""
if (-not [string]::IsNullOrWhiteSpace($LeaseManifestPath)) {
    $candidateLeaseManifestPath = if ([System.IO.Path]::IsPathRooted($LeaseManifestPath)) {
        $LeaseManifestPath
    }
    else {
        [System.IO.Path]::Combine($resolvedWorkingDirectory, $LeaseManifestPath)
    }
    $resolvedLeaseManifestPath = [System.IO.Path]::GetFullPath($candidateLeaseManifestPath)
}

Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

public static class JobMemoryNative {
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    public struct STARTUPINFO {
        public UInt32 cb;
        public string lpReserved;
        public string lpDesktop;
        public string lpTitle;
        public UInt32 dwX;
        public UInt32 dwY;
        public UInt32 dwXSize;
        public UInt32 dwYSize;
        public UInt32 dwXCountChars;
        public UInt32 dwYCountChars;
        public UInt32 dwFillAttribute;
        public UInt32 dwFlags;
        public UInt16 wShowWindow;
        public UInt16 cbReserved2;
        public IntPtr lpReserved2;
        public IntPtr hStdInput;
        public IntPtr hStdOutput;
        public IntPtr hStdError;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct PROCESS_INFORMATION {
        public IntPtr hProcess;
        public IntPtr hThread;
        public UInt32 dwProcessId;
        public UInt32 dwThreadId;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct IO_COUNTERS {
        public UInt64 ReadOperationCount;
        public UInt64 WriteOperationCount;
        public UInt64 OtherOperationCount;
        public UInt64 ReadTransferCount;
        public UInt64 WriteTransferCount;
        public UInt64 OtherTransferCount;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct JOBOBJECT_BASIC_LIMIT_INFORMATION {
        public Int64 PerProcessUserTimeLimit;
        public Int64 PerJobUserTimeLimit;
        public UInt32 LimitFlags;
        public UIntPtr MinimumWorkingSetSize;
        public UIntPtr MaximumWorkingSetSize;
        public UInt32 ActiveProcessLimit;
        public UIntPtr Affinity;
        public UInt32 PriorityClass;
        public UInt32 SchedulingClass;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION {
        public JOBOBJECT_BASIC_LIMIT_INFORMATION BasicLimitInformation;
        public IO_COUNTERS IoInfo;
        public UIntPtr ProcessMemoryLimit;
        public UIntPtr JobMemoryLimit;
        public UIntPtr PeakProcessMemoryUsed;
        public UIntPtr PeakJobMemoryUsed;
    }

    public const UInt32 CREATE_SUSPENDED = 0x00000004;
    public const UInt32 JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100;
    public const UInt32 JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200;
    public const UInt32 JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000;
    public const UInt32 INFINITE = 0xFFFFFFFF;
    public const int JobObjectExtendedLimitInformation = 9;

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static extern bool CreateProcess(
        string lpApplicationName,
        string lpCommandLine,
        IntPtr lpProcessAttributes,
        IntPtr lpThreadAttributes,
        [MarshalAs(UnmanagedType.Bool)] bool bInheritHandles,
        UInt32 dwCreationFlags,
        IntPtr lpEnvironment,
        string lpCurrentDirectory,
        ref STARTUPINFO lpStartupInfo,
        out PROCESS_INFORMATION lpProcessInformation
    );

    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern IntPtr CreateJobObject(IntPtr lpJobAttributes, string lpName);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static extern bool SetInformationJobObject(
        IntPtr hJob,
        int JobObjectInformationClass,
        IntPtr lpJobObjectInformation,
        UInt32 cbJobObjectInformationLength
    );

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static extern bool AssignProcessToJobObject(IntPtr hJob, IntPtr hProcess);

    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern UInt32 ResumeThread(IntPtr hThread);

    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern UInt32 WaitForSingleObject(IntPtr hHandle, UInt32 dwMilliseconds);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static extern bool GetExitCodeProcess(IntPtr hProcess, out UInt32 lpExitCode);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static extern bool CloseHandle(IntPtr hObject);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static extern bool TerminateProcess(IntPtr hProcess, UInt32 uExitCode);
}
"@

function Quote-CommandArgument {
    param([string]$Value)

    if ($null -eq $Value -or $Value -eq "") {
        return '""'
    }

    if ($Value -notmatch '[\s"]') {
        return $Value
    }

    $escaped = $Value -replace '(\\*)"', '$1$1\"'
    $escaped = $escaped -replace '(\\+)$', '$1$1'
    return '"' + $escaped + '"'
}

function Get-Win32ErrorMessage {
    param([string]$Prefix)

    $code = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
    $message = ([ComponentModel.Win32Exception]::new($code)).Message
    return "$Prefix (Win32 error ${code}: $message)"
}

function Get-ChildPythonProcessIds {
    param([int[]]$ParentProcessIds)

    $childProcessIds = [System.Collections.Generic.HashSet[int]]::new()
    foreach ($parentProcessId in @($ParentProcessIds | Where-Object { $_ -gt 0 })) {
        $childRows = Get-CimInstance Win32_Process -Filter "ParentProcessId = $parentProcessId" -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -like 'python*' }
        foreach ($childRow in $childRows) {
            try {
                [void]$childProcessIds.Add([int]$childRow.ProcessId)
            }
            catch {
            }
        }
    }
    return @($childProcessIds)
}

function Get-UtcTimestamp {
    return (Get-Date).ToUniversalTime().ToString("o")
}

function Write-LeaseManifest {
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

$memoryLimitBytes = [UInt64]$MemoryLimitMB * 1MB
$jobHandle = [IntPtr]::Zero
$processHandle = [IntPtr]::Zero
$threadHandle = [IntPtr]::Zero
$watchdogExitCode = 137
$leaseManifest = $null

try {
    $startupInfo = [JobMemoryNative+STARTUPINFO]::new()
    $startupInfo.cb = [UInt32][Runtime.InteropServices.Marshal]::SizeOf([type][JobMemoryNative+STARTUPINFO])
    $processInfo = [JobMemoryNative+PROCESS_INFORMATION]::new()

    $quotedArgs = @($ArgumentList | ForEach-Object { Quote-CommandArgument -Value ([string]$_) })
    $commandLine = '"' + $resolvedFilePath + '"'
    if ($quotedArgs.Count -gt 0) {
        $commandLine += " " + ($quotedArgs -join " ")
    }

    $created = [JobMemoryNative]::CreateProcess(
        $resolvedFilePath,
        $commandLine,
        [IntPtr]::Zero,
        [IntPtr]::Zero,
        $false,
        [JobMemoryNative]::CREATE_SUSPENDED,
        [IntPtr]::Zero,
        $resolvedWorkingDirectory,
        [ref]$startupInfo,
        [ref]$processInfo
    )

    if (-not $created) {
        throw (Get-Win32ErrorMessage -Prefix "CreateProcess failed")
    }

    $processHandle = $processInfo.hProcess
    $threadHandle = $processInfo.hThread

    $jobHandle = [JobMemoryNative]::CreateJobObject([IntPtr]::Zero, $null)
    if ($jobHandle -eq [IntPtr]::Zero) {
        throw (Get-Win32ErrorMessage -Prefix "CreateJobObject failed")
    }

    $limitInfo = [JobMemoryNative+JOBOBJECT_EXTENDED_LIMIT_INFORMATION]::new()
    $limitInfo.BasicLimitInformation.LimitFlags = `
        [JobMemoryNative]::JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    $limitInfo.ProcessMemoryLimit = [UIntPtr]$memoryLimitBytes
    $limitInfo.JobMemoryLimit = [UIntPtr]$memoryLimitBytes

    $limitInfoSize = [Runtime.InteropServices.Marshal]::SizeOf([type][JobMemoryNative+JOBOBJECT_EXTENDED_LIMIT_INFORMATION])
    $limitInfoPtr = [Runtime.InteropServices.Marshal]::AllocHGlobal($limitInfoSize)
    try {
        [Runtime.InteropServices.Marshal]::StructureToPtr($limitInfo, $limitInfoPtr, $false)
        $setInfo = [JobMemoryNative]::SetInformationJobObject(
            $jobHandle,
            [JobMemoryNative]::JobObjectExtendedLimitInformation,
            $limitInfoPtr,
            [UInt32]$limitInfoSize
        )
        if (-not $setInfo) {
            throw (Get-Win32ErrorMessage -Prefix "SetInformationJobObject failed")
        }
    }
    finally {
        if ($limitInfoPtr -ne [IntPtr]::Zero) {
            [Runtime.InteropServices.Marshal]::FreeHGlobal($limitInfoPtr)
        }
    }

    $assigned = [JobMemoryNative]::AssignProcessToJobObject($jobHandle, $processHandle)
    if (-not $assigned) {
        throw (Get-Win32ErrorMessage -Prefix "AssignProcessToJobObject failed")
    }

    $leaseManifest = [ordered]@{
        schema_version = 1
        role = [string]$LeaseRole
        topology = if ([string]::IsNullOrWhiteSpace($LeaseTopology)) { $null } else { [string]$LeaseTopology }
        started_at_utc = Get-UtcTimestamp
        updated_at_utc = Get-UtcTimestamp
        status = "starting"
        watchdog_triggered = $false
        memory_limit_mb = [int]$MemoryLimitMB
        working_directory = $resolvedWorkingDirectory
        resolved_file_path = $resolvedFilePath
        argument_list = @($ArgumentList | ForEach-Object { [string]$_ })
        command_line = $commandLine
        root_process_pid = [int]$processInfo.dwProcessId
        primary_tracked_process_id = [int]$processInfo.dwProcessId
        tracked_child_pids = @()
        tracked_pids = @([int]$processInfo.dwProcessId)
        aggregate_working_set_mb = 0.0
    }
    Write-LeaseManifest -Payload $leaseManifest

    Write-Host ("[job-mem-limit] pid={0} cap_mb={1} cwd={2}" -f $processInfo.dwProcessId, $MemoryLimitMB, $resolvedWorkingDirectory)
    Write-Host ("[job-mem-limit] command={0}" -f $commandLine)

    $resumeResult = [JobMemoryNative]::ResumeThread($threadHandle)
    if ($resumeResult -eq [uint32]::MaxValue) {
        throw (Get-Win32ErrorMessage -Prefix "ResumeThread failed")
    }

    $process = [System.Diagnostics.Process]::GetProcessById([int]$processInfo.dwProcessId)
    $trackedProcesses = @{}
    $trackedProcesses[[int]$process.Id] = $process
    $primaryTrackedProcessId = [int]$process.Id
    $watchdogTriggered = $false
    while ($true) {
        $knownProcessIds = @($trackedProcesses.Keys | ForEach-Object { [int]$_ })
        $childProcessIds = @(Get-ChildPythonProcessIds -ParentProcessIds $knownProcessIds)
        foreach ($childProcessId in $childProcessIds) {
            $childProcessId = [int]$childProcessId
            if ($trackedProcesses.ContainsKey($childProcessId)) {
                continue
            }
            try {
                $childProcess = [System.Diagnostics.Process]::GetProcessById($childProcessId)
            }
            catch {
                continue
            }
            $trackedProcesses[$childProcessId] = $childProcess
            $primaryTrackedProcessId = $childProcessId
            try {
                $null = $childProcess.Handle
                [void][JobMemoryNative]::AssignProcessToJobObject($jobHandle, $childProcess.Handle)
            }
            catch {
            }
            Write-Host ("[job-mem-limit] tracking-child pid={0}" -f $childProcessId)
            if ($null -ne $leaseManifest) {
                $leaseManifest["updated_at_utc"] = Get-UtcTimestamp
                $leaseManifest["status"] = "running"
                $leaseManifest["primary_tracked_process_id"] = $primaryTrackedProcessId
                $leaseManifest["tracked_child_pids"] = @(
                    $trackedProcesses.Keys |
                        ForEach-Object { [int]$_ } |
                        Where-Object { $_ -ne [int]$process.Id } |
                        Sort-Object
                )
                $leaseManifest["tracked_pids"] = @(
                    $trackedProcesses.Keys |
                        ForEach-Object { [int]$_ } |
                        Sort-Object
                )
                Write-LeaseManifest -Payload $leaseManifest
            }
        }

        $liveProcesses = @()
        [UInt64]$aggregateWorkingSetBytes = 0
        foreach ($trackedProcessId in @($trackedProcesses.Keys | ForEach-Object { [int]$_ })) {
            $trackedProcess = $trackedProcesses[$trackedProcessId]
            try {
                $trackedProcess.Refresh()
            }
            catch {
            }
            if ($trackedProcess.HasExited) {
                continue
            }
            $liveProcesses += $trackedProcess
            $aggregateWorkingSetBytes += [UInt64]([Math]::Max([int64]0, [int64]$trackedProcess.WorkingSet64))
        }

        if ($liveProcesses.Count -eq 0) {
            break
        }

        if ($null -ne $leaseManifest) {
            $leaseManifest["updated_at_utc"] = Get-UtcTimestamp
            $leaseManifest["status"] = "running"
            $leaseManifest["primary_tracked_process_id"] = $primaryTrackedProcessId
            $leaseManifest["tracked_child_pids"] = @(
                $trackedProcesses.Keys |
                    ForEach-Object { [int]$_ } |
                    Where-Object { $_ -ne [int]$process.Id } |
                    Sort-Object
            )
            $leaseManifest["tracked_pids"] = @($liveProcesses | ForEach-Object { [int]$_.Id } | Sort-Object)
            $leaseManifest["aggregate_working_set_mb"] = [Math]::Round(($aggregateWorkingSetBytes / 1MB), 2)
            Write-LeaseManifest -Payload $leaseManifest
        }

        if ($aggregateWorkingSetBytes -gt $memoryLimitBytes) {
            $liveProcessIds = @($liveProcesses | ForEach-Object { $_.Id }) -join ","
            Write-Host ("[job-mem-limit] limit-exceeded tracked_pids={0} aggregate_working_set_mb={1:N2} cap_mb={2}" -f `
                $liveProcessIds, `
                ($aggregateWorkingSetBytes / 1MB), `
                $MemoryLimitMB)
            foreach ($liveProcess in $liveProcesses) {
                try {
                    Stop-Process -Id $liveProcess.Id -Force -ErrorAction Stop
                }
                catch {
                }
            }
            $watchdogTriggered = $true
            if ($null -ne $leaseManifest) {
                $leaseManifest["updated_at_utc"] = Get-UtcTimestamp
                $leaseManifest["status"] = "limit_exceeded"
                $leaseManifest["watchdog_triggered"] = $true
                $leaseManifest["tracked_pids"] = @($liveProcesses | ForEach-Object { [int]$_.Id } | Sort-Object)
                $leaseManifest["aggregate_working_set_mb"] = [Math]::Round(($aggregateWorkingSetBytes / 1MB), 2)
                Write-LeaseManifest -Payload $leaseManifest
            }
            break
        }
        Start-Sleep -Milliseconds 100
    }

    if ($watchdogTriggered) {
        if ($null -ne $leaseManifest) {
            $leaseManifest["updated_at_utc"] = Get-UtcTimestamp
            $leaseManifest["exit_code"] = [int]$watchdogExitCode
            Write-LeaseManifest -Payload $leaseManifest
        }
        exit $watchdogExitCode
    }

    $primaryProcess = $trackedProcesses[[int]$primaryTrackedProcessId]
    if ($null -eq $primaryProcess) {
        $primaryProcess = $process
    }
    $primaryProcess.WaitForExit()
    $exitCode = [int]([int64]$primaryProcess.ExitCode)
    if ($null -ne $leaseManifest) {
        $leaseManifest["updated_at_utc"] = Get-UtcTimestamp
        $leaseManifest["status"] = "exited"
        $leaseManifest["primary_tracked_process_id"] = [int]$primaryProcess.Id
        $leaseManifest["tracked_child_pids"] = @(
            $trackedProcesses.Keys |
                ForEach-Object { [int]$_ } |
                Where-Object { $_ -ne [int]$process.Id } |
                Sort-Object
        )
        $leaseManifest["tracked_pids"] = @(
            $trackedProcesses.Keys |
                ForEach-Object { [int]$_ } |
                Sort-Object
        )
        $leaseManifest["exit_code"] = $exitCode
        Write-LeaseManifest -Payload $leaseManifest
    }
    exit $exitCode
}
catch {
    if ($null -ne $leaseManifest) {
        $leaseManifest["updated_at_utc"] = Get-UtcTimestamp
        $leaseManifest["status"] = "launch_error"
        $leaseManifest["watchdog_triggered"] = $false
        $leaseManifest["error"] = $_.Exception.Message
        Write-LeaseManifest -Payload $leaseManifest
    }
    if ($processHandle -ne [IntPtr]::Zero) {
        [void][JobMemoryNative]::TerminateProcess($processHandle, 1)
    }
    throw
}
finally {
    if ($threadHandle -ne [IntPtr]::Zero) {
        [void][JobMemoryNative]::CloseHandle($threadHandle)
    }
    if ($processHandle -ne [IntPtr]::Zero) {
        [void][JobMemoryNative]::CloseHandle($processHandle)
    }
    if ($jobHandle -ne [IntPtr]::Zero) {
        [void][JobMemoryNative]::CloseHandle($jobHandle)
    }
}
