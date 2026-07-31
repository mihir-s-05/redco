param(
    [Parameter(Mandatory = $true)]
    [string]$PodId,
    [Parameter(Mandatory = $true)]
    [string]$CreatedAtUtc,
    [Parameter(Mandatory = $true)]
    [ValidateRange(1, 86400)]
    [int]$TotalBilledLifetimeCapSeconds,
    [Parameter(Mandatory = $true)]
    [string]$LogPath,
    [ValidateRange(60, 900)]
    [int]$TerminationGraceSeconds = 300,
    [ValidateRange(5, 120)]
    [int]$PollSeconds = 30
)

$ErrorActionPreference = "Stop"
$created = [DateTimeOffset]::Parse($CreatedAtUtc).ToUniversalTime()
$absoluteCap = $created.AddSeconds($TotalBilledLifetimeCapSeconds)
$terminationTrigger = $absoluteCap.AddSeconds(-$TerminationGraceSeconds)
$logFile = [System.IO.Path]::GetFullPath($LogPath)
$logDirectory = [System.IO.Path]::GetDirectoryName($logFile)
[System.IO.Directory]::CreateDirectory($logDirectory) | Out-Null

function Write-GuardLog {
    param([string]$Message)
    $timestamp = [DateTimeOffset]::UtcNow.ToString("o")
    Add-Content -LiteralPath $logFile -Value "$timestamp`t$Message"
}

if ($terminationTrigger -le [DateTimeOffset]::UtcNow) {
    throw "termination trigger is not in the future"
}
Write-GuardLog "START pod=$PodId created=$($created.ToString('o')) cap=$($absoluteCap.ToString('o')) trigger=$($terminationTrigger.ToString('o'))"

while ([DateTimeOffset]::UtcNow -lt $terminationTrigger) {
    $statusOutput = & prime --plain pods status $PodId --output json 2>&1
    if ($LASTEXITCODE -eq 0) {
        $status = ($statusOutput -join "`n") | ConvertFrom-Json
        Write-GuardLog "STATUS $($status.status)"
        if ($status.status -in @("TERMINATED", "FAILED", "DELETED")) {
            Write-GuardLog "EXIT pod already terminal"
            exit 0
        }
    }
    else {
        Write-GuardLog "STATUS_ERROR $($statusOutput -join ' ')"
    }
    $remaining = [int][Math]::Ceiling(
        ($terminationTrigger - [DateTimeOffset]::UtcNow).TotalSeconds
    )
    if ($remaining -gt 0) {
        Start-Sleep -Seconds ([Math]::Min($PollSeconds, $remaining))
    }
}

Write-GuardLog "TERMINATE trigger reached"
$shutdownDeadline = $absoluteCap
do {
    $terminateOutput = & prime --plain pods terminate $PodId --yes 2>&1
    Write-GuardLog "TERMINATE_RESULT exit=$LASTEXITCODE $($terminateOutput -join ' ')"
    Start-Sleep -Seconds 10
    $statusOutput = & prime --plain pods status $PodId --output json 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-GuardLog "CONFIRMED absent"
        exit 0
    }
    $status = ($statusOutput -join "`n") | ConvertFrom-Json
    Write-GuardLog "CONFIRM_STATUS $($status.status)"
    if ($status.status -in @("TERMINATED", "FAILED", "DELETED")) {
        Write-GuardLog "CONFIRMED terminal"
        exit 0
    }
} while ([DateTimeOffset]::UtcNow -lt $shutdownDeadline)

Write-GuardLog "FATAL provider did not confirm termination before absolute cap"
exit 20
