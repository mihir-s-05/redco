$ErrorActionPreference = "Stop"

$preflight = "/mnt/c/Users/mihir/Documents/redco/scripts/preflight_stage_d_runtime_paths_v4_8.sh"
$arguments = @(
    "-d", "Ubuntu", "--", "env",
    "REDCO_REPO_ROOT=/workspace/redco",
    "REDCO_RUNTIME_ROOT=/workspace/redco/.runtime/stage-d-v4-8",
    "REDCO_RUNTIME_PREFLIGHT_REPORT=/workspace/redco/.runtime/stage-d-v4-8/runtime-path-preflight.tsv",
    "REDCO_MINIMUM_FREE_KIB=47185920",
    "bash", "-x", $preflight
)
$trace = (& wsl @arguments 2>&1) -join "`n"
$status = $LASTEXITCODE

if ($status -eq 0) {
    throw "contract-only invocation unexpectedly ran the host-specific body"
}
$required = @(
    "+ test /workspace/redco = /workspace/redco",
    "+ test /workspace/redco/.runtime/stage-d-v4-8 = /workspace/redco/.runtime/stage-d-v4-8",
    "+ test /workspace/redco/.runtime/stage-d-v4-8/runtime-path-preflight.tsv = /workspace/redco/.runtime/stage-d-v4-8/runtime-path-preflight.tsv",
    "+ test mihir = ubuntu"
)
foreach ($line in $required) {
    if (-not $trace.Contains($line)) {
        throw "preflight did not reach the expected contract trace: $line"
    }
}
if ($trace.Contains("+ mkdir -p")) {
    throw "contract-only invocation made a filesystem mutation"
}

Write-Output "Stage D v4.10 runtime path contract passed"
