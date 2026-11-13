# Keep the CPU training task alive.
#
# A Windows scheduled task can still die (session teardown, OOM, a stray taskkill
# sweep of python processes). This watchdog runs every 10 minutes, and if no python
# process is alive while the run is unfinished, it starts the training task again.
#
# Register it once:
#   schtasks /create /tn "bitter-lesson-watchdog" /sc minute /mo 10 /tr \
#     "powershell -NoProfile -ExecutionPolicy Bypass -File <this file>" /f
#
# It is deliberately dumb: it does not resume from a checkpoint itself, it just
# re-runs the registered task, which already carries --resume-from.

$ErrorActionPreference = "Stop"
$root = "C:\Users\Public\WorkPlace\mid-eml-toy\tiny-stories-byte-lm"
$metrics = Join-Path $root "checkpoints\cpu-cont\metrics.jsonl"
$targetSteps = 15752
$taskName = "bitter-lesson-cpu-cont"

$running = @(Get-Process -Name python -ErrorAction SilentlyContinue)
if ($running.Count -gt 0) {
    Write-Output "training alive, nothing to do"
    exit 0
}

$step = 0
if (Test-Path -LiteralPath $metrics) {
    $last = Get-Content -LiteralPath $metrics -Tail 1
    if ($last) {
        $step = ($last | ConvertFrom-Json).step
    }
}

if ($step -ge $targetSteps) {
    Write-Output "run finished at step $step, watchdog idle"
    exit 0
}

$latest = Get-ChildItem -LiteralPath (Join-Path $root "checkpoints\cpu-cont") -Filter "checkpoint-*.pt" |
    Sort-Object Name | Select-Object -Last 1
if (-not $latest) {
    Write-Output "no checkpoint yet, not starting"
    exit 0
}

# Point the registered action at the newest checkpoint before starting it.
$task = Get-ScheduledTask -TaskName $taskName
$argument = $task.Actions[0].Arguments -replace "checkpoint-\d{8}\.pt", $latest.Name
Set-ScheduledTask -TaskName $taskName -Action (New-ScheduledTaskAction -Execute "cmd.exe" -Argument $argument -WorkingDirectory $root) | Out-Null
Start-ScheduledTask -TaskName $taskName
Write-Output "restarted from $($latest.Name) at step $step"
