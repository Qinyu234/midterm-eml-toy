# Build image and run full SCORE suite in Docker (results → ./results).
param(
    [string]$Service = "eml",
    [switch]$Build
)

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$args = @("compose")
if ($Build) { $args += "build" }
$args += @("run", "--rm", $Service)

& docker @args
