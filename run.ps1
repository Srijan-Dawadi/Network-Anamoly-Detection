<#
run.ps1 — one-command helper for the network-anomaly autoencoder.

Usage (from the project root):
    .\run.ps1 setup                # download real NSL-KDD data (data/)
    .\run.ps1 train                # train on data/KDDTrain+.csv -> runs/<timestamp>/
    .\run.ps1 infer                # classify data/sample_infer.csv with the latest run
    .\run.ps1 infer -Data path.csv # classify a specific CSV
    .\run.ps1 infer -RunDir runs/<ts> [-Data path.csv]
    .\run.ps1 report               # print a summary of the latest run
    .\run.ps1 report -Open         # ... and open its plots
    .\run.ps1 report -RunDir runs/<ts> [-Open]
    .\run.ps1 open                 # same as `report -Open` (alias)
    .\run.ps1 test                 # run the full test suite with coverage

`-Open` with no action is treated as `open`. PYTHONPATH is set automatically.
#>

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet("setup", "train", "infer", "test", "report", "open", "help")]
    [string]$Action = "help",

    [string]$Config = "configs/nsl_kdd_default.yaml",
    [string]$Data = "data/sample_infer.csv",
    [string]$RunDir = $null,
    [switch]$Open
)

$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot
$env:PYTHONPATH = $RepoRoot

function Invoke-Py {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$PyArgs)
    & python @PyArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Invoke-Report {
    param([switch]$ForceOpen)
    $base = @("main.py", "report", "--config", $Config)
    if ($RunDir) { $base += @("--run-dir", $RunDir) }
    if ($ForceOpen -or $Open) { $base += @("--open") }
    Invoke-Py @base
}

# `-Open` without an action word means "open the latest run's plots".
if ($Action -eq "help" -and $Open) { $Action = "open" }

switch ($Action) {
    "setup" {
        Invoke-Py "scripts/make_data.py"
    }
    "train" {
        Invoke-Py "main.py" "train" "--config" $Config
    }
    "infer" {
        $base = @("main.py", "infer", "--config", $Config, "--data", $Data)
        if ($RunDir) { $base += @("--run-dir", $RunDir) }
        Invoke-Py @base
    }
    "report" {
        Invoke-Report
    }
    "open" {
        Invoke-Report -ForceOpen
    }
    "test" {
        Invoke-Py "-m" "pytest" "tests/" "--cov=network_anomaly_autoencoder" "-q" "-p" "no:cacheprovider"
    }
    default {
        Write-Host "See README.md for the full quickstart:"
        Get-Content "$RepoRoot\README.md" -Encoding UTF8 -TotalCount 60
    }
}