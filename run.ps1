<#
run.ps1 — one-command helper for the network-anomaly autoencoder.

Usage (from the project root):
    .\run.ps1 setup                # download real NSL-KDD data (data/)
    .\run.ps1 train                # train on data/KDDTrain+.csv -> runs/<timestamp>/
    .\run.ps1 infer                # classify data/sample_infer.csv with the latest run
    .\run.ps1 infer -Data path.csv # classify a specific CSV
    .\run.ps1 infer -RunDir runs/<ts> [-Data path.csv]
    .\run.ps1 test                 # run the full test suite with coverage

PYTHONPATH is set automatically, so no environment setup is required.
#>

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet("setup", "train", "infer", "test", "help")]
    [string]$Action = "help",

    [string]$Config = "configs/nsl_kdd_default.yaml",
    [string]$Data = "data/sample_infer.csv",
    [string]$RunDir = $null
)

$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot
$env:PYTHONPATH = $RepoRoot

function Invoke-Py {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$PyArgs)
    & python @PyArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

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
    "test" {
        Invoke-Py "-m" "pytest" "tests/" "--cov=network_anomaly_autoencoder" "-q" "-p" "no:cacheprovider"
    }
    default {
        Write-Host "See README.md for the full quickstart:"
        Get-Content "$RepoRoot\README.md" -TotalCount 60
    }
}