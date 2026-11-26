param(
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$CliArgs
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "Environment not found. Run .\INSTALL.ps1 first." }
& $python -m sciintel.cli @CliArgs
exit $LASTEXITCODE
