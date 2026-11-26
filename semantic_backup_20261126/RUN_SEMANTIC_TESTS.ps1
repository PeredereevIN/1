[CmdletBinding()]
param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path $ProjectRoot).Path
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "Project venv Python not found: $python" }

Push-Location $PSScriptRoot
try {
    Write-Host "Running semantic evidence and relabeling tests..." -ForegroundColor Cyan
    & $python -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) { throw "Semantic tests failed with exit code $LASTEXITCODE" }
    Write-Host "All semantic tests passed." -ForegroundColor Green
}
finally {
    Pop-Location
}
