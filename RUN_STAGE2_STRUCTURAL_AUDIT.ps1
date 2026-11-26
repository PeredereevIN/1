[CmdletBinding()]
param(
    [string]$ProjectRoot = (Get-Location).Path,
    [string]$RunId = "20261120_015622_graph_b63e68",
    [int]$MaxBoundary = 150,
    [int]$SilhouetteSample = 1500,
    [int]$Seed = 42
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path $ProjectRoot).Path
$python = Join-Path $root ".venv\Scripts\python.exe"
$auditScript = Join-Path $PSScriptRoot "stage2_structural_audit.py"

if (-not (Test-Path $python)) {
    throw "Project venv Python not found: $python"
}
if (-not (Test-Path $auditScript)) {
    throw "Stage 2 audit script not found: $auditScript"
}

Write-Host "Checking Python dependencies..." -ForegroundColor Cyan
& $python -c "import numpy, pandas, scipy, sklearn; print('numpy', numpy.__version__); print('pandas', pandas.__version__); print('scipy', scipy.__version__); print('scikit-learn', sklearn.__version__)"
if ($LASTEXITCODE -ne 0) {
    throw "Required Python dependencies are missing in the project venv. No packages were installed automatically."
}

Write-Host "Starting Stage 2 structural audit..." -ForegroundColor Cyan
& $python $auditScript `
    --root $root `
    --run-id $RunId `
    --max-boundary $MaxBoundary `
    --silhouette-sample $SilhouetteSample `
    --seed $Seed

if ($LASTEXITCODE -ne 0) {
    throw "Stage 2 structural audit failed with exit code $LASTEXITCODE"
}

Write-Host "Stage 2 audit completed." -ForegroundColor Green
