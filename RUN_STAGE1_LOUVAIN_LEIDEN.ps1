[CmdletBinding()]
param(
    [string]$ProjectRoot = (Get-Location).Path,
    [string]$RunId = "20261120_015622_graph_b63e68",
    [double]$Resolution = 1.0,
    [int]$MinCommunitySize = 5,
    [int]$Seed = 42,
    [int]$Iterations = -1
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path $ProjectRoot).Path
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Project venv Python not found: $python"
}
$script = Join-Path $PSScriptRoot "compare_louvain_leiden.py"
if (-not (Test-Path $script)) {
    throw "Comparison script not found: $script"
}

& $python -c "import igraph; print('igraph', igraph.__version__)" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing pinned python-igraph package..." -ForegroundColor Yellow
    & $python -m pip install "igraph==1.0.0"
    if ($LASTEXITCODE -ne 0) { throw "igraph installation failed" }
}

& $python $script --root $root --run-id $RunId --resolution $Resolution --min-community-size $MinCommunitySize --seed $Seed --n-iterations $Iterations
if ($LASTEXITCODE -ne 0) {
    throw "Stage 1 comparison failed with exit code $LASTEXITCODE"
}
