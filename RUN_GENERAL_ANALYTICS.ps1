param(
    [ValidateSet("cuda","auto","cpu")][string]$Device = "cuda",
    [ValidateSet("hdbscan","graph")][string]$Method = "hdbscan"
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "Project Python environment not found. Run .\INSTALL.ps1 first." }

Write-Host "[1/4] Ingesting papers..." -ForegroundColor Cyan
& $python -m sciintel.cli ingest
if ($LASTEXITCODE -ne 0) { throw "Ingest failed." }

Write-Host "[2/4] Extracting YAKE/RAKE terms..." -ForegroundColor Cyan
& $python -m sciintel.cli keywords
if ($LASTEXITCODE -ne 0) { throw "Keyword extraction failed." }

Write-Host "[3/4] Creating SPECTER embeddings..." -ForegroundColor Cyan
& $python -m sciintel.cli embed --device $Device --batch-size 16
if ($LASTEXITCODE -ne 0) { throw "Embedding generation failed." }

Write-Host "[4/4] Clustering..." -ForegroundColor Cyan
if ($Method -eq "graph") {
    & $python -m sciintel.cli graph-cluster --device $Device --k 15 --min-similarity 0.35 --resolution 1.0 --min-community-size 5
} else {
    & $python -m sciintel.cli hdbscan --device $Device --min-cluster-size 12 --min-samples 5 --selection eom --reduction umap --umap-components 10 --umap-neighbors 30 --umap-min-dist 0.0
}
if ($LASTEXITCODE -ne 0) { throw "Clustering failed." }

Write-Host "Pipeline completed. Start the dashboard with .\START_DASHBOARD.ps1" -ForegroundColor Green
