$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "Virtual environment not found. Run .\INSTALL.ps1 first." }
$latest = Join-Path $PSScriptRoot "artifacts\latest_run.txt"
if (Test-Path $latest) {
    $runId = (Get-Content -Raw -Encoding UTF8 $latest).Trim()
    $assignments = Join-Path $PSScriptRoot "artifacts\runs\$runId\clusters.csv"
} else {
    $candidate = Get-ChildItem (Join-Path $PSScriptRoot "artifacts\runs") -Filter "clusters.csv" -Recurse -File |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $candidate) { throw "No cluster assignments found. Run HDBSCAN or graph-cluster first." }
    $assignments = $candidate.FullName
}
if (-not (Test-Path $assignments)) { throw "Cluster assignments not found: $assignments" }
$output = Join-Path (Split-Path $assignments -Parent) "semantic_profiles"
& $python (Join-Path $PSScriptRoot "run_topic_understanding.py") --project-root $PSScriptRoot --assignments $assignments --output-dir $output
if ($LASTEXITCODE -ne 0) { throw "Semantic profile generation failed." }
Write-Host "Semantic profiles saved to: $output" -ForegroundColor Green
