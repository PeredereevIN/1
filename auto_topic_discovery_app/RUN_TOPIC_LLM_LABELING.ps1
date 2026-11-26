[CmdletBinding()]
param(
    [string]$ProjectRoot = (Get-Location).Path,
    [string]$RunId = "20261120_015622_graph_b63e68",
    [string]$SourceResult = "latest",
    [string]$LabelerModel = "qwen3.5:4b",
    [string]$OllamaUrl = "http://127.0.0.1:11434",
    [int]$OllamaTimeout = 300,
    [int]$Seed = 42
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path $ProjectRoot).Path
$python = Join-Path $root ".venv\Scripts\python.exe"
$script = Join-Path $PSScriptRoot "label_existing_topics.py"

if (-not (Test-Path $python)) {
    throw "Python venv not found: $python"
}
if (-not (Test-Path $script)) {
    throw "Labeling script not found: $script"
}

Write-Host "`n=== LLM TOPIC RELABELING (NO RECLUSTERING) ===" -ForegroundColor Cyan
Write-Host "Graph run: $RunId"
Write-Host "Source result: $SourceResult"
Write-Host "Local model: $LabelerModel"
Write-Host "This step will NOT rerun Leiden or UMAP." -ForegroundColor Green

& $python $script `
    --root $root `
    --run-id $RunId `
    --source-result $SourceResult `
    --ollama-model $LabelerModel `
    --ollama-url $OllamaUrl `
    --ollama-timeout $OllamaTimeout `
    --seed $Seed

if ($LASTEXITCODE -ne 0) {
    throw "LLM topic relabeling failed with exit code $LASTEXITCODE"
}
