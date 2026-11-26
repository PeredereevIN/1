[CmdletBinding()]
param(
    [string]$ProjectRoot = (Get-Location).Path,
    [string]$RunId = "20261120_015622_graph_b63e68",
    [string]$SourceResult = "latest",
    [string]$LabelerModel = "qwen3.5:4b",
    [string]$OllamaUrl = "http://127.0.0.1:11434",
    [int]$OllamaTimeout = 300,
    [int]$Seed = 42,
    [ValidateSet("auto", "local", "openalex", "off")]
    [string]$AbstractSource = "auto",
    [int]$OpenAlexTimeout = 15,
    [int]$MaxOpenAlexBatches = 50,
    [int]$MaxOpenAlexTitleLookups = 100,
    [int]$MaxCrossrefLookups = 50,
    [switch]$DisableCrossrefFallback
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path $ProjectRoot).Path
$python = Join-Path $root ".venv\Scripts\python.exe"
$script = Join-Path $PSScriptRoot "label_existing_topics.py"

if (-not (Test-Path $python)) { throw "Python venv not found: $python" }
if (-not (Test-Path $script)) { throw "Labeling script not found: $script" }

Write-Host "`n=== SEMANTIC EVIDENCE + LLM REVIEW (NO RECLUSTERING) ===" -ForegroundColor Cyan
Write-Host "Graph run: $RunId"
Write-Host "Source result: $SourceResult"
Write-Host "Local model: $LabelerModel"
Write-Host "Abstract source: $AbstractSource"
Write-Host "OpenAlex API key configured: $([bool]$env:OPENALEX_API_KEY) (optional)"
Write-Host "Crossref fallback enabled: $(-not $DisableCrossrefFallback)"
Write-Host "Two LLM passes per topic: generation + same-model critical review"
Write-Host "External APIs receive DOI/OpenAlex IDs; only when IDs are absent, a limited publication-title search is used. Full texts are not uploaded."
Write-Host "Leiden, graph, embeddings and UMAP will NOT be rerun." -ForegroundColor Green

$argsList = @(
    $script, "--root", $root, "--run-id", $RunId, "--source-result", $SourceResult,
    "--ollama-model", $LabelerModel, "--ollama-url", $OllamaUrl,
    "--ollama-timeout", "$OllamaTimeout", "--seed", "$Seed",
    "--abstract-source", $AbstractSource, "--openalex-timeout", "$OpenAlexTimeout",
    "--max-openalex-batches", "$MaxOpenAlexBatches", "--max-openalex-title-lookups", "$MaxOpenAlexTitleLookups",
    "--max-crossref-lookups", "$MaxCrossrefLookups"
)
if ($DisableCrossrefFallback) { $argsList += "--disable-crossref-fallback" }
& $python @argsList
if ($LASTEXITCODE -ne 0) { throw "Semantic enrichment and labeling failed with exit code $LASTEXITCODE" }
