[CmdletBinding()]
param(
    [string]$ProjectRoot = (Get-Location).Path,
    [string]$RunId = "20261120_015622_graph_b63e68",
    [double]$ResolutionMin = 0.35,
    [double]$ResolutionMax = 2.0,
    [int]$ResolutionSteps = 12,
    [int]$Repeats = 3,
    [int]$Seed = 42,
    [ValidateSet("en")]
    [string]$Language = "en",
    [ValidateSet("ollama", "heuristic")]
    [string]$Labeler = "ollama",
    [string]$LabelerModel = "qwen3.5:4b",
    [string]$OllamaUrl = "http://127.0.0.1:11434",
    [int]$OllamaTimeout = 300,
    [switch]$SkipDependencyInstall
)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path $ProjectRoot).Path
$python = Join-Path $root ".venv\Scripts\python.exe"
$script = Join-Path $PSScriptRoot "auto_topic_discovery.py"
$requirements = Join-Path $PSScriptRoot "requirements-auto-topics.txt"
$dependencyCheck = Join-Path $PSScriptRoot "check_addon_dependencies.py"
if (-not (Test-Path $python)) { throw "Python venv not found: $python" }
if (-not (Test-Path $script)) { throw "Script not found: $script" }
if (-not (Test-Path $requirements)) { throw "Requirements not found: $requirements" }
if (-not (Test-Path $dependencyCheck)) { throw "Dependency checker not found: $dependencyCheck" }

# Run a real Python file, not a multiline -c payload. This is reliable in Windows PowerShell 5.1.
& $python $dependencyCheck
$checkExit = $LASTEXITCODE
if ($checkExit -ne 0) {
    if ($SkipDependencyInstall) {
        throw "Missing required add-on dependencies. Run: .\.venv\Scripts\python.exe -m pip install -r `"$requirements`""
    }
    Write-Host "Installing only core topic-discovery dependencies..." -ForegroundColor Yellow
    & $python -m pip install -r $requirements
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed" }
    & $python $dependencyCheck
    if ($LASTEXITCODE -ne 0) { throw "Required dependencies are still missing after installation" }
}

Write-Host "`n=== AUTOMATIC TOPIC DISCOVERY ===" -ForegroundColor Cyan
Write-Host "Graph run: $RunId"
Write-Host "Resolution sweep: $ResolutionMin .. $ResolutionMax ($ResolutionSteps points)"
Write-Host "Seed repeats: $Repeats"
Write-Host "Target cluster count: NOT SET" -ForegroundColor Green
Write-Host "Topic labeler: $Labeler"
if ($Labeler -eq "ollama") {
    Write-Host "Local LLM: $LabelerModel via $OllamaUrl"
}
& $python $script `
    --root $root `
    --run-id $RunId `
    --resolution-min $ResolutionMin `
    --resolution-max $ResolutionMax `
    --resolution-steps $ResolutionSteps `
    --repeats $Repeats `
    --seed $Seed `
    --language $Language `
    --labeler $Labeler `
    --ollama-model $LabelerModel `
    --ollama-url $OllamaUrl `
    --ollama-timeout $OllamaTimeout
if ($LASTEXITCODE -ne 0) { throw "Automatic topic discovery failed with exit code $LASTEXITCODE" }
