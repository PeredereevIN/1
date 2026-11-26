param(
    [int]$Target = 1870,
    [int]$Seed = 20261119,
    [string]$Email = ""
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "Project Python environment not found. Run .\INSTALL.ps1 first." }
$argsList = @(".\download_random_arxiv.py", "--target", $Target, "--seed", $Seed)
if ($Email) { $argsList += @("--email", $Email) }
& $python @argsList
exit $LASTEXITCODE
