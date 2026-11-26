[CmdletBinding()]
param(
    [string]$ProjectRoot = (Get-Location).Path,
    [int]$Port = 8501,
    [switch]$SkipDependencyInstall,
    [switch]$StrictPort
)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path $ProjectRoot).Path
$python = Join-Path $root ".venv\Scripts\python.exe"
$app = Join-Path $PSScriptRoot "streamlit_topic_explorer.py"
$requirements = Join-Path $PSScriptRoot "requirements-auto-topics.txt"
$dependencyCheck = Join-Path $PSScriptRoot "check_addon_dependencies.py"
if (-not (Test-Path $python)) { throw "Python venv not found: $python" }
if (-not (Test-Path $app)) { throw "Streamlit app not found: $app" }
if (-not (Test-Path $dependencyCheck)) { throw "Dependency checker not found: $dependencyCheck" }

& $python $dependencyCheck
if ($LASTEXITCODE -ne 0) {
    if ($SkipDependencyInstall) { throw "Required Streamlit/Plotly dependencies are missing." }
    & $python -m pip install -r $requirements
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed" }
    & $python $dependencyCheck
    if ($LASTEXITCODE -ne 0) { throw "Required dependencies are still missing after installation" }
}

function Test-LocalPortAvailable([int]$CandidatePort) {
    $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $CandidatePort)
    try {
        $listener.Start()
        return $true
    }
    catch {
        return $false
    }
    finally {
        try { $listener.Stop() } catch { }
    }
}

$requestedPort = $Port
if (-not (Test-LocalPortAvailable $Port)) {
    if ($StrictPort) {
        throw "Port $Port is already occupied. Close the existing process or choose another -Port."
    }
    do {
        $Port++
        if ($Port -gt 65535) { throw "No free TCP port found after $requestedPort." }
    } until (Test-LocalPortAvailable $Port)
    Write-Warning "Порт $requestedPort уже занят. Streamlit будет запущен на свободном порту $Port."
}

$url = "http://localhost:$Port"
Write-Host "Starting Streamlit topic explorer on $url" -ForegroundColor Cyan
$previousProjectRoot = $env:SIL_PROJECT_ROOT
$env:SIL_PROJECT_ROOT = $root
try {
    & $python -m streamlit run $app --server.address 127.0.0.1 --server.port $Port
    if ($LASTEXITCODE -ne 0) { throw "Streamlit exited with code $LASTEXITCODE" }
}
finally {
    if ($null -eq $previousProjectRoot) { Remove-Item Env:SIL_PROJECT_ROOT -ErrorAction SilentlyContinue }
    else { $env:SIL_PROJECT_ROOT = $previousProjectRoot }
}
