param(
    [switch]$CpuOnly
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "=== Scientific Intelligence Lab installer ===" -ForegroundColor Cyan
$py = Get-Command py -ErrorAction SilentlyContinue
if (-not $py) { throw "Python Launcher (py.exe) not found. Install Python 3.12 from python.org first." }
py -3.12 --version
if ($LASTEXITCODE -ne 0) { throw "Python 3.12 is required. Install it and rerun INSTALL.ps1." }

if (-not (Test-Path ".venv")) { py -3.12 -m venv .venv }
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $python -m pip install --upgrade pip setuptools wheel

if (-not $CpuOnly) {
    Write-Host "Installing PyTorch 2.14.0 with CUDA 13.2 wheels..." -ForegroundColor Yellow
    & $python -m pip install --index-url https://download.pytorch.org/whl/cu132 "torch==2.14.0"
    if ($LASTEXITCODE -ne 0) { throw "CUDA PyTorch install failed. Check NVIDIA driver and Python 3.12, then retry or run INSTALL.ps1 -CpuOnly." }
} else {
    & $python -m pip install torch==2.14.0
}

& $python -m pip install -r requirements.txt
& $python -m pip install -e .
if (-not (Test-Path ".env")) { Copy-Item ".env.example" ".env" }
New-Item -ItemType Directory -Force -Path "data\inbox","data\metadata","data\processed","artifacts","reports" | Out-Null
Write-Host "Installation complete. CUDA diagnostic:" -ForegroundColor Green
if ($CpuOnly) {
    & $python -m sciintel.cli doctor
} else {
    & $python -m sciintel.cli doctor --require-cuda
    if ($LASTEXITCODE -ne 0) { throw "PyTorch was installed but CUDA is not available. Check the NVIDIA driver before running the pipeline." }
}
