$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "Environment not found. Run .\INSTALL.ps1 first." }
& $python -m streamlit run .\dashboard\app.py
