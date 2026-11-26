$ErrorActionPreference = "Stop"
$Python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Project venv Python not found: $Python" }
& $Python ".\run_topic_understanding.py" @args
if ($LASTEXITCODE -ne 0) { throw "Cluster Topic Understanding failed with exit code $LASTEXITCODE" }
