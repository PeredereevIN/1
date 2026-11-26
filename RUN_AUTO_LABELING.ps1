$ErrorActionPreference = "Stop"
$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Не найден Python виртуального окружения: $python. Запустите скрипт из корня проекта."
}
& $python ".\run_automatic_labeling.py"
if ($LASTEXITCODE -ne 0) {
    throw "Automatic Cluster Labeling завершился с кодом $LASTEXITCODE"
}
