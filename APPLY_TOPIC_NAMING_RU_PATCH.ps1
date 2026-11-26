[CmdletBinding()]
param(
    [string]$ProjectRoot = (Get-Location).Path
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $ProjectRoot -PathType Container)) {
    throw "Project root does not exist: $ProjectRoot"
}
$ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
$SourceRoot = Join-Path $PSScriptRoot "files"
if (-not (Test-Path -LiteralPath $SourceRoot -PathType Container)) {
    throw "Patch payload folder not found: $SourceRoot"
}

$requiredMarkers = @(
    "RUN_TOPIC_UNDERSTANDING.ps1",
    "START_DASHBOARD.ps1",
    "src\sciintel\db.py",
    "dashboard\app.py"
)
foreach ($marker in $requiredMarkers) {
    $markerPath = Join-Path $ProjectRoot $marker
    if (-not (Test-Path -LiteralPath $markerPath -PathType Leaf)) {
        throw "This does not look like the expected SIL project root. Missing: $marker"
    }
}

$payloadFiles = @(Get-ChildItem -LiteralPath $SourceRoot -Recurse -File)
if ($payloadFiles.Count -eq 0) {
    throw "Patch payload is empty."
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$BackupRoot = Join-Path $ProjectRoot (Join-Path ".sil_backups" ("topic_naming_ru_" + $stamp))
$copied = 0
$backedUp = 0

foreach ($sourceFile in $payloadFiles) {
    $relativePath = $sourceFile.FullName.Substring($SourceRoot.Length).TrimStart('\', '/')
    $targetPath = Join-Path $ProjectRoot $relativePath
    $targetDirectory = Split-Path -Parent $targetPath
    if (-not (Test-Path -LiteralPath $targetDirectory -PathType Container)) {
        New-Item -ItemType Directory -Path $targetDirectory -Force | Out-Null
    }

    if (Test-Path -LiteralPath $targetPath -PathType Leaf) {
        $backupPath = Join-Path $BackupRoot $relativePath
        $backupDirectory = Split-Path -Parent $backupPath
        if (-not (Test-Path -LiteralPath $backupDirectory -PathType Container)) {
            New-Item -ItemType Directory -Path $backupDirectory -Force | Out-Null
        }
        Copy-Item -LiteralPath $targetPath -Destination $backupPath -Force
        $backedUp++
    }

    Copy-Item -LiteralPath $sourceFile.FullName -Destination $targetPath -Force
    $copied++
}

Write-Host "SIL topic naming patch applied." -ForegroundColor Green
Write-Host "Project: $ProjectRoot"
Write-Host "Files copied: $copied"
Write-Host "Existing files backed up: $backedUp"
if ($backedUp -gt 0) {
    Write-Host "Backup folder: $BackupRoot"
}
Write-Host "Next: .\RUN_TOPIC_UNDERSTANDING.ps1"
