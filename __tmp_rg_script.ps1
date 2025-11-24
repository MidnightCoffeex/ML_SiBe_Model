param(
    [string]$ModelPath = "Modelle/ALL/xgb/3/model.joblib",
    [string]$OutputDir = "Modell-Visual",
    [int]$TreesPerTarget = 3,
    [int]$StartDepth = 5,
    [int]$MaxSamples = 7500,
    [string]$FeaturesRoot = "Final_Features_Test",
    [string]$Part = "ALL",
    [string]$FeatureFile = "",
    [string]$Targets = "",
    [int]$SampleRows = 10000,
    [int]$TreeOffset = 0,
    [switch]$Tail
)

$ErrorActionPreference = "Stop"

function Resolve-RelativePath {
    param([string]$PathValue)

    if ([string]::IsNullOrWhiteSpace($PathValue)) {
        return $PathValue
    }

    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return $PathValue
    }

    return Join-Path -Path $PSScriptRoot -ChildPath $PathValue
}

$solutionRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path -Path $solutionRoot -ChildPath ".venv\Scripts\python.exe"

if (-not (Test-Path -Path $pythonPath)) {
    throw "Python-Interpreter in der .venv wurde nicht gefunden: $pythonPath"
}

$scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "temp_xgb_pyvis.py"
if (-not (Test-Path -Path $scriptPath)) {
    throw "Visualisierungsskript fehlt: $scriptPath"
}

$resolvedModel = Resolve-RelativePath -PathValue $ModelPath
$resolvedOutput = Resolve-RelativePath -PathValue $OutputDir
$resolvedFeaturesRoot = Resolve-RelativePath -PathValue $FeaturesRoot
$resolvedFeatureFile = Resolve-RelativePath -PathValue $FeatureFile

Write-Host "Starte SuperTree-Visualisierung..." -ForegroundColor Cyan

$arguments = @(
    "--no-prompt",
    "--model", $resolvedModel,
    "--output-dir", $resolvedOutput,
    "--trees-per-target", $TreesPerTarget,
    "--start-depth", $StartDepth,
    "--max-samples", $MaxSamples
)

if (-not [string]::IsNullOrWhiteSpace($Targets)) {
    $arguments += @("--targets", $Targets)
}

if (-not [string]::IsNullOrWhiteSpace($FeatureFile)) {
    $arguments += @("--feature-file", $resolvedFeatureFile)
} else {
    if (-not [string]::IsNullOrWhiteSpace($FeaturesRoot)) {
        $arguments += @("--features-root", $resolvedFeaturesRoot)
    }
    if (-not [string]::IsNullOrWhiteSpace($Part)) {
        $arguments += @("--part", $Part)
    }
}

if ($SampleRows -ge 0) {
    $arguments += @("--sample-rows", $SampleRows)
}

if ($TreeOffset -gt 0) {
    $arguments += @("--tree-offset", $TreeOffset)
}

if ($Tail.IsPresent) {
    $arguments += @("--tail")
}

& $pythonPath $scriptPath @arguments

if ($LASTEXITCODE -ne 0) {
    throw "Das Visualisierungsskript wurde mit Exitcode $LASTEXITCODE beendet."
}

Write-Host "Visualisierung abgeschlossen. Dateien liegen in $resolvedOutput." -ForegroundColor Green
