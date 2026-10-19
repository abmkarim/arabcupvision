<#
.SYNOPSIS
    Point d'entrée unique des commandes du projet ArabCupVision (Windows / PowerShell).

.EXAMPLE
    .\tasks.ps1 setup
    .\tasks.ps1 test
    .\tasks.ps1 process -Video data\videos\match.mp4 -MaxSeconds 20
#>
param(
    [Parameter(Position = 0)]
    [ValidateSet("help", "setup", "test", "lint", "format", "train", "process", "synth",
                 "app", "eval", "benchmark", "docker", "secrets", "clean")]
    [string]$Target = "help",
    [string]$Video = "",
    [string]$Config = "configs/default.yaml",
    [double]$MaxSeconds = 0,
    [string]$Gt = "",
    [string]$Labels = ""
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$LogDir = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Invoke-Py {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$PyArgs)
    if (-not (Test-Path $VenvPython)) {
        throw "Environnement virtuel absent : lancez d'abord '.\tasks.ps1 setup'."
    }
    $env:PYTHONPATH = Join-Path $PSScriptRoot "src"
    & $VenvPython @PyArgs
    if ($LASTEXITCODE -ne 0) { throw "La commande Python a échoué (code $LASTEXITCODE)." }
}

function Find-DemoVideo {
    $dir = Join-Path $PSScriptRoot "data\videos"
    if (Test-Path $dir) {
        $found = Get-ChildItem -Path $dir -File |
            Where-Object { $_.Extension -in ".mp4", ".avi", ".mov", ".mkv" } |
            Sort-Object Name | Select-Object -First 1
        if ($found) { return $found.FullName }
    }
    return ""
}

switch ($Target) {
    "help" {
        Write-Host "Cibles : setup, test, lint, format, train, process, synth, app, eval, benchmark, docker, secrets, clean"
    }
    "setup" {
        if (-not (Test-Path $VenvPython)) {
            py -3.11 -m venv .venv
            if ($LASTEXITCODE -ne 0) { throw "Impossible de créer le venv Python 3.11." }
        }
        & $VenvPython -m pip install --upgrade pip
        & $VenvPython -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
        if ($LASTEXITCODE -ne 0) { throw "Installation de torch (CPU) échouée." }
        & $VenvPython -m pip install -e ".[train,eval,dev]"
        if ($LASTEXITCODE -ne 0) { throw "Installation des dépendances échouée." }
        Write-Host "Environnement prêt."
    }
    "test" {
        Invoke-Py -m ruff check src tests app training
        Invoke-Py -m pytest
    }
    "lint" { Invoke-Py -m ruff check src tests app training }
    "format" { Invoke-Py -m ruff format src tests app training }
    "train" {
        Invoke-Py training/train_all.py --config $Config
    }
    "synth" {
        Invoke-Py -m arabcupvision.cli synth --output data/videos/synthetic_demo.mp4
    }
    "process" {
        if (-not $Video) { $Video = Find-DemoVideo }
        if (-not $Video) {
            Write-Host "Aucune vidéo dans data/videos : génération d'une vidéo synthétique."
            Invoke-Py -m arabcupvision.cli synth --output data/videos/synthetic_demo.mp4
            $Video = "data/videos/synthetic_demo.mp4"
        }
        $pyArgs = @("-m", "arabcupvision.cli", "process", $Video, "--config", $Config)
        if ($MaxSeconds -gt 0) { $pyArgs += @("--max-seconds", "$MaxSeconds") }
        Invoke-Py @pyArgs
    }
    "app" {
        $env:PYTHONPATH = Join-Path $PSScriptRoot "src"
        & $VenvPython -m streamlit run app/streamlit_app.py --server.port 8501
    }
    "eval" {
        if ($Gt) {
            if (-not $Video) { throw "Précisez -Video (vidéo correspondant à la vérité terrain)." }
            Invoke-Py -m arabcupvision.cli eval-tracking --video $Video --gt $Gt --config $Config
        }
        if ($Labels) {
            Invoke-Py -m arabcupvision.cli eval-teams --labels $Labels --predictions outputs/latest/tracks.csv
        }
        if (-not $Gt -and -not $Labels) {
            Write-Host "Utilisation : .\tasks.ps1 eval -Video <video> -Gt <gt.txt>  et/ou  -Labels <labels.csv>"
        }
    }
    "benchmark" {
        if (-not $Video) { $Video = Find-DemoVideo }
        if (-not $Video) { throw "Aucune vidéo trouvée : précisez -Video ou lancez '.\tasks.ps1 synth'." }
        Invoke-Py -m arabcupvision.cli benchmark $Video --config $Config
    }
    "docker" {
        docker compose build
        if ($LASTEXITCODE -ne 0) { throw "Construction de l'image Docker échouée." }
        Write-Host "Lancez 'docker compose up' puis ouvrez http://localhost:8501"
    }
    "secrets" {
        Invoke-Py -m detect_secrets scan --exclude-files "(\.venv|data|models|logs|runs|outputs)[/\\]"
    }
    "clean" {
        foreach ($p in @("outputs", ".pytest_cache", ".ruff_cache")) {
            if (Test-Path $p) { Remove-Item -Recurse -Force $p }
        }
    }
}
