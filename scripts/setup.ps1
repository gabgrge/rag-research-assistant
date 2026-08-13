Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$PSScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$EnvPath = Join-Path $ProjectRoot ".env"

Write-Host "--- Configuration de l'Assistant Documentaire RAG ---" -ForegroundColor Cyan

if (-not (Test-Path $EnvPath)) {
    Write-Host "[INFO] Création du fichier .env à partir de .env.example..." -ForegroundColor Yellow

    $ExamplePath = Join-Path $ProjectRoot ".env.example"
    Copy-Item -Path $ExamplePath -Destination $EnvPath

    Write-Host "[OK] Fichier .env créé. Veuillez l'ouvrir pour remplir vos accès !" -ForegroundColor Green
} else {
    Write-Host "[INFO] Fichier .env déjà existant. Étape ignorée."
}

Write-Host "`n[INFO] Initialisation de l'environnement Docker (Téléchargement des composants)..." -ForegroundColor Yellow
docker info >$null 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Docker Desktop ne semble pas lancé. Veuillez l'allumer pour terminer l'installation."
} else {
    docker compose build
    Write-Host "[OK] Image Docker compilée avec succès !" -ForegroundColor Green
}

Write-Host "`n[SUCCÈS] Installation terminée !" -ForegroundColor Green
