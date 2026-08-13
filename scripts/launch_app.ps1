$PSScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$dockerCheck = docker info 2>$null
if ($LASTEXITCODE -ne 0) {
    Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    while ($LASTEXITCODE -ne 0) {
        Start-Sleep -Seconds 5
        docker info 2>$null
    }
}

docker compose up -d
Start-Sleep -Seconds 2
Start-Process "http://localhost:8501"
