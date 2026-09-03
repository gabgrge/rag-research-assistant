Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location $ProjectRoot

$StartAppVbs = Join-Path $ProjectRoot "scripts\launch_app.vbs"

if (-not (Test-Path $StartAppVbs)) {
    Write-Error "Script de lancement introuvable : $StartAppVbs. L'installation est peut-être incomplète."
    exit 1
}

$ProgramsDir = [Environment]::GetFolderPath("Programs")
$ShortcutPath = Join-Path $ProgramsDir "Assistant documentaire.lnk"

$Shell = New-Object -ComObject WScript.Shell
$Shortcut = $Shell.CreateShortcut($ShortcutPath)

$Shortcut.TargetPath = "wscript.exe"
$Shortcut.Arguments = "`"$StartAppVbs`""
$Shortcut.WorkingDirectory = $ProjectRoot
$Shortcut.IconLocation = "shell32.dll,22"

$Shortcut.Save()

Write-Host "[SUCCÈS] Raccourci créé dans le menu Démarrer : $ShortcutPath" -ForegroundColor Green
