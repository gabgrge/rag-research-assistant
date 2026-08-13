Set objShell = CreateObject("WScript.Shell")

scriptPath = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptTarget)
if scriptPath = "" then scriptPath = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)

objShell.Run "powershell.exe -NoProfile -ExecutionPolicy Bypass -File """ & scriptPath & "\launch_app.ps1""", 0, False
