# PowerShell script to create a desktop shortcut for Protocol AI
# Right-click this file and select "Run with PowerShell"

$WshShell = New-Object -ComObject WScript.Shell
$DesktopPath = [System.Environment]::GetFolderPath('Desktop')
$ShortcutPath = Join-Path $DesktopPath "Protocol AI.lnk"
$TargetPath = Join-Path $PSScriptRoot "run_protocol_ai.bat"
$IconPath = Join-Path $PSScriptRoot "gui\icon.ico"

# Create the shortcut
$Shortcut = $WshShell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = $TargetPath
$Shortcut.WorkingDirectory = $PSScriptRoot
$Shortcut.Description = "Protocol AI - Governance Layer System"

# Set icon if it exists
if (Test-Path $IconPath) {
    $Shortcut.IconLocation = $IconPath
}

$Shortcut.Save()

Write-Host "========================================" -ForegroundColor Green
Write-Host "Desktop shortcut created successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Shortcut location: $ShortcutPath"
Write-Host ""
Write-Host "You can now double-click 'Protocol AI' on your desktop to launch the system."
Write-Host ""
Pause
