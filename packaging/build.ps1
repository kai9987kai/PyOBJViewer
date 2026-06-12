# Builds the standalone exe and the Windows installer.
# Run from the repo root:  powershell -ExecutionPolicy Bypass -File packaging\build.ps1
# Requires: Python with PyInstaller + Pillow, Inno Setup 6.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# 1) Application icon
py packaging\make_icon.py

# 2) Standalone exe (numpy + Pillow are bundled automatically if installed)
py -m PyInstaller --noconfirm --clean --onefile --windowed `
    --name PyOBJViewer `
    --icon assets\icon.ico `
    --version-file packaging\version_info.txt `
    main.py

# 3) Installer
$iscc = @(
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
    "C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
    "C:\Program Files\Inno Setup 6\ISCC.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $iscc) { throw "Inno Setup 6 not found - install it (winget install JRSoftware.InnoSetup)" }

& $iscc "packaging\installer.iss"

Write-Host ""
Write-Host "Portable exe : dist\PyOBJViewer.exe"
Write-Host "Installer    : dist\installer\PyOBJViewer-Setup-2.0.0.exe"
