@echo off
REM Automated build script for Protocol AI distribution packages
REM Builds all three distribution methods

echo ========================================
echo Protocol AI - Distribution Builder
echo ========================================
echo.
echo This will create distributable packages that users can run
echo with ZERO installation - just double-click an icon!
echo.
echo Choose distribution method:
echo.
echo [1] Portable Package (RECOMMENDED)
echo     - Size: ~2-3 GB
echo     - Users: Extract ZIP and double-click
echo     - Best for: Non-technical users
echo.
echo [2] Ultimate Launcher
echo     - Size: ~50 MB exe
echo     - Downloads everything on first run
echo     - Best for: Minimal download size
echo.
echo [3] Professional Installer
echo     - Size: ~1-2 GB installer
echo     - Professional install wizard
echo     - Best for: Commercial distribution
echo.
set /p choice="Enter choice (1-3): "

if "%choice%"=="1" goto portable
if "%choice%"=="2" goto ultimate
if "%choice%"=="3" goto professional

echo Invalid choice
pause
exit /b 1

:portable
echo.
echo ========================================
echo Building Portable Package
echo ========================================
echo.
echo This will:
echo - Download embedded Python (~25 MB)
echo - Install all dependencies (~500 MB)
echo - Copy all project files
echo - Create ready-to-distribute folder
echo.
echo Time required: 10-15 minutes
echo Internet required: Yes (for downloads)
echo.
pause

python build_portable_package.py

echo.
echo ========================================
echo Build Complete!
echo ========================================
echo.
echo Next steps:
echo.
echo 1. Copy your model file to:
echo    ProtocolAI_Portable\models\
echo.
echo 2. Test the package:
echo    cd ProtocolAI_Portable
echo    "Launch Protocol AI.bat"
echo.
echo 3. If it works, create ZIP file:
echo    - Right-click ProtocolAI_Portable folder
echo    - Send to ^> Compressed (zipped) folder
echo.
echo 4. Distribute the ZIP file to users!
echo    They extract and double-click "Launch Protocol AI.bat"
echo.
pause
exit /b 0

:ultimate
echo.
echo ========================================
echo Building Ultimate Launcher
echo ========================================
echo.
echo This will:
echo - Create launcher source code
echo - Compile to single .exe file
echo - Bundle project files
echo.
echo Time required: 2-3 minutes
echo Internet required: No (but users need it on first run)
echo.
pause

REM Check for PyInstaller
python -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo Installing PyInstaller...
    pip install pyinstaller
)

REM Generate launcher source
python build_ultimate_launcher.py

REM Build executable
echo.
echo Building executable...
pyinstaller ultimate_launcher.py --onefile --windowed --name=ProtocolAI ^
    --add-data="modules;modules" ^
    --add-data="gui;gui" ^
    --add-data="tools;tools" ^
    --add-data="analytical_frameworks.txt;." ^
    --add-data="protocol_ai.py;." ^
    --add-data="report_formatter.py;."

echo.
echo ========================================
echo Build Complete!
echo ========================================
echo.
echo Executable location: dist\ProtocolAI.exe
echo.
echo Distribute this single file to users!
echo On first run, it will:
echo - Download Python runtime
echo - Install dependencies
echo - Prompt for model file
echo - Create desktop shortcut
echo - Launch GUI
echo.
echo First run requires internet connection.
echo.
pause
exit /b 0

:professional
echo.
echo ========================================
echo Building Professional Installer
echo ========================================
echo.
echo This requires Inno Setup to be installed.
echo Download from: https://jrsoftware.org/isinfo.php
echo.
pause

REM Check for PyInstaller
python -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo Installing PyInstaller...
    pip install pyinstaller
)

REM Build main executable
echo Building main executable...
python build_executable.py

REM Check for Inno Setup
set INNO="C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
if not exist %INNO% (
    echo.
    echo [ERROR] Inno Setup not found!
    echo Please install Inno Setup from:
    echo https://jrsoftware.org/isinfo.php
    echo.
    pause
    exit /b 1
)

REM Compile installer
echo.
echo Compiling installer...
%INNO% installer.iss

echo.
echo ========================================
echo Build Complete!
echo ========================================
echo.
echo Installer location: installer_output\ProtocolAI_Setup.exe
echo.
echo Distribute this installer to users!
echo They will:
echo - Double-click ProtocolAI_Setup.exe
echo - Follow installation wizard
echo - Get desktop shortcut
echo - Launch from Start Menu or desktop
echo.
pause
exit /b 0
