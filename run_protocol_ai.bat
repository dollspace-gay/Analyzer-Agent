@echo off
REM Protocol AI Launcher
REM Double-click this file to start the Protocol AI system

echo ========================================
echo Protocol AI - Governance Layer System
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.9+ from https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

echo [1/4] Python detected
python --version

REM Check if virtual environment exists
if not exist "venv\" (
    echo.
    echo [2/4] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created
) else (
    echo [2/4] Virtual environment found
)

REM Activate virtual environment
echo.
echo [3/4] Activating virtual environment...
call venv\Scripts\activate.bat

REM Install/update dependencies
echo.
echo [4/4] Checking dependencies...
pip install -r requirements.txt --quiet --upgrade
if errorlevel 1 (
    echo [WARNING] Some dependencies may have failed to install
)

echo.
echo ========================================
echo Starting Protocol AI GUI...
echo ========================================
echo.

REM Start the GUI
cd gui
python app.py

REM If GUI exits, pause to show any error messages
if errorlevel 1 (
    echo.
    echo [ERROR] Protocol AI exited with an error
    pause
)
