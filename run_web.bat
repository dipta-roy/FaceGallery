@echo off
setlocal enabledelayedexpansion

:: ─────────────────────────────────────────────────────────────────────────────
:: FaceGallery Web-Only Launcher
:: ─────────────────────────────────────────────────────────────────────────────

echo [^>] Initializing FaceGallery Web Server (Headless)...

:: Set project root
set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%"

:: Create necessary directories
if not exist "logs" (
    echo [^>] Creating logs directory...
    mkdir "logs"
)

:: Virtual environment directory
set "VENV_DIR=venv"

:: 1. Create venv if it doesn't exist
if not exist "%VENV_DIR%" (
    echo [^>] Creating virtual environment in %VENV_DIR%...
    python -m venv "%VENV_DIR%"
)

:: 2. Activate venv
echo [^>] Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"

:: 3. Install dependencies if not already installed (pip handles caching)
echo [^>] Installing/Upgrading Python dependencies...
pip install -r requirements.txt

:: 4. Run application
echo [^>] Starting Web Server...
echo.
python run_web.py
if !errorlevel! neq 0 (
    echo.
    echo [!] Application exited with error code !errorlevel!
    pause
)

exit /b 0
