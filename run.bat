@echo off
setlocal enabledelayedexpansion

:: ─────────────────────────────────────────────────────────────────────────────
:: FaceGallery Launcher
:: ─────────────────────────────────────────────────────────────────────────────

echo [^>] Initializing FaceGallery...

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
    if !errorlevel! neq 0 (
        echo [!] Failed to create virtual environment. 
        echo [!] Please ensure Python is installed and in your PATH.
        pause
        exit /b 1
    )
)

:: 2. Activate venv
echo [^>] Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"
if !errorlevel! neq 0 (
    echo [!] Failed to activate virtual environment.
    pause
    exit /b 1
)

:: 3. Install/Update dependencies
echo [^>] Checking dependencies...
python -m pip install --upgrade pip
if exist "requirements.txt" (
    echo [^>] Installing requirements from requirements.txt...
    pip install -r requirements.txt
    if !errorlevel! neq 0 (
        echo [!] Failed to install dependencies.
        pause
        exit /b 1
    )
) else (
    echo [!] requirements.txt not found! Skipping dependency installation.
)

:: 4. Run application
echo [^>] Starting FaceGallery...
echo.
python main.py
if !errorlevel! neq 0 (
    echo.
    echo [!] Application exited with error code !errorlevel!
    echo [!] Check the logs in the 'logs' folder for details.
    pause
)

exit /b 0
