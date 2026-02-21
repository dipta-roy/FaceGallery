@echo off
setlocal enabledelayedexpansion

:: ─────────────────────────────────────────────────────────────────────────────
:: FaceGallery Clean-up Tool
:: ─────────────────────────────────────────────────────────────────────────────

echo [^>] FaceGallery Repository Cleaner
echo ===================================
echo.

:: 1. Ask about venv
set /p CLEAN_VENV="[?] Delete virtual environment (venv folder)? (y/n): "

:: 2. Ask about Database
set /p CLEAN_DB="[?] Delete FaceGallery application data (Database/Logs in AppData)? (y/n): "

echo.
echo [^>] Cleaning temporary files...

:: Delete python caches
echo     - Removing __pycache__ folders...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"

echo     - Removing .pyc files...
del /s /q *.pyc >nul 2>&1

:: Clean local project logs
if exist logs (
    echo     - Cleaning local logs...
    rd /s /q logs
    mkdir logs
)

:: Clean debug scripts created during troubleshooting
if exist debug_deepface.py del debug_deepface.py
if exist debug_detectors.py del debug_detectors.py
if exist test_mp_import.py del test_mp_import.py
if exist debug_detect_image.py del debug_detect_image.py

:: 3. Delete venv if requested
if /i "%CLEAN_VENV%"=="y" (
    if exist venv (
        echo [^>] Deleting virtual environment...
        rd /s /q venv
        echo     - venv deleted successfully.
    ) else (
        echo     - venv folder NOT found, skipping.
    )
)

:: 4. Delete DB if requested
if /i "%CLEAN_DB%"=="y" (
    echo [^>] Deleting application data...
    set "FG_DATA=%APPDATA%\FaceGallery"
    if exist "!FG_DATA!" (
        rd /s /q "!FG_DATA!"
        echo     - AppData/FaceGallery deleted successfully.
    ) else (
        echo     - Application data folder NOT found in AppData, skipping.
    )
)

echo.
echo ===================================
echo [OK] Project repository cleaned.
echo.
pause
