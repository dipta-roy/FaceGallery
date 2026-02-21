#!/bin/bash

# ─────────────────────────────────────────────────────────────────────────────
# FaceGallery Launcher (Linux/macOS)
# ─────────────────────────────────────────────────────────────────────────────

echo "[>] Initializing FaceGallery..."

# Set project root
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$PROJECT_ROOT"

# Create necessary directories
if [ ! -d "logs" ]; then
    echo "[>] Creating logs directory..."
    mkdir "logs"
fi

# Virtual environment directory
VENV_DIR="venv"

# 1. Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "[>] Creating virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

# 2. Activate venv
echo "[>] Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# 3. Install requirements
if [ -f "requirements.txt" ]; then
    echo "[>] Installing dependencies..."
    pip install -r requirements.txt
fi

# 4. Run application
echo "[>] Starting FaceGallery..."
echo ""
python3 main.py

if [ $? -ne 0 ]; then
    echo ""
    echo "[!] Application exited with error code $?"
fi
