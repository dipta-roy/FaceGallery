# 📷 FaceGallery

**FaceGallery** is a cross-platform, privacy-focused offline desktop photo manager. It uses advanced AI to automatically detect, cluster, and organize faces in your local photo collection—all without your data ever leaving your machine.

---

## 🌟 Features

- **Privacy First**: 100% offline. No cloud uploads, no external APIs.
- **Smart Face Detection**: Support for multiple state-of-the-art backends:
  - **InsightFace** (High accuracy)
  - **Mediapipe** & **DeepFace** (Fast & Efficient)
  - **Dlib** (Classic & Reliable)
- **Face Clustering**: Automatically groups similar unknown faces together, making it easy to name hundreds of photos in just a few clicks.
- **Local Web Server**: Host a local gallery on your Wi-Fi so other devices (phones, tablets, laptops) can browse your photos securely.
- **Web Management & Upload**: 
  - Create new server folders and upload photos directly from your browser.
  - User-specific upload directories: Folders are created under a central `/uploads` directory, with user-specific prefixes (e.g., `/uploads/username-myfolder`). The web UI displays only the relative folder name (`myfolder`).
  - Role-based access: Administrators can delegate upload permissions to specific users.
  - Ownership: Users can be restricted to managing only the folders they created.
- **Web Interface Face Naming & Editing**: Name unknown faces directly in the web browser with AI-powered grouping and auto-matching, and also edit existing face-to-person assignments or rename people directly from image views.
- **Multi-User Access**: Secure the web interface with per-user PIN codes and granular roles (Admin, Uploader, Viewer).
- **Fast Search**: Browse your library by person, folder, or date.
- **Export Tools**: Easily package selected photos into ZIP files.
- **Rich User Interface**: A modern, dark-themed GUI built with PyQt6.

---

## 🚀 Getting Started

### Prerequisites

- **Python**: 3.9 or higher.
- **OS**: Windows, macOS, or Linux.
- **Build Tools**: [Visual Studio C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (Required for installing native dependencies like `dlib` or `insightface` on Windows).

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/dipta-roy/FaceGallery.git
   cd FaceGallery
   ```

2. **Setup virtual environment**:
   - **Windows**: `run.bat` or `run_web.bat` will create and activate the virtual environment and install dependencies from `requirements.txt` automatically.
   - **Linux/macOS**: `./run.sh` or `./run_web.sh` will create and activate the virtual environment and install dependencies from `requirements.txt` automatically.

   *Manual setup (optional):*
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   pip install -r requirements.txt
   ```

3. **Setup AI Models (Manual Setup)**:
   For offline usage, place the face recognition weights manually:
   - **Windows**: Create the directory `%USERPROFILE%\.deepface\weights\` and place `facenet512_weights.h5` inside.
   - **Linux**: Create the directory `~/.deepface/weights/` and place `facenet512_weights.h5` inside.
   - **macOS**: Create the directory `~/.deepface/weights/` and place `facenet512_weights.h5` inside.
   - Download `facenet512_weights.h5` and place it in the respective folder.

---

## 🎮 Running FaceGallery

### Desktop GUI (Full Experience)
- **Windows**: Double-click `run.bat`
- **Linux/macOS**: Run `./run.sh`
- **CLI**: `python main.py`

### Standalone Web Server (Headless/Server Mode)
If you want to run FaceGallery as a server without the desktop interface:
- **Windows**: Double-click `run_web.bat`
- **Linux/macOS**: Run `./run_web.sh`
- **CLI**: `python run_web.py`

---

## ⌨️ Keyboard Shortcuts (Desktop)

| Action | Shortcut |
| :--- | :--- |
| **Scan Folders** | `Ctrl + O` |
| **Name Faces** | `Ctrl + N` |
| **Select All** | `Ctrl + A` |
| **Clear Selection** | `Esc` |
| **Refresh List** | `F5` or `Ctrl + R` |
| **Start Web Server** | `Ctrl + W` |
| **Open Web Browser** | `Ctrl + B` |
| **Manage Users** | `Ctrl + U` |
| **Preferences** | `Ctrl + ,` |
| **About Software** | `F1` |
| **Exit** | `Alt + F4` |

---

## 🛠 Project Structure

- `src/core/`: Business logic, scanner, and application state.
- `src/db/`: SQLite schema and data management.
- `src/face_engine/`: AI detection and clustering logic.
- `src/gui/`: PyQt6 desktop interface components.
- `src/web/`: Flask web server, templates, and API.
- `run_web.py`: Standalone web server entry point.
- `main.py`: Full Desktop GUI entry point.

---

## 📋 Technical Details

- **Technical Details**:
- **Static File Serving**: Improved and robust static file serving for web assets like logos and favicons.
- **Database**: SQLite with Write-Ahead Logging (WAL).
- **Path Handling**: Standardized absolute pathing for cross-platform and multi-environment reliability.
- **Security**: PIN-based authentication with salted SHA-256 hashing.
- **AI Backend**: Modular design allowing fallback between InsightFace, DeepFace, and Dlib.

---

## 👨‍💻 Author

**Dipta Roy**  
Version: 1.1.0  
License: MIT
