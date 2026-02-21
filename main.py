"""
Application entry point for FaceGallery.
"""

import sys
import logging
import os
from pathlib import Path

# ── Bootstrap Python path ──────────────────────────────────────────────────
ROOT = Path(__file__).parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Suppress AI Noises ──────────────────────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'      # 3 = Fatal only
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'    # Disable oneDNN performance logs
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*sparse_softmax_cross_entropy.*")

# Silence noisy libraries
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)


def _setup_logging():
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Root level is INFO for the file log
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # File handler: keep everything (INFO+)
    fh = logging.FileHandler(log_dir / "facegallery.log", encoding="utf-8")
    fh.setFormatter(formatter)
    root.addHandler(fh)

    # Console handler: only show WARNING and above to keep CMD clean
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.WARNING)
    ch.setFormatter(formatter)
    root.addHandler(ch)


def _get_db_path() -> Path:
    from src.utils.helpers import data_dir
    return data_dir() / "facegallery.db"


def _ensure_default_user(db):
    """If no users exist, create a default 'admin' user with PIN 1234."""
    if db.user_count() == 0:
        from src.utils.helpers import hash_pin
        db.create_user("admin", hash_pin("1234"), role="admin")
        logging.getLogger(__name__).info(
            "Created default user 'admin' with PIN 1234. "
            "Please change this PIN in the Users menu."
        )


def main():
    _setup_logging()
    # Clean console message for the user
    print("\n🚀 FaceGallery is starting... (Please wait while AI models load)\n")
    
    logger = logging.getLogger(__name__)

    # ── Database & core ───────────────────────────────────────────────── #
    from src.db.manager import DatabaseManager
    from src.core.app_core import AppCore

    db_path = _get_db_path()
    logger.info("Database: %s", db_path)
    db = DatabaseManager(db_path)
    _ensure_default_user(db)
    app_core = AppCore(db)

    # ── Qt application ───────────────────────────────────────────────── #
    try:
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtGui import QFont
        from PyQt6.QtCore import Qt

        app = QApplication(sys.argv)
        app.setApplicationName("FaceGallery")
        app.setApplicationVersion("1.0.0")
        app.setOrganizationName("FaceGallery")

        # Set AppUserModelID to show icon in taskbar on Windows
        if sys.platform == "win32":
            import ctypes
            myappid = u'facegallery.photo.manager' 
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

        from PyQt6.QtGui import QIcon
        app.setWindowIcon(QIcon("resources/icons/app-logo.png"))

        # Apply global font
        font = QFont("Segoe UI", 10)
        app.setFont(font)

        from src.gui.main_window import MainWindow
        window = MainWindow(db, app_core)
        window.show()

        sys.exit(app.exec())

    except ImportError as e:
        logger.error("PyQt6 not found: %s", e)
        print(
            "\n⚠ PyQt6 is not installed.\n"
            "Install it with: pip install PyQt6\n"
            "\nThe web server can still be used standalone with Flask.\n"
        )
        # Fallback: just start the web server
        _run_headless(db, app_core)


def _run_headless(db, app_core):
    """Run the web server only (no GUI)."""
    from src.web.server import start_server
    import time

    port = app_core.get_web_port()
    bind_all = app_core.get_web_bind_all()
    start_server(db, app_core, port=port, bind_all=bind_all)

    from src.utils.helpers import get_local_ip
    ip = get_local_ip()
    print(f"\n🌐 FaceGallery web server running:")
    print(f"   Local:   http://localhost:{port}")
    print(f"   Network: http://{ip}:{port}")
    print("   Press Ctrl+C to exit.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
