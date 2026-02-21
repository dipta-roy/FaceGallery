"""
Directly start the FaceGallery web server without the GUI.
Usage: python run_web.py
"""

import sys
import logging
import time
import os
from pathlib import Path

# -- Bootstrap Python path --
ROOT = Path(__file__).parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# -- Suppress AI Noises --
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.db.manager import DatabaseManager
from src.core.app_core import AppCore
from src.web.server import start_server
from src.utils.helpers import data_dir, get_local_ip, hash_pin

def setup_logging():
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    
    fh = logging.FileHandler(log_dir / "web_server.log", encoding="utf-8")
    fh.setFormatter(formatter)
    root.addHandler(fh)
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    root.addHandler(ch)

def main():
    setup_logging()
    logger = logging.getLogger("WebRunner")
    
    print("\n🚀 Starting FaceGallery Web Server...\n")
    
    # Init DB
    db_path = data_dir() / "facegallery.db"
    db = DatabaseManager(db_path)
    
    # Ensure admin user
    if db.user_count() == 0:
        db.create_user("admin", hash_pin("1234"), role="admin")
        print("Initialised default user 'admin' with PIN 1234")
    
    # Init Core
    app_core = AppCore(db)
    
    # Start Server
    port = app_core.get_web_port()
    bind_all = app_core.get_web_bind_all()
    
    start_server(db, app_core, port=port, bind_all=bind_all)
    
    ip = get_local_ip()
    print(f"\n🌐 FaceGallery web server is now ACTIVE:")
    print(f"   Local:   http://localhost:{port}")
    if bind_all:
        print(f"   Network: http://{ip}:{port}")
    print("\n   Press Ctrl+C to terminate the server.\n")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping web server...")

if __name__ == "__main__":
    main()
