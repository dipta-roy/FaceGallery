"""
Utility helpers used across the application.
"""

import hashlib
import os
import struct
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif",
    ".webp", ".heic", ".heif", ".gif"
}


def is_image_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS


def walk_images(folder: str | Path):
    """Yield absolute paths of image files found recursively under *folder*."""
    for root, _dirs, files in os.walk(str(folder)):
        for f in files:
            fp = os.path.join(root, f)
            if is_image_file(fp):
                yield fp


def compute_file_hash(path: str | Path, algo: str = "sha256") -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_pin(pin: str) -> str:
    """Return SHA-256 hex digest of the PIN string."""
    return hashlib.sha256(pin.encode()).hexdigest()


def verify_pin(pin: str, pin_hash: str) -> bool:
    return hash_pin(pin) == pin_hash


def embedding_to_bytes(vec) -> bytes:
    """Serialise a list/array of floats to raw bytes."""
    import numpy as np
    arr = np.array(vec, dtype=np.float32)
    return arr.tobytes()


def bytes_to_embedding(data: bytes):
    """Deserialise raw bytes back to a numpy float32 array."""
    import numpy as np
    return np.frombuffer(data, dtype=np.float32)


def cosine_similarity(a, b) -> float:
    import numpy as np
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def euclidean_distance(a, b) -> float:
    import numpy as np
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    return float(np.linalg.norm(a - b))


def make_thumbnail(image_path: str | Path, size: tuple = (256, 256)) -> Optional[bytes]:
    """Return JPEG bytes of a thumbnail, or None on failure."""
    try:
        from PIL import Image, ExifTags
        import io
        img = Image.open(str(image_path))
        # Auto-rotate based on EXIF
        try:
            for orientation_tag in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation_tag] == "Orientation":
                    break
            exif = img._getexif()
            if exif:
                orientation_val = exif.get(orientation_tag)
                if orientation_val == 3:
                    img = img.rotate(180, expand=True)
                elif orientation_val == 6:
                    img = img.rotate(270, expand=True)
                elif orientation_val == 8:
                    img = img.rotate(90, expand=True)
        except Exception:
            pass
        img.thumbnail(size, Image.LANCZOS)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=70)
        return buf.getvalue()
    except Exception as exc:
        logger.debug("Thumbnail error for %s: %s", image_path, exc)
        return None


def get_image_info(image_path: str | Path):
    """Return (width, height, date_taken, orientation) or (None,None,None,1)."""
    try:
        from PIL import Image, ExifTags
        img = Image.open(str(image_path))
        w, h = img.size
        date_taken = None
        orientation = 1
        try:
            exif_data = img._getexif() or {}
            for tag_id, val in exif_data.items():
                tag = ExifTags.TAGS.get(tag_id, "")
                if tag == "DateTimeOriginal":
                    date_taken = str(val)
                elif tag == "Orientation":
                    orientation = int(val)
        except Exception:
            pass
        return w, h, date_taken, orientation
    except Exception:
        return None, None, None, 1


def get_local_ip() -> str:
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def data_dir() -> Path:
    """Return platform-appropriate user data directory."""
    import sys
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home()))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    d = base / "FaceGallery"
    d.mkdir(parents=True, exist_ok=True)
    return d
