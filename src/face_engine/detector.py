"""
Face recognition engine – detects faces and computes embeddings.

Supports three backends (tried in order of preference):
  1. insightface  (best accuracy, recommended)
  2. face_recognition / dlib  (classic, widely available)
  3. DeepFace  (fallback)

If none is installed the engine operates in "no-op" mode (no faces detected).
"""

import logging
import io
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Backend detection
# ─────────────────────────────────────────────────────────────────────────────

_BACKEND = None  # "insightface" | "face_recognition" | "deepface" | "mediapipe" | "opencv" | None

MAX_DETECTION_SIZE = 1600  # Increased for better detection of small faces in high-res images
MIN_FACE_AREA_PCT = 0.0004  # 0.04% - smaller to capture faces in group photos
MIN_CONFIDENCE = 0.7       # Increased from 0.5 to be much stricter about what is a face
MAX_ASPECT_RATIO = 2.0     # Faces are generally square-ish; reject long/wide strips


def _detect_backend() -> Optional[str]:
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    
    # Priority 1: insightface
    try:
        import insightface  # noqa: F401
        _BACKEND = "insightface"
        logger.info("Face engine backend: insightface")
        return _BACKEND
    except (ImportError, Exception):
        pass

    # Priority 2: face_recognition
    try:
        import face_recognition  # noqa: F401
        _BACKEND = "face_recognition"
        logger.info("Face engine backend: face_recognition")
        return _BACKEND
    except (ImportError, Exception):
        pass

    # Priority 3: DeepFace
    try:
        from deepface import DeepFace  # noqa: F401
        _BACKEND = "deepface"
        logger.info("Face engine backend: deepface")
        return _BACKEND
    except (ImportError, Exception):
        pass

    # Priority 4: Mediapipe
    try:
        import mediapipe as mp  # noqa: F401
        _BACKEND = "mediapipe"
        logger.info("Face engine backend: mediapipe")
        return _BACKEND
    except (ImportError, Exception):
        pass

    # Priority 5: OpenCV (Fallback)
    try:
        import cv2
        _BACKEND = "opencv"
        logger.info("Face engine backend: opencv (Haar Cascade)")
        return _BACKEND
    except (ImportError, Exception):
        pass

    _BACKEND = None
    return None


# ─────────────────────────────────────────────────────────────────────────────
# insightface helpers
# ─────────────────────────────────────────────────────────────────────────────

_insight_app = None


def _get_insight_app():
    global _insight_app
    if _insight_app is None:
        import insightface
        _insight_app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )
        _insight_app.prepare(ctx_id=0, det_size=(1024, 1024))
    return _insight_app


def _detect_insightface(img_rgb: np.ndarray) -> List[dict]:
    app = _get_insight_app()
    faces = app.get(img_rgb)
    results = []
    for face in faces:
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        emb = face.embedding.astype(np.float32)
        conf = float(face.det_score) if hasattr(face, "det_score") else 1.0
        results.append({
            "bbox": (x1, y1, w, h),
            "embedding": emb,
            "confidence": conf,
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# face_recognition helpers
# ─────────────────────────────────────────────────────────────────────────────

def _detect_face_recognition(img_rgb: np.ndarray) -> List[dict]:
    import face_recognition as fr
    locations = fr.face_locations(img_rgb, model="hog")
    encodings = fr.face_encodings(img_rgb, locations)
    results = []
    for (top, right, bottom, left), enc in zip(locations, encodings):
        x, y, w, h = left, top, right - left, bottom - top
        results.append({
            "bbox": (x, y, w, h),
            "embedding": np.array(enc, dtype=np.float32),
            "confidence": 1.0,
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# DeepFace helpers
# ─────────────────────────────────────────────────────────────────────────────

def _detect_deepface(img_rgb: np.ndarray) -> List[dict]:
    from deepface import DeepFace
    import cv2
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    try:
        # Use 'retinaface' or 'mtcnn' for better detection over 'opencv' (avoids clothes, catches groups)
        objs = DeepFace.represent(
            img_path=img_bgr,
            model_name="Facenet512",
            detector_backend="retinaface", # Much better at rejecting false positives and detecting small faces
            enforce_detection=False,
        )
        results = []
        for obj in objs:
            region = obj.get("facial_area", {})
            x = region.get("x", 0)
            y = region.get("y", 0)
            w = region.get("w", 0)
            h = region.get("h", 0)
            
            # DeepFace returns a single whole-image 'face' if it is told enforce_detection=False 
            # and finds nothing. We ignore those 100% crops.
            if x == 0 and y == 0 and w == img_rgb.shape[1] and h == img_rgb.shape[0]:
                continue
                
            emb = np.array(obj["embedding"], dtype=np.float32) if "embedding" in obj else None
            results.append({
                "bbox": (x, y, w, h),
                "embedding": emb,
                "confidence": 1.0,
            })
        return results
    except Exception as exc:
        logger.debug("DeepFace error: %s", exc)
        return []

# ─────────────────────────────────────────────────────────────────────────────
# POST-PROCESSING: Compute embeddings for detected boxes
# ─────────────────────────────────────────────────────────────────────────────

def _get_embedding_for_box(img_rgb: np.ndarray, x: int, y: int, w: int, h: int) -> Optional[np.ndarray]:
    """Attempt to get an embedding for a specific box using DeepFace."""
    try:
        from deepface import DeepFace
        import cv2
        # Crop with a slight margin
        margin = int(min(w, h) * 0.1)
        x1, y1 = max(0, x - margin), max(0, y - margin)
        x2, y2 = min(img_rgb.shape[1], x + w + margin), min(img_rgb.shape[0], y + h + margin)
        face_img = img_rgb[y1:y2, x1:x2]
        if face_img.size == 0:
            return None
        
        # Convert to BGR for DeepFace
        face_bgr = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
        
        objs = DeepFace.represent(
            img_path=face_bgr,
            model_name="Facenet512",
            detector_backend="skip", # We already detected it
            enforce_detection=False,
            align=True
        )
        if objs:
            return np.array(objs[0]["embedding"], dtype=np.float32)
    except Exception as exc:
        logger.debug("Failed to get embedding for box: %s", exc)
        if "facenet512_weights.h5" in str(exc):
            logger.warning("DeepFace embedding weights (Facenet512) are missing. Clustering won't work until they are downloaded.")
    return None


def _filter_faces(results: List[dict], img_w: int, img_h: int) -> List[dict]:
    """Filter out small false positives and low-confidence detections."""
    total_area = img_w * img_h
    filtered = []
    for f in results:
        x, y, w, h = f["bbox"]
        face_area = w * h
        
        # Filter by size
        if face_area < total_area * MIN_FACE_AREA_PCT:
            continue
            
        # Filter by confidence
        if f.get("confidence", 1.0) < MIN_CONFIDENCE:
            continue

        # Filter by aspect ratio (faces shouldn't be very elongated)
        aspect_ratio = max(w / h, h / w)
        if aspect_ratio > MAX_ASPECT_RATIO:
            continue
            
        filtered.append(f)
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# Mediapipe helpers
# ─────────────────────────────────────────────────────────────────────────────

def _detect_mediapipe(img_rgb: np.ndarray) -> List[dict]:
    # Try New Tasks API first
    try:
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core import base_options
        import mediapipe as mp
        
        # We need a model file for Tasks API. If not found, fall back to solutions.
        # Note: Face detection model usually needs to be downloaded.
        # To avoid complexity here, we'll try the old solutions API if it exists.
    except (ImportError, Exception):
        pass

    try:
        import mediapipe as mp
        from mediapipe.solutions import face_detection as mp_face_detection
    except ImportError:
        try:
            import mediapipe as mp
            from mediapipe.python.solutions import face_detection as mp_face_detection
        except ImportError:
            # logger.error("Mediapipe solutions not found. Please install mediapipe.")
            return []

    results = []
    try:
        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7) as fd:
            h_img, w_img, _ = img_rgb.shape
            detection_results = fd.process(img_rgb)
            if detection_results.detections:
                for detection in detection_results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * w_img)
                    y = int(bbox.ymin * h_img)
                    w = int(bbox.width * w_img)
                    h = int(bbox.height * h_img)
                    
                    emb = _get_embedding_for_box(img_rgb, x, y, w, h)
                    
                    results.append({
                        "bbox": (x, y, w, h),
                        "embedding": emb,
                        "confidence": detection.score[0],
                    })
    except Exception as exc:
        logger.debug("Mediapipe process failed: %s", exc)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# OpenCV (Haar Cascades) helper
# ─────────────────────────────────────────────────────────────────────────────

def _detect_opencv(img_rgb: np.ndarray) -> List[dict]:
    import cv2
    import os
    
    # Load built-in Haar Cascade (alt2 is generally more accurate with fewer false positives)
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
    if not os.path.exists(cascade_path):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if not os.path.exists(cascade_path):
            return []
        
    face_cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # scaleFactor=1.1 and minNeighbors=5 is a good balance. 
    # minSize=(30,30) allows detection of smaller faces in group photos.
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    results = []
    for (x, y, w, h) in faces:
        # Slightly expand the Haar box as it tends to be very tight on the face
        pad_w = int(w * 0.1)
        pad_h = int(h * 0.1)
        nx, ny = max(0, x - pad_w), max(0, y - pad_h)
        nw, nh = w + 2*pad_w, h + 2*pad_h
        
        emb = _get_embedding_for_box(img_rgb, int(nx), int(ny), int(nw), int(nh))
        results.append({
            "bbox": (int(nx), int(ny), int(nw), int(nh)),
            "embedding": emb,
            "confidence": 0.8, # Haar doesn't provide a score; we assume 0.8 for matched boxes
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def detect_faces(image_path: str | Path) -> List[dict]:
    """
    Detect faces in *image_path* and return a list of dicts with keys:
      - bbox       : (x, y, w, h)
      - embedding  : np.ndarray float32
      - confidence : float
    Returns [] if no face library is found or no faces detected.
    """
    try:
        img_orig = Image.open(str(image_path))
        img_orig = ImageOps.exif_transpose(img_orig) # Correct orientation
        img_orig = img_orig.convert("RGB")
        img_full = np.array(img_orig)
        
        # Calculate resize factor for fast detection
        w, h = img_orig.size
        scale = 1.0
        if max(w, h) > MAX_DETECTION_SIZE:
            scale = MAX_DETECTION_SIZE / max(w, h)
            img_detect = img_orig.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
            img_detect_np = np.array(img_detect)
        else:
            img_detect_np = img_full
            
    except Exception as exc:
        logger.error("Cannot open image %s: %s", image_path, exc)
        return []

    # Try backends in order of preference (Fastest first)
    all_backends = ["insightface", "face_recognition", "mediapipe", "deepface", "opencv"]
    
    for backend in all_backends:
        try:
            if backend == "insightface":
                import insightface # noqa
            elif backend == "face_recognition":
                import face_recognition # noqa
            elif backend == "deepface":
                from deepface import DeepFace # noqa
            elif backend == "mediapipe":
                import mediapipe # noqa
            elif backend == "opencv":
                import cv2 # noqa
            else:
                continue
        except (ImportError, Exception):
            continue

        try:
            results = []
            # We run detection on img_detect_np (potentially smaller)
            if backend == "insightface":
                results = _detect_insightface(img_detect_np)
            elif backend == "face_recognition":
                results = _detect_face_recognition(img_detect_np)
            elif backend == "deepface":
                results = _detect_deepface(img_detect_np)
            elif backend == "mediapipe":
                results = _detect_mediapipe(img_detect_np)
            elif backend == "opencv":
                results = _detect_opencv(img_detect_np)
            
            if results:
                # If we detected on a scaled image, scale coordinates back up
                if scale != 1.0:
                    for f in results:
                        x, y, w, h = f["bbox"]
                        f["bbox"] = (int(x / scale), int(y / scale), int(w / scale), int(h / scale))
                        # Note: embeddings were already computed on the scaled img in some helpers,
                        # but _get_embedding_for_box usually handles full img. 
                        # To be safe, let's re-run embedding on full image for high quality if they are missing.
                        if f["embedding"] is None:
                             f["embedding"] = _get_embedding_for_box(img_full, *f["bbox"])

                # Filter by size
                results = _filter_faces(results, img_full.shape[1], img_full.shape[0])
                
                if results:
                    logger.info("Found %d face(s) in %s using %s", len(results), image_path, backend)
                    return results
        except Exception as exc:
            logger.debug("%s backend failed for %s: %s", backend, image_path, exc)
            continue
            
    return []


def backend_name() -> str:
    """Return the active backend name, or 'none'."""
    return _detect_backend() or "none"


def get_face_thumbnail(image_path: str | Path, bbox: Tuple[int, int, int, int],
                       size: Tuple[int, int] = (128, 128)) -> Optional[bytes]:
    """Crop the face region and return JPEG bytes."""
    try:
        img = Image.open(str(image_path))
        img = ImageOps.exif_transpose(img) # Correct orientation
        img = img.convert("RGB")
        x, y, w, h = bbox
        
        # Increase margin to capture more of the head (standard for galleries)
        margin_w = int(w * 0.45)
        margin_h = int(h * 0.45)
        
        x1 = max(0, x - margin_w)
        y1 = max(0, y - margin_h)
        x2 = min(img.width, x + w + margin_w)
        y2 = min(img.height, y + h + margin_h)
        
        face_img = img.crop((x1, y1, x2, y2))
        face_img.thumbnail(size, Image.LANCZOS)
        buf = io.BytesIO()
        face_img.save(buf, format="JPEG", quality=85) # High quality
        return buf.getvalue()
    except Exception as exc:
        logger.debug("Face thumbnail error: %s", exc)
        return None
