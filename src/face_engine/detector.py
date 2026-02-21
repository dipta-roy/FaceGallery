"""
Face recognition engine – detects faces and computes embeddings.

Supports three backends (tried in order of preference):
  1. insightface  (best accuracy, recommended)
  2. face_recognition / dlib  (classic, widely available)
  3. DeepFace  (fallback, uses retinaface detector)
  4. Mediapipe  (lightweight fallback)
  5. OpenCV Haar Cascade  (last resort)

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

# Detection image size – insightface gets a bigger canvas so small group faces are seen
MAX_DETECTION_SIZE = 1920

# Minimum face area as a fraction of image area.
# 0.03% is permissive enough for small faces in group photos.
MIN_FACE_AREA_PCT = 0.0003

# Per-backend confidence thresholds. Insightface scores are on [0,1] and very
# reliable. Haar/OpenCV scores are hardcoded 0.8 so we skip confidence filtering
# for that backend and rely on minNeighbors / aspect ratio instead.
_CONF_THRESHOLD = {
    "insightface":    0.50,   # very reliable; 0.5 already means "a face was found"
    "face_recognition": 0.0,  # no confidence returned; always pass
    "deepface":       0.0,    # retinaface already filters internally
    "mediapipe":      0.75,   # mediapipe confidence is a soft probability
    "opencv":         0.0,    # hardcoded 0.8 fake value; filter by other means
}

MAX_ASPECT_RATIO = 2.5       # Faces ≤ 2.5:1 (wider tolerance for tilted heads)


def _detect_backend() -> Optional[str]:
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND

    for name, probe in [
        ("insightface",    lambda: __import__("insightface")),
        ("face_recognition", lambda: __import__("face_recognition")),
        ("deepface",       lambda: __import__("deepface")),
        ("mediapipe",      lambda: __import__("mediapipe")),
        ("opencv",         lambda: __import__("cv2")),
    ]:
        try:
            probe()
            _BACKEND = name
            logger.info("Face engine backend: %s", name)
            return _BACKEND
        except (ImportError, Exception):
            continue

    _BACKEND = None
    return None


# ─────────────────────────────────────────────────────────────────────────────
# insightface
# ─────────────────────────────────────────────────────────────────────────────

_insight_app = None


def _get_insight_app():
    global _insight_app
    if _insight_app is None:
        import insightface
        _insight_app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        # Use a larger det_size so small faces in high-res images are found
        _insight_app.prepare(ctx_id=0, det_size=(640, 640))
    return _insight_app


def _detect_insightface(img_rgb: np.ndarray) -> List[dict]:
    app = _get_insight_app()
    h, w = img_rgb.shape[:2]

    # Run at native size first
    faces = app.get(img_rgb)

    # If nothing found and image is large, try a downscaled version as well
    # (insightface det_size=(640,640) clips detections for giant images)
    if not faces and max(w, h) > 1280:
        scale = 1280 / max(w, h)
        from PIL import Image as _PIL
        small = np.array(
            _PIL.fromarray(img_rgb).resize((int(w * scale), int(h * scale)), _PIL.BILINEAR)
        )
        faces_small = app.get(small)
        # Scale boxes back up
        for f in faces_small:
            f.bbox = f.bbox / scale
        faces = faces_small if faces_small else faces

    results = []
    for face in faces:
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        # Clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        fw, fh = x2 - x1, y2 - y1
        if fw <= 0 or fh <= 0:
            continue
        emb = face.embedding.astype(np.float32) if face.embedding is not None else None
        conf = float(face.det_score) if hasattr(face, "det_score") else 1.0
        results.append({"bbox": (x1, y1, fw, fh), "embedding": emb, "confidence": conf})
    return results


# ─────────────────────────────────────────────────────────────────────────────
# face_recognition (dlib)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_face_recognition(img_rgb: np.ndarray) -> List[dict]:
    import face_recognition as fr
    # CNN model is significantly more accurate than HOG, especially for:
    #   - faces at angles, partially occluded, smaller faces in groups
    # It is slower but that is acceptable for a background scanner.
    try:
        locations = fr.face_locations(img_rgb, model="cnn")
    except Exception:
        locations = fr.face_locations(img_rgb, model="hog")

    encodings = fr.face_encodings(img_rgb, locations, num_jitters=2)
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
# DeepFace (retinaface detector – best at rejecting clothes / objects)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_deepface(img_rgb: np.ndarray) -> List[dict]:
    from deepface import DeepFace
    import cv2
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Prefer retinaface; fall back to mtcnn if unavailable
    for det_backend in ("retinaface", "mtcnn", "opencv"):
        try:
            objs = DeepFace.represent(
                img_path=img_bgr,
                model_name="Facenet512",
                detector_backend=det_backend,
                enforce_detection=False,
                align=True,
            )
            break
        except Exception as exc:
            logger.debug("DeepFace %s failed: %s", det_backend, exc)
            objs = []

    results = []
    img_h, img_w = img_rgb.shape[:2]
    for obj in objs:
        region = obj.get("facial_area", {})
        x = region.get("x", 0)
        y = region.get("y", 0)
        w = region.get("w", 0)
        h = region.get("h", 0)

        # Ignore the "whole image is a face" fallback DeepFace returns
        if x == 0 and y == 0 and abs(w - img_w) < 5 and abs(h - img_h) < 5:
            continue
        if w <= 0 or h <= 0:
            continue

        # Confidence from facial_area if available (retinaface provides it)
        conf = float(region.get("confidence") or obj.get("face_confidence") or 1.0)

        emb = np.array(obj["embedding"], dtype=np.float32) if obj.get("embedding") else None
        results.append({"bbox": (x, y, w, h), "embedding": emb, "confidence": conf})
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Mediapipe
# ─────────────────────────────────────────────────────────────────────────────

def _detect_mediapipe(img_rgb: np.ndarray) -> List[dict]:
    try:
        from mediapipe.solutions import face_detection as mp_fd
    except ImportError:
        try:
            from mediapipe.python.solutions import face_detection as mp_fd
        except ImportError:
            return []

    results = []
    try:
        # model_selection=1 targets faces up to 5 m away (full-body shots)
        with mp_fd.FaceDetection(model_selection=1, min_detection_confidence=0.6) as fd:
            h_img, w_img = img_rgb.shape[:2]
            det = fd.process(img_rgb)
            if det.detections:
                for detection in det.detections:
                    bb = detection.location_data.relative_bounding_box
                    x = int(bb.xmin * w_img)
                    y = int(bb.ymin * h_img)
                    w = int(bb.width * w_img)
                    h = int(bb.height * h_img)
                    x, y = max(0, x), max(0, y)
                    w = min(w, w_img - x)
                    h = min(h, h_img - y)
                    if w <= 0 or h <= 0:
                        continue
                    emb = _get_embedding_for_box(img_rgb, x, y, w, h)
                    results.append({
                        "bbox": (x, y, w, h),
                        "embedding": emb,
                        "confidence": float(detection.score[0]),
                    })
    except Exception as exc:
        logger.debug("Mediapipe failed: %s", exc)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# OpenCV Haar Cascade (last resort)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_opencv(img_rgb: np.ndarray) -> List[dict]:
    import cv2, os

    # alt2 has better recall than default; LBP is faster but noisier
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
    if not os.path.exists(cascade_path):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if not os.path.exists(cascade_path):
            return []

    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Equalise histogram to handle poor lighting
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)

    # Two-pass detection: standard params + a tighter pass to reduce noise
    # minNeighbors=6 is a good middle ground
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=6, minSize=(40, 40),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    results = []
    h_img, w_img = img_rgb.shape[:2]
    for (x, y, w, h) in faces:
        # ── Landmark-based false-positive rejection ──────────────────────
        # Extract the candidate crop and check if it has eye-like features.
        # This rejects shirts, bags, and textured backgrounds.
        eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
        if os.path.exists(eye_cascade_path):
            eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
            roi = gray[y:y+h, x:x+w]
            upper_roi = roi[:h//2, :]   # eyes are in the upper half
            eyes = eye_cascade.detectMultiScale(upper_roi, scaleFactor=1.1, minNeighbors=3)
            if len(eyes) == 0:
                logger.debug("Rejecting Haar face box %s:%s – no eyes found", (x, y), (w, h))
                continue   # Not a face – skip

        # Pad box slightly
        pad_w = int(w * 0.08)
        pad_h = int(h * 0.08)
        nx, ny = max(0, x - pad_w), max(0, y - pad_h)
        nw = min(w_img - nx, w + 2 * pad_w)
        nh = min(h_img - ny, h + 2 * pad_h)

        emb = _get_embedding_for_box(img_rgb, nx, ny, nw, nh)
        results.append({
            "bbox": (nx, ny, nw, nh),
            "embedding": emb,
            "confidence": 0.85,
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Embedding helper (for backends that only detect boxes, not embeddings)
# ─────────────────────────────────────────────────────────────────────────────

def _get_embedding_for_box(img_rgb: np.ndarray, x: int, y: int, w: int, h: int) -> Optional[np.ndarray]:
    """Compute a Facenet512 embedding for a detected bounding box crop."""
    try:
        from deepface import DeepFace
        import cv2
        # Generous margin for forehead / chin
        margin = int(min(w, h) * 0.15)
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(img_rgb.shape[1], x + w + margin)
        y2 = min(img_rgb.shape[0], y + h + margin)
        face_img = img_rgb[y1:y2, x1:x2]
        if face_img.size == 0:
            return None
        face_bgr = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
        objs = DeepFace.represent(
            img_path=face_bgr,
            model_name="Facenet512",
            detector_backend="skip",
            enforce_detection=False,
            align=True,
        )
        if objs:
            return np.array(objs[0]["embedding"], dtype=np.float32)
    except Exception as exc:
        if "facenet512_weights" in str(exc).lower():
            logger.warning("DeepFace Facenet512 weights missing – embeddings won't work.")
        else:
            logger.debug("Embedding extraction failed: %s", exc)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing filter
# ─────────────────────────────────────────────────────────────────────────────

def _filter_faces(results: List[dict], img_w: int, img_h: int, backend: str) -> List[dict]:
    """Filter detections by size, confidence, and aspect ratio."""
    total_area = img_w * img_h
    conf_threshold = _CONF_THRESHOLD.get(backend, 0.5)
    filtered = []
    for f in results:
        x, y, w, h = f["bbox"]
        if w <= 0 or h <= 0:
            continue

        # Size filter
        face_area = w * h
        if face_area < total_area * MIN_FACE_AREA_PCT:
            continue

        # Confidence filter (only when the backend actually provides a score)
        if conf_threshold > 0 and f.get("confidence", 1.0) < conf_threshold:
            continue

        # Aspect ratio – reject very elongated detections
        aspect_ratio = max(w / h, h / w)
        if aspect_ratio > MAX_ASPECT_RATIO:
            continue

        filtered.append(f)
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def detect_faces(image_path: str | Path) -> List[dict]:
    """
    Detect faces in *image_path* and return a list of dicts:
      - bbox       : (x, y, w, h) in full-resolution pixel coords
      - embedding  : np.ndarray float32 (None if unavailable)
      - confidence : float
    Returns [] if no library is available or no faces found.
    """
    try:
        img_orig = Image.open(str(image_path))
        img_orig = ImageOps.exif_transpose(img_orig)
        img_orig = img_orig.convert("RGB")
        img_full = np.array(img_orig)
        orig_w, orig_h = img_orig.size
    except Exception as exc:
        logger.error("Cannot open image %s: %s", image_path, exc)
        return []

    # Resize for detection if too large (keeps processing fast)
    scale = 1.0
    if max(orig_w, orig_h) > MAX_DETECTION_SIZE:
        scale = MAX_DETECTION_SIZE / max(orig_w, orig_h)
        img_detect = img_orig.resize(
            (int(orig_w * scale), int(orig_h * scale)), Image.LANCZOS
        )
        img_detect_np = np.array(img_detect)
    else:
        img_detect_np = img_full

    all_backends = [
        ("insightface",     lambda: __import__("insightface"),            img_detect_np),
        ("face_recognition", lambda: __import__("face_recognition"),      img_detect_np),
        ("deepface",        lambda: __import__("deepface"),               img_detect_np),
        ("mediapipe",       lambda: __import__("mediapipe"),              img_detect_np),
        ("opencv",          lambda: __import__("cv2"),                    img_detect_np),
    ]

    for backend, probe, detect_img in all_backends:
        try:
            probe()
        except (ImportError, Exception):
            continue

        try:
            if backend == "insightface":
                results = _detect_insightface(detect_img)
            elif backend == "face_recognition":
                results = _detect_face_recognition(detect_img)
            elif backend == "deepface":
                results = _detect_deepface(detect_img)
            elif backend == "mediapipe":
                results = _detect_mediapipe(detect_img)
            elif backend == "opencv":
                results = _detect_opencv(detect_img)
            else:
                continue

            if not results:
                continue

            # Scale bounding boxes back to full-resolution coordinates
            if scale != 1.0:
                for f in results:
                    bx, by, bw, bh = f["bbox"]
                    f["bbox"] = (
                        int(bx / scale), int(by / scale),
                        int(bw / scale), int(bh / scale),
                    )

            # For backends that don't produce embeddings, extract them from full-res crop
            if backend in ("mediapipe", "opencv"):
                for f in results:
                    if f.get("embedding") is None:
                        f["embedding"] = _get_embedding_for_box(img_full, *f["bbox"])

            # For insightface with a scaled image, re-extract embeddings from full-res
            # for higher quality (insightface already provided them but at lower res)
            if backend == "insightface" and scale != 1.0:
                for f in results:
                    if f.get("embedding") is None:
                        f["embedding"] = _get_embedding_for_box(img_full, *f["bbox"])

            # Filter noise / false positives
            results = _filter_faces(results, orig_w, orig_h, backend)

            if results:
                logger.info("Found %d face(s) in %s using %s",
                            len(results), Path(image_path).name, backend)
                return results

        except Exception as exc:
            logger.debug("%s backend failed for %s: %s", backend, image_path, exc)
            continue

    return []


def backend_name() -> str:
    """Return the active backend name, or 'none'."""
    return _detect_backend() or "none"


def get_face_thumbnail(image_path: str | Path, bbox: Tuple[int, int, int, int],
                       size: Tuple[int, int] = (160, 160)) -> Optional[bytes]:
    """Crop the face region with generous margin and return high-quality JPEG bytes."""
    try:
        img = Image.open(str(image_path))
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        x, y, w, h = bbox

        # 50% margin on each side so we capture forehead, chin, and ears
        margin_w = int(w * 0.50)
        margin_h = int(h * 0.50)

        x1 = max(0, x - margin_w)
        y1 = max(0, y - margin_h)
        x2 = min(img.width,  x + w + margin_w)
        y2 = min(img.height, y + h + margin_h)

        face_img = img.crop((x1, y1, x2, y2))
        face_img.thumbnail(size, Image.LANCZOS)
        buf = io.BytesIO()
        face_img.save(buf, format="JPEG", quality=92)
        return buf.getvalue()
    except Exception as exc:
        logger.debug("Face thumbnail error: %s", exc)
        return None
