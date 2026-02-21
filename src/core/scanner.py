"""
Photo scanner – walks folders, indexes photos into the database,
and dispatches face detection.

Designed to run in a background thread and report progress via callbacks.
"""

import logging
import os
import time
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from ..db.manager import DatabaseManager
from ..face_engine.detector import detect_faces, get_face_thumbnail
from ..utils.helpers import (
    walk_images, compute_file_hash, make_thumbnail, get_image_info,
    embedding_to_bytes
)

logger = logging.getLogger(__name__)


class PhotoScanner:
    """
    Scans one or more directories, indexes photos, and detects faces.

    Constructor params
    ------------------
    db : DatabaseManager
    progress_cb  : called with (current, total, message) after each photo
    cancelled_cb : should return True if the user wants to cancel
    """

    def __init__(
        self,
        db: DatabaseManager,
        progress_cb: Callable[[int, int, str], None] = None,
        face_found_cb: Callable[[bytes], None] = None,
        cancelled_cb: Callable[[], bool] = None,
    ):
        self.db = db
        self._progress_cb = progress_cb or (lambda c, t, m: None)
        self._face_found_cb = face_found_cb or (lambda b: None)
        self._cancelled_cb = cancelled_cb or (lambda: False)

    # ------------------------------------------------------------------ #
    # Public                                                               #
    # ------------------------------------------------------------------ #

    def scan(self, folders: List[str], force_rescan: bool = False) -> dict:
        """
        Scan *folders* for images, index them, detect faces.
        If force_rescan is True, re-detect faces even for already-indexed photos.

        Returns a summary dict.
        """
        all_images = []
        for folder in folders:
            all_images.extend(walk_images(folder))

        total = len(all_images)
        logger.info("Scanner: found %d images in %d folders", total, len(folders))

        added = 0
        skipped = 0
        faces_found = 0
        auto_assigned = 0

        # Load known faces once for fast auto-matching during scan
        known_faces = self._load_known_faces()

        for idx, img_path in enumerate(all_images, start=1):
            if self._cancelled_cb():
                logger.info("Scanner cancelled by user at %d/%d", idx, total)
                break

            self._progress_cb(idx, total, f"Indexing {os.path.basename(img_path)}")

            try:
                photo_id = self._index_photo(img_path)
                
                # If photo is new/changed, OR if it's already indexed but has 0 faces,
                # we run (or re-run) face detection.
                should_scan_faces = False
                target_id = photo_id
                
                if photo_id:
                    should_scan_faces = True
                else:
                    # Check if already in DB but has no faces
                    existing = self.db.get_photo_by_path(img_path)
                    if existing:
                        target_id = existing["photo_id"]
                        faces_in_db = self.db.get_faces_for_photo(target_id)
                        if not faces_in_db:
                            should_scan_faces = True
                
                if (should_scan_faces or force_rescan) and target_id:
                    faces, matched = self._index_faces(img_path, target_id, known_faces)
                    faces_found += faces
                    auto_assigned += matched
                    added += 1
                else:
                    skipped += 1
            except Exception as exc:
                logger.error("Error processing %s: %s", img_path, exc)
                skipped += 1

        for folder in folders:
            self.db.mark_folder_scanned(folder)

        summary = {
            "total": total,
            "added": added,
            "skipped": skipped,
            "faces": faces_found,
            "matched": auto_assigned,
        }
        logger.info("Scan complete: %s", summary)
        return summary

    # ------------------------------------------------------------------ #
    # Private                                                              #
    # ------------------------------------------------------------------ #

    def _load_known_faces(self) -> List[tuple]:
        """Load ALL embeddings for ALL named people for maximum matching accuracy."""
        from ..utils.helpers import bytes_to_embedding
        
        known = []
        persons = self.db.get_all_persons()
        for p in persons:
            faces = self.db.get_faces_for_person(p["person_id"])
            for f in faces:
                if f["embedding"]:
                    emb = bytes_to_embedding(f["embedding"])
                    known.append((p["person_id"], emb))
        return known

    def _find_match(self, face_emb: np.ndarray, known_faces: List[tuple]) -> Optional[int]:
        """Simple cosine similarity check against known faces."""
        if not known_faces or face_emb is None:
            return None
        
        from ..face_engine.clusterer import find_best_match
        
        # known_faces is already list of (id, np_array)
        return find_best_match(face_emb, known_faces)

    def _index_photo(self, img_path: str) -> Optional[int]:
        """Insert/update a photo record; return photo_id."""
        file_size = os.path.getsize(img_path)
        phash = compute_file_hash(img_path)

        # Check if already indexed (by path + hash)
        existing = self.db.get_photo_by_path(img_path)
        if existing and existing["phash"] == phash:
            return None  # unchanged, skip face re-detection

        w, h, date_taken, orientation = get_image_info(img_path)
        thumbnail = make_thumbnail(img_path)

        photo_id = self.db.upsert_photo(
            path=img_path,
            phash=phash,
            file_size=file_size,
            width=w,
            height=h,
            orientation=orientation,
            date_taken=date_taken,
            thumbnail=thumbnail,
        )
        return photo_id

    def _index_faces(self, img_path: str, photo_id: int, known_faces: List[tuple]) -> tuple[int, int]:
        """Detect faces in *img_path* and store them in the database."""
        # Remove old face records for this photo before re-detecting
        self.db.delete_faces_for_photo(photo_id)

        faces = detect_faces(img_path)
        count = 0
        matched_count = 0
        for face in faces:
            bbox = face["bbox"]
            emb_bytes = embedding_to_bytes(face["embedding"]) if face["embedding"] is not None else None
            face_thumb = get_face_thumbnail(img_path, bbox)
            if face_thumb:
                self._face_found_cb(face_thumb)
            
            face_id = self.db.insert_face(
                photo_id=photo_id,
                bbox_x=bbox[0],
                bbox_y=bbox[1],
                bbox_w=bbox[2],
                bbox_h=bbox[3],
                embedding=emb_bytes,
                confidence=face["confidence"],
                face_thumb=face_thumb,
            )
            
            # Auto-match if possible
            person_id = self._find_match(face["embedding"], known_faces)
            if person_id:
                self.db.map_face_to_person(person_id, face_id)
                matched_count += 1

            count += 1
        return count, matched_count
