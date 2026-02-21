"""
Application Core – high-level business logic bridging the DB and face engine.
"""

import logging
import os
import shutil
import zipfile
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from ..db.manager import DatabaseManager
from ..face_engine.clusterer import (
    cluster_embeddings, find_best_match, SIMILARITY_THRESHOLD
)
from ..utils.helpers import bytes_to_embedding

logger = logging.getLogger(__name__)


class AppCore:
    """Central business-logic facade used by both GUI and web layers."""

    def __init__(self, db: DatabaseManager):
        self.db = db

    # ─────────────────────────────────────────────────────────────────── #
    # Face clustering / person assignment                                  #
    # ─────────────────────────────────────────────────────────────────── #

    def cluster_unknown_faces(self, threshold: float = SIMILARITY_THRESHOLD) -> List[dict]:
        """
        Cluster all unassigned faces and return groups with person suggestions.
        Each group contains face_ids and an optional suggestion (person_id).
        """
        rows = self.db.get_unassigned_faces()
        if not rows:
            return []

        face_ids = []
        embeddings = []
        standalone_face_ids = []

        for row in rows:
            if row["embedding"]:
                face_ids.append(row["face_id"])
                embeddings.append(bytes_to_embedding(row["embedding"]))
            else:
                standalone_face_ids.append(row["face_id"])

        results = []
        known_embs = self._get_all_known_embeddings() if embeddings else []

        if embeddings:
            raw_clusters = cluster_embeddings(embeddings, threshold=threshold)
            for cluster_indices in raw_clusters:
                c_face_ids = [face_ids[idx] for idx in cluster_indices]
                c_embs = [embeddings[idx] for idx in cluster_indices]
                
                # Suggest a person for the whole cluster
                suggestion = None
                if known_embs:
                    # Match the cluster's centroid against known individuals
                    centroid = np.mean(c_embs, axis=0)
                    norm = np.linalg.norm(centroid)
                    if norm > 0: centroid /= norm
                    suggestion = find_best_match(centroid, known_embs, threshold=threshold)
                
                results.append({
                    "face_ids": c_face_ids,
                    "suggestion": suggestion
                })
        
        # Standalone faces (no embeddings) get no suggestions
        for fid in standalone_face_ids:
            results.append({
                "face_ids": [fid],
                "suggestion": None
            })

        return results

    def auto_assign_face(self, face_id: int, threshold: float = SIMILARITY_THRESHOLD) -> Optional[int]:
        """
        Try to match *face_id* against known persons.
        Returns person_id if matched, None otherwise.
        """
        face_row = self.db.get_face(face_id)
        if not face_row or not face_row["embedding"]:
            return None

        query_emb = bytes_to_embedding(face_row["embedding"])
        known = []
        for person in self.db.get_all_persons():
            p_faces = self.db.get_faces_for_person(person["person_id"])
            for pf in p_faces:
                if pf["embedding"]:
                    emb = bytes_to_embedding(pf["embedding"])
                    known.append((person["person_id"], emb))

        return find_best_match(query_emb, known, threshold=threshold)

    def assign_faces_to_person(self, face_ids: List[int], person_id: int) -> int:
        """Assign a list of face_ids to an existing person, unmapping previous assignments."""
        for fid in face_ids:
            self.db.unmap_face(fid) # Remove existing mapping
            self.db.map_face_to_person(person_id, fid) # Add new mapping
        
        # Update person's representative face to the first one in the list (newly assigned)
        if face_ids:
            self.db.update_person(person_id, profile_face_id=face_ids[0])
            
        # Re-index: find other faces that match this person
        # self.auto_match_all_unassigned() # Removed as it aggressively matches wrongly in the background
        return 0

    def create_person_from_faces(self, face_ids: List[int], name: str,
                                 notes: str = None) -> Tuple[int, int]:
        """Create a new person and assign faces. Returns (person_id, 0)."""
        profile_face_id = face_ids[0] if face_ids else None
        person_id = self.db.create_person(name=name, notes=notes,
                                          profile_face_id=profile_face_id)
        for fid in face_ids:
            self.db.unmap_face(fid)
            self.db.map_face_to_person(person_id, fid)
            
        # Re-index: find other faces that match this new person
        # self.auto_match_all_unassigned() # Removed as it aggressively matches wrongly in the background
        return person_id, 0

    def _get_known_centroids(self) -> List[Tuple[int, np.ndarray]]:
        """
        Calculate a single representative embedding (centroid) for each person.
        This is much faster for matching than comparing against every face.
        """
        centroids = []
        for person in self.db.get_all_persons():
            p_faces = self.db.get_faces_for_person(person["person_id"])
            embs = []
            for pf in p_faces:
                if pf["embedding"]:
                    embs.append(bytes_to_embedding(pf["embedding"]))
            
            if embs:
                # Average all embeddings (centroids are more robust than single points)
                avg_emb = np.mean(embs, axis=0)
                # Re-normalize
                norm = np.linalg.norm(avg_emb)
                if norm > 0:
                    avg_emb = avg_emb / norm
                centroids.append((person["person_id"], avg_emb))
        return centroids

    def _get_all_known_embeddings(self) -> List[Tuple[int, np.ndarray]]:
        """Retrieve ALL embeddings for ALL known persons."""
        known = []
        for person in self.db.get_all_persons():
            p_faces = self.db.get_faces_for_person(person["person_id"])
            for pf in p_faces:
                if pf["embedding"]:
                    known.append((person["person_id"], bytes_to_embedding(pf["embedding"])))
        return known

    def auto_match_all_unassigned(self, threshold: float = SIMILARITY_THRESHOLD) -> int:
        """
        Try to auto-match all unassigned faces to existing persons.
        Uses all known embeddings for higher accuracy than simple centroids.
        """
        unassigned = self.db.get_unassigned_faces()
        if not unassigned:
            return 0
        
        # Using ALL known embeddings is more robust to different angles/lighting than centroids
        known = self._get_all_known_embeddings()
        if not known:
            return 0
            
        matches = 0
        for face in unassigned:
            if not face["embedding"]:
                continue
            emb = bytes_to_embedding(face["embedding"])
            pid = find_best_match(emb, known, threshold=threshold)
            if pid:
                self.db.map_face_to_person(pid, face["face_id"])
                matches += 1
                
        if matches > 0:
            logger.info("Auto-assigned %d faces to known persons", matches)
        return matches

    def remove_false_positive(self, face_ids: List[int]):
        """Remove face detections that are not actually faces."""
        for fid in face_ids:
            self.db.delete_face(fid)

    # ─────────────────────────────────────────────────────────────────── #
    # Photo browsing / filtering                                           #
    # ─────────────────────────────────────────────────────────────────── #

    def get_photos(self, person_ids: List[int] = None, use_union: bool = False, 
                   only_groups: bool = False, only_solos: bool = False) -> list:
        """
        Return photos optionally filtered by person IDs.
        If person_ids is None or empty, return all photos.
        If use_union is True, return photos containing ANY of the specified persons.
        If only_groups is True, return only photos with >1 person.
        If only_solos is True, return only photos with exactly 1 person.
        """
        # Start with the set of all photo IDs or filtered photo IDs
        if person_ids:
            sets = [
                {r["photo_id"] for r in self.db.get_photos_for_person(pid)}
                for pid in person_ids
            ]
            if not sets:
                target_ids = set()
            elif use_union:
                target_ids = set().union(*sets)
            else:
                target_ids = sets[0].intersection(*sets[1:])
        else:
            all_photos = self.db.get_all_photos()
            target_ids = {p["photo_id"] for p in all_photos}

        # Apply group filter
        if only_groups:
            group_ids = set(self.db.get_group_photo_ids())
            target_ids = target_ids.intersection(group_ids)
        elif only_solos:
            solo_ids = set(self.db.get_solo_photo_ids())
            target_ids = target_ids.intersection(solo_ids)

        if not target_ids:
            return []

        # Maintain original order (Photos are sorted by date in get_all_photos)
        all_ordered = self.db.get_all_photos()
        return [p for p in all_ordered if p["photo_id"] in target_ids]

    def get_persons(self, allowed_ids: List[int] = None) -> list:
        """Return persons, optionally filtered by a list of allowed IDs."""
        all_p = self.db.get_all_persons()
        if allowed_ids is None:
            return all_p
        return [p for p in all_p if p["person_id"] in allowed_ids]

    def get_person_face_thumbnail(self, person_id: int) -> Optional[bytes]:
        """Return the profile face thumbnail blob for a person."""
        person = self.db.get_person(person_id)
        if not person:
            return None
        face_id = person["profile_face_id"]
        if not face_id:
            # Fall back to first mapped face
            faces = self.db.get_faces_for_person(person_id)
            if faces:
                face_id = faces[0]["face_id"]
        if not face_id:
            return None
        face = self.db.get_face(face_id)
        return face["face_thumb"] if face else None

    # ─────────────────────────────────────────────────────────────────── #
    # Export                                                               #
    # ─────────────────────────────────────────────────────────────────── #

    def export_photos_to_folder(self, photo_paths: List[str],
                                dest_dir: str) -> int:
        """Copy photos to *dest_dir*. Returns number of files copied."""
        os.makedirs(dest_dir, exist_ok=True)
        count = 0
        for src in photo_paths:
            if os.path.isfile(src):
                dst = os.path.join(dest_dir, os.path.basename(src))
                # Avoid overwriting by adding index suffix
                base, ext = os.path.splitext(dst)
                idx = 1
                while os.path.exists(dst):
                    dst = f"{base}_{idx}{ext}"
                    idx += 1
                shutil.copy2(src, dst)
                count += 1
        return count

    def export_photos_to_zip(self, photo_paths: List[str]) -> str:
        """Create a temp ZIP file and return its path."""
        tmp = tempfile.NamedTemporaryFile(
            suffix=".zip", prefix="facegallery_export_", delete=False)
        tmp.close()
        with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as zf:
            for src in photo_paths:
                if os.path.isfile(src):
                    zf.write(src, os.path.basename(src))
        return tmp.name

    # ─────────────────────────────────────────────────────────────────── #
    # Settings helpers                                                     #
    # ─────────────────────────────────────────────────────────────────── #

    def get_web_port(self) -> int:
        return int(self.db.get_setting("web_port") or "5050")

    def set_web_port(self, port: int):
        self.db.set_setting("web_port", str(port))

    def get_web_bind_all(self) -> bool:
        return self.db.get_setting("web_bind_all", "1") == "1"

    def set_web_bind_all(self, value: bool):
        self.db.set_setting("web_bind_all", "1" if value else "0")

    def get_theme(self) -> str:
        return self.db.get_setting("theme", "dark")

    def set_theme(self, theme: str):
        self.db.set_setting("theme", theme)

    def reset_project(self):
        """Clear all data. The default admin user will be recreated by main.py if missing."""
        self.db.clear_all_data(keep_users=True)
