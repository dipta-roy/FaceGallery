"""
Database manager – thin wrapper around sqlite3 providing connection management,
table initialisation, and common CRUD helpers.
"""

import os
import sqlite3
import logging
import threading
from pathlib import Path
from typing import Any, Iterator, List, Optional, Tuple

from .schema import ALL_TABLES, INDEXES

logger = logging.getLogger(__name__)

_thread_local = threading.local()


class DatabaseManager:
    """Thread-safe SQLite connection manager."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._local = threading.local()
        self._init_db()

    # ------------------------------------------------------------------ #
    # Connection helpers                                                   #
    # ------------------------------------------------------------------ #

    def _get_conn(self) -> sqlite3.Connection:
        """Return a per-thread connection, creating one if needed."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            self._local.conn = conn
        return self._local.conn

    @property
    def conn(self) -> sqlite3.Connection:
        return self._get_conn()

    def close(self):
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    # ------------------------------------------------------------------ #
    # Schema initialisation                                                #
    # ------------------------------------------------------------------ #

    def _init_db(self):
        with self.conn:
            for sql in ALL_TABLES:
                self.conn.execute(sql)
            for idx in INDEXES:
                self.conn.execute(idx)
            self._migrate_db()
        logger.info("Database initialised at %s", self.db_path)

    def _migrate_db(self):
        """Add missing columns to existing tables."""
        # Add can_upload to users
        try:
            self.conn.execute("ALTER TABLE users ADD COLUMN can_upload INTEGER NOT NULL DEFAULT 0")
        except sqlite3.OperationalError: pass
        
        # Add created_by to scan_folders
        try:
            self.conn.execute("ALTER TABLE scan_folders ADD COLUMN created_by INTEGER REFERENCES users(user_id) ON DELETE SET NULL")
        except sqlite3.OperationalError: pass

        # Convert relative paths to absolute for consistency
        try:
            fixes = 0
            folders = self.fetchall("SELECT folder_id, path FROM scan_folders")
            for f in folders:
                old_path = f["path"]
                if not old_path: continue
                
                new_path = old_path
                if not os.path.isabs(old_path):
                    new_path = os.path.abspath(old_path)
                
                if not os.path.isdir(new_path):
                    # Try stripping common incorrect mid-paths
                    for bad_segment in ["src\\web\\", "src/web/", "src\\web", "src/web"]:
                        if bad_segment in new_path:
                            repaired = os.path.normpath(new_path.replace(bad_segment, ""))
                            if os.path.isdir(repaired):
                                new_path = repaired
                                break
                
                if new_path != old_path:
                    self.conn.execute("UPDATE scan_folders SET path=? WHERE folder_id=?", 
                                     (new_path, f["folder_id"]))
                    fixes += 1
            
            photos = self.fetchall("SELECT photo_id, path FROM photos")
            for p in photos:
                old_path = p["path"]
                if not old_path: continue
                
                new_path = old_path
                if not os.path.isabs(old_path):
                    new_path = os.path.abspath(old_path)
                
                if not os.path.isfile(new_path):
                    for bad_segment in ["src\\web\\", "src/web/", "src\\web", "src/web"]:
                        if bad_segment in new_path:
                            repaired = os.path.normpath(new_path.replace(bad_segment, ""))
                            if os.path.isfile(repaired):
                                new_path = repaired
                                break
                
                if new_path != old_path:
                    self.conn.execute("UPDATE photos SET path=? WHERE photo_id=?", 
                                     (new_path, p["photo_id"]))
                    fixes += 1
            
            if fixes > 0:
                logger.info("Path migration: repaired %d database entries.", fixes)
        except Exception as e:
            logger.warning("Path migration failed: %s", e)

    # ------------------------------------------------------------------ #
    # Generic query helpers                                                #
    # ------------------------------------------------------------------ #

    def execute(self, sql: str, params: Tuple = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, params)

    def executemany(self, sql: str, params_seq) -> sqlite3.Cursor:
        return self.conn.executemany(sql, params_seq)

    def fetchone(self, sql: str, params: Tuple = ()) -> Optional[sqlite3.Row]:
        return self.conn.execute(sql, params).fetchone()

    def fetchall(self, sql: str, params: Tuple = ()) -> List[sqlite3.Row]:
        return self.conn.execute(sql, params).fetchall()

    def commit(self):
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # Photos                                                               #
    # ------------------------------------------------------------------ #

    def upsert_photo(self, path: str, phash: str = None, file_size: int = None,
                     width: int = None, height: int = None,
                     orientation: int = 1, date_taken: str = None,
                     thumbnail: bytes = None) -> int:
        with self.conn:
            cur = self.conn.execute(
                """INSERT INTO photos (path, phash, file_size, width, height,
                       orientation, date_taken, thumbnail)
                   VALUES (?,?,?,?,?,?,?,?)
                   ON CONFLICT(path) DO UPDATE SET
                       phash=excluded.phash,
                       file_size=excluded.file_size,
                       width=excluded.width,
                       height=excluded.height,
                       orientation=excluded.orientation,
                       date_taken=excluded.date_taken,
                       thumbnail=excluded.thumbnail""",
                (path, phash, file_size, width, height, orientation, date_taken, thumbnail)
            )
            return cur.lastrowid or self.fetchone(
                "SELECT photo_id FROM photos WHERE path=?", (path,))["photo_id"]

    def get_photo(self, photo_id: int) -> Optional[sqlite3.Row]:
        return self.fetchone("SELECT * FROM photos WHERE photo_id=?", (photo_id,))

    def get_photo_by_path(self, path: str) -> Optional[sqlite3.Row]:
        return self.fetchone("SELECT * FROM photos WHERE path=?", (path,))

    def get_all_photos(self) -> List[sqlite3.Row]:
        return self.fetchall("SELECT * FROM photos ORDER BY date_taken DESC, date_added DESC")

    def get_photos_for_person(self, person_id: int) -> List[sqlite3.Row]:
        return self.fetchall(
            """SELECT DISTINCT p.* FROM photos p
               JOIN faces f ON f.photo_id = p.photo_id
               JOIN person_face_mappings pfm ON pfm.face_id = f.face_id
               WHERE pfm.person_id = ?
               ORDER BY p.date_taken DESC""",
            (person_id,)
        )

    def delete_photo(self, photo_id: int):
        with self.conn:
            self.conn.execute("DELETE FROM photos WHERE photo_id=?", (photo_id,))

    # ------------------------------------------------------------------ #
    # Faces                                                                #
    # ------------------------------------------------------------------ #

    def insert_face(self, photo_id: int, bbox_x: int, bbox_y: int,
                    bbox_w: int, bbox_h: int, embedding: bytes = None,
                    confidence: float = None, face_thumb: bytes = None) -> int:
        with self.conn:
            cur = self.conn.execute(
                """INSERT INTO faces (photo_id,bbox_x,bbox_y,bbox_w,bbox_h,
                       embedding,confidence,face_thumb)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (photo_id, bbox_x, bbox_y, bbox_w, bbox_h,
                 embedding, confidence, face_thumb)
            )
            return cur.lastrowid

    def get_faces_for_photo(self, photo_id: int) -> List[sqlite3.Row]:
        return self.fetchall("SELECT * FROM faces WHERE photo_id=?", (photo_id,))

    def get_all_faces(self) -> List[sqlite3.Row]:
        return self.fetchall("SELECT * FROM faces")

    def get_unassigned_faces(self) -> List[sqlite3.Row]:
        return self.fetchall(
            """SELECT f.* FROM faces f
               LEFT JOIN person_face_mappings pfm ON pfm.face_id = f.face_id
               WHERE pfm.face_id IS NULL"""
        )

    def get_face(self, face_id: int) -> Optional[sqlite3.Row]:
        return self.fetchone("SELECT * FROM faces WHERE face_id=?", (face_id,))

    def delete_faces_for_photo(self, photo_id: int):
        with self.conn:
            self.conn.execute("DELETE FROM faces WHERE photo_id=?", (photo_id,))

    def delete_face(self, face_id: int):
        with self.conn:
            self.conn.execute("DELETE FROM faces WHERE face_id=?", (face_id,))

    def get_photo_people_map(self, photo_ids: List[int]) -> dict:
        """Return {photo_id: [(person_id, name), ...]} for given photos."""
        if not photo_ids: return {}
        placeholders = ','.join(['?'] * len(photo_ids))
        rows = self.fetchall(f"""
            SELECT f.photo_id, p.person_id, p.name 
            FROM persons p
            JOIN person_face_mappings pfm ON pfm.person_id = p.person_id
            JOIN faces f ON f.face_id = pfm.face_id
            WHERE f.photo_id IN ({placeholders})
            ORDER BY p.name
        """, tuple(photo_ids))
        res = {}
        for r in rows:
            if r["photo_id"] not in res: res[r["photo_id"]] = []
            res[r["photo_id"]].append((r["person_id"], r["name"]))
        return res

    def get_group_photo_ids(self) -> List[int]:
        """Return IDs of photos that have more than one named person."""
        rows = self.fetchall("""
            SELECT f.photo_id FROM faces f
            JOIN person_face_mappings pfm ON pfm.face_id = f.face_id
            GROUP BY f.photo_id HAVING COUNT(DISTINCT pfm.person_id) > 1
        """)
        return [r["photo_id"] for r in rows]

    def get_solo_photo_ids(self) -> List[int]:
        """Return IDs of photos that have exactly one named person."""
        rows = self.fetchall("""
            SELECT f.photo_id FROM faces f
            JOIN person_face_mappings pfm ON pfm.face_id = f.face_id
            GROUP BY f.photo_id HAVING COUNT(DISTINCT pfm.person_id) = 1
        """)
        return [r["photo_id"] for r in rows]

    # ------------------------------------------------------------------ #
    # Persons                                                              #
    # ------------------------------------------------------------------ #

    def create_person(self, name: str, notes: str = None,
                      profile_face_id: int = None) -> int:
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO persons (name,notes,profile_face_id) VALUES (?,?,?)",
                (name, notes, profile_face_id)
            )
            return cur.lastrowid

    def update_person(self, person_id: int, name: str = None,
                      notes: str = None, profile_face_id: int = None):
        fields, vals = [], []
        if name is not None:
            fields.append("name=?"); vals.append(name)
        if notes is not None:
            fields.append("notes=?"); vals.append(notes)
        if profile_face_id is not None:
            fields.append("profile_face_id=?"); vals.append(profile_face_id)
        if not fields:
            return
        vals.append(person_id)
        with self.conn:
            self.conn.execute(
                f"UPDATE persons SET {','.join(fields)} WHERE person_id=?", vals)

    def delete_person(self, person_id: int):
        with self.conn:
            self.conn.execute("DELETE FROM persons WHERE person_id=?", (person_id,))

    def get_all_persons(self) -> List[sqlite3.Row]:
        return self.fetchall("SELECT * FROM persons ORDER BY name")

    def get_person(self, person_id: int) -> Optional[sqlite3.Row]:
        return self.fetchone("SELECT * FROM persons WHERE person_id=?", (person_id,))

    def get_person_by_name(self, name: str) -> Optional[sqlite3.Row]:
        return self.fetchone("SELECT * FROM persons WHERE name=?", (name,))

    # ------------------------------------------------------------------ #
    # Person–Face mappings                                                 #
    # ------------------------------------------------------------------ #

    def map_face_to_person(self, person_id: int, face_id: int):
        with self.conn:
            self.conn.execute(
                """INSERT OR IGNORE INTO person_face_mappings (person_id, face_id)
                   VALUES (?,?)""",
                (person_id, face_id)
            )

    def unmap_face(self, face_id: int):
        with self.conn:
            self.conn.execute(
                "DELETE FROM person_face_mappings WHERE face_id=?", (face_id,))

    def get_faces_for_person(self, person_id: int) -> List[sqlite3.Row]:
        return self.fetchall(
            """SELECT f.* FROM faces f
               JOIN person_face_mappings pfm ON pfm.face_id = f.face_id
               WHERE pfm.person_id = ?""",
            (person_id,)
        )

    def get_person_for_face(self, face_id: int) -> Optional[sqlite3.Row]:
        return self.fetchone(
            """SELECT p.* FROM persons p
               JOIN person_face_mappings pfm ON pfm.person_id = p.person_id
               WHERE pfm.face_id = ?""",
            (face_id,)
        )

    # ------------------------------------------------------------------ #
    # Users                                                                #
    # ------------------------------------------------------------------ #

    def get_user_by_id(self, user_id: int) -> Optional[sqlite3.Row]:
        return self.fetchone("SELECT * FROM users WHERE user_id=?", (user_id,))

    def create_user(self, username: str, pin_hash: str, role: str = "viewer", can_upload: int = 0) -> int:
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO users (username,pin_hash,role,can_upload) VALUES (?,?,?,?)",
                (username, pin_hash, role, can_upload)
            )
            return cur.lastrowid

    def get_user(self, username: str) -> Optional[sqlite3.Row]:
        return self.fetchone("SELECT * FROM users WHERE username=?", (username,))

    def get_all_users(self) -> List[sqlite3.Row]:
        return self.fetchall("SELECT user_id,username,role,can_upload,created_at,last_login FROM users")

    def update_user(self, username: str, pin_hash: str = None, role: str = None, can_upload: int = None):
        fields, vals = [], []
        if pin_hash is not None:
            fields.append("pin_hash=?"); vals.append(pin_hash)
        if role is not None:
            fields.append("role=?"); vals.append(role)
        if can_upload is not None:
            fields.append("can_upload=?"); vals.append(can_upload)
        if not fields:
            return
        vals.append(username)
        with self.conn:
            self.conn.execute(
                f"UPDATE users SET {','.join(fields)} WHERE username=?", vals)

    def update_user_pin(self, username: str, new_pin_hash: str):
        with self.conn:
            self.conn.execute(
                "UPDATE users SET pin_hash=? WHERE username=?",
                (new_pin_hash, username)
            )

    def delete_user(self, username: str):
        with self.conn:
            self.conn.execute("DELETE FROM users WHERE username=?", (username,))

    def update_last_login(self, username: str):
        with self.conn:
            self.conn.execute(
                "UPDATE users SET last_login=datetime('now') WHERE username=?",
                (username,)
            )

    def user_count(self) -> int:
        row = self.fetchone("SELECT COUNT(*) AS c FROM users")
        return row["c"] if row else 0

    # ------------------------------------------------------------------ #
    # Permissions                                                          #
    # ------------------------------------------------------------------ #

    def add_user_permission(self, user_id: int, person_id: int):
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO user_person_permissions (user_id, person_id) VALUES (?,?)",
                (user_id, person_id)
            )

    def remove_user_permissions(self, user_id: int):
        with self.conn:
            self.conn.execute("DELETE FROM user_person_permissions WHERE user_id=?", (user_id,))

    def get_user_permissions(self, user_id: int) -> List[int]:
        """Return list of person_ids allowed for this user."""
        rows = self.fetchall("SELECT person_id FROM user_person_permissions WHERE user_id=?", (user_id,))
        return [r["person_id"] for r in rows]

    # ------------------------------------------------------------------ #
    # Scan folders                                                         #
    # ------------------------------------------------------------------ #

    def add_scan_folder(self, path: str, created_by: int = None) -> int:
        with self.conn:
            cur = self.conn.execute(
                "INSERT OR IGNORE INTO scan_folders (path, created_by) VALUES (?,?)", (path, created_by))
            return cur.lastrowid or self.fetchone(
                "SELECT folder_id FROM scan_folders WHERE path=?", (path,))["folder_id"]

    def get_scan_folders(self, created_by: int = None) -> List[sqlite3.Row]:
        if created_by is not None:
             return self.fetchall("SELECT sf.*, u.username FROM scan_folders sf LEFT JOIN users u ON sf.created_by = u.user_id WHERE sf.created_by=?", (created_by,))
        return self.fetchall("SELECT sf.*, u.username FROM scan_folders sf LEFT JOIN users u ON sf.created_by = u.user_id")

    def remove_scan_folder(self, path: str):
        with self.conn:
            self.conn.execute("DELETE FROM scan_folders WHERE path=?", (path,))

    def delete_photos_for_folder(self, folder_path: str) -> int:
        """Delete all photos (and their faces) whose path starts with folder_path.
        
        person_face_mappings and persons are intentionally NOT touched so trained
        recognition data is preserved.
        Returns number of photos deleted.
        """
        # Normalise separator so LIKE works cross-platform
        norm = folder_path.rstrip("/\\").replace("\\", "/")
        rows = self.fetchall(
            "SELECT photo_id FROM photos WHERE replace(path,'\\\\','/') LIKE ?",
            (norm + "/%",)
        )
        if not rows:
            return 0
        photo_ids = [r["photo_id"] for r in rows]
        with self.conn:
            self.conn.execute("PRAGMA foreign_keys=OFF;")
            placeholders = ",".join(["?"] * len(photo_ids))
            # Remove face mappings first (FK), then faces, then photos
            self.conn.execute(
                f"DELETE FROM person_face_mappings WHERE face_id IN "
                f"(SELECT face_id FROM faces WHERE photo_id IN ({placeholders}))",
                photo_ids
            )
            self.conn.execute(
                f"DELETE FROM faces WHERE photo_id IN ({placeholders})",
                photo_ids
            )
            self.conn.execute(
                f"DELETE FROM photos WHERE photo_id IN ({placeholders})",
                photo_ids
            )
            self.conn.execute("PRAGMA foreign_keys=ON;")
        return len(photo_ids)

    def mark_folder_scanned(self, path: str):
        with self.conn:
            self.conn.execute(
                "UPDATE scan_folders SET last_scanned=datetime('now') WHERE path=?",
                (path,)
            )

    # ------------------------------------------------------------------ #
    # Settings                                                             #
    # ------------------------------------------------------------------ #

    def get_setting(self, key: str, default: str = None) -> Optional[str]:
        row = self.fetchone("SELECT value FROM settings WHERE key=?", (key,))
        return row["value"] if row else default

    def set_setting(self, key: str, value: str):
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO settings (key,value) VALUES (?,?)",
                (key, value)
            )

    def clear_all_data(self, keep_users: bool = True):
        """Wipe all project data (photos, faces, persons, folders)."""
        tables = ["person_face_mappings", "faces", "photos", "persons", "scan_folders"]
        if not keep_users:
            tables.append("users")
        
        with self.conn:
            # Disable foreign keys temporarily to allow truncating in any order
            self.conn.execute("PRAGMA foreign_keys=OFF;")
            for table in tables:
                self.conn.execute(f"DELETE FROM {table}")
                # Reset autoincrement counters
                self.conn.execute(f"DELETE FROM sqlite_sequence WHERE name='{table}'")
            self.conn.execute("PRAGMA foreign_keys=ON;")
        logger.info("Project data cleared.")
