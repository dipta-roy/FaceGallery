"""
Database schema definitions (CREATE TABLE statements) for FaceGallery.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Table creation SQL
# ─────────────────────────────────────────────────────────────────────────────

CREATE_PHOTOS_TABLE = """
CREATE TABLE IF NOT EXISTS photos (
    photo_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    path        TEXT    NOT NULL UNIQUE,
    phash       TEXT,
    file_size   INTEGER,
    width       INTEGER,
    height      INTEGER,
    orientation INTEGER DEFAULT 1,
    date_taken  TEXT,
    date_added  TEXT    NOT NULL DEFAULT (datetime('now')),
    thumbnail   BLOB
);
"""

CREATE_FACES_TABLE = """
CREATE TABLE IF NOT EXISTS faces (
    face_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_id    INTEGER NOT NULL REFERENCES photos(photo_id) ON DELETE CASCADE,
    bbox_x      INTEGER,
    bbox_y      INTEGER,
    bbox_w      INTEGER,
    bbox_h      INTEGER,
    embedding   BLOB,
    confidence  REAL,
    face_thumb  BLOB
);
"""

CREATE_PERSONS_TABLE = """
CREATE TABLE IF NOT EXISTS persons (
    person_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name                TEXT    NOT NULL,
    notes               TEXT,
    profile_face_id     INTEGER REFERENCES faces(face_id) ON DELETE SET NULL,
    created_at          TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""

CREATE_PERSON_FACE_MAPPINGS_TABLE = """
CREATE TABLE IF NOT EXISTS person_face_mappings (
    mapping_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id   INTEGER NOT NULL REFERENCES persons(person_id) ON DELETE CASCADE,
    face_id     INTEGER NOT NULL REFERENCES faces(face_id)     ON DELETE CASCADE,
    UNIQUE(person_id, face_id)
);
"""

CREATE_USERS_TABLE = """
CREATE TABLE IF NOT EXISTS users (
    user_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    username    TEXT    NOT NULL UNIQUE,
    pin_hash    TEXT    NOT NULL,
    role        TEXT    NOT NULL DEFAULT 'viewer',
    can_upload  INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    last_login  TEXT
);
"""

CREATE_USER_PERSON_PERMISSIONS_TABLE = """
CREATE TABLE IF NOT EXISTS user_person_permissions (
    user_id     INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    person_id   INTEGER NOT NULL REFERENCES persons(person_id) ON DELETE CASCADE,
    PRIMARY KEY (user_id, person_id)
);
"""

CREATE_SCAN_FOLDERS_TABLE = """
CREATE TABLE IF NOT EXISTS scan_folders (
    folder_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    path        TEXT    NOT NULL UNIQUE,
    last_scanned TEXT,
    created_by  INTEGER REFERENCES users(user_id) ON DELETE SET NULL
);
"""

CREATE_SETTINGS_TABLE = """
CREATE TABLE IF NOT EXISTS settings (
    key     TEXT PRIMARY KEY,
    value   TEXT
);
"""

ALL_TABLES = [
    CREATE_PHOTOS_TABLE,
    CREATE_FACES_TABLE,
    CREATE_PERSONS_TABLE,
    CREATE_PERSON_FACE_MAPPINGS_TABLE,
    CREATE_USERS_TABLE,
    CREATE_USER_PERSON_PERMISSIONS_TABLE,
    CREATE_SCAN_FOLDERS_TABLE,
    CREATE_SETTINGS_TABLE,
]

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_faces_photo_id ON faces(photo_id);",
    "CREATE INDEX IF NOT EXISTS idx_pfm_person    ON person_face_mappings(person_id);",
    "CREATE INDEX IF NOT EXISTS idx_pfm_face      ON person_face_mappings(face_id);",
    "CREATE INDEX IF NOT EXISTS idx_photos_path   ON photos(path);",
]
