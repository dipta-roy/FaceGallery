"""
FaceGallery – architecture overview and module guide.

Run with:
  python main.py

Architecture (plain text diagram):

┌────────────────────────────────────────────────────────────────────────┐
│                         FaceGallery Application                        │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │              5. Desktop GUI Layer (PyQt6)                        │  │
│  │  MainWindow → person sidebar, photo grid, toolbar, menus         │  │
│  │  Dialogs    → ScanDialog, FaceNameDialog, UserMgmt, Settings     │  │
│  │  Widgets    → PhotoThumbnailWidget, FaceThumbnailWidget          │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                          │ calls                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │    4. Application Core / Business Logic (src/core)               │  │
│  │  AppCore  → face clustering, person assign, photo filter, export │  │
│  │  Scanner  → walks FS, indexes photos, triggers face detection    │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│          │ reads/writes              │ calls                            │
│  ┌───────────────────────┐  ┌───────────────────────────────────────┐  │
│  │  3. Data Layer        │  │  2. Face Recognition Layer            │  │
│  │  (src/db)             │  │  (src/face_engine)                    │  │
│  │  DatabaseManager      │  │  detector.py  → insightface /         │  │
│  │  schema.py (SQL DDL)  │  │               face_recognition /      │  │
│  │  SQLite WAL mode      │  │               deepface / mediapipe / opencv (auto-detect)      │  │
│  └───────────────────────┘  │  clusterer.py → cosine-distance       │  │
│                              │               greedy clustering        │  │
│                              └───────────────────────────────────────┘  │
│                                        │ calls                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │    1. File System & Photo Scanner Layer (src/utils)              │  │
│  │  helpers.py → walk_images, make_thumbnail, compute_file_hash,    │  │
│  │               get_image_info, embedding_to_bytes, hash_pin       │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │    6. Local Web Server Layer (src/web)                           │  │
│  │  Flask app running in daemon thread (port 5050, 0.0.0.0)        │  │
│  │  PIN-based session auth, user management, person/photo browse,   │  │
│  │  ZIP export, face editing (rename people, reassign faces),       │  │
│  │  user-specific folder management & uploads (relative paths in UI),│  │
│  │  real-time scan progress, photo filtering (groups/solos).        │  │
│  │  REST API: /api/persons, /api/photos, /thumb/<id>, /photo/<id>, │  │
│  │            /face_thumb/<id>, /api/photo/<id>/faces,              │  │
│  │            /api/photo/<id>/faces/manual (manual face adding).    │  │
│  │  Routes:  /admin, /admin/users, /admin/folders, /admin/naming,  │  │
│  │            /admin/person/<id>/faces, /admin/face/<id>/edit.    │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘


Database Schema (SQLite – facegallery.db)
─────────────────────────────────────────

photos            – image file records (path, hash, EXIF, thumbnail)
faces             – detected faces with bbox + embedding blobs
persons           – named individuals
person_face_mappings – many-to-many faces ↔ persons
users             – web access accounts (username + PIN hash, role, can_upload, created_at, last_login)
user_person_permissions – maps which persons a user can view
scan_folders      – persisted folder list for re-indexing (includes created_by for ownership)
settings          – key/value app configuration


Folder / Module Structure
─────────────────────────

FaceGallary/
├── main.py                   ← entry point
├── requirements.txt
├── ARCHITECTURE.py           ← this file
├── task-complete.md          ← task checklist
├── uploads/                  ← centralized directory for user-managed photos
├── src/
│   ├── db/
│   │   ├── schema.py         ← Defines SQLite table schemas (SQL DDL).
│   │   └── manager.py        ← Manages SQLite connection and provides CRUD operations for all tables.
│   ├── face_engine/
│   │   ├── detector.py       ← Handles multi-backend face detection (InsightFace, DeepFace, Mediapipe, Dlib, OpenCV) and embedding extraction.
│   │   └── clusterer.py      ← Implements cosine-distance greedy clustering for face grouping.
│   ├── core/
│   │   ├── scanner.py        ← Scans filesystem for photos, indexes them, and dispatches face detection.
│   │   └── app_core.py       ← Central business logic facade, bridging DB and face engine for clustering, person assignment, photo filtering, and export.
│   ├── gui/
│   │   ├── main_window.py    ← Main PyQt6 desktop application window, orchestrates UI.
│   │   ├── dialogs.py        ← PyQt6 dialogs for scanning, face naming, user management, and settings.
│   │   └── widgets.py        ← Reusable PyQt6 custom widgets, like photo and face thumbnails.
│   ├── web/
│   │   └── server.py         ← Flask web server, defines all web routes, API endpoints, and HTML templates.
│   └── utils/
│       └── helpers.py        ← Collection of shared utility functions (image processing, hashing, network).
├── resources/
│   └── icons/                ← (reserved for icons)
└── tests/                    ← (reserved for unit tests)


Important Python Packages & Versions
──────────────────────────────────────

Package            Version    Purpose
─────────────────────────────────────────────────────────────────
PyQt6              >=6.5.0    Desktop GUI
Pillow             >=10.0.0   Image loading, thumbnails, EXIF
Flask              >=3.0.0    Local web server
numpy              >=1.24.0   Embedding math, clustering
insightface        >=0.7.3    Face detection (Option A – preferred)
onnxruntime        >=1.16.0   Required by insightface
face_recognition   >=1.3.0    Face detection (Option B – dlib)
deepface           >=0.0.89   Face detection (Option C – TF)
pillow-heif        >=0.13.0   HEIC/HEIF support (optional)
pyinstaller        >=6.0.0    Packaging (dev only)
"""
