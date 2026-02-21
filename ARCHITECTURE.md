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
│  │  SQLite WAL mode      │  │               deepface (auto-detect)   │  │
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
│  │  PIN-based session auth, person/photo browse, ZIP export         │  │
│  │  User-specific folder management & uploads (relative paths in UI)│  │
│  │  Face editing capabilities: rename people, reassign faces        │  │
│  │  Improved static file serving                                    │  │
│  │  REST API: /api/persons  /api/photos  /thumb/<id>  /photo/<id>  │  │
│  │            /api/photo/<id>/faces                               │  │
│  │  Routes:  /admin/face/<id>/edit                                │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘


Database Schema (SQLite – facegallery.db)
─────────────────────────────────────────

photos            – image file records (path, hash, EXIF, thumbnail)
faces             – detected faces with bbox + embedding blobs
persons           – named individuals
person_face_mappings – many-to-many faces ↔ persons
users             – web access accounts (username + PIN hash)
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
│   │   ├── schema.py         ← CREATE TABLE SQL
│   │   └── manager.py        ← DatabaseManager (CRUD)
│   ├── face_engine/
│   │   ├── detector.py       ← multi-backend face detection
│   │   └── clusterer.py      ← cosine-distance clustering
│   ├── core/
│   │   ├── scanner.py        ← filesystem + photo indexer
│   │   └── app_core.py       ← business logic facade
│   ├── gui/
│   │   ├── main_window.py    ← MainWindow (PyQt6)
│   │   ├── dialogs.py        ← Scan / FaceNaming / Users / Settings dialogs
│   │   └── widgets.py        ← reusable thumbnail widgets
│   ├── web/
│   │   └── server.py         ← Flask web server
│   └── utils/
│       └── helpers.py        ← shared utility functions
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
