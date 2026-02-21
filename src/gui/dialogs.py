"""
Dialog windows used by the FaceGallery desktop application.
"""

import os
from typing import List, Optional, Any, Set
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QPixmap, QIcon, QColor
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QProgressBar, QScrollArea, QWidget, QComboBox,
    QGridLayout, QTextEdit, QMessageBox, QFileDialog, QCheckBox,
    QGroupBox, QFormLayout, QSizePolicy, QFrame, QListWidget,
    QListWidgetItem, QTabWidget, QSpinBox, QApplication
)

from ..face_engine.clusterer import cluster_embeddings
from ..utils.helpers import bytes_to_embedding

from .widgets import (PhotoThumbnailWidget, FaceThumbnailWidget,
                       bytes_to_pixmap, _placeholder_pixmap)


# ─────────────────────────────────────────────────────────────────────────────
# Clustering worker thread
# ─────────────────────────────────────────────────────────────────────────────

class ClusterWorker(QThread):
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, app_core):
        super().__init__()
        self.app_core = app_core

    def run(self):
        try:
            clusters = self.app_core.cluster_unknown_faces()
            self.finished.emit(clusters)
        except Exception as exc:
            self.error.emit(str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Scanning worker thread
# ─────────────────────────────────────────────────────────────────────────────

class ScanWorker(QThread):
    progress = pyqtSignal(int, int, str)
    face_found = pyqtSignal(bytes)
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, scanner, folders: List[str], force_rescan: bool = False):
        super().__init__()
        self._scanner = scanner
        self._folders = folders
        self._force_rescan = force_rescan
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            summary = self._scanner.scan(self._folders, force_rescan=self._force_rescan)
            self.finished.emit(summary)
        except Exception as exc:
            self.error.emit(str(exc))


class ScanDialog(QDialog):
    """Dialog for selecting folders and running the indexing scan."""

    def __init__(self, db, app_core, parent=None):
        super().__init__(parent)
        self.db = db
        self.app_core = app_core
        self._worker: Optional[ScanWorker] = None
        self.setWindowTitle("📁 Scan Photos")
        self.setMinimumWidth(560)
        self._build_ui()
        self._load_folders()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        # Header
        lbl = QLabel("Index Photo Folders")
        lbl.setStyleSheet("font-size:18px;font-weight:700;")
        layout.addWidget(lbl)

        # Folder list
        self._folder_list = QListWidget()
        self._folder_list.setMinimumHeight(120)
        layout.addWidget(self._folder_list)

        # Folder buttons
        btn_row = QHBoxLayout()
        self._btn_add = QPushButton("➕ Add Folder")
        self._btn_remove = QPushButton("🗑 Remove")
        for b in (self._btn_add, self._btn_remove):
            b.setStyleSheet(
                "QPushButton{background:#252840;color:#e2e8f0;border:1px solid #2d3748;"
                "border-radius:6px;padding:7px 16px;font-size:13px;}"
                "QPushButton:hover{background:#6c63ff;}")
        btn_row.addWidget(self._btn_add)
        btn_row.addWidget(self._btn_remove)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Progress
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setStyleSheet(
            "QProgressBar{background:#1a1d2e;border-radius:6px;border:1px solid #2d3748;height:18px;}"
            "QProgressBar::chunk{background:#6c63ff;border-radius:6px;}")
        layout.addWidget(self._progress)

        self._status_lbl = QLabel("Ready.")
        self._status_lbl.setStyleSheet("color:#94a3b8;font-size:12px;")
        layout.addWidget(self._status_lbl)

        # Preview area for faces found
        self._preview_row = QHBoxLayout()
        self._preview_row.setContentsMargins(10, 0, 10, 0)
        self._face_preview = QLabel()
        self._face_preview.setFixedSize(80, 80)
        self._face_preview.setStyleSheet("background:#1a1d2e; border:1px solid #2d3748; border-radius:40px;")
        self._face_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self._preview_info = QLabel("Faces will appear here as they are detected...")
        self._preview_info.setStyleSheet("color:#4a5568; font-style:italic; font-size:12px;")
        
        self._preview_row.addWidget(self._face_preview)
        self._preview_row.addWidget(self._preview_info)
        self._preview_row.addStretch()
        layout.addLayout(self._preview_row)

        # Force rescan option
        self._force_cb = QCheckBox("Force re-scan faces (Find missed faces in group shots)")
        self._force_cb.setStyleSheet("color:#94a3b8; font-size:12px; margin-top:8px;")
        layout.addWidget(self._force_cb)

        # Action buttons
        action_row = QHBoxLayout()
        self._btn_scan = QPushButton("🔍 Start Scan")
        self._btn_cancel = QPushButton("✕ Cancel")
        self._btn_close = QPushButton("Close")
        self._btn_cancel.setEnabled(False)
        for b, color in [(self._btn_scan, "#6c63ff"),
                         (self._btn_cancel, "#e53e3e"),
                         (self._btn_close, "#2d3748")]:
            b.setStyleSheet(
                f"QPushButton{{background:{color};color:#fff;border:none;"
                f"border-radius:8px;padding:9px 24px;font-size:14px;font-weight:600;}}"
                f"QPushButton:hover{{opacity:.85;}} QPushButton:disabled{{opacity:.4;}}")
        self._btn_scan.clicked.connect(self._start_scan)
        self._btn_cancel.clicked.connect(self._cancel_scan)
        self._btn_close.clicked.connect(self.accept)
        self._btn_add.clicked.connect(self._add_folder)
        self._btn_remove.clicked.connect(self._remove_folder)
        action_row.addWidget(self._btn_scan)
        action_row.addWidget(self._btn_cancel)
        action_row.addStretch()
        action_row.addWidget(self._btn_close)
        layout.addLayout(action_row)

    def _load_folders(self):
        self._folder_list.clear()
        for row in self.db.get_scan_folders():
            self._folder_list.addItem(row["path"])

    def _add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Photo Folder")
        if folder:
            self.db.add_scan_folder(folder)
            self._load_folders()

    def _remove_folder(self):
        item = self._folder_list.currentItem()
        if item:
            # Desktop app: ONLY removes the folder from the scan list in the DB.
            # Physical image files on disk are NEVER deleted from the desktop app.
            self.db.remove_scan_folder(item.text())
            self._load_folders()

    def _start_scan(self):
        folders = [self._folder_list.item(i).text()
                   for i in range(self._folder_list.count())]
        if not folders:
            QMessageBox.warning(self, "No Folders", "Add at least one folder first.")
            return

        from ..core.scanner import PhotoScanner

        self._cancelled = False
        scanner = PhotoScanner(
            self.db,
            progress_cb=lambda c, t, m: self._worker.progress.emit(c, t, m),
            face_found_cb=lambda b: self._worker.face_found.emit(b),
            cancelled_cb=lambda: self._cancelled,
        )
        self._worker = ScanWorker(scanner, folders, force_rescan=self._force_cb.isChecked())
        self._worker.progress.connect(self._on_progress)
        self._worker.face_found.connect(self._on_face_found)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

        self._btn_scan.setEnabled(False)
        self._btn_cancel.setEnabled(True)

    def closeEvent(self, event):
        """Ensure scan is cancelled if dialog is closed."""
        self._cancel_scan()
        if self._worker and self._worker.isRunning():
            self._worker.wait(500)
        super().closeEvent(event)

    def _cancel_scan(self):
        self._cancelled = True
        self._btn_cancel.setEnabled(False)
        self._status_lbl.setText("Cancelling…")

    def _on_progress(self, current, total, msg):
        if total > 0:
            pct = int(current / total * 100)
            self._progress.setValue(pct)
        self._status_lbl.setText(f"[{current}/{total}] {msg}")

    def _on_face_found(self, thumb_bytes):
        from .widgets import make_circular_pixmap
        pm = make_circular_pixmap(thumb_bytes, 80)
        self._face_preview.setPixmap(pm)
        self._preview_info.setText("Scanning... Face detected!")
        self._preview_info.setStyleSheet("color:#e2e8f0; font-weight:600;")

    def _on_finished(self, summary):
        self._progress.setValue(100)
        self._status_lbl.setText(
            f"Done! {summary['added']} photos indexed, {summary['faces']} faces detected.")
        self._btn_scan.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        QMessageBox.information(self, "Scan Complete",
            f"✅ Scan finished!\n\n"
            f"Photos indexed: {summary['added']}\n"
            f"Photos skipped: {summary['skipped']}\n"
            f"Faces detected: {summary['faces']}")

    def _on_error(self, msg):
        self._btn_scan.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        QMessageBox.critical(self, "Scan Error", f"Scan failed:\n{msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Face Naming Dialog
# ─────────────────────────────────────────────────────────────────────────────

class FaceNameDialog(QDialog):
    """Show clusters of unknown faces for user to name."""

    def __init__(self, db, app_core, parent=None):
        super().__init__(parent)
        self.db = db
        self.app_core = app_core
        self.setWindowTitle("👥 Name Unknown Faces")
        self.setMinimumSize(1000, 700)
        self._face_widgets: List[FaceThumbnailWidget] = []
        self._selected_faces: Set[int] = set()
        self._last_selected_fid: Optional[int] = None
        self._combos: List[QComboBox] = []
        self._build_ui()
        self._load_unassigned_faces()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        hdr = QLabel("Name Unknown Faces")
        hdr.setStyleSheet("font-size:20px; font-weight:800; margin-bottom:2px;")
        layout.addWidget(hdr)

        self._progress_lbl = QLabel("Identifying your library...")
        self._progress_lbl.setStyleSheet("font-size:13px; margin-bottom:10px;")
        layout.addWidget(self._progress_lbl)

        # Scroll area for faces
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setStyleSheet("QScrollArea{background:transparent;}")
        
        self._inner = QWidget()
        self._inner.setStyleSheet("background:transparent;")
        self._inner_layout = QVBoxLayout(self._inner)
        self._inner_layout.setContentsMargins(0, 0, 0, 0)
        self._inner_layout.setSpacing(10)
        self._inner_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._grid_layout = QGridLayout()
        self._grid_layout.setSpacing(10)
        self._grid_layout.setContentsMargins(10, 10, 10, 10)
        self._inner_layout.addLayout(self._grid_layout)

        self._scroll.setWidget(self._inner)
        layout.addWidget(self._scroll)

        btn_row = QHBoxLayout()
        self._btn_refresh = QPushButton("🔄 Refresh Faces")
        self._btn_automatch = QPushButton("🤖 Auto Match All")
        self._btn_close = QPushButton("Close")
        for b, color in [(self._btn_refresh, "#6c63ff"), 
                         (self._btn_automatch, "#38a169"),
                         (self._btn_close, "#2d3748")]:
            b.setStyleSheet(
                f"QPushButton{{background:{color};color:#fff;border:none;"
                f"border-radius:8px;padding:9px 20px;font-size:13px;font-weight:600;}}"
                f"QPushButton:disabled{{opacity:.4;}}")
        self._btn_refresh.clicked.connect(self._load_unassigned_faces)
        self._btn_automatch.clicked.connect(self._auto_match_click)
        self._btn_close.clicked.connect(self.accept)
        btn_row.addWidget(self._btn_refresh)
        btn_row.addWidget(self._btn_automatch)
        btn_row.addStretch()
        btn_row.addWidget(self._btn_close)
        layout.addLayout(btn_row)

        # Bulk assignment panel (hidden by default)
        self._bulk_panel = QFrame()
        self._bulk_panel.setStyleSheet(
            "QFrame{background:#1a1d2e; border-top: 2px solid #6c63ff; border-radius:0;}")
        self._bulk_panel.setFixedHeight(70)
        self._bulk_panel.hide()
        bulk_layout = QHBoxLayout(self._bulk_panel)
        
        self._bulk_lbl = QLabel("0 selected")
        self._bulk_lbl.setStyleSheet("color:#e2e8f0; font-weight:700; font-size:14px;")
        
        self._bulk_combo = QComboBox()
        self._bulk_combo.setMinimumWidth(300) # Increased from 250
        self._bulk_combo.setStyleSheet(
            "QComboBox{background:#252840;color:#e2e8f0;border:1px solid #2d3748;"
            "border-radius:6px;padding:6px 10px;font-size:13px;}")
        self._combos.append(self._bulk_combo)
        
        self._bulk_name = QLineEdit()
        self._bulk_name.setPlaceholderText("Or new person name...")
        self._bulk_name.setStyleSheet(
            "QLineEdit{background:#252840;color:#e2e8f0;border:1px solid #2d3748;"
            "border-radius:6px;padding:6px 12px;font-size:13px;}")
        
        self._btn_bulk_save = QPushButton("💾 Assign Selected")
        self._btn_bulk_save.setStyleSheet(
            "QPushButton{background:#6c63ff;color:#fff;border:none;"
            "border-radius:8px;padding:9px 20px;font-size:13px;font-weight:600;}")
        self._btn_bulk_save.clicked.connect(self._bulk_assign_click)
        
        self._btn_bulk_reject = QPushButton("🗑 Not a Face")
        self._btn_bulk_reject.setStyleSheet(
            "QPushButton{background:#e53e3e;color:#fff;border:none;"
            "border-radius:8px;padding:9px 15px;font-size:13px;font-weight:600;}")
        self._btn_bulk_reject.clicked.connect(self._bulk_reject_click)
        
        self._btn_bulk_clear = QPushButton("✕")
        self._btn_bulk_clear.setFixedWidth(40)
        self._btn_bulk_clear.setStyleSheet("QPushButton{background:#2d3748; color:#fff; border-radius:8px;}")
        self._btn_bulk_clear.clicked.connect(self._clear_selection)

        bulk_layout.addWidget(self._bulk_lbl)
        bulk_layout.addWidget(QLabel("Assign all to:"))
        bulk_layout.addWidget(self._bulk_combo)
        bulk_layout.addWidget(self._bulk_name)
        bulk_layout.addWidget(self._btn_bulk_save)
        bulk_layout.addWidget(self._btn_bulk_reject)
        bulk_layout.addStretch()
        bulk_layout.addWidget(self._btn_bulk_clear)
        layout.addWidget(self._bulk_panel)

    def _load_unassigned_faces(self):
        self._btn_refresh.setEnabled(False)
        self._btn_refresh.setText("⏳ Loading…")

        # Clear existing grid
        while self._grid_layout.count():
            item = self._grid_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()
        
        self._face_widgets.clear()
        self._combos.clear()
        self._combos.append(self._bulk_combo)
        
        self._bulk_combo.clear()
        self._bulk_combo.addItem("— Create new person —", None)
        for p in self.db.get_all_persons():
            self._bulk_combo.addItem(p["name"], p["person_id"])

        faces = self.db.get_unassigned_faces()
        
        if not faces:
            self._btn_refresh.setEnabled(True)
            self._btn_refresh.setText("🔄 Refresh")
            self._progress_lbl.setText("No unknown faces found.")
            lbl = QLabel("No unknown faces found. All faces have been named.")
            lbl.setStyleSheet("color:#94a3b8;font-size:14px;padding:40px;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._grid_layout.addWidget(lbl, 0, 0)
            return

        # Sort faces by similarity (clustering)
        # Prepare embeddings
        face_list = list(faces)
        embs = []
        for f in face_list:
            if f["embedding"]:
                embs.append(bytes_to_embedding(f["embedding"]))
            else:
                # Dummy embedding for faces without one (shouldn't happen)
                embs.append(np.zeros(512, dtype=np.float32))

        # Perform clustering to group similar faces
        import numpy as np
        clusters = cluster_embeddings(embs)
        
        # Flatten clusters into a sorted face list
        sorted_faces = []
        for cluster_indices in clusters:
            for idx in cluster_indices:
                sorted_faces.append(face_list[idx])

        self._progress_lbl.setText(f"Found <b>{len(sorted_faces)}</b> unknown faces, sorted by similarity.")
        
        for face in sorted_faces:
            fid = face["face_id"]
            fw = FaceThumbnailWidget(fid, face["face_thumb"], size=100)
            fw.selected.connect(self._on_face_selected)
            if fid in self._selected_faces:
                fw.set_selected(True)
            self._face_widgets.append(fw)

        self._rearrange_grid(force=True)

        self._btn_refresh.setEnabled(True)
        self._btn_refresh.setText("🔄 Refresh")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Rearrange grid when window is resized
        self._rearrange_grid()

    def _rearrange_grid(self, force=False):
        if not self._face_widgets:
            return

        # Calculate columns based on scroll area width
        w = self._scroll.viewport().width()
        if w < 100: w = self.width() - 40
        
        col_width = 110 # Item width (100) + spacing (10)
        cols = max(1, w // col_width)
        
        # Don't update if nothing changed (unless forced during refresh)
        if not force and hasattr(self, "_last_cols") and self._last_cols == cols:
            return
        self._last_cols = cols

        # Remove from layout without deleting
        for i in reversed(range(self._grid_layout.count())):
            item = self._grid_layout.takeAt(i)
            if item and item.widget():
                item.widget().setParent(None)

        # Re-add in new positions
        for i, fw in enumerate(self._face_widgets):
            r, c = divmod(i, cols)
            self._grid_layout.addWidget(fw, r, c)
            fw.show()

        self._inner.adjustSize()

    def _auto_match_click(self):
        self._btn_automatch.setEnabled(False)
        self._btn_automatch.setText("⏳ Matching…")
        QApplication.processEvents()
        
        matches = self.app_core.auto_match_all_unassigned()
        
        self._btn_automatch.setEnabled(True)
        self._btn_automatch.setText("🤖 Auto Match All")
        
        if matches > 0:
            QMessageBox.information(self, "Auto Match Complete", 
                                     f"Successfully matched and assigned {matches} face(s)!")
            self._load_unassigned_faces()
        else:
            QMessageBox.information(self, "Auto Match", "No additional matches found.")

    def _add_person_to_dropdowns(self, name: str, person_id: int):
        """Update all active dropdowns with a newly created person."""
        for cb in self._combos:
            exists = False
            for idx in range(cb.count()):
                if cb.itemData(idx) == person_id:
                    exists = True
                    break
            if not exists:
                cb.addItem(name, person_id)

    def _on_face_selected(self, face_id: int):
        modifiers = QApplication.keyboardModifiers()
        
        if (modifiers & Qt.KeyboardModifier.ShiftModifier) and (self._last_selected_fid is not None):
            # Range selection
            idx_start = -1
            idx_end = -1
            for i, fw in enumerate(self._face_widgets):
                if fw.face_id == self._last_selected_fid: idx_start = i
                if fw.face_id == face_id: idx_end = i
            
            if idx_start != -1 and idx_end != -1:
                lo, hi = min(idx_start, idx_end), max(idx_start, idx_end)
                for i in range(lo, hi + 1):
                    fw = self._face_widgets[i]
                    fid = fw.face_id
                    if fid not in self._selected_faces:
                        self._selected_faces.add(fid)
                        fw.set_selected(True)
            self._last_selected_fid = face_id
        else:
            # Standard toggle
            if face_id in self._selected_faces:
                self._selected_faces.remove(face_id)
            else:
                self._selected_faces.add(face_id)
            self._last_selected_fid = face_id
            
        self._update_bulk_panel_state()

    def _update_bulk_panel_state(self):
        count = len(self._selected_faces)
        if count > 0:
            self._bulk_lbl.setText(f"👤 {count} selected")
            self._bulk_panel.show()
        else:
            self._bulk_panel.hide()

    def _clear_selection(self):
        self._selected_faces.clear()
        self._last_selected_fid = None
        for fw in self._face_widgets:
            fw.set_selected(False)
        self._bulk_panel.hide()

    def _bulk_assign_click(self):
        if not self._selected_faces: return
        
        person_id = self._bulk_combo.currentData()
        name = self._bulk_name.text().strip()
        face_ids = list(self._selected_faces)
        
        if person_id is not None and person_id != -1 and name:
            QMessageBox.warning(self, "Conflicting Input", "Please either select an existing person or type a new name—not both.")
            return

        if person_id == -1:
            for fid in face_ids:
                self.db.unmap_face(fid)
        elif person_id is None:
            if not name:
                QMessageBox.warning(self, "Name Required", "Enter a name for the new person.")
                return
            new_id, matched = self.app_core.create_person_from_faces(face_ids, name)
            self._add_person_to_dropdowns(name, new_id)
        else:
            matched = self.app_core.assign_faces_to_person(face_ids, person_id)
        
        self._selected_faces.clear()
        self._last_selected_fid = None
        self._bulk_name.clear()
        self._load_unassigned_faces()
        self._update_bulk_panel_state()

    def _bulk_reject_click(self):
        if not self._selected_faces: return
        
        ans = QMessageBox.question(self, "Confirm Rejection", 
                                   f"Are you sure you want to remove {len(self._selected_faces)} selected face(s)?",
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if ans == QMessageBox.StandardButton.Yes:
            self.app_core.remove_false_positive(list(self._selected_faces))
            self._selected_faces.clear()
            self._last_selected_fid = None
            self._load_unassigned_faces()
            self._update_bulk_panel_state()


# ─────────────────────────────────────────────────────────────────────────────
# User Management Dialog
# ─────────────────────────────────────────────────────────────────────────────

class UserManagementDialog(QDialog):
    """Create / delete web users and reset PINs."""

    def __init__(self, db, parent=None):
        super().__init__(parent)
        self.db = db
        self._current_user_id = None
        self.setWindowTitle("👤 User Management")
        self.setMinimumWidth(550)
        self._build_ui()
        self._refresh()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(14)

        hdr = QLabel("Web Access Users")
        hdr.setStyleSheet("font-size:18px;font-weight:700;color:#e2e8f0;")
        layout.addWidget(hdr)

        self._user_list = QListWidget()
        self._user_list.setStyleSheet(
            "QListWidget{background:#1a1d2e;border:1px solid #2d3748;border-radius:8px;"
            "color:#e2e8f0;font-size:13px;padding:4px;}"
            "QListWidget::item{padding:8px;border-radius:4px;}"
            "QListWidget::item:selected{background:#6c63ff;}")
        self._user_list.setMinimumHeight(140)
        self._user_list.itemSelectionChanged.connect(self._on_user_selected)
        layout.addWidget(self._user_list)

        # New user form
        form_group = QGroupBox("Add / Update User")
        form_layout = QFormLayout(form_group)

        self._username_edit = QLineEdit()
        self._pin_edit = QLineEdit()
        self._pin_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._pin_edit.setPlaceholderText("4–8 digits")
        self._pin_edit.setMaxLength(8)
        self._role_combo = QComboBox()
        self._role_combo.addItems(["viewer", "admin"])

        for w in (self._username_edit, self._pin_edit):
            pass # Styles handled by global theme

        self._can_upload_cb = QCheckBox("Can Upload & Manage Folders")
        self._can_upload_cb.setToolTip("Allows user to create folders and upload photos from web UI.")
        
        form_layout.addRow("Username:", self._username_edit)
        form_layout.addRow("PIN:", self._pin_edit)
        form_layout.addRow("Role:", self._role_combo)
        form_layout.addRow("", self._can_upload_cb)

        # Permissions sub-group
        self._perm_group = QGroupBox("Allowed People (Only for Viewers)")
        perm_layout = QVBoxLayout(self._perm_group)
        
        self._person_list = QListWidget()
        self._person_list.setMaximumHeight(150)
        perm_layout.addWidget(self._person_list)
        
        form_layout.addRow(self._perm_group)
        self._role_combo.currentTextChanged.connect(self._on_role_changed)

        layout.addWidget(form_group)

        btn_row = QHBoxLayout()
        btn_add = QPushButton("➕ Add / Update")
        btn_del = QPushButton("🗑 Delete Selected")
        btn_close = QPushButton("Close")
        for b, color in [(btn_add, "#6c63ff"), (btn_del, "#e53e3e"),
                         (btn_close, "#2d3748")]:
            b.setStyleSheet(
                f"QPushButton{{background:{color};color:#fff;border:none;"
                f"border-radius:8px;padding:9px 20px;font-size:13px;font-weight:600;}}")
        btn_add.clicked.connect(self._add_user)
        btn_del.clicked.connect(self._del_user)
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_add)
        btn_row.addWidget(btn_del)
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

    def _on_role_changed(self, role):
        self._perm_group.setEnabled(role == "viewer")

    def _on_user_selected(self):
        item = self._user_list.currentItem()
        if not item:
            self._username_edit.clear()
            self._username_edit.setReadOnly(False)
            self._role_combo.setCurrentText("viewer")
            self._current_user_id = None
            for idx in range(self._person_list.count()):
                self._person_list.item(idx).setCheckState(Qt.CheckState.Unchecked)
            return

        username = item.data(Qt.ItemDataRole.UserRole)
        user = self.db.get_user(username)
        if user:
            self._current_user_id = user["user_id"]
            self._username_edit.setText(user["username"])
            self._username_edit.setReadOnly(True)
            self._role_combo.setCurrentText(user["role"])
            self._pin_edit.clear()
            self._pin_edit.setPlaceholderText("(Leave blank to keep current PIN)")
            self._can_upload_cb.setChecked(bool(user["can_upload"]))
            
            allowed = self.db.get_user_permissions(user["user_id"])
            for idx in range(self._person_list.count()):
                p_item = self._person_list.item(idx)
                pid = p_item.data(Qt.ItemDataRole.UserRole)
                p_item.setCheckState(Qt.CheckState.Checked if pid in allowed else Qt.CheckState.Unchecked)

    def _refresh(self):
        self._user_list.clear()
        for u in self.db.get_all_users():
            item = QListWidgetItem(
                f"👤  {u['username']}   [{u['role']}]   "
                f"Last login: {u['last_login'] or 'never'}")
            item.setData(Qt.ItemDataRole.UserRole, u["username"])
            self._user_list.addItem(item)
        
        # Fresh load persons
        self._person_list.clear()
        for p in self.db.get_all_persons():
            p_item = QListWidgetItem(p["name"])
            p_item.setData(Qt.ItemDataRole.UserRole, p["person_id"])
            p_item.setFlags(p_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            p_item.setCheckState(Qt.CheckState.Unchecked)
            self._person_list.addItem(p_item)
        
        self._username_edit.clear()
        self._username_edit.setReadOnly(False)
        self._pin_edit.clear()
        self._pin_edit.setPlaceholderText("4–8 digits")
        self._can_upload_cb.setChecked(False)
        self._current_user_id = None

    def _add_user(self):
        from ..utils.helpers import hash_pin
        username = self._username_edit.text().strip()
        pin = self._pin_edit.text().strip()
        role = self._role_combo.currentText()
        can_upload = 1 if self._can_upload_cb.isChecked() else 0
        
        if not username:
            QMessageBox.warning(self, "Input Required", "Username is required.")
            return

        existing = self.db.get_user(username)
        
        if not existing and not pin:
            QMessageBox.warning(self, "Input Required", "PIN is required for new users.")
            return

        if pin:
            if not pin.isdigit() or not (4 <= len(pin) <= 8):
                QMessageBox.warning(self, "Invalid PIN", "PIN must be 4–8 digits.")
                return
        if existing:
            if pin:
                self.db.update_user(username, pin_hash=hash_pin(pin), role=role, can_upload=can_upload)
            else:
                self.db.update_user(username, role=role, can_upload=can_upload)
            
            user_id = existing["user_id"]
            self.db.remove_user_permissions(user_id)
            if role == "viewer":
                for idx in range(self._person_list.count()):
                    p_item = self._person_list.item(idx)
                    if p_item.checkState() == Qt.CheckState.Checked:
                        self.db.add_user_permission(user_id, p_item.data(Qt.ItemDataRole.UserRole))
            
            QMessageBox.information(self, "Updated", f"User '{username}' updated.")
        else:
            user_id = self.db.create_user(username, hash_pin(pin), role, can_upload=can_upload)
            if role == "viewer":
                for idx in range(self._person_list.count()):
                    p_item = self._person_list.item(idx)
                    if p_item.checkState() == Qt.CheckState.Checked:
                        self.db.add_user_permission(user_id, p_item.data(Qt.ItemDataRole.UserRole))
            QMessageBox.information(self, "Added", f"User '{username}' created.")
        self._refresh()

    def _del_user(self):
        item = self._user_list.currentItem()
        if not item:
            return
        username = item.data(Qt.ItemDataRole.UserRole)
        if QMessageBox.question(self, "Delete User",
                                f"Delete user '{username}'?") == QMessageBox.StandardButton.Yes:
            self.db.delete_user(username)
            self._refresh()


# ─────────────────────────────────────────────────────────────────────────────
# Settings Dialog
# ─────────────────────────────────────────────────────────────────────────────

class SettingsDialog(QDialog):
    """App settings: web server port, binding, etc."""

    def __init__(self, db, app_core, parent=None):
        super().__init__(parent)
        self.db = db
        self.app_core = app_core
        self.setWindowTitle("⚙ Settings")
        self.setMinimumWidth(420)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        hdr = QLabel("Application Settings")
        hdr.setStyleSheet("font-size:18px;font-weight:700;color:#e2e8f0;")
        layout.addWidget(hdr)

        grp = QGroupBox("Web Server")
        grp.setStyleSheet(
            "QGroupBox{color:#94a3b8;border:1px solid #2d3748;border-radius:8px;"
            "margin-top:8px;padding:12px;font-size:13px;}"
            "QGroupBox::title{subcontrol-origin:margin;left:10px;color:#94a3b8;}")
        form = QFormLayout(grp)

        self._port_spin = QSpinBox()
        self._port_spin.setRange(1024, 65535)
        self._port_spin.setValue(self.app_core.get_web_port())
        self._port_spin.setStyleSheet(
            "QSpinBox{background:#252840;color:#e2e8f0;border:1px solid #2d3748;"
            "border-radius:6px;padding:7px;}")

        self._bind_all_cb = QCheckBox("Bind to all interfaces (LAN access)")
        self._bind_all_cb.setChecked(self.app_core.get_web_bind_all())
        self._bind_all_cb.setStyleSheet("color:#e2e8f0;")

        form.addRow("Port:", self._port_spin)
        form.addRow(self._bind_all_cb)
        layout.addWidget(grp)

        # Maintenance Section
        maint_grp = QGroupBox("Maintenance")
        maint_grp.setStyleSheet(grp.styleSheet())
        maint_layout = QVBoxLayout(maint_grp)
        
        maint_info = QLabel("Dangerous actions. Use with caution.")
        maint_info.setStyleSheet("color: #94a3b8; font-size: 11px; margin-bottom: 4px;")
        maint_layout.addWidget(maint_info)

        btn_reset = QPushButton("🗑 Reset Project (Clear All Data)")
        btn_reset.setStyleSheet(
            "QPushButton{background:#ef4444;color:#fff;border:none;"
            "border-radius:6px;padding:8px;font-size:12px;font-weight:600;}"
            "QPushButton:hover{background:#dc2626;}")
        btn_reset.clicked.connect(self._reset_click)
        maint_layout.addWidget(btn_reset)
        layout.addWidget(maint_grp)

        btn_row = QHBoxLayout()
        btn_save = QPushButton("💾 Save")
        btn_close = QPushButton("Cancel")
        for b, color in [(btn_save, "#6c63ff"), (btn_close, "#2d3748")]:
            b.setStyleSheet(
                f"QPushButton{{background:{color};color:#fff;border:none;"
                f"border-radius:8px;padding:9px 20px;font-size:13px;font-weight:600;}}")
        btn_save.clicked.connect(self._save)
        btn_close.clicked.connect(self.reject)
        btn_row.addWidget(btn_save)
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

    def _reset_click(self):
        ans = QMessageBox.warning(
            self, "Confirm Reset",
            "<p align='center'><b>Are you sure you want to RESET the project?</b></p>"
            "<p>This will permanently delete all indexed photos, face data, and person names from the database.</p>"
            "<p><i>Your actual photo files on disk will NOT be deleted.</i></p>",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if ans == QMessageBox.StandardButton.Yes:
            self.app_core.reset_project()
            QMessageBox.information(self, "Reset Complete", "Project data has been cleared. The app will now refresh.")
            self.accept()
            if self.parent() and hasattr(self.parent(), "refresh_ui"):
                self.parent().refresh_ui()

    def _save(self):
        self.app_core.set_web_port(self._port_spin.value())
        self.app_core.set_web_bind_all(self._bind_all_cb.isChecked())
        QMessageBox.information(self, "Saved",
            "Settings saved. Restart the web server for changes to take effect.")
        self.accept()


# ─────────────────────────────────────────────────────────────────────────────
# Person Face Manager Dialog
# ─────────────────────────────────────────────────────────────────────────────

class PersonFaceManagerDialog(QDialog):
    """View and remove faces assigned to a person."""
    def __init__(self, db, person_id, parent=None):
        super().__init__(parent)
        self.db = db
        self.person_id = person_id
        self._selected_faces = set()
        
        person = self.db.get_person(person_id)
        self.person_name = person["name"] if person else "Unknown"
        
        self.setWindowTitle(f"Manage Faces: {self.person_name}")
        self.setMinimumSize(700, 500)
        self._build_ui()
        self._load_faces()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        hdr = QLabel(f"Faces assigned to <b>{self.person_name}</b>")
        hdr.setStyleSheet("font-size:16px;")
        layout.addWidget(hdr)

        info = QLabel("Select faces that were assigned incorrectly and click 'Remove Selected'. "
                      "They will become 'Unknown' again.")
        info.setStyleSheet("font-size:12px;")
        info.setWordWrap(True)
        layout.addWidget(info)

        # Scrollable area for faces
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._inner = QWidget()
        self._inner_layout = QGridLayout(self._inner)
        self._inner_layout.setSpacing(10)
        self._inner_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._scroll.setWidget(self._inner)
        layout.addWidget(self._scroll)

        # Bottom buttons
        bottom = QHBoxLayout()
        self._btn_remove = QPushButton("🗑 Unmap Selected")
        self._btn_remove.setEnabled(False)
        self._btn_remove.setStyleSheet(
            "QPushButton{background:#4a5568; color:#fff; border:none; border-radius:8px; padding:10px 18px; font-weight:600;}"
            "QPushButton:disabled{opacity:.4;}")
        self._btn_remove.clicked.connect(self._remove_selected)
        
        self._btn_delete = QPushButton("🗑 Delete (Not a Face)")
        self._btn_delete.setEnabled(False)
        self._btn_delete.setStyleSheet(
            "QPushButton{background:#e53e3e; color:#fff; border:none; border-radius:8px; padding:10px 18px; font-weight:600;}"
            "QPushButton:disabled{opacity:.4;}")
        self._btn_delete.clicked.connect(self._delete_selected)
        
        btn_close = QPushButton("Close")
        btn_close.setStyleSheet(
            "QPushButton{background:#2d3748; color:#fff; border:none; border-radius:8px; padding:10px 20px;}")
        btn_close.clicked.connect(self.accept)

        bottom.addWidget(self._btn_remove)
        bottom.addWidget(self._btn_delete)
        bottom.addStretch()
        bottom.addWidget(btn_close)
        layout.addLayout(bottom)

    def _load_faces(self):
        # Clear existing
        for i in reversed(range(self._inner_layout.count())):
            self._inner_layout.itemAt(i).widget().deleteLater()
        
        self._selected_faces.clear()
        self._btn_remove.setEnabled(False)

        faces = self.db.get_faces_for_person(self.person_id)
        COLS = 6
        for idx, row in enumerate(faces):
            from .widgets import FaceThumbnailWidget
            fw = FaceThumbnailWidget(row["face_id"], row["face_thumb"], size=100)
            fw.selected.connect(self._on_face_selected)
            r, c = divmod(idx, COLS)
            self._inner_layout.addWidget(fw, r, c)
        
        if not faces:
            empty = QLabel("No faces assigned to this person.")
            empty.setStyleSheet("color:#4a5568; font-style:italic;")
            self._inner_layout.addWidget(empty, 0, 0)

    def _on_face_selected(self, face_id: int):
        if face_id in self._selected_faces:
            self._selected_faces.remove(face_id)
        else:
            self._selected_faces.add(face_id)
        
        enabled = len(self._selected_faces) > 0
        self._btn_remove.setEnabled(enabled)
        self._btn_delete.setEnabled(enabled)
        
        txt_idx = f" ({len(self._selected_faces)})" if enabled else ""
        self._btn_remove.setText(f"🗑 Unmap{txt_idx}")
        self._btn_delete.setText(f"🗑 Delete{txt_idx}")

    def _remove_selected(self):
        count = len(self._selected_faces)
        ans = QMessageBox.question(
            self, "Confirm Removal",
            f"Are you sure you want to remove {count} face(s) from {self.person_name}?\n\n"
            "They will return to the 'Unknown Faces' pool.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if ans == QMessageBox.StandardButton.Yes:
            for fid in self._selected_faces:
                self.db.unmap_face(fid)
            QMessageBox.information(self, "Success", f"Removed {count} face(s) from {self.person_name}.")
            self._load_faces()

    def _delete_selected(self):
        count = len(self._selected_faces)
        ans = QMessageBox.question(
            self, "Confirm Deletion",
            f"Are you sure you want to PERMANENTLY DELETE {count} detections?\n\n"
            "Only do this if they are not actually faces (e.g., dress patterns, shadows).\n"
            "This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if ans == QMessageBox.StandardButton.Yes:
            for fid in self._selected_faces:
                self.db.delete_face(fid)
            QMessageBox.information(self, "Success", f"Deleted {count} false detection(s).")
            self._load_faces()


class KeyboardShortcutsDialog(QDialog):
    """Show a list of available keyboard shortcuts."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("⌨ Keyboard Shortcuts")
        self.setMinimumWidth(400)
        self.setStyleSheet("background:#0f1117; color:#e2e8f0;")
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        hdr = QLabel("Application Shortcuts")
        hdr.setStyleSheet("font-size:18px; font-weight:700; color:#6c63ff;")
        layout.addWidget(hdr)

        grid = QGridLayout()
        grid.setSpacing(10)
        
        shortcuts = [
            ("Ctrl + O", "Scan Photo Folders"),
            ("Ctrl + N", "Name Unknown Faces"),
            ("Ctrl + A", "Select All Photos"),
            ("Esc", "Clear Selection"),
            ("F5 / Ctrl + R", "Refresh View"),
            ("Ctrl + W", "Start Web Server"),
            ("Ctrl + Shift + W", "Stop Web Server"),
            ("Ctrl + B", "Open in Browser"),
            ("Ctrl + U", "Manage Users"),
            ("Ctrl + ,", "Preferences / Settings"),
            ("F1", "About FaceGallery"),
            ("Alt + F4", "Exit Application"),
        ]

        for i, (key, desc) in enumerate(shortcuts):
            k_lbl = QLabel(key)
            k_lbl.setStyleSheet("font-weight:700; color:#e2e8f0; background:#2d3748; padding:4px 8px; border-radius:4px;")
            d_lbl = QLabel(desc)
            d_lbl.setStyleSheet("color:#94a3b8;")
            grid.addWidget(k_lbl, i, 0)
            grid.addWidget(d_lbl, i, 1)

        layout.addLayout(grid)
        layout.addStretch()

        btn_close = QPushButton("Close")
        btn_close.setStyleSheet(
            "QPushButton{background:#6c63ff; color:#fff; border:none; border-radius:8px; padding:10px 20px; font-weight:600;}")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close, alignment=Qt.AlignmentFlag.AlignRight)
