"""
Main application window for FaceGallery.
"""

import os
import sys
import webbrowser
from typing import List, Optional, Set

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer
from PyQt6.QtGui import (QAction, QIcon, QPixmap, QColor,
                          QPalette, QKeySequence)
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QScrollArea, QPushButton, QComboBox, QFileDialog, QMessageBox,
    QStatusBar, QToolBar, QSplitter, QListWidget, QListWidgetItem,
    QFrame, QLineEdit, QGridLayout, QSizePolicy, QApplication, QCheckBox
)

from .widgets import PhotoThumbnailWidget, FaceThumbnailWidget, bytes_to_pixmap
from .dialogs import (
    ScanDialog, FaceNameDialog, UserManagementDialog, SettingsDialog,
    PersonFaceManagerDialog, KeyboardShortcutsDialog
)
from ..utils.helpers import get_local_ip


# ─────────────────────────────────────────────────────────────────────────────
# Thumbnail loader thread
# ─────────────────────────────────────────────────────────────────────────────

class ThumbLoader(QThread):
    loaded = pyqtSignal(int, bytes)  # photo_id, thumb_bytes

    def __init__(self, db, photo_ids: List[int]):
        super().__init__()
        self._db = db
        self._photo_ids = photo_ids

    def run(self):
        for pid in self._photo_ids:
            row = self._db.fetchone(
                "SELECT thumbnail FROM photos WHERE photo_id=?", (pid,))
            if row and row["thumbnail"]:
                self.loaded.emit(pid, bytes(row["thumbnail"]))


# ─────────────────────────────────────────────────────────────────────────────
# Main Window
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self, db, app_core):
        super().__init__()
        self.db = db
        self.app_core = app_core
        self._selected_photos: Set[int] = set()
        self._current_person_id: Optional[int] = None
        self._web_running = False
        self._thumb_widgets: dict = {}

        self.setWindowTitle("FaceGallery")
        self.setWindowIcon(QIcon("resources/icons/app-logo.png"))
        self.setMinimumSize(1100, 680)
        self._build_ui()
        self._apply_theme(self.app_core.get_theme())
        self._update_status_bar()

    # ─────────────────────────────────────────────────────── #
    # Theme                                                   #
    # ─────────────────────────────────────────────────────── #

    def _apply_theme(self, theme: str):
        self.app_core.set_theme(theme)
        is_dark = theme == "dark"
        
        # Base colors
        bg = "#0f1117" if is_dark else "#f1f5f9"
        card = "#1a1d2e" if is_dark else "#ffffff"
        text = "#e2e8f0" if is_dark else "#0f172a"
        sub = "#94a3b8" if is_dark else "#64748b"
        border = "#2d3748" if is_dark else "#cbd5e1"
        accent = "#6c63ff" if is_dark else "#4f46e5"
        item_hover = "#252840" if is_dark else "#e2e8f0"

        self.setStyleSheet(f"""
        QMainWindow, QWidget {{
            background: {bg};
            color: {text};
            font-family: 'Segoe UI', system-ui, sans-serif;
        }}
        QMenuBar {{
            background: {card};
            color: {text};
            border-bottom: 1px solid {border};
            padding: 2px 4px;
        }}
        QMenuBar::item {{
            background: transparent;
            padding: 6px 12px;
            margin: 2px;
            border-radius: 4px;
        }}
        QMenuBar::item:selected {{ background: {accent}; color: #fff; }}
        QMenu {{
            background: {card};
            color: {text};
            border: 1px solid {border};
            padding: 5px;
        }}
        QMenu::item {{
            padding: 6px 28px 6px 12px;
            margin: 2px;
            border-radius: 4px;
        }}
        QMenu::item:selected {{ background: {accent}; color: #fff; }}
        QMenu::separator {{ height: 1px; background: {border}; margin: 5px 10px; }}
        QToolBar {{
            background: {card};
            border-bottom: 1px solid {border};
            spacing: 6px;
            padding: 4px 8px;
        }}
        QStatusBar {{
            background: {card};
            color: {sub};
            border-top: 1px solid {border};
            font-size: 12px;
        }}
        QScrollBar:vertical {{
            background: {card}; width: 8px;
        }}
        QScrollBar::handle:vertical {{
            background: {border}; border-radius: 4px; min-height: 20px;
        }}
        QScrollBar:horizontal {{
            background: {card}; height: 8px;
        }}
        QScrollBar::handle:horizontal {{
            background: {border}; border-radius: 4px;
        }}
        QPushButton#toolbar-btn {{
            background: {item_hover}; color: {text}; border: 1px solid {border};
            border-radius: 8px; padding: 7px 16px; font-size: 13px;
        }}
        QPushButton#toolbar-btn:hover {{
            background: {accent}; border-color: {accent}; color: #fff;
        }}
        QFrame#PhotoThumb {{
            background: {item_hover}; border-radius: 8px; border: 1px solid {border};
        }}
        QFrame#PhotoThumb:hover {{ border-color: {accent}; }}
        QFrame#PhotoThumb[selected="true"] {{ border: 3px solid {accent}; }}
        
        QFrame#FaceThumb {{
            background: {item_hover}; border: 1px solid {border};
        }}
        QFrame#FaceThumb:hover {{ border-color: {accent}; }}
        QFrame#FaceThumb[selected="true"] {{ border: 3px solid {accent}; }}
        
        QDialog {{ background: {bg}; color: {text}; }}
        QListWidget {{ 
            background: {card}; border: 1px solid {border}; border-radius: 8px; 
            color: {text}; font-size: 13px; outline: none;
        }}
        QListWidget::item {{ padding: 8px; border-radius: 4px; }}
        QListWidget::item:selected {{ background: {accent}; color: #fff; }}
        QGroupBox {{
            color: {sub}; border: 1px solid {border}; border-radius: 8px;
            margin-top: 10px; padding: 12px; font-size: 13px;
        }}
        QGroupBox::title {{ subcontrol-origin: margin; left: 10px; color: {sub}; }}
        QLineEdit, QComboBox, QSpinBox {{
            background: {item_hover}; color: {text}; border: 1px solid {border};
            border-radius: 6px; padding: 7px 12px; font-size: 13px;
        }}
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus {{ border-color: {accent}; }}
        QCheckBox {{ font-size: 13px; color: {text}; }}
        QCheckBox::indicator {{ width: 18px; height: 18px; border: 1px solid {border}; border-radius: 4px; background: {card}; }}
        QCheckBox::indicator:checked {{ background: {accent}; border-color: {accent}; }}
        """)
        QApplication.instance().setStyleSheet(self.styleSheet())
        
        # After stylesheet change, we might need to refresh parts of the UI
        # if they have hardcoded styles. We'll refresh the whole UI.
        if hasattr(self, "_person_list"):
            self._refresh_ui_styles(theme)

    # ─────────────────────────────────────────────────────── #
    # UI construction                                         #
    # ─────────────────────────────────────────────────────── #

    def _build_ui(self):
        self._build_menubar()
        self._build_toolbar()

        # Central widget = splitter (sidebar | content)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet("QSplitter::handle{background:transparent;}")

        # ── Left sidebar (person list) ── #
        self._sidebar = QFrame()
        self._sidebar.setFixedWidth(220)
        sb_layout = QVBoxLayout(self._sidebar)
        sb_layout.setContentsMargins(0, 0, 0, 0)
        sb_layout.setSpacing(0)

        self._sb_hdr = QLabel("  People")
        sb_layout.addWidget(self._sb_hdr)

        self._person_list = QListWidget()
        self._person_list.currentRowChanged.connect(self._on_person_selected)
        self._person_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._person_list.customContextMenuRequested.connect(self._on_person_context_menu)
        sb_layout.addWidget(self._person_list)
        splitter.addWidget(self._sidebar)

        # ── Right content area ── #
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(16, 12, 16, 12)
        content_layout.setSpacing(12)

        # Toolbar within content
        bar = QHBoxLayout()
        self._title_lbl = QLabel("All Photos")
        self._title_lbl.setStyleSheet("font-size:18px;font-weight:700;")
        self._count_lbl = QLabel("")
        self._count_lbl.setStyleSheet("font-size:13px;")
        bar.addWidget(self._title_lbl)
        bar.addWidget(self._count_lbl)
        bar.addStretch()

        self._only_groups_cb = QCheckBox("Only Group Photos")
        self._only_groups_cb.setStyleSheet("QCheckBox{font-size:13px; margin-right:10px;}")
        
        self._only_solos_cb = QCheckBox("Only Solo Photos")
        self._only_solos_cb.setStyleSheet("QCheckBox{font-size:13px; margin-right:10px;}")

        def on_group_change(state):
            if state == 2: # Checked
                self._only_solos_cb.blockSignals(True)
                self._only_solos_cb.setChecked(False)
                self._only_solos_cb.blockSignals(False)
            self._load_photos()

        def on_solo_change(state):
            if state == 2: # Checked
                self._only_groups_cb.blockSignals(True)
                self._only_groups_cb.setChecked(False)
                self._only_groups_cb.blockSignals(False)
            self._load_photos()

        self._only_groups_cb.stateChanged.connect(on_group_change)
        self._only_solos_cb.stateChanged.connect(on_solo_change)
        
        bar.addWidget(self._only_groups_cb)
        bar.addWidget(self._only_solos_cb)

        self._export_btn = QPushButton("⬇ Export Selected")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._export_selected)
        self._select_all_btn = QPushButton("Select All")
        self._select_all_btn.clicked.connect(self._select_all)
        self._clear_sel_btn = QPushButton("Clear")
        self._clear_sel_btn.clicked.connect(self._clear_selection)

        bar.addWidget(self._select_all_btn)
        bar.addWidget(self._clear_sel_btn)
        bar.addWidget(self._export_btn)
        content_layout.addLayout(bar)

        # Photo grid (scrollable)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._grid_widget = QWidget()
        self._grid_layout = QGridLayout(self._grid_widget)
        self._grid_layout.setSpacing(10)
        self._scroll.setWidget(self._grid_widget)
        content_layout.addWidget(self._scroll)

        splitter.addWidget(content)
        splitter.setSizes([220, 880])
        self.setCentralWidget(splitter)

        self._status = QStatusBar()
        self.setStatusBar(self._status)

        # Load data
        self._refresh_persons()
        self._load_photos()

    def _build_menubar(self):
        mb = self.menuBar()

        # ── FILE ──
        file_menu = mb.addMenu("File")
        a_scan = QAction("📁 Scan Photo Folders…     ", self)
        a_scan.setShortcut(QKeySequence("Ctrl+O"))
        a_scan.triggered.connect(self._open_scan)
        a_reset = QAction("🗑 Clear All Data…", self)
        a_reset.triggered.connect(self._reset_project_click)
        a_exit = QAction("Exit                       ", self)
        a_exit.setShortcut(QKeySequence("Alt+F4"))
        a_exit.triggered.connect(self.close)
        file_menu.addAction(a_scan)
        file_menu.addAction(a_reset)
        file_menu.addSeparator()
        file_menu.addAction(a_exit)

        # ── FACES ──
        face_menu = mb.addMenu("Faces")
        a_name = QAction("👥 Name Unknown Faces…     ", self)
        a_name.setShortcut(QKeySequence("Ctrl+N"))
        a_name.triggered.connect(self._open_face_naming)
        face_menu.addAction(a_name)

        # ── WEB ──
        web_menu = mb.addMenu("Web")
        self._a_start_web = QAction("🌐 Start Web Server        ", self)
        self._a_start_web.setShortcut(QKeySequence("Ctrl+W"))
        self._a_stop_web = QAction("⏹ Stop Web Server         ", self)
        self._a_stop_web.setShortcut(QKeySequence("Ctrl+Shift+W"))
        a_open_web = QAction("🔗 Open in Browser         ", self)
        a_open_web.setShortcut(QKeySequence("Ctrl+B"))
        self._a_start_web.triggered.connect(self._start_web_server)
        self._a_stop_web.triggered.connect(self._stop_web_server)
        self._a_stop_web.setEnabled(False)
        a_open_web.triggered.connect(self._open_browser)
        web_menu.addAction(self._a_start_web)
        web_menu.addAction(self._a_stop_web)
        web_menu.addSeparator()
        web_menu.addAction(a_open_web)

        # ── USERS ──
        user_menu = mb.addMenu("Users")
        a_users = QAction("👤 Manage Web Users…       ", self)
        a_users.setShortcut(QKeySequence("Ctrl+U"))
        a_users.triggered.connect(self._open_user_mgmt)
        user_menu.addAction(a_users)

        # ── VIEW ──
        view_menu = mb.addMenu("View")
        a_refresh = QAction("🔄 Refresh View            ", self)
        a_refresh.setShortcuts([QKeySequence("F5"), QKeySequence("Ctrl+R")])
        a_refresh.triggered.connect(self._refresh_all)
        view_menu.addAction(a_refresh)
        view_menu.addSeparator()
        
        theme_menu = view_menu.addMenu("Theme")
        a_light = QAction("☀️ Light Mode", self)
        a_dark = QAction("🌙 Dark Mode", self)
        a_light.triggered.connect(lambda: self._apply_theme("light"))
        a_dark.triggered.connect(lambda: self._apply_theme("dark"))
        theme_menu.addAction(a_light)
        theme_menu.addAction(a_dark)
        
        # ── EDIT ──
        edit_menu = mb.addMenu("Edit")
        a_sel_all = QAction("Select All                 ", self)
        a_sel_all.setShortcut(QKeySequence("Ctrl+A"))
        a_sel_all.triggered.connect(self._select_all)
        a_clear_sel = QAction("Clear Selection            ", self)
        a_clear_sel.setShortcut(QKeySequence("Esc"))
        a_clear_sel.triggered.connect(self._clear_selection)
        edit_menu.addAction(a_sel_all)
        edit_menu.addAction(a_clear_sel)

        # ── SETTINGS ──
        settings_menu = mb.addMenu("Settings")
        a_settings = QAction("⚙ Preferences…             ", self)
        a_settings.setShortcut(QKeySequence("Ctrl+,"))
        a_settings.triggered.connect(self._open_settings)
        settings_menu.addAction(a_settings)

        # ── HELP ──
        help_menu = mb.addMenu("Help")
        a_shortcuts = QAction("⌨ Keyboard Shortcuts      ", self)
        a_shortcuts.setShortcut(QKeySequence("Alt+K"))
        a_shortcuts.triggered.connect(self._open_shortcuts)
        a_about = QAction("ℹ About FaceGallery       ", self)
        a_about.setShortcut(QKeySequence("F1"))
        a_about.triggered.connect(self._on_about)
        help_menu.addAction(a_shortcuts)
        help_menu.addAction(a_about)

    def _build_toolbar(self):
        tb = QToolBar("Main Toolbar")
        tb.setIconSize(QSize(20, 20))
        tb.setMovable(False)
        self.addToolBar(tb)

        def _tb_btn(text: str, slot, tip: str = "") -> QPushButton:
            btn = QPushButton(text)
            btn.setToolTip(tip)
            btn.clicked.connect(slot)
            # Use a class selector or dynamic update. 
            # Better to use a common stylesheet rule for toolbar buttons in _apply_theme
            btn.setObjectName("toolbar-btn")
            return btn

        tb.addWidget(_tb_btn("📁 Scan",  self._open_scan,      "Scan photo folders"))
        tb.addWidget(_tb_btn("👥 Name Faces", self._open_face_naming, "Name unknown faces"))
        tb.addSeparator()
        self._web_btn = _tb_btn("🌐 Start Web", self._toggle_web_server,
                                "Start / stop the local web server")
        tb.addWidget(self._web_btn)
        tb.addWidget(_tb_btn("🔗 Browser", self._open_browser, "Open web UI in browser"))
        tb.addSeparator()
        tb.addWidget(_tb_btn("🔄 Refresh",  self._refresh_all,  "Refresh photo list"))

    # ─────────────────────────────────────────────────────── #
    # Data loading                                            #
    # ─────────────────────────────────────────────────────── #

    def _refresh_persons(self):
        self._person_list.blockSignals(True)
        self._person_list.clear()

        all_item = QListWidgetItem("🖼  All Photos")
        all_item.setData(Qt.ItemDataRole.UserRole, None)
        self._person_list.addItem(all_item)

        for p in self.db.get_all_persons():
            photo_count = len(self.db.get_photos_for_person(p["person_id"]))
            item = QListWidgetItem(f"👤  {p['name']}   ({photo_count})")
            item.setData(Qt.ItemDataRole.UserRole, p["person_id"])
            self._person_list.addItem(item)

        self._person_list.setCurrentRow(0)
        self._person_list.blockSignals(False)

    def _on_person_selected(self, row: int):
        item = self._person_list.item(row)
        if item is None:
            return
        self._current_person_id = item.data(Qt.ItemDataRole.UserRole)
        self._load_photos()

    def _on_person_context_menu(self, pos):
        item = self._person_list.itemAt(pos)
        if not item:
            return
        
        person_id = item.data(Qt.ItemDataRole.UserRole)
        if person_id is None: # "All Photos" item
            return

        from PyQt6.QtWidgets import QMenu
        menu = QMenu(self)
        a_manage = menu.addAction("👤 Manage Faces (Fix Mistakes)")
        a_rename = menu.addAction("✏ Rename Person")
        a_delete = menu.addAction("🗑 Delete Person")
        
        action = menu.exec(self._person_list.mapToGlobal(pos))
        if action == a_manage:
            self._manage_faces_click(person_id)
        elif action == a_rename:
            self._rename_person_click(person_id)
        elif action == a_delete:
            self._delete_person_click(person_id)

    def _manage_faces_click(self, person_id):
        dlg = PersonFaceManagerDialog(self.db, person_id, self)
        dlg.exec()
        self._refresh_all()

    def _rename_person_click(self, person_id):
        person = self.db.get_person(person_id)
        if not person: return
        
        from PyQt6.QtWidgets import QInputDialog
        new_name, ok = QInputDialog.getText(self, "Rename Person", "Enter new name:", 
                                           text=person["name"])
        if ok and new_name.strip():
            self.db.update_person(person_id, name=new_name.strip())
            self._refresh_persons()

    def _delete_person_click(self, person_id):
        person = self.db.get_person(person_id)
        if not person: return
        
        ans = QMessageBox.question(
            self, "Delete Person",
            f"Are you sure you want to delete '{person['name']}'?\n\n"
            "This will only remove the person's name and identity. "
            "The photos and detected faces will remain in the database as 'Unknown'.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if ans == QMessageBox.StandardButton.Yes:
            self.db.delete_person(person_id)
            self._refresh_persons()
            self._load_photos()

    def _load_photos(self, preserve_selection: bool = False):
        # Clear grid
        for i in reversed(range(self._grid_layout.count())):
            item = self._grid_layout.itemAt(i)
            if item and item.widget():
                item.widget().deleteLater()
        
        self._thumb_widgets.clear()
        if not preserve_selection:
            self._selected_photos.clear()
            self._export_btn.setEnabled(False)
            self._export_btn.setText("⬇ Export Selected")
        else:
            # We keep the set, but some IDs might no longer exist in the new view.
            # We'll filter them during widget creation.
            pass

        pid = [self._current_person_id] if self._current_person_id else None
        only_groups = self._only_groups_cb.isChecked()
        only_solos = self._only_solos_cb.isChecked()
        photos = self.app_core.get_photos(pid, only_groups=only_groups, only_solos=only_solos)

        name = "All Photos"
        if self._current_person_id:
            p = self.db.get_person(self._current_person_id)
            name = p["name"] if p else "Person"
        self._title_lbl.setText(name)
        self._count_lbl.setText(f"  {len(photos)} photo(s)")

        COLS = 5
        for idx, photo in enumerate(photos):
            ph_id = photo["photo_id"]
            thumb_data = bytes(photo["thumbnail"]) if photo["thumbnail"] else None
            widget = PhotoThumbnailWidget(ph_id, thumb_data, size=168)
            widget.selected.connect(self._on_photo_selected)
            
            # Restore visual selection state
            if ph_id in self._selected_photos:
                widget.set_selected(True)
                
            self._thumb_widgets[ph_id] = widget
            row, col = divmod(idx, COLS)
            self._grid_layout.addWidget(widget, row, col)

        # Stretch spacer
        self._grid_layout.setRowStretch(len(photos) // COLS + 1, 1)

    def _on_photo_selected(self, photo_id: int):
        if photo_id in self._selected_photos:
            self._selected_photos.discard(photo_id)
        else:
            self._selected_photos.add(photo_id)
        self._export_btn.setEnabled(bool(self._selected_photos))
        self._export_btn.setText(
            f"⬇ Export ({len(self._selected_photos)})" if self._selected_photos
            else "⬇ Export Selected")

    def _select_all(self):
        for ph_id, widget in self._thumb_widgets.items():
            widget.set_selected(True)
            self._selected_photos.add(ph_id)
        self._export_btn.setEnabled(bool(self._selected_photos))
        self._export_btn.setText(f"⬇ Export ({len(self._selected_photos)})")

    def _clear_selection(self):
        for widget in self._thumb_widgets.values():
            widget.set_selected(False)
        self._selected_photos.clear()
        self._export_btn.setEnabled(False)
        self._export_btn.setText("⬇ Export Selected")

    # ─────────────────────────────────────────────────────── #
    # Actions                                                 #
    # ─────────────────────────────────────────────────────── #

    def _open_scan(self):
        dlg = ScanDialog(self.db, self.app_core, self)
        dlg.exec()
        self._refresh_all()

    def _open_face_naming(self):
        dlg = FaceNameDialog(self.db, self.app_core, self)
        dlg.exec()
        self._refresh_all()

    def _open_user_mgmt(self):
        dlg = UserManagementDialog(self.db, self)
        dlg.exec()

    def _open_settings(self):
        dlg = SettingsDialog(self.db, self.app_core, self)
        dlg.exec()

    def _open_shortcuts(self):
        dlg = KeyboardShortcutsDialog(self)
        dlg.exec()

    def _on_about(self):
        msg = QMessageBox(self)
        msg.setWindowTitle("About FaceGallery")
        logo = QPixmap("resources/icons/app-logo.png").scaled(64, 64, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        msg.setIconPixmap(logo)
        msg.setText(
            "<h2>FaceGallery v1.0</h2>"
            "<p><b>Author:</b> Dipta Roy</p>"
            "<hr>"
            "<p>FaceGallery is a cross-platform, offline desktop photo manager "
            "designed for privacy-conscious users. It automatically indexes your "
            "local photos, detects faces using AI, and clusters them "
            "for easy organization.</p>"
            "<p>It also features a built-in local web server for sharing the "
            "gallery with other devices on your local network.</p>"
            "<p><i>Built with Python, PyQt6, and SQLite.</i></p>"
        )
        msg.exec()

    def _reset_project_click(self):
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
            QMessageBox.information(self, "Reset Complete", "Project data has been cleared.")
            self.refresh_ui()

    def _export_selected(self):
        if not self._selected_photos:
            return
        paths = []
        for photo_id in self._selected_photos:
            row = self.db.fetchone(
                "SELECT path FROM photos WHERE photo_id=?", (photo_id,))
            if row:
                paths.append(row["path"])

        choice = QMessageBox.question(
            self, "Export Photos",
            f"Export {len(paths)} photo(s).\n\nChoose destination type:",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No |
            QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes
        )
        if choice == QMessageBox.StandardButton.Cancel:
            return

        if choice == QMessageBox.StandardButton.Yes:
            # Copy to folder
            dest = QFileDialog.getExistingDirectory(self, "Select Export Folder")
            if dest:
                count = self.app_core.export_photos_to_folder(paths, dest)
                QMessageBox.information(self, "Export Complete",
                                         f"✅ {count} photos copied to:\n{dest}")
        else:
            # Save ZIP
            zip_dest, _ = QFileDialog.getSaveFileName(
                self, "Save ZIP As", "facegallery_export.zip",
                "ZIP Archive (*.zip)")
            if zip_dest:
                import shutil
                tmp = self.app_core.export_photos_to_zip(paths)
                shutil.move(tmp, zip_dest)
                QMessageBox.information(self, "Export Complete",
                                         f"✅ ZIP saved to:\n{zip_dest}")

    def refresh_ui(self):
        """Public alias for refreshing all data in the window."""
        self._refresh_all()

    def _refresh_all(self):
        # Preserve selection across refresh
        old_sel = set(self._selected_photos)
        self._refresh_persons()
        self._selected_photos = old_sel
        # _load_photos normally clears selection, but we can bypass it
        self._load_photos(preserve_selection=True)

    def _refresh_ui_styles(self, theme: str):
        is_dark = theme == "dark"
        card = "#1a1d2e" if is_dark else "#ffffff"
        text = "#e2e8f0" if is_dark else "#0f172a"
        sub = "#94a3b8" if is_dark else "#64748b"
        border = "#2d3748" if is_dark else "#cbd5e1"
        accent = "#6c63ff" if is_dark else "#4f46e5"
        item_hover = "#252840" if is_dark else "#e2e8f0"
        bg = "#0f1117" if is_dark else "#f1f5f9"

        if hasattr(self, "_sidebar"):
            self._sidebar.setStyleSheet(f"QFrame{{background:{card};border-right:1px solid {border};}}")
        if hasattr(self, "_person_list"):
            self._person_list.setStyleSheet(f"""
                QListWidget {{ background: transparent; border: none; color: {text}; }}
                QListWidget::item {{ padding: 10px 12px; border-bottom: 1px solid {border}; }}
                QListWidget::item:hover {{ background: {item_hover}; }}
                QListWidget::item:selected {{ background: {accent}; color: #fff; }}
            """)
        if hasattr(self, "_title_lbl"):
            self._title_lbl.setStyleSheet(f"font-size:18px;font-weight:700;color:{text};")
        if hasattr(self, "_count_lbl"):
            self._count_lbl.setStyleSheet(f"color:{sub};font-size:13px;")
        if hasattr(self, "_sb_hdr"):
            self._sb_hdr.setStyleSheet(f"font-size:13px;font-weight:700;color:{sub};padding:14px 12px 8px;")
        if hasattr(self, "_scroll"):
            self._scroll.setStyleSheet(f"QScrollArea{{border:none;background:{bg};}}")
        if hasattr(self, "_grid_widget"):
            self._grid_widget.setStyleSheet(f"background:{bg};")
            
        # Action Buttons
        btn_base = f"border-radius:8px;padding:8px 14px;font-size:13px;font-weight:600;"
        if hasattr(self, "_export_btn"):
            self._export_btn.setStyleSheet(f"""
                QPushButton {{ background: {accent}; color: #fff; border: none; {btn_base} }}
                QPushButton:hover {{ background: {accent}cc; }}
                QPushButton:disabled {{ opacity: 0.4; }}
            """)
        if hasattr(self, "_select_all_btn"):
             self._select_all_btn.setStyleSheet(f"""
                QPushButton {{ background: {item_hover}; color: {text}; border: 1px solid {border}; {btn_base} }}
                QPushButton:hover {{ background: {accent}; color: #fff; }}
            """)
        if hasattr(self, "_clear_sel_btn"):
             self._clear_sel_btn.setStyleSheet(f"""
                QPushButton {{ background: {item_hover}; color: {text}; border: 1px solid {border}; {btn_base} }}
                QPushButton:hover {{ background: #e53e3e; color: #fff; }}
            """)

    def closeEvent(self, event):
        """Shutdown web server on app exit."""
        if self._web_running:
            self._stop_web_server()
        super().closeEvent(event)

    # ─────────────────────────────────────────────────────── #
    # Web server                                              #
    # ─────────────────────────────────────────────────────── #

    def _start_web_server(self):
        from ..web.server import start_server
        port = self.app_core.get_web_port()
        bind_all = self.app_core.get_web_bind_all()
        start_server(self.db, self.app_core, port=port, bind_all=bind_all)
        self._web_running = True
        self._web_btn.setText("⏹ Stop Web")
        self._a_start_web.setEnabled(False)
        self._a_stop_web.setEnabled(True)
        ip = get_local_ip()
        self._update_status_bar()
        QMessageBox.information(
            self, "Web Server Started",
            f"🌐 Web server is running.\n\n"
            f"Local:   http://localhost:{port}\n"
            f"Network: http://{ip}:{port}\n\n"
            f"Share the Network URL with others on your Wi-Fi."
        )

    def _stop_web_server(self):
        from ..web.server import stop_server
        stop_server()
        self._web_running = False
        self._web_btn.setText("🌐 Start Web")
        self._a_start_web.setEnabled(True)
        self._a_stop_web.setEnabled(False)
        self._update_status_bar()

    def _toggle_web_server(self):
        if self._web_running:
            self._stop_web_server()
        else:
            self._start_web_server()

    def _open_browser(self):
        port = self.app_core.get_web_port()
        webbrowser.open(f"http://localhost:{port}")

    # ─────────────────────────────────────────────────────── #
    # Status bar                                              #
    # ─────────────────────────────────────────────────────── #

    def _update_status_bar(self):
        total_photos = self.db.fetchone("SELECT COUNT(*) AS c FROM photos")
        total_faces = self.db.fetchone("SELECT COUNT(*) AS c FROM faces")
        total_persons = self.db.fetchone("SELECT COUNT(*) AS c FROM persons")
        web_status = (f"🌐 Web: http://localhost:{self.app_core.get_web_port()}"
                      if self._web_running else "🌐 Web: offline")
        self._status.showMessage(
            f"📷 {total_photos['c'] if total_photos else 0} photos   "
            f"👤 {total_persons['c'] if total_persons else 0} people   "
            f"😊 {total_faces['c'] if total_faces else 0} faces   │   {web_status}"
        )
