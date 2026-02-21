"""
PyQt6 GUI components – widgets shared across dialog windows.
"""

from typing import Optional
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QPixmap, QImage, QColor, QPainter, QBrush, QPen, QFont
from PyQt6.QtWidgets import (
    QLabel, QFrame, QWidget, QVBoxLayout, QHBoxLayout,
    QSizePolicy, QPushButton, QScrollArea
)


def bytes_to_pixmap(data: bytes, size: QSize = None) -> QPixmap:
    """Convert raw JPEG bytes to a QPixmap, optionally scaled."""
    pm = QPixmap()
    pm.loadFromData(data)
    if size and not pm.isNull():
        pm = pm.scaled(size, Qt.AspectRatioMode.KeepAspectRatio,
                       Qt.TransformationMode.SmoothTransformation)
    return pm


def make_circular_pixmap(data: bytes, diameter: int = 80) -> QPixmap:
    """Return a circular-cropped pixmap from JPEG bytes."""
    src = bytes_to_pixmap(data)
    if src.isNull():
        return _placeholder_pixmap(diameter)
    src = src.scaled(diameter, diameter,
                     Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                     Qt.TransformationMode.SmoothTransformation)
    result = QPixmap(diameter, diameter)
    result.fill(Qt.GlobalColor.transparent)
    painter = QPainter(result)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QBrush(src))
    painter.setPen(Qt.PenStyle.NoPen)
    x_off = (src.width() - diameter) // 2
    y_off = (src.height() - diameter) // 2
    painter.drawEllipse(0, 0, diameter, diameter)
    painter.end()
    return result


def _placeholder_pixmap(size: int = 80) -> QPixmap:
    pm = QPixmap(size, size)
    pm.fill(QColor("#6c63ff"))
    painter = QPainter(pm)
    painter.setPen(QColor("#ffffff"))
    # Ensure point size is greater than 0 to avoid QFont::setPointSize warnings
    font = QFont()
    font_size = max(1, size // 3)
    font.setPointSize(font_size)
    painter.setFont(font)
    painter.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, "👤")
    painter.end()
    return pm


class ClickableLabel(QLabel):
    """QLabel that emits a signal when clicked."""
    from PyQt6.QtCore import pyqtSignal
    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class PhotoThumbnailWidget(QFrame):
    """
    A thumbnail widget showing one photo.
    Emits 'selected' signal (photo_id) on click.
    """
    from PyQt6.QtCore import pyqtSignal
    selected = pyqtSignal(int)

    def __init__(self, photo_id: int, thumb_data: Optional[bytes],
                 parent=None, size: int = 180):
        super().__init__(parent)
        self._photo_id = photo_id
        self._selected = False
        self._size = size
        self.setFixedSize(size, size)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setObjectName("PhotoThumb")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._img_label = QLabel()
        self._img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img_label.setScaledContents(False)
        layout.addWidget(self._img_label)

        if thumb_data:
            pm = bytes_to_pixmap(thumb_data, QSize(size, size))
        else:
            pm = _placeholder_pixmap(size)
        self._img_label.setPixmap(pm)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._selected = not self._selected
            self._update_style()
            self.selected.emit(self._photo_id)
        super().mousePressEvent(event)

    def set_selected(self, state: bool):
        self._selected = state
        self._update_style()

    def _update_style(self):
        # We manually set state property for CSS selectors if needed,
        # but for simplicity, we'll just toggle a property and re-polish
        self.setProperty("selected", self._selected)
        self.style().unpolish(self)
        self.style().polish(self)

    @property
    def photo_id(self): return self._photo_id


class FaceThumbnailWidget(QFrame):
    """A small square widget showing a detected face."""
    from PyQt6.QtCore import pyqtSignal
    selected = pyqtSignal(int)  # face_id

    def __init__(self, face_id: int, thumb_data: Optional[bytes],
                 parent=None, size: int = 120):
        super().__init__(parent)
        self._face_id = face_id
        self._selected = False
        self.setFixedSize(size, size)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._radius = size // 2
        self.setObjectName("FaceThumb")
        self.setProperty("radius", self._radius)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        lbl = QLabel()
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if thumb_data:
            pm = make_circular_pixmap(thumb_data, size)
        else:
            pm = _placeholder_pixmap(size)
        lbl.setPixmap(pm)
        layout.addWidget(lbl)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._selected = not self._selected
            self._update_style()
            self.selected.emit(self._face_id)
        super().mousePressEvent(event)

    def set_selected(self, state: bool):
        self._selected = state
        self._update_style()

    def _update_style(self):
        self.setProperty("selected", self._selected)
        self.style().unpolish(self)
        self.style().polish(self)

    @property
    def face_id(self): return self._face_id
