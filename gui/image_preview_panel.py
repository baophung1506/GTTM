"""
===== FILE: gui/image_preview_panel.py =====
Panel xem truoc anh vi pham va bien so xe.
"""

import os

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWidgets import (QGroupBox, QLabel, QSizePolicy, QVBoxLayout,
                              QWidget)

from models.violation import Violation


class ImagePreviewPanel(QGroupBox):
    """
    Hien thi:
        - Anh cat vung vi pham (ben trai)
        - Anh bien so xe (ket qua OCR)
        - Van ban bien so + loai vi pham
    """

    def __init__(self, parent=None):
        super().__init__("🖼  Xem Ảnh Vi Phạm", parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Nhan tieu de vi pham
        self._lbl_violation_type = QLabel("Chọn vi phạm để xem ảnh")
        self._lbl_violation_type.setAlignment(Qt.AlignCenter)
        self._lbl_violation_type.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self._lbl_violation_type.setStyleSheet("color: #ffcc80;")
        self._lbl_violation_type.setWordWrap(True)
        layout.addWidget(self._lbl_violation_type)

        # Anh vi pham chinh
        self._img_violation = QLabel()
        self._img_violation.setAlignment(Qt.AlignCenter)
        self._img_violation.setMinimumHeight(180)
        self._img_violation.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._img_violation.setStyleSheet("background-color: #0a0a1a; border: 1px solid #3a3a5c; border-radius: 4px;")
        self._img_violation.setText("Chưa có ảnh")
        layout.addWidget(self._img_violation, 3)

        # Anh bien so
        self._lbl_plate_title = QLabel("Biển số xe:")
        self._lbl_plate_title.setFont(QFont("Segoe UI", 10, QFont.Bold))
        layout.addWidget(self._lbl_plate_title)

        self._img_plate = QLabel()
        self._img_plate.setAlignment(Qt.AlignCenter)
        self._img_plate.setFixedHeight(60)
        self._img_plate.setStyleSheet("background-color: #0a0a1a; border: 1px solid #3a3a5c; border-radius: 4px;")
        self._img_plate.setText("—")
        layout.addWidget(self._img_plate)

        # Van ban bien so
        self._lbl_plate_text = QLabel("—")
        self._lbl_plate_text.setAlignment(Qt.AlignCenter)
        self._lbl_plate_text.setFont(QFont("Courier New", 16, QFont.Bold))
        self._lbl_plate_text.setStyleSheet("color: #69f0ae; letter-spacing: 4px;")
        layout.addWidget(self._lbl_plate_text)

        # Thong tin them
        self._lbl_info = QLabel("")
        self._lbl_info.setAlignment(Qt.AlignCenter)
        self._lbl_info.setWordWrap(True)
        self._lbl_info.setStyleSheet("color: #a0a0c0; font-size: 11px;")
        layout.addWidget(self._lbl_info)

    def show_violation(self, violation: Violation) -> None:
        """Hien thi anh va thong tin cua mot vi pham."""
        # Tieu de
        self._lbl_violation_type.setText(f"⚠ {violation.violation_type}")

        # Anh vi pham
        if violation.image_path and os.path.exists(violation.image_path):
            pixmap = QPixmap(violation.image_path)
            if not pixmap.isNull():
                scaled = pixmap.scaled(
                    self._img_violation.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
                self._img_violation.setPixmap(scaled)
            else:
                self._img_violation.setText("Không tải được ảnh")
        else:
            self._img_violation.setText("Không có ảnh")
            self._img_violation.setPixmap(QPixmap())

        # Anh bien so
        if violation.plate_image_path and os.path.exists(violation.plate_image_path):
            plate_px = QPixmap(violation.plate_image_path)
            if not plate_px.isNull():
                scaled_plate = plate_px.scaled(
                    self._img_plate.width(), self._img_plate.height(),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation,
                )
                self._img_plate.setPixmap(scaled_plate)
            else:
                self._img_plate.setText("—")
        else:
            self._img_plate.setText("—")
            self._img_plate.setPixmap(QPixmap())

        # Bien so text
        plate_text = violation.license_plate or "Chưa nhận dạng"
        self._lbl_plate_text.setText(plate_text)

        # Thong tin them
        info = (
            f"Loại xe: {violation.vehicle_type}  |  "
            f"Track ID: {violation.track_id}  |  "
            f"{violation.get_formatted_timestamp()}"
        )
        self._lbl_info.setText(info)

    def clear(self) -> None:
        """Xoa noi dung hien thi."""
        self._lbl_violation_type.setText("Chọn vi phạm để xem ảnh")
        self._img_violation.setText("Chưa có ảnh")
        self._img_violation.setPixmap(QPixmap())
        self._img_plate.setText("—")
        self._img_plate.setPixmap(QPixmap())
        self._lbl_plate_text.setText("—")
        self._lbl_info.setText("")
