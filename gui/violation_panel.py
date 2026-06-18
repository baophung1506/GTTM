"""
===== FILE: gui/violation_panel.py =====
Panel danh sach vi pham - hien thi, tim kiem va loc cac vi pham
da duoc luu trong database.
"""

from typing import Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (QComboBox, QGroupBox, QHBoxLayout, QHeaderView,
                              QLabel, QLineEdit, QPushButton, QTableWidget,
                              QTableWidgetItem, QVBoxLayout, QWidget)

from core.database_manager import DatabaseManager
from models.violation import Violation, ViolationType


# Mau sac theo loai vi pham
VIOLATION_ROW_COLORS = {
    ViolationType.RED_LIGHT_VIOLATION.value:     "#3a0a0a",
    ViolationType.STOP_LINE_VIOLATION.value:      "#3a1a0a",
    ViolationType.WRONG_LANE.value:     "#1a1a3a",
    ViolationType.WRONG_WAY.value:      "#2a0a2a",
    ViolationType.NO_HELMET.value:      "#2a2a0a",
    ViolationType.ILLEGAL_PARKING.value:"#0a2a2a",
    ViolationType.ILLEGAL_U_TURN.value:  "#1a2a0a",
}

COLUMNS = ["ID", "Thời gian", "Loại vi phạm", "Loại xe", "Biển số", "Trạng thái email"]


class ViolationPanel(QGroupBox):
    """
    Bang danh sach vi pham voi chuc nang:
        - Hien thi theo thoi gian thuc (them dong khi co vi pham moi)
        - Tim kiem theo bien so
        - Loc theo loai vi pham
        - Click de xem anh vi pham (phat tin hieu len MainWindow)
    """

    violation_selected = pyqtSignal(object)  # Violation duoc chon

    def __init__(self, database_manager: DatabaseManager, parent=None):
        super().__init__("⚠️  Danh Sách Vi Phạm", parent)
        self._db = database_manager
        self._setup_ui()
        self._load_history()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        # -- Thanh tim kiem / loc --
        filter_row = QHBoxLayout()

        filter_row.addWidget(QLabel("Tìm biển số:"))
        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("Nhập biển số xe...")
        self._search_edit.setFixedWidth(160)
        self._search_edit.textChanged.connect(self._apply_filter)
        filter_row.addWidget(self._search_edit)

        filter_row.addWidget(QLabel("Lọc vi phạm:"))
        self._combo_type = QComboBox()
        self._combo_type.addItem("Tất cả", None)
        for vtype in ViolationType:
            self._combo_type.addItem(vtype.value, vtype.value)
        self._combo_type.setFixedWidth(200)
        self._combo_type.currentIndexChanged.connect(self._apply_filter)
        filter_row.addWidget(self._combo_type)

        self._btn_refresh = QPushButton("🔄 Làm mới")
        self._btn_refresh.clicked.connect(self._load_history)
        filter_row.addWidget(self._btn_refresh)

        self._btn_clear = QPushButton("🗑 Xóa tất cả")
        self._btn_clear.clicked.connect(self._clear_all)
        filter_row.addWidget(self._btn_clear)

        filter_row.addStretch(1)
        layout.addLayout(filter_row)

        # -- Bang --
        self._table = QTableWidget()
        self._table.setColumnCount(len(COLUMNS))
        self._table.setHorizontalHeaderLabels(COLUMNS)
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSortingEnabled(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setWordWrap(False)

        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)

        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self._table)

        # -- So luong --
        self._lbl_count = QLabel("Tổng: 0 vi phạm")
        self._lbl_count.setAlignment(Qt.AlignRight)
        layout.addWidget(self._lbl_count)

    # ------------------------------------------------------------------
    # API cong khai
    # ------------------------------------------------------------------

    def add_violation(self, violation: Violation) -> None:
        """Them mot vi pham moi len dau bang (real-time)."""
        self._table.setSortingEnabled(False)
        self._table.insertRow(0)
        self._populate_row(0, violation)
        self._table.setSortingEnabled(True)
        self._update_count()

    def _load_history(self) -> None:
        """Tai lai lich su vi pham tu database."""
        self._table.setSortingEnabled(False)
        self._table.setRowCount(0)

        plate_filter = self._search_edit.text().strip() or None
        type_filter = self._combo_type.currentData()

        if plate_filter:
            violations = self._db.search_by_plate(plate_filter)
        elif type_filter:
            violations = self._db.filter_violations(violation_type=type_filter)
        else:
            violations = self._db.load_history(limit=500)

        for v in violations:
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._populate_row(row, v)

        self._table.setSortingEnabled(True)
        self._update_count()

    def _apply_filter(self) -> None:
        self._load_history()

    def _populate_row(self, row: int, violation: Violation) -> None:
        """Dien du lieu vao mot hang cua bang."""
        values = [
            str(violation.violation_id or ""),
            violation.get_formatted_timestamp(),
            violation.violation_type,
            violation.vehicle_type,
            violation.license_plate or "—",
            violation.email_status,
        ]
        bg_color = QColor(VIOLATION_ROW_COLORS.get(violation.violation_type, "#1a1a2e"))

        for col, text in enumerate(values):
            item = QTableWidgetItem(text)
            item.setBackground(bg_color)
            if col == 2:
                item.setFont(QFont("Segoe UI", 11, QFont.Bold))
                item.setForeground(QColor("#ffcc80"))
            self._table.setItem(row, col, item)

        # Luu doi tuong Violation vao item de xem anh khi click
        id_item = self._table.item(row, 0)
        if id_item:
            id_item.setData(Qt.UserRole, violation)

    def _on_selection_changed(self) -> None:
        selected = self._table.selectedItems()
        if not selected:
            return
        row = self._table.currentRow()
        id_item = self._table.item(row, 0)
        if id_item:
            violation = id_item.data(Qt.UserRole)
            if violation:
                self.violation_selected.emit(violation)

    def _update_count(self) -> None:
        count = self._table.rowCount()
        self._lbl_count.setText(f"Tổng: {count} vi phạm")

    def _clear_all(self) -> None:
        self._table.setRowCount(0)
        self._update_count()

    def update_email_status(self, violation_id: int, status: str) -> None:
        """Cap nhat trang thai email cho hang co violation_id tuong ung."""
        for row in range(self._table.rowCount()):
            id_item = self._table.item(row, 0)
            if id_item and id_item.text() == str(violation_id):
                status_item = self._table.item(row, 5)
                if status_item:
                    status_item.setText(status)
                break
