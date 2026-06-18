"""
===== FILE: gui/traffic_light_panel.py =====
Panel kiem soat den tin hieu giao thong - cho phep nguoi dung
chuyen doi trang thai DEN DO / VANG / XANH bang tay.
"""

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QButtonGroup, QGroupBox, QHBoxLayout, QLabel,
                              QPushButton, QSpinBox, QVBoxLayout, QWidget)

from gui.styles import TRAFFIC_LIGHT_ACTIVE, TRAFFIC_LIGHT_INACTIVE
from models.traffic_light_state import TrafficLightController, TrafficLightState


class TrafficLightPanel(QGroupBox):
    """
    Panel hien thi va dieu khien trang thai den giao thong.
    Cho phep:
        - Click thu cong vao den de chuyen trang thai
        - Bat/tat che do tu dong (tu dong xoay vong)
    """

    state_changed = pyqtSignal(object)  # TrafficLightState

    def __init__(self, controller: TrafficLightController, parent=None):
        super().__init__("🚦  Đèn Giao Thông", parent)
        self._controller = controller
        self._auto_timer = QTimer(self)
        self._auto_timer.timeout.connect(self._auto_cycle)
        self._auto_running = False
        self._auto_sequence = [
            (TrafficLightState.RED, 30),
            (TrafficLightState.YELLOW, 5),
            (TrafficLightState.GREEN, 30),
            (TrafficLightState.YELLOW, 5),
        ]
        self._auto_index = 0
        self._setup_ui()

        # Lang nghe thay doi trang thai tu controller
        self._controller.register_observer(self._on_state_changed)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # -- 3 nut den --
        light_row = QHBoxLayout()
        light_row.setSpacing(14)
        light_row.addStretch(1)

        self._btn_red = QPushButton()
        self._btn_red.setObjectName("btnRed")
        self._btn_red.setCheckable(True)
        self._btn_red.setFixedSize(70, 70)
        self._btn_red.setToolTip("Đèn ĐỎ - Click để bật")
        self._btn_red.clicked.connect(lambda: self._manual_set(TrafficLightState.RED))

        self._btn_yellow = QPushButton()
        self._btn_yellow.setObjectName("btnYellow")
        self._btn_yellow.setCheckable(True)
        self._btn_yellow.setFixedSize(70, 70)
        self._btn_yellow.setToolTip("Đèn VÀNG - Click để bật")
        self._btn_yellow.clicked.connect(lambda: self._manual_set(TrafficLightState.YELLOW))

        self._btn_green = QPushButton()
        self._btn_green.setObjectName("btnGreen")
        self._btn_green.setCheckable(True)
        self._btn_green.setFixedSize(70, 70)
        self._btn_green.setToolTip("Đèn XANH - Click để bật")
        self._btn_green.clicked.connect(lambda: self._manual_set(TrafficLightState.GREEN))

        light_row.addWidget(self._btn_red)
        light_row.addWidget(self._btn_yellow)
        light_row.addWidget(self._btn_green)
        light_row.addStretch(1)
        layout.addLayout(light_row)

        # -- Nhan trang thai --
        self._lbl_state = QLabel("● ĐÈN ĐỎ")
        self._lbl_state.setAlignment(Qt.AlignCenter)
        self._lbl_state.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self._lbl_state.setStyleSheet("color: #ff5252;")
        layout.addWidget(self._lbl_state)

        # -- Che do tu dong --
        auto_row = QHBoxLayout()
        self._btn_auto = QPushButton("▶ Tự động")
        self._btn_auto.setCheckable(True)
        self._btn_auto.clicked.connect(self._toggle_auto)
        auto_row.addWidget(self._btn_auto)

        auto_row.addWidget(QLabel("Chu kỳ đỏ (s):"))
        self._spin_red = QSpinBox()
        self._spin_red.setRange(5, 120)
        self._spin_red.setValue(30)
        self._spin_red.setFixedWidth(60)
        auto_row.addWidget(self._spin_red)

        auto_row.addWidget(QLabel("Xanh (s):"))
        self._spin_green = QSpinBox()
        self._spin_green.setRange(5, 120)
        self._spin_green.setValue(30)
        self._spin_green.setFixedWidth(60)
        auto_row.addWidget(self._spin_green)

        layout.addLayout(auto_row)

        self._update_light_display(TrafficLightState.RED)

    def _manual_set(self, state: TrafficLightState) -> None:
        """Dat trang thai thu cong (dung tu dong neu dang chay)."""
        if self._auto_running:
            self._stop_auto()
        self._controller.set_state(state)

    def _on_state_changed(self, state: TrafficLightState) -> None:
        """Callback khi TrafficLightController thay doi trang thai."""
        self._update_light_display(state)
        self.state_changed.emit(state)

    def _update_light_display(self, state: TrafficLightState) -> None:
        """Cap nhat giao dien nut va nhan theo trang thai."""
        for btn, key in [
            (self._btn_red, "RED"),
            (self._btn_yellow, "YELLOW"),
            (self._btn_green, "GREEN"),
        ]:
            is_active = (
                (key == "RED" and state == TrafficLightState.RED) or
                (key == "YELLOW" and state == TrafficLightState.YELLOW) or
                (key == "GREEN" and state == TrafficLightState.GREEN)
            )
            btn.setChecked(is_active)
            style = TRAFFIC_LIGHT_ACTIVE[key] if is_active else TRAFFIC_LIGHT_INACTIVE[key]
            btn.setStyleSheet(style + " min-width:70px; min-height:70px; max-width:70px; max-height:70px;")

        state_labels = {
            TrafficLightState.RED:    ("● ĐÈN ĐỎ",   "#ff5252"),
            TrafficLightState.YELLOW: ("● ĐÈN VÀNG", "#ffd600"),
            TrafficLightState.GREEN:  ("● ĐÈN XANH", "#69f0ae"),
        }
        text, color = state_labels[state]
        self._lbl_state.setText(text)
        self._lbl_state.setStyleSheet(f"color: {color};")

    def _toggle_auto(self, checked: bool) -> None:
        if checked:
            self._start_auto()
        else:
            self._stop_auto()

    def _start_auto(self) -> None:
        self._auto_running = True
        self._auto_index = 0
        self._btn_auto.setText("⏹ Dừng tự động")
        self._run_auto_step()

    def _stop_auto(self) -> None:
        self._auto_running = False
        self._auto_timer.stop()
        self._btn_auto.setChecked(False)
        self._btn_auto.setText("▶ Tự động")

    def _run_auto_step(self) -> None:
        state, _ = self._auto_sequence[self._auto_index]
        # Lay thoi gian tu spin box cho do va xanh
        if state == TrafficLightState.RED:
            duration_ms = self._spin_red.value() * 1000
        elif state == TrafficLightState.GREEN:
            duration_ms = self._spin_green.value() * 1000
        else:
            duration_ms = 5000  # Vang mac dinh 5s

        self._controller.set_state(state)
        self._auto_timer.start(duration_ms)

    def _auto_cycle(self) -> None:
        self._auto_timer.stop()
        self._auto_index = (self._auto_index + 1) % len(self._auto_sequence)
        self._run_auto_step()

    def get_controller(self) -> TrafficLightController:
        return self._controller

    def cleanup(self) -> None:
        self._stop_auto()
        self._controller.unregister_observer(self._on_state_changed)
