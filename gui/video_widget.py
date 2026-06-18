"""
===== FILE: gui/video_widget.py =====

VideoWidget - Hien thi khung hinh video kem overlay bbox, track_id,
trang thai den tin hieu, FPS, so luong phuong tien.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from models.vehicle import Detection, Vehicle, VehicleClass
from models.traffic_light_state import TrafficLightState

# Mau sac cho tung loai phuong tien
CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    VehicleClass.MOTORCYCLE.value: (0, 200, 255),   # Vang-nhat
    VehicleClass.CAR.value:        (50, 220, 50),    # Xanh la
    VehicleClass.BUS.value:        (255, 150, 0),    # Cam
    VehicleClass.PERSON.value:     (200, 100, 255),  # Tim
    VehicleClass.HELMET.value:     (0, 255, 150),    # Xanh ngoc
    VehicleClass.NO_HELMET.value:  (0, 60, 255),     # Do
}

DEFAULT_COLOR = (180, 180, 180)
VIOLATION_COLOR = (0, 0, 255)


def _cv2_bgr_to_qimage(frame: np.ndarray) -> QImage:
    """Chuyen doi numpy BGR array (OpenCV) thanh QImage (PyQt5)."""
    h, w, ch = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)


class VideoWidget(QLabel):
    """
    Widget hien thi video va overlay giao thong.
    Nhan frame (numpy BGR) tu MainWindow va ve tat ca thong tin len do.
    """

    clicked = pyqtSignal(int, int)  # x, y (toa do tren frame goc)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(640, 480)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #0a0a1a; color: #e0e0e0;")
        self.setText("📷  Chọn video để bắt đầu giám sát")
        self.setFont(QFont("Segoe UI", 14))

        self._current_frame: Optional[np.ndarray] = None
        self._vehicles: List[Vehicle] = []
        self._helmet_dets: List[Detection] = []
        self._fps: float = 0.0
        self._frame_number: int = 0
        self._traffic_light_state: TrafficLightState = TrafficLightState.RED
        self._stop_line: List[List[float]] = []

    def update_frame(
        self,
        frame: np.ndarray,
        vehicles: List[Vehicle],
        helmet_dets: List[Detection],
        fps: float,
        frame_number: int,
        traffic_light_state: TrafficLightState,
        stop_line: Optional[List[List[float]]] = None,
    ) -> None:
        """Cap nhat du lieu va ve lai widget."""
        self._current_frame = frame.copy()
        self._vehicles = vehicles
        self._helmet_dets = helmet_dets
        self._fps = fps
        self._frame_number = frame_number
        self._traffic_light_state = traffic_light_state
        if stop_line is not None:
            self._stop_line = stop_line
        self._render()

    def _render(self) -> None:
        """Ve overlay len frame va hien thi."""
        if self._current_frame is None:
            return

        frame = self._current_frame.copy()
        self._draw_stop_line(frame)
        self._draw_vehicles(frame)
        self._draw_hud(frame)

        qimg = _cv2_bgr_to_qimage(frame)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.setPixmap(pixmap)

    def _draw_stop_line(self, frame: np.ndarray) -> None:
        """Ve vach dung."""
        if len(self._stop_line) < 2:
            return
        p1 = (int(self._stop_line[0][0]), int(self._stop_line[0][1]))
        p2 = (int(self._stop_line[1][0]), int(self._stop_line[1][1]))
        color = (0, 0, 255) if self._traffic_light_state == TrafficLightState.RED else (0, 255, 0)
        cv2.line(frame, p1, p2, color, 3)
        cv2.putText(frame, "VACH DUNG", p1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def _draw_vehicles(self, frame: np.ndarray) -> None:
        """Ve bbox, track_id va nhan loai cho tung vehicle."""
        for vehicle in self._vehicles:
            x1, y1, x2, y2 = [int(v) for v in vehicle.bbox]
            has_violation = len(vehicle.violations_flagged) > 0
            color = VIOLATION_COLOR if has_violation else CLASS_COLORS.get(vehicle.class_name, DEFAULT_COLOR)

            # Bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Nhan (label)
            label = vehicle.to_overlay_label()
            label_bg_y = max(y1 - 22, 0)
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
            cv2.rectangle(frame, (x1, label_bg_y), (x1 + text_size[0] + 4, y1), color, -1)
            text_color = (0, 0, 0) if sum(color) > 380 else (255, 255, 255)
            cv2.putText(frame, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 1)

            # Diem tam
            cx, cy = int(vehicle.center_point[0]), int(vehicle.center_point[1])
            cv2.circle(frame, (cx, cy), 3, color, -1)

            # Mui ten huong di chuyen
            mv = vehicle.movement_vector
            if mv is not None and (abs(mv[0]) + abs(mv[1])) > 3:
                arrow_end = (cx + int(mv[0] * 5), cy + int(mv[1] * 5))
                h, w = frame.shape[:2]
                if 0 <= arrow_end[0] < w and 0 <= arrow_end[1] < h:
                    cv2.arrowedLine(frame, (cx, cy), arrow_end, color, 2, tipLength=0.3)

    def _draw_hud(self, frame: np.ndarray) -> None:
        """Ve HUD: FPS, so luong xe, trang thai den."""
        h, w = frame.shape[:2]

        # Nen mo (semi-transparent) cho HUD
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 8), (260, 90), (10, 10, 30), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # FPS & Frame
        cv2.putText(frame, f"FPS: {self._fps:.1f}", (14, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 180), 2)
        cv2.putText(frame, f"Frame: {self._frame_number}", (14, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # So luong xe
        count = len(self._vehicles)
        cv2.putText(frame, f"Phuong tien: {count}", (14, 74),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # Trang thai den giao thong (goc tren phai)
        tl_color = self._traffic_light_state.to_color_rgb()
        tl_bgr = (tl_color[2], tl_color[1], tl_color[0])
        tl_label = self._traffic_light_state.value.upper()
        tl_text_size = cv2.getTextSize(tl_label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
        tl_x = w - tl_text_size[0] - 60
        cv2.circle(frame, (w - 30, 28), 18, tl_bgr, -1)
        cv2.circle(frame, (w - 30, 28), 18, (255, 255, 255), 1)
        cv2.putText(frame, tl_label, (tl_x, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, tl_bgr, 2)

    def mousePressEvent(self, event):
        """Chuyen doi toa do click thanh toa do tren frame goc."""
        if self._current_frame is None:
            return super().mousePressEvent(event)
        pixmap = self.pixmap()
        if pixmap is None:
            return super().mousePressEvent(event)

        pw, ph = pixmap.width(), pixmap.height()
        lw, lh = self.width(), self.height()
        offset_x = (lw - pw) // 2
        offset_y = (lh - ph) // 2
        click_x = event.x() - offset_x
        click_y = event.y() - offset_y

        if 0 <= click_x <= pw and 0 <= click_y <= ph:
            fh, fw = self._current_frame.shape[:2]
            frame_x = int(click_x * fw / pw)
            frame_y = int(click_y * fh / ph)
            self.clicked.emit(frame_x, frame_y)

        super().mousePressEvent(event)

    def clear_display(self) -> None:
        """Xoa man hinh ve trang thai ban dau."""
        self._current_frame = None
        self._vehicles = []
        self._helmet_dets = []
        self.clear()
        self.setText("📷  Chọn video để bắt đầu giám sát")
