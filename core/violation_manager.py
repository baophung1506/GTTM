"""
===== FILE: core/violation_manager.py =====

ViolationManager - Phat hien va xu ly 7 loai vi pham giao thong:
1. Vuot den do (red light)
2. De vach dung (stop line)
3. Di sai lan (wrong lane)
4. Di nguoc chieu (wrong way)
5. Khong doi mu bao hiem (no helmet)
6. Dung do sai quy dinh (illegal parking)
7. Quay dau sai noi quy dinh (illegal U-turn)

Nhan tin hieu tracking_ready tu TrackingManager, kiem tra tung Vehicle
voi cac quy tac vi pham, luu anh + tao ban ghi Violation,
chuyen cho DatabaseManager / OCRManager / EmailManager.
"""

import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

from core.config_manager import ConfigManager
from core.database_manager import DatabaseManager
from core.email_manager import EmailManager
from core.logger_manager import LoggerManager
from core.ocr_manager import OCRManager
from models.traffic_light_state import TrafficLightController, TrafficLightState
from models.vehicle import Detection, Vehicle, VehicleClass
from models.violation import Violation, ViolationType


# =====================================================================
# Ham tien ich hinh hoc
# =====================================================================

def _point_in_polygon(point: Tuple[float, float], polygon: List[List[float]]) -> bool:
    """Kiem tra diem co nam trong da giac (polygon) hay khong (ray casting)."""
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i][0], polygon[i][1]
        xj, yj = polygon[j][0], polygon[j][1]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi):
            inside = not inside
        j = i
    return inside


def _segments_intersect(
    p1: Tuple[float, float], p2: Tuple[float, float],
    p3: Tuple[float, float], p4: Tuple[float, float],
) -> bool:
    """Kiem tra hai doan thang [p1,p2] va [p3,p4] co giao nhau khong."""
    def _cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    d1 = _cross(p3, p4, p1)
    d2 = _cross(p3, p4, p2)
    d3 = _cross(p1, p2, p3)
    d4 = _cross(p1, p2, p4)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    return False


def _angle_diff(a: float, b: float) -> float:
    """Tinh chenh lech goc (do) giua hai huong, trong [-180, 180]."""
    diff = (a - b + 180) % 360 - 180
    return diff


def _trajectory_has_uturn(trajectory: List[Tuple[float, float]], min_reversal_angle: float = 120.0) -> bool:
    """
    Kiem tra xem quy dao co chua hanh dong quay dau hay khong.
    Neu huong di chuyen thay doi >= min_reversal_angle do -> quay dau.
    """
    if len(trajectory) < 5:
        return False

    vectors = []
    for i in range(1, len(trajectory)):
        dx = trajectory[i][0] - trajectory[i - 1][0]
        dy = trajectory[i][1] - trajectory[i - 1][1]
        if abs(dx) < 1 and abs(dy) < 1:
            continue
        angle = float(np.degrees(np.arctan2(dy, dx)))
        vectors.append(angle)

    if len(vectors) < 4:
        return False

    # Chia vectors thanh nua dau va nua sau, so sanh trung binh
    half = len(vectors) // 2
    first_angles = vectors[:half]
    last_angles = vectors[half:]

    avg_first = float(np.mean(first_angles))
    avg_last = float(np.mean(last_angles))

    return abs(_angle_diff(avg_last, avg_first)) >= min_reversal_angle


# =====================================================================
# ViolationManager
# =====================================================================
class ViolationManager(QObject):
    """
    Kiem tra tung Vehicle voi 7 quy tac vi pham.
    Hoat dong trong GUI thread (duoc goi truc tiep tu slot nhan
    tin hieu tracking_ready), khong phai QThread rieng biet.
    """

    violation_detected = pyqtSignal(object)  # phat ra Violation

    # Thoi gian cho giua 2 lan phat hien vi pham cung loai cua cung xe (giay)
    COOLDOWN_SECONDS = 10.0

    def __init__(
        self,
        config_manager: ConfigManager,
        database_manager: DatabaseManager,
        ocr_manager: OCRManager,
        email_manager: EmailManager,
        traffic_light_controller: TrafficLightController,
        logger: LoggerManager,
        parent=None,
    ):
        super().__init__(parent)

        self._cfg = config_manager
        self._db = database_manager
        self._ocr = ocr_manager
        self._email = email_manager
        self._tl = traffic_light_controller
        self._logger = logger

        # (track_id, ViolationType) -> timestamp lan vi pham cuoi
        self._cooldown_map: Dict[Tuple[int, ViolationType], float] = {}

        self._stop_line: List[List[float]] = []
        self._lanes: List[dict] = []
        self._wrong_way_zones: List[dict] = []
        self._forbidden_parking_zones: List[dict] = []
        self._forbidden_uturn_zones: List[dict] = []

        self._violation_images_dir = ""
        self._reload_config()

        # Ket noi OCR callback
        self._ocr.plate_recognized.connect(self._on_plate_recognized)

        # Map tam: placeholder_id -> Violation (cho OCR async)
        self._pending_violations: Dict[int, Violation] = {}
        self._pending_counter = 0

    def reload_config(self) -> None:
        """Tai lai tat ca cau hinh khi nguoi dung thay doi settings."""
        self._reload_config()

    def _reload_config(self) -> None:
        self._stop_line = self._cfg.get_stop_line()
        self._lanes = self._cfg.get_lanes()
        self._wrong_way_zones = self._cfg.get_wrong_way_zones()
        self._forbidden_parking_zones = self._cfg.get_forbidden_parking_zones()
        self._forbidden_uturn_zones = self._cfg.get_forbidden_uturn_zones()
        self._violation_images_dir = self._cfg.get_violation_images_dir()

    # ------------------------------------------------------------
    # Entry point chinh: duoc goi tu slot cua MainWindow
    # ------------------------------------------------------------
    def process_frame(
        self,
        frame_payload: dict,
        vehicles: List[Vehicle],
        helmet_dets: List[Detection],
        frame: np.ndarray,
    ) -> None:
        """
        Kiem tra toan bo danh sach vehicles trong frame hien tai.
        frame la anh goc (BGR numpy array) de cat anh vi pham.
        """
        current_time = frame_payload.get("timestamp", time.time())
        frame_number = frame_payload.get("frame_number", 0)

        for vehicle in vehicles:
            self._check_vehicle(vehicle, frame, frame_number, current_time)

    def _check_vehicle(
        self,
        vehicle: Vehicle,
        frame: np.ndarray,
        frame_number: int,
        current_time: float,
    ) -> None:
        """Kiem tra mot vehicle voi tat ca 7 loai vi pham."""
        # ---- 1. Vuot den do ----
        if self._tl.is_red():
            if self._check_red_light(vehicle):
                self._flag(vehicle, ViolationType.RED_LIGHT_VIOLATION, frame, frame_number, current_time)

        # ---- 2. De vach dung ----
        if self._check_stop_line(vehicle):
            self._flag(vehicle, ViolationType.STOP_LINE_VIOLATION, frame, frame_number, current_time)

        # ---- 3. Di sai lan ----
        wrong_lane = self._check_wrong_lane(vehicle)
        if wrong_lane:
            self._flag(vehicle, ViolationType.WRONG_LANE, frame, frame_number, current_time)

        # ---- 4. Di nguoc chieu ----
        if self._check_wrong_way(vehicle):
            self._flag(vehicle, ViolationType.WRONG_WAY, frame, frame_number, current_time)

        # ---- 5. Khong doi mu bao hiem ----
        if self._check_no_helmet(vehicle):
            self._flag(vehicle, ViolationType.NO_HELMET, frame, frame_number, current_time)

        # ---- 6. Dung do sai quy dinh ----
        if self._check_illegal_parking(vehicle, current_time):
            self._flag(vehicle, ViolationType.ILLEGAL_PARKING, frame, frame_number, current_time)

        # ---- 7. Quay dau sai quy dinh ----
        if self._check_illegal_uturn(vehicle):
            self._flag(vehicle, ViolationType.ILLEGAL_U_TURN, frame, frame_number, current_time)

    # ================================================================
    # Cac phuong thuc kiem tra vi pham
    # ================================================================

    def _check_red_light(self, vehicle: Vehicle) -> bool:
        """Xe co vuot vach dung trong luc den do khong?"""
        if not vehicle.crossed_stop_line:
            return False
        return True

    def _check_stop_line(self, vehicle: Vehicle) -> bool:
        """Kiem tra xe co cat qua vach dung hay khong."""
        if len(self._stop_line) < 2:
            return False

        traj = vehicle.get_trajectory()
        if len(traj) < 2:
            return False

        p3 = (self._stop_line[0][0], self._stop_line[0][1])
        p4 = (self._stop_line[1][0], self._stop_line[1][1])

        # Kiem tra cac doan lien tiep trong quy dao
        for i in range(len(traj) - 1):
            if _segments_intersect(traj[i], traj[i + 1], p3, p4):
                vehicle.crossed_stop_line = True
                return True

        return False

    def _check_wrong_lane(self, vehicle: Vehicle) -> bool:
        """Xe co di sai lan quy dinh khong?"""
        if not self._lanes:
            return False

        cx, cy = vehicle.center_point
        for lane in self._lanes:
            polygon = lane.get("polygon", [])
            allowed_classes: List[str] = lane.get("allowed_classes", [])

            if not polygon or not allowed_classes:
                continue

            if _point_in_polygon((cx, cy), polygon):
                if vehicle.class_name not in allowed_classes:
                    return True
                return False

        return False

    def _check_wrong_way(self, vehicle: Vehicle) -> bool:
        """Xe co di nguoc chieu trong vung cam di nguoc chieu khong?"""
        if not self._wrong_way_zones:
            return False

        cx, cy = vehicle.center_point
        direction = vehicle.get_direction_degrees()
        if direction is None:
            return False

        speed = vehicle.get_speed_pixels_per_frame()
        if speed < 2.0:
            return False

        for zone in self._wrong_way_zones:
            polygon = zone.get("polygon", [])
            allowed_deg = zone.get("allowed_direction_degrees", 0)
            tolerance = zone.get("tolerance_degrees", 30)

            if not polygon:
                continue

            if _point_in_polygon((cx, cy), polygon):
                diff = abs(_angle_diff(direction, allowed_deg))
                if diff > tolerance:
                    return True

        return False

    def _check_no_helmet(self, vehicle: Vehicle) -> bool:
        """Nguoi di xe may co khong doi mu bao hiem khong?"""
        if vehicle.class_name != VehicleClass.MOTORCYCLE.value:
            return False
        return vehicle.has_helmet is False

    def _check_illegal_parking(self, vehicle: Vehicle, current_time: float) -> bool:
        """Xe co dung/do qua lau trong vung cam dung do khong?"""
        if not self._forbidden_parking_zones:
            return False

        cx, cy = vehicle.center_point
        stationary_secs = vehicle.get_stationary_duration(current_time)

        for zone in self._forbidden_parking_zones:
            polygon = zone.get("polygon", [])
            max_sec = zone.get("max_stationary_seconds", 30)

            if not polygon:
                continue

            if _point_in_polygon((cx, cy), polygon):
                if stationary_secs >= max_sec:
                    return True

        return False

    def _check_illegal_uturn(self, vehicle: Vehicle) -> bool:
        """Xe co quay dau trong vung cam quay dau khong?"""
        if not self._forbidden_uturn_zones:
            return False

        cx, cy = vehicle.center_point
        traj = vehicle.get_trajectory()

        for zone in self._forbidden_uturn_zones:
            polygon = zone.get("polygon", [])
            if not polygon:
                continue

            if _point_in_polygon((cx, cy), polygon):
                if _trajectory_has_uturn(list(traj)):
                    return True

        return False

    # ================================================================
    # Ghi nhan va xu ly vi pham
    # ================================================================

    def _flag(
        self,
        vehicle: Vehicle,
        vtype: ViolationType,
        frame: np.ndarray,
        frame_number: int,
        current_time: float,
    ) -> None:
        """
        Kiem tra cooldown, tao ban ghi Violation, luu anh,
        gui vao Database / OCR / Email.
        """
        key = (vehicle.track_id, vtype)
        last_time = self._cooldown_map.get(key, 0.0)
        if (current_time - last_time) < ViolationManager.COOLDOWN_SECONDS:
            return  # con trong cooldown -> bo qua

        self._cooldown_map[key] = current_time
        vehicle.mark_violation(vtype.value)

        # Cat anh vi pham
        image_path = self._save_violation_image(vehicle, vtype, frame, frame_number)

        violation = Violation.create(
            vehicle_type=vehicle.class_name,
            track_id=vehicle.track_id,
            violation_type=vtype,
            image_path=image_path,
            frame_number=frame_number,
        )

        # Luu vao database
        self._db.insert_violation_async(violation)

        # Gui cho OCR de nhan bien so (bat dong bo)
        if image_path and os.path.exists(image_path):
            self._pending_counter += 1
            placeholder_id = self._pending_counter
            self._pending_violations[placeholder_id] = violation
            vehicle_crop = self._crop_vehicle(vehicle, frame)
            if vehicle_crop is not None:
                self._ocr.enqueue_ocr_task(
                    placeholder_id=placeholder_id,
                    vehicle_image=vehicle_crop,
                    output_dir=self._violation_images_dir,
                    file_prefix=f"plate_{vehicle.track_id}_{frame_number}",
                )

        # Gui email canh bao
        self._email.enqueue_violation(violation)

        # Phat tin hieu len GUI
        self.violation_detected.emit(violation)

        self._logger.log_violation("ViolationManager", 
            f"[{vtype.value}] Track#{vehicle.track_id} "
            f"({vehicle.class_name}) frame#{frame_number}"
        )

    def _save_violation_image(
        self,
        vehicle: Vehicle,
        vtype: ViolationType,
        frame: np.ndarray,
        frame_number: int,
    ) -> str:
        """Cat va luu anh vung vi pham, tra ve duong dan file."""
        os.makedirs(self._violation_images_dir, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{vtype.code}_{vehicle.track_id}_{timestamp_str}.jpg"
        filepath = os.path.join(self._violation_images_dir, filename)

        try:
            h_frame, w_frame = frame.shape[:2]
            x1, y1, x2, y2 = vehicle.bbox
            pad = 20
            x1c = max(0, int(x1) - pad)
            y1c = max(0, int(y1) - pad)
            x2c = min(w_frame, int(x2) + pad)
            y2c = min(h_frame, int(y2) + pad)
            crop = frame[y1c:y2c, x1c:x2c]
            if crop.size > 0:
                cv2.imwrite(filepath, crop)
                return filepath
        except Exception as exc:
            self._logger.log_error("ViolationManager", f"Khong the luu anh vi pham: {exc}")
        return ""

    def _crop_vehicle(self, vehicle: Vehicle, frame: np.ndarray) -> Optional[np.ndarray]:
        """Cat phan anh chua phuong tien de gui cho OCR."""
        try:
            h_frame, w_frame = frame.shape[:2]
            x1, y1, x2, y2 = vehicle.bbox
            x1c = max(0, int(x1))
            y1c = max(0, int(y1))
            x2c = min(w_frame, int(x2))
            y2c = min(h_frame, int(y2))
            crop = frame[y1c:y2c, x1c:x2c]
            if crop.size > 0:
                return crop
        except Exception:
            pass
        return None

    def _on_plate_recognized(self, placeholder_id: int, license_plate, plate_image_path: str) -> None:
        """Nhan ket qua OCR va cap nhat ban ghi Violation trong DB."""
        violation = self._pending_violations.pop(placeholder_id, None)
        if violation is None:
            return

        if license_plate is not None and license_plate.is_valid_format:
            self._db.update_plate_async(
                violation.violation_id,
                str(license_plate),
                plate_image_path,
            )

    def reset(self) -> None:
        """Xoa lich su cooldown khi he thong reset."""
        self._cooldown_map.clear()
        self._pending_violations.clear()


