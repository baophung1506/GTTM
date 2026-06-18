"""
===== FILE: models/vehicle.py =====

Module: Vehicle Model
Mo ta: Dinh nghia lop du lieu Vehicle (phuong tien duoc theo doi boi
       ByteTrack). Moi doi tuong Vehicle luu lai day du thong tin can
       thiet de ViolationManager phan tich hanh vi: vi tri, lich su di
       chuyen, vector chuyen dong, thoi gian xuat hien/mat dang...

       Ngoai ra cung dinh nghia VehicleClass (enum cac loai doi tuong
       YOLOv8 nhan dien duoc) va Detection (ket qua tho tu detection
       model truoc khi duoc gan track_id).
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, List, Optional, Tuple


class VehicleClass(str, Enum):
    """Cac lop doi tuong ma YOLOv8 duoc huan luyen de nhan dien."""

    MOTORCYCLE = "motorcycle"
    CAR = "car"
    BUS = "bus"
    PERSON = "person"
    HELMET = "helmet"
    NO_HELMET = "no_helmet"

    @classmethod
    def vehicle_classes(cls) -> List["VehicleClass"]:
        """Cac lop duoc xem la 'phuong tien' co the bi theo doi (track)."""
        return [cls.MOTORCYCLE, cls.CAR, cls.BUS]

    @classmethod
    def helmet_related_classes(cls) -> List["VehicleClass"]:
        return [cls.HELMET, cls.NO_HELMET]


BoundingBox = Tuple[float, float, float, float]  # (x1, y1, x2, y2)
Point2D = Tuple[float, float]


@dataclass
class Detection:
    """Ket qua tho tra ve tu DetectionManager (chua co track_id).

    Attributes:
        class_name: Ten lop doi tuong (vd: "car", "motorcycle"...).
        bbox: Bounding box dang (x1, y1, x2, y2) toa do pixel tren frame.
        confidence: Do tin cay cua detection (0.0 - 1.0).
        class_id: ID lop trong model YOLO (su dung de truyen vao ByteTrack).
    """

    class_name: str
    bbox: BoundingBox
    confidence: float
    class_id: int = -1

    @property
    def center(self) -> Point2D:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def to_xywh(self) -> Tuple[float, float, float, float]:
        """Chuyen bbox (x1,y1,x2,y2) -> (cx, cy, w, h)."""
        x1, y1, x2, y2 = self.bbox
        w = x2 - x1
        h = y2 - y1
        return (x1 + w / 2.0, y1 + h / 2.0, w, h)


@dataclass
class Vehicle:
    """Doi tuong phuong tien duoc theo doi qua nhieu frame boi ByteTrack.

    Day la "model" trung tam duoc TrackingManager duy tri va
    ViolationManager su dung de phan tich vi pham.

    Attributes:
        track_id: ID duy nhat do ByteTrack gan cho phuong tien.
        class_name: Loai phuong tien (motorcycle / car / bus / person).
        bbox: Bounding box hien tai (x1, y1, x2, y2).
        center_point: Tam bounding box hien tai (cx, cy).
        history_positions: Hang doi (deque) luu lai cac tam diem gan nhat,
            dung de tinh quy dao, vector chuyen dong, phat hien dung do...
        first_seen: Thoi diem (epoch second) lan dau phat hien.
        last_seen: Thoi diem (epoch second) lan cuoi cap nhat.
        movement_vector: Vector chuyen dong trung binh (dx, dy) duoc
            tinh tu history_positions, dung de phat hien di nguoc chieu
            va quay dau sai quy dinh.
        has_helmet: Trang thai non bao hiem (None = chua xac dinh,
            True = co non, False = khong co non) - chi ap dung cho
            motorcycle.
        stationary_since: Thoi diem (epoch second) phuong tien bat dau
            duoc xem la dung yen (None neu dang di chuyen).
        confidence: Do tin cay detection gan nhat.
        violations_flagged: Tap hop cac loai vi pham da duoc ghi nhan
            cho track nay (de tranh ghi trung lap lien tuc).
    """

    track_id: int
    class_name: str
    bbox: BoundingBox
    confidence: float = 0.0
    center_point: Point2D = field(default_factory=lambda: (0.0, 0.0))
    history_positions: Deque[Point2D] = field(default_factory=lambda: deque(maxlen=60))
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    movement_vector: Point2D = field(default_factory=lambda: (0.0, 0.0))
    has_helmet: Optional[bool] = None
    stationary_since: Optional[float] = None
    violations_flagged: set = field(default_factory=set)
    crossed_stop_line: bool = False
    last_lane_id: Optional[str] = None

    HISTORY_MAXLEN: int = 60
    STATIONARY_PIXEL_THRESHOLD: float = 3.0

    def __post_init__(self) -> None:
        if not self.history_positions:
            self.history_positions = deque(maxlen=self.HISTORY_MAXLEN)
        self.center_point = self._compute_center(self.bbox)
        self.history_positions.append(self.center_point)

    @staticmethod
    def _compute_center(bbox: BoundingBox) -> Point2D:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def update(self, bbox: BoundingBox, confidence: float = 0.0, timestamp: Optional[float] = None, class_name: Optional[str] = None, frame_number: int = 0) -> None:
        """Cap nhat trang thai phuong tien voi detection moi nhat.

        Args:
            bbox: Bounding box moi (x1, y1, x2, y2).
            confidence: Do tin cay detection.
            timestamp: Thoi diem cap nhat (mac dinh la thoi gian hien tai).
        """
        ts = timestamp if timestamp is not None else time.time()
        if class_name is not None:
            self.class_name = class_name
        new_center = self._compute_center(bbox)

        # Kiem tra dung yen: so sanh tam diem moi voi tam diem cu.
        if self.history_positions:
            prev_center = self.history_positions[-1]
            dist = math.hypot(new_center[0] - prev_center[0], new_center[1] - prev_center[1])
            if dist <= self.STATIONARY_PIXEL_THRESHOLD:
                if self.stationary_since is None:
                    self.stationary_since = ts
            else:
                self.stationary_since = None

        if class_name is not None:
            self.class_name = class_name
        self.bbox = bbox
        self.confidence = confidence
        self.center_point = new_center
        self.history_positions.append(new_center)
        self.last_seen = ts
        self._update_movement_vector()

    def _update_movement_vector(self) -> None:
        """Tinh vector chuyen dong trung binh tu N diem gan nhat.

        Vector duoc chuan hoa (don vi: pixel/frame trung binh) va dung de
        xac dinh huong di chuyen tong quat cua phuong tien - phuc vu phat
        hien "di nguoc chieu" va "quay dau sai quy dinh".
        """
        n = min(len(self.history_positions), 10)
        if n < 2:
            self.movement_vector = (0.0, 0.0)
            return

        points = list(self.history_positions)[-n:]
        dx_total = 0.0
        dy_total = 0.0
        for i in range(1, len(points)):
            dx_total += points[i][0] - points[i - 1][0]
            dy_total += points[i][1] - points[i - 1][1]

        count = len(points) - 1
        self.movement_vector = (dx_total / count, dy_total / count)

    def get_speed_pixels_per_frame(self) -> float:
        """Toc do (pixel/frame) dua tren movement_vector hien tai."""
        dx, dy = self.movement_vector
        return math.hypot(dx, dy)

    def get_direction_degrees(self) -> Optional[float]:
        """Goc huong di chuyen (do), 0 = sang phai, 90 = xuong duoi.

        Tra ve None neu phuong tien dung yen (vector qua nho).
        """
        speed = self.get_speed_pixels_per_frame()
        if speed < 0.3:
            return None
        dx, dy = self.movement_vector
        angle = math.degrees(math.atan2(dy, dx))
        return angle % 360.0

    def get_trajectory(self) -> List[Point2D]:
        """Tra ve danh sach cac diem quy dao da di qua."""
        return list(self.history_positions)

    def get_stationary_duration(self, current_time=None) -> float:
        """Thoi gian (giay) phuong tien dung yen lien tuc cho den hien tai.

        Tra ve 0.0 neu phuong tien dang di chuyen.
        """
        if self.stationary_since is None:
            return 0.0
        t = current_time if current_time is not None else time.time()
        return t - self.stationary_since

    def mark_violation(self, violation_type: str) -> bool:
        """Ghi nhan loai vi pham da xu ly cho track nay.

        Returns:
            True neu day la lan dau ghi nhan loai vi pham nay (chua ton tai
            trong violations_flagged), False neu da ton tai truoc do.
        """
        if violation_type in self.violations_flagged:
            return False
        self.violations_flagged.add(violation_type)
        return True

    def has_been_flagged(self, violation_type: str) -> bool:
        return violation_type in self.violations_flagged

    def to_overlay_label(self) -> str:
        """Chuoi nhan hien thi tren overlay video: 'ID:3 car'."""
        return f"ID:{self.track_id} {self.class_name}"

    def to_dict(self) -> dict:
        """Chuyen doi sang dict (phuc vu logging / debug)."""
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "bbox": self.bbox,
            "center_point": self.center_point,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "movement_vector": self.movement_vector,
            "has_helmet": self.has_helmet,
            "stationary_seconds": self.get_stationary_duration(),
        }




