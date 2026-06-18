"""
===== FILE: models/violation.py =====

Module: Violation Model
Mo ta: Dinh nghia lop du lieu Violation - dai dien cho mot ban ghi vi
       pham giao thong duoc he thong phat hien va luu vao SQLite.

       Cung dinh nghia ViolationType (enum 7 loai vi pham theo yeu cau
       de tai) va EmailStatus (enum trang thai gui email mo phong).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ViolationType(str, Enum):
    """7 loai vi pham giao thong he thong phai phat hien."""

    RED_LIGHT_VIOLATION = "Vuot den do"
    STOP_LINE_VIOLATION = "De vach dung"
    WRONG_LANE = "Di sai lan"
    WRONG_WAY = "Di nguoc chieu"
    NO_HELMET = "Khong doi mu bao hiem"
    ILLEGAL_PARKING = "Dung do sai quy dinh"
    ILLEGAL_U_TURN = "Quay dau sai noi quy dinh"

    @property
    def code(self) -> str:
        """Ma ngan (khong dau, dung cho ten file)."""
        mapping = {
            ViolationType.RED_LIGHT_VIOLATION: "RED_LIGHT",
            ViolationType.STOP_LINE_VIOLATION: "STOP_LINE",
            ViolationType.WRONG_LANE: "WRONG_LANE",
            ViolationType.WRONG_WAY: "WRONG_WAY",
            ViolationType.NO_HELMET: "NO_HELMET",
            ViolationType.ILLEGAL_PARKING: "ILLEGAL_PARKING",
            ViolationType.ILLEGAL_U_TURN: "ILLEGAL_U_TURN",
        }
        return mapping[self]


class EmailStatus(str, Enum):
    """Trang thai gui email mo phong cho vi pham."""

    PENDING = "Pending"
    SENT = "Email sent successfully"
    FAILED = "Email failed"


@dataclass
class Violation:
    """Ban ghi mot vi pham giao thong.

    Attributes:
        violation_id: ID duy nhat (None truoc khi luu vao DB, DB se
            tu sinh khi INSERT - AUTOINCREMENT).
        timestamp: Thoi gian phat hien vi pham (epoch seconds).
        vehicle_type: Loai phuong tien (car / motorcycle / bus).
        track_id: ID theo doi cua phuong tien tai thoi diem vi pham.
        license_plate: Chuoi bien so da nhan dien (hoac "UNKNOWN").
        violation_type: Loai vi pham (ViolationType).
        image_path: Duong dan anh chup lai khoanh khac vi pham.
        plate_image_path: Duong dan anh crop bien so (neu co).
        email_status: Trang thai gui email mo phong.
        frame_number: So thu tu frame video tai thoi diem vi pham.
        confidence: Do tin cay phat hien doi tuong tai thoi diem vi pham.
        extra_info: Thong tin bo sung dang text (vd: toc do, vi tri...).
    """

    timestamp: float
    vehicle_type: str
    track_id: int
    license_plate: str
    violation_type: ViolationType
    image_path: str
    violation_id: Optional[int] = None
    plate_image_path: Optional[str] = None
    email_status: EmailStatus = EmailStatus.PENDING
    frame_number: int = 0
    confidence: float = 0.0
    extra_info: str = ""

    @classmethod
    def create(
        cls,
        vehicle_type: str,
        track_id: int,
        violation_type: ViolationType,
        image_path: str,
        license_plate: str = "UNKNOWN",
        frame_number: int = 0,
        confidence: float = 0.0,
        extra_info: str = "",
        plate_image_path: Optional[str] = None,
    ) -> "Violation":
        """Tao moi mot Violation voi timestamp hien tai.

        Day la factory method tien loi de ViolationManager tao ban ghi
        moi truoc khi ghi vao database (violation_id se duoc DB gan sau).
        """
        return cls(
            timestamp=time.time(),
            vehicle_type=vehicle_type,
            track_id=track_id,
            license_plate=license_plate,
            violation_type=violation_type,
            image_path=image_path,
            plate_image_path=plate_image_path,
            email_status=EmailStatus.PENDING,
            frame_number=frame_number,
            confidence=confidence,
            extra_info=extra_info,
        )

    def to_db_tuple(self) -> tuple:
        """Chuyen doi sang tuple theo dung thu tu cot trong bang `violations`
        (khong bao gom violation_id - se duoc AUTOINCREMENT).

        Thu tu cot: timestamp, vehicle_type, track_id, license_plate,
        violation_type, image_path, plate_image_path, email_status,
        frame_number, confidence, extra_info
        """
        return (
            self.timestamp,
            self.vehicle_type,
            self.track_id,
            self.license_plate,
            self.violation_type.value,
            self.image_path,
            self.plate_image_path or "",
            self.email_status.value,
            self.frame_number,
            self.confidence,
            self.extra_info,
        )

    @classmethod
    def from_db_row(cls, row: tuple) -> "Violation":
        """Tao Violation tu mot dong (row) du lieu SQLite.

        Thu tu cot trong row: violation_id, timestamp, vehicle_type,
        track_id, license_plate, violation_type, image_path,
        plate_image_path, email_status, frame_number, confidence, extra_info
        """
        (
            violation_id,
            timestamp,
            vehicle_type,
            track_id,
            license_plate,
            violation_type_str,
            image_path,
            plate_image_path,
            email_status_str,
            frame_number,
            confidence,
            extra_info,
        ) = row

        try:
            violation_type = ViolationType(violation_type_str)
        except ValueError:
            violation_type = ViolationType.WRONG_LANE  # fallback an toan

        try:
            email_status = EmailStatus(email_status_str)
        except ValueError:
            email_status = EmailStatus.PENDING

        return cls(
            violation_id=violation_id,
            timestamp=timestamp,
            vehicle_type=vehicle_type,
            track_id=track_id,
            license_plate=license_plate,
            violation_type=violation_type,
            image_path=image_path,
            plate_image_path=plate_image_path or None,
            email_status=email_status,
            frame_number=frame_number,
            confidence=confidence,
            extra_info=extra_info or "",
        )

    def get_formatted_timestamp(self) -> str:
        """Tra ve timestamp dang chuoi 'YYYY-mm-dd HH:MM:SS'."""
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp))

    def to_dict(self) -> dict:
        return {
            "violation_id": self.violation_id,
            "timestamp": self.get_formatted_timestamp(),
            "vehicle_type": self.vehicle_type,
            "track_id": self.track_id,
            "license_plate": self.license_plate,
            "violation_type": self.violation_type.value,
            "image_path": self.image_path,
            "plate_image_path": self.plate_image_path,
            "email_status": self.email_status.value,
            "frame_number": self.frame_number,
            "confidence": self.confidence,
            "extra_info": self.extra_info,
        }
