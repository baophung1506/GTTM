"""
===== FILE: models/license_plate.py =====

Module: License Plate Model
Mo ta: Dinh nghia lop du lieu LicensePlate dai dien cho ket qua nhan dien
       bien so xe (OCR). Lop nay duoc OCRManager tao ra sau khi:
         1. Crop phuong tien vi pham.
         2. Phat hien vung bien so (plate localization).
         3. Crop vung bien so.
         4. Chay EasyOCR.
         5. Chuan hoa chuoi ket qua.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

BoundingBox = Tuple[float, float, float, float]


@dataclass
class LicensePlate:
    """Ket qua nhan dien bien so xe.

    Attributes:
        raw_text: Chuoi van ban tho do EasyOCR tra ve (truoc khi chuan hoa).
        normalized_text: Chuoi bien so sau khi chuan hoa (vd: "29A1-12345").
        confidence: Do tin cay trung binh cua OCR (0.0 - 1.0).
        bbox: Bounding box vung bien so trong anh crop phuong tien
            (x1, y1, x2, y2), toa do tuong doi so voi anh crop.
        plate_image_path: Duong dan anh crop bien so da luu (neu co).
        is_valid_format: Co khop voi mau bien so xe Viet Nam pho bien
            hay khong (dung de canh bao OCR loi).
    """

    raw_text: str
    normalized_text: str
    confidence: float
    bbox: Optional[BoundingBox] = None
    plate_image_path: Optional[str] = None
    is_valid_format: bool = False

    # Mau bien so xe may / xe oto Viet Nam pho bien:
    #   29A-123.45 , 29A1-123.45 , 29-A1 123.45 , 30G-001.23 ...
    _PLATE_REGEX = re.compile(r"^[0-9]{2}[A-Z]{1,2}[0-9]?-?[0-9]{3,5}$")

    @staticmethod
    def normalize_text(raw_text: str) -> str:
        """Chuan hoa chuoi ket qua OCR thanh dang bien so chuan.

        Buoc xu ly:
            1. Chuyen toan bo ky tu sang in hoa.
            2. Loai bo khoang trang, ky tu khong phai chu/so/gach ngang/cham.
            3. Thay the cac ky tu OCR thuong nham (O->0, I->1, ...) trong
               phan so (neu can) - o day giu don gian: chi loai bo ky tu la
               thua va chuan hoa dau gach ngang.
            4. Chen dau '-' giua phan chu va phan so neu chua co, theo
               dinh dang XXY-NNNNN.

        Args:
            raw_text: Chuoi tho tu EasyOCR (co the chua nhieu dong, ghep
                bang khoang trang truoc khi truyen vao).

        Returns:
            Chuoi bien so da chuan hoa, vi du "29A1-12345". Neu khong the
            chuan hoa, tra ve chuoi da loai bo ky tu thua (in hoa).
        """
        if not raw_text:
            return ""

        # Buoc 1-2: in hoa, loai bo cac ky tu khong mong muon.
        cleaned = raw_text.upper()
        cleaned = re.sub(r"[^A-Z0-9\-\.]", "", cleaned)
        cleaned = cleaned.replace(".", "")

        # Neu da co dau '-' thi giu nguyen cau truc, chi loai bo ky tu trung.
        if "-" in cleaned:
            parts = cleaned.split("-")
            cleaned = "-".join(p for p in parts if p)
            return cleaned

        # Neu chua co dau '-', tim diem chuyen tiep tu chu sang so cuoi cung
        # de chen dau '-' (vi du "29A112345" -> "29A1-12345").
        match = re.match(r"^([0-9]{2}[A-Z]{1,2}[0-9]?)([0-9]{3,6})$", cleaned)
        if match:
            return f"{match.group(1)}-{match.group(2)}"

        return cleaned

    @classmethod
    def from_ocr_result(
        cls,
        ocr_fragments: List[Tuple[str, float]],
        bbox: Optional[BoundingBox] = None,
        plate_image_path: Optional[str] = None,
    ) -> "LicensePlate":
        """Tao LicensePlate tu danh sach cac doan text EasyOCR tra ve.

        Args:
            ocr_fragments: Danh sach (text, confidence) tu easyocr.readtext.
                Co the gom nhieu dong (bien so 2 dong).
            bbox: Vi tri vung bien so trong anh crop.
            plate_image_path: Duong dan file anh bien so da luu.

        Returns:
            Doi tuong LicensePlate da duoc chuan hoa.
        """
        if not ocr_fragments:
            return cls(
                raw_text="",
                normalized_text="",
                confidence=0.0,
                bbox=bbox,
                plate_image_path=plate_image_path,
                is_valid_format=False,
            )

        raw_text = " ".join(fragment_text for fragment_text, _ in ocr_fragments)
        avg_conf = sum(conf for _, conf in ocr_fragments) / len(ocr_fragments)
        normalized = cls.normalize_text(raw_text)
        is_valid = bool(cls._PLATE_REGEX.match(normalized))

        return cls(
            raw_text=raw_text,
            normalized_text=normalized,
            confidence=avg_conf,
            bbox=bbox,
            plate_image_path=plate_image_path,
            is_valid_format=is_valid,
        )

    @classmethod
    def unknown(cls) -> "LicensePlate":
        """Tra ve doi tuong dai dien cho truong hop khong nhan dien duoc."""
        return cls(
            raw_text="",
            normalized_text="UNKNOWN",
            confidence=0.0,
            bbox=None,
            plate_image_path=None,
            is_valid_format=False,
        )

    def __str__(self) -> str:
        return self.normalized_text if self.normalized_text else "UNKNOWN"
