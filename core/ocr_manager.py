"""
===== FILE: core/ocr_manager.py =====

Module: OCR Manager
Mo ta: QThread chuyen trach nhan dien bien so xe (License Plate
       Recognition) - CHI duoc kich hoat khi ViolationManager phat hien
       mot vi pham moi.

       Quy trinh xu ly cho moi vi pham:
           1. Nhan anh crop phuong tien vi pham tu hang doi.
           2. Phat hien vung bien so bang model YOLO rieng (best.pt).
           3. Crop vung bien so, phong to anh.
           4. TACH RIENG dong tren / dong duoi (bien so VN 2 dong)
              de tang do chinh xac OCR.
           5. Chay EasyOCR rieng tren tung dong.
           6. CHUAN HOA text theo dung dinh dang bien so VN (regex)
              de sua loi OCR thuong gap (dau '-' bi doc nham thanh
              ky tu khac).
           7. Luu anh bien so va phat signal ket qua cho ViolationManager.
"""

from __future__ import annotations

import os
import queue
import re
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from core.logger_manager import LoggerManager
from models.license_plate import LicensePlate


PLATE_MODEL_PATH = "resources/best.pt"


class OCRTask:
    """Mot tac vu OCR trong hang doi cua OCRManager."""

    def __init__(
        self,
        violation_id_placeholder: int,
        vehicle_image: np.ndarray,
        output_dir: str,
        file_prefix: str,
    ) -> None:
        self.violation_id_placeholder = violation_id_placeholder
        self.vehicle_image = vehicle_image
        self.output_dir = output_dir
        self.file_prefix = file_prefix


class OCRManager(QThread):
    """QThread thuc hien OCR bien so xe theo hang doi (thread-safe).

    Signals:
        plate_recognized(int, object, str): Phat ra (placeholder_id,
            LicensePlate, plate_image_path) sau khi xu ly xong mot task.
        ocr_error(str): Phat ra khi co loi trong qua trinh OCR.
    """

    plate_recognized = pyqtSignal(int, object, str)
    ocr_error = pyqtSignal(str)

    def __init__(
        self,
        languages: Optional[List[str]] = None,
        use_gpu: bool = False,
        min_confidence: float = 0.3,
        plate_confidence: float = 0.25,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.languages = languages or ["en"]
        self.use_gpu = use_gpu
        self.min_confidence = min_confidence
        self.plate_confidence = plate_confidence
        self.logger = LoggerManager.get_instance()

        self._task_queue: "queue.Queue[Optional[OCRTask]]" = queue.Queue()
        self._running = False
        self._reader = None       # easyocr.Reader - khoi tao lazy trong run()
        self._plate_model = None  # ultralytics.YOLO - khoi tao lazy trong run()

    # ------------------------------------------------------------------
    # QThread lifecycle
    # ------------------------------------------------------------------
    def run(self) -> None:
        self._running = True
        self._init_reader()
        self._init_plate_model()
        self.logger.log_info("OCRManager", "OCR thread da khoi dong.")

        while self._running:
            try:
                task = self._task_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if task is None:
                break

            try:
                self._process_task(task)
            except Exception as exc:  # noqa: BLE001
                self.logger.log_error("OCRManager", f"Loi xu ly OCR: {exc}")
                self.ocr_error.emit(str(exc))

        self.logger.log_info("OCRManager", "OCR thread da dung.")

    def stop(self) -> None:
        self._running = False
        self._task_queue.put(None)

    def _init_reader(self) -> None:
        try:
            import easyocr

            self._reader = easyocr.Reader(self.languages, gpu=self.use_gpu)
            self.logger.log_info(
                "OCRManager", f"Da khoi tao EasyOCR voi ngon ngu {self.languages}."
            )
        except Exception as exc:  # noqa: BLE001
            self._reader = None
            self.logger.log_error(
                "OCRManager",
                f"Khong the khoi tao EasyOCR: {exc}. OCR se tra ve UNKNOWN.",
            )

    def _init_plate_model(self) -> None:
        """Khoi tao model YOLO phat hien bien so."""
        if not PLATE_MODEL_PATH or not os.path.exists(PLATE_MODEL_PATH):
            self._plate_model = None
            self.logger.log_warning(
                "OCRManager",
                f"Khong tim thay plate model: {PLATE_MODEL_PATH}. Dung heuristic.",
            )
            return

        try:
            from ultralytics import YOLO

            self._plate_model = YOLO(PLATE_MODEL_PATH)
            self.logger.log_info("OCRManager", f"Da nap plate model: {PLATE_MODEL_PATH}")
        except Exception as exc:  # noqa: BLE001
            self._plate_model = None
            self.logger.log_error(
                "OCRManager",
                f"Khong the nap plate model '{PLATE_MODEL_PATH}': {exc}. Dung heuristic.",
            )

    # ------------------------------------------------------------------
    # API - dua tac vu vao hang doi
    # ------------------------------------------------------------------
    def enqueue_ocr_task(
        self,
        placeholder_id: int,
        vehicle_image: np.ndarray,
        output_dir: str,
        file_prefix: str,
    ) -> None:
        self._task_queue.put(
            OCRTask(placeholder_id, vehicle_image.copy(), output_dir, file_prefix)
        )

    # ------------------------------------------------------------------
    # Xu ly chinh
    # ------------------------------------------------------------------
    def _process_task(self, task: OCRTask) -> None:
        plate_region = self._locate_plate_region(task.vehicle_image)

        if plate_region is None:
            plate = LicensePlate.unknown()
            self.plate_recognized.emit(task.violation_id_placeholder, plate, "")
            return

        x1, y1, x2, y2 = plate_region
        plate_crop = task.vehicle_image[y1:y2, x1:x2]

        if plate_crop.size == 0:
            plate = LicensePlate.unknown()
            self.plate_recognized.emit(task.violation_id_placeholder, plate, "")
            return

        plate_image_path = self._save_plate_image(plate_crop, task.output_dir, task.file_prefix)
        plate = self._run_ocr(plate_crop, bbox=plate_region, plate_image_path=plate_image_path)

        self.logger.log_info(
            "OCRManager",
            f"OCR hoan tat cho '{task.file_prefix}': "
            f"'{plate.normalized_text}' (conf={plate.confidence:.2f})",
        )
        self.plate_recognized.emit(task.violation_id_placeholder, plate, plate_image_path)

    # ------------------------------------------------------------------
    # Phat hien vung bien so - uu tien model YOLO, fallback heuristic
    # ------------------------------------------------------------------
    def _locate_plate_region(
        self, vehicle_image: np.ndarray
    ) -> Optional[Tuple[int, int, int, int]]:
        if vehicle_image is None or vehicle_image.size == 0:
            return None

        h, w = vehicle_image.shape[:2]
        if h < 10 or w < 10:
            return None

        if self._plate_model is not None:
            box = self._locate_plate_region_yolo(vehicle_image)
            if box is not None:
                return box

        return self._locate_plate_region_heuristic(vehicle_image)

    def _locate_plate_region_yolo(
        self, vehicle_image: np.ndarray
    ) -> Optional[Tuple[int, int, int, int]]:
        try:
            results = self._plate_model.predict(
                vehicle_image, conf=self.plate_confidence, verbose=False
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.log_error("OCRManager", f"Loi chay plate model: {exc}")
            return None

        if not results:
            return None

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

        h, w = vehicle_image.shape[:2]
        best_idx = int(boxes.conf.argmax())
        xyxy = boxes.xyxy[best_idx].tolist()
        x1, y1, x2, y2 = [int(v) for v in xyxy]

        pad_x = max(2, int((x2 - x1) * 0.05))
        pad_y = max(2, int((y2 - y1) * 0.08))
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        if x2 <= x1 or y2 <= y1:
            return None

        return (x1, y1, x2, y2)

    def _locate_plate_region_heuristic(
        self, vehicle_image: np.ndarray
    ) -> Optional[Tuple[int, int, int, int]]:
        """Fallback: heuristic OpenCV khi khong co model bien so."""
        if vehicle_image is None or vehicle_image.size == 0:
            return None

        h, w = vehicle_image.shape[:2]
        if h < 10 or w < 10:
            return None

        gray = cv2.cvtColor(vehicle_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        edges = cv2.Canny(gray, 50, 150)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        best_box: Optional[Tuple[int, int, int, int]] = None
        best_score = -1.0

        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            if cw <= 0 or ch <= 0:
                continue
            area = cw * ch
            aspect_ratio = cw / float(ch)
            if not (1.2 <= aspect_ratio <= 5.5):
                continue
            if area < 0.01 * w * h or area > 0.35 * w * h:
                continue
            vertical_bias = (y + ch / 2.0) / float(h)
            score = area * (0.5 + vertical_bias)
            if score > best_score:
                best_score = score
                best_box = (x, y, x + cw, y + ch)

        if best_box is not None:
            x1, y1, x2, y2 = best_box
            pad_x = max(2, int((x2 - x1) * 0.05))
            pad_y = max(2, int((y2 - y1) * 0.05))
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            return (x1, y1, x2, y2)

        fallback_y1 = int(h * 0.6)
        if fallback_y1 < h:
            return (0, fallback_y1, w, h)
        return (0, 0, w, h)

    # ------------------------------------------------------------------
    # OCR - tach dong tren/duoi + chuan hoa text bien so VN
    # ------------------------------------------------------------------
    def _run_ocr(
        self,
        plate_image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        plate_image_path: str,
    ) -> LicensePlate:
        """Chay EasyOCR tren anh crop bien so, tach dong + chuan hoa ket qua."""
        if self._reader is None:
            return LicensePlate.unknown()

        h, w = plate_image.shape[:2]
        scale = max(1, int(300 / max(h, 1)))
        big = cv2.resize(plate_image, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC) \
            if scale > 1 else plate_image.copy()
        bh, bw = big.shape[:2]

        gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # Ty le rong/cao cua bien so quyet dinh co phai bien 2 dong hay khong.
        # Bien so VN 2 dong (xe may) co ty le ~0.8-1.4; bien 1 dong (oto) dai hon ~3-4.
        aspect_ratio = bw / float(bh) if bh > 0 else 0

        try:
            if aspect_ratio < 2.0:
                # Bien 2 dong: tach rieng tren/duoi de tang do chinh xac
                top_half = gray[0:int(bh * 0.52), :]
                bottom_half = gray[int(bh * 0.48):bh, :]

                results_top = self._reader.readtext(top_half)
                results_bottom = self._reader.readtext(bottom_half)

                top_text, top_conf = self._best_fragment(results_top)
                bottom_text, bottom_conf = self._best_fragment(results_bottom)

                normalized = self._normalize_plate_text(top_text, bottom_text)
                avg_conf = (top_conf + bottom_conf) / 2 if (top_text or bottom_text) else 0.0

                if not normalized:
                    return LicensePlate.unknown()

                return LicensePlate(
                    raw_text=f"{top_text} {bottom_text}".strip(),
                    normalized_text=normalized,
                    confidence=avg_conf,
                    bbox=bbox,
                    plate_image_path=plate_image_path,
                )
            else:
                # Bien 1 dong: OCR truc tiep toan bo
                results = self._reader.readtext(gray)
                fragments = [
                    (text, conf) for (_, text, conf) in results if conf >= self.min_confidence
                ]
                if not fragments:
                    return LicensePlate.unknown()
                return LicensePlate.from_ocr_result(
                    fragments, bbox=bbox, plate_image_path=plate_image_path
                )
        except Exception as exc:  # noqa: BLE001
            self.logger.log_error("OCRManager", f"Loi EasyOCR readtext: {exc}")
            return LicensePlate.unknown()

    @staticmethod
    def _best_fragment(ocr_results) -> Tuple[str, float]:
        """Lay fragment text co confidence cao nhat tu ket qua EasyOCR."""
        if not ocr_results:
            return "", 0.0
        best = max(ocr_results, key=lambda r: r[2])
        return best[1].strip().upper(), float(best[2])

    @staticmethod
    def _normalize_plate_text(top_text: str, bottom_text: str) -> str:
        """
        Chuan hoa text bien so VN theo dung dinh dang: XXA-YYYYY hoac XXA-YYY.YY

        Quy tac xu ly loi OCR thuong gap:
            - Dong tren dang: 2 so + 1 chu cai + (so seri) -> tu dong chen
              dau '-' dung vi tri, khong phu thuoc OCR doc dung ky tu do.
            - Dong duoi: chi giu lai chu so, tu dong dinh dang XXX.XX neu
              du 5 chu so.
        """
        top = (top_text or "").strip().upper()
        bottom = (bottom_text or "").strip().upper()

        top_clean = re.sub(r'[^A-Z0-9]', '', top)

        match = re.match(r'^(\d{2})([A-Z]\d?)$', top_clean)
        if match:
            province_code, series = match.groups()
            top_normalized = f"{province_code}-{series}"
        elif len(top_clean) >= 2:
            top_normalized = f"{top_clean[:2]}-{top_clean[2:]}"
        else:
            top_normalized = top_clean

        bottom_clean = re.sub(r'[^0-9]', '', bottom)
        if len(bottom_clean) == 5:
            bottom_normalized = f"{bottom_clean[:3]}.{bottom_clean[3:]}"
        else:
            bottom_normalized = bottom_clean

        result = f"{top_normalized} {bottom_normalized}".strip()
        # Neu ket qua qua ngan / rong, coi nhu khong doc duoc
        if len(re.sub(r'[^A-Z0-9]', '', result)) < 4:
            return ""
        return result

    # ------------------------------------------------------------------
    # Luu anh
    # ------------------------------------------------------------------
    @staticmethod
    def _save_plate_image(plate_crop: np.ndarray, output_dir: str, file_prefix: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{file_prefix}_plate_{int(time.time() * 1000)}.jpg"
        full_path = os.path.join(output_dir, filename)
        cv2.imwrite(full_path, plate_crop)
        return full_path