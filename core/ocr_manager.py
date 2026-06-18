"""
===== FILE: core/ocr_manager.py =====
"""

from __future__ import annotations

import os
import queue
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from core.logger_manager import LoggerManager
from models.license_plate import LicensePlate


class OCRTask:
    def __init__(self, violation_id_placeholder, vehicle_image, output_dir, file_prefix):
        self.violation_id_placeholder = violation_id_placeholder
        self.vehicle_image = vehicle_image
        self.output_dir = output_dir
        self.file_prefix = file_prefix


class OCRManager(QThread):
    plate_recognized = pyqtSignal(int, object, str)
    ocr_error = pyqtSignal(str)
    PLATE_MODEL_PATH = "resources/plate_model.pt"

    def __init__(self, languages=None, use_gpu=False, min_confidence=0.3, parent=None):
        super().__init__(parent)
        self.languages = languages or ["en"]
        self.use_gpu = use_gpu
        self.min_confidence = min_confidence
        self.logger = LoggerManager.get_instance()
        self._task_queue = queue.Queue()
        self._running = False
        self._reader = None
        self._plate_model = None

    def run(self):
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
            except Exception as exc:
                self.logger.log_error("OCRManager", f"Loi xu ly OCR: {exc}")
                self.ocr_error.emit(str(exc))
        self.logger.log_info("OCRManager", "OCR thread da dung.")

    def stop(self):
        self._running = False
        self._task_queue.put(None)

    def _init_reader(self):
        try:
            import easyocr
            self._reader = easyocr.Reader(self.languages, gpu=self.use_gpu)
            self.logger.log_info("OCRManager", f"Da khoi tao EasyOCR voi ngon ngu {self.languages}.")
        except Exception as exc:
            self._reader = None
            self.logger.log_error("OCRManager", f"Khong the khoi tao EasyOCR: {exc}.")

    def _init_plate_model(self):
        try:
            if os.path.exists(self.PLATE_MODEL_PATH):
                from ultralytics import YOLO
                self._plate_model = YOLO(self.PLATE_MODEL_PATH)
                self.logger.log_info("OCRManager", f"Da nap plate model: {self.PLATE_MODEL_PATH}")
            else:
                self.logger.log_warning("OCRManager", "Khong tim thay plate_model.pt, dung heuristic.")
        except Exception as exc:
            self._plate_model = None
            self.logger.log_error("OCRManager", f"Loi nap plate model: {exc}")

    def enqueue_ocr_task(self, placeholder_id, vehicle_image, output_dir, file_prefix):
        self._task_queue.put(OCRTask(placeholder_id, vehicle_image.copy(), output_dir, file_prefix))

    def _process_task(self, task):
        plate_region = self._locate_plate_region(task.vehicle_image)
        if plate_region is None:
            self.plate_recognized.emit(task.violation_id_placeholder, LicensePlate.unknown(), "")
            return
        x1, y1, x2, y2 = plate_region
        plate_crop = task.vehicle_image[y1:y2, x1:x2]
        if plate_crop.size == 0:
            self.plate_recognized.emit(task.violation_id_placeholder, LicensePlate.unknown(), "")
            return
        plate_image_path = self._save_plate_image(plate_crop, task.output_dir, task.file_prefix)
        plate = self._run_ocr(plate_crop, bbox=plate_region, plate_image_path=plate_image_path)
        self.logger.log_info("OCRManager", f"OCR hoan tat: '{plate.normalized_text}' (conf={plate.confidence:.2f})")
        self.plate_recognized.emit(task.violation_id_placeholder, plate, plate_image_path)

    def _locate_plate_region(self, vehicle_image):
        if vehicle_image is None or vehicle_image.size == 0:
            return None
        h, w = vehicle_image.shape[:2]
        if h < 10 or w < 10:
            return None

        if self._plate_model is not None:
            try:
                results = self._plate_model.predict(source=vehicle_image, conf=0.3, verbose=False)
                if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                    boxes = results[0].boxes
                    best_idx = int(boxes.conf.argmax())
                    x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[best_idx].tolist()]
                    pad_x = max(2, int((x2-x1)*0.05))
                    pad_y = max(2, int((y2-y1)*0.05))
                    return (max(0,x1-pad_x), max(0,y1-pad_y), min(w,x2+pad_x), min(h,y2+pad_y))
            except Exception as exc:
                self.logger.log_warning("OCRManager", f"YOLO plate loi: {exc}")

        gray = cv2.cvtColor(vehicle_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        best_box, best_score = None, -1.0
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            if cw <= 0 or ch <= 0:
                continue
            ar = cw / float(ch)
            area = cw * ch
            if not (1.2 <= ar <= 5.5) or area < 0.01*w*h or area > 0.35*w*h:
                continue
            score = area * (0.5 + (y+ch/2.0)/float(h))
            if score > best_score:
                best_score = score
                best_box = (x, y, x+cw, y+ch)
        if best_box:
            x1,y1,x2,y2 = best_box
            px,py = max(2,int((x2-x1)*0.05)), max(2,int((y2-y1)*0.05))
            return (max(0,x1-px), max(0,y1-py), min(w,x2+px), min(h,y2+py))
        fy = int(h*0.6)
        return (0, fy, w, h) if fy < h else (0, 0, w, h)

    def _run_ocr(self, plate_image, bbox, plate_image_path):
        if self._reader is None:
            return LicensePlate.unknown()
        h, w = plate_image.shape[:2]
        scale = max(1, int(200/max(h,1)))
        if scale > 1:
            plate_image = cv2.resize(plate_image, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
        gray = cv2.equalizeHist(cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY))
        try:
            results = self._reader.readtext(gray)
        except Exception as exc:
            self.logger.log_error("OCRManager", f"Loi EasyOCR: {exc}")
            return LicensePlate.unknown()
        fragments = [(t, c) for (_,t,c) in results if c >= self.min_confidence]
        if not fragments:
            return LicensePlate.unknown()
        return LicensePlate.from_ocr_result(fragments, bbox=bbox, plate_image_path=plate_image_path)

    @staticmethod
    def _save_plate_image(plate_crop, output_dir, file_prefix):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{file_prefix}_plate_{int(time.time()*1000)}.jpg")
        cv2.imwrite(path, plate_crop)
        return path

