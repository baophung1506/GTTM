"""
===== FILE: core/detection_manager.py =====

Module: Detection Manager
Mo ta: QThread (Detection Thread) chiu trach nhiem chay mo hinh YOLOv8 de
       phat hien cac doi tuong tren tung frame video:
           - Model phuong tien (yolov8n.pt - pretrained COCO): car,
             motorcycle, bus, person.
           - Model non bao hiem (helmet_model.pt - custom trained):
             helmet, no_helmet.

       Ket qua duoc chuan hoa thanh danh sach models.vehicle.Detection va
       phat (emit) cho TrackingManager xu ly tiep.

       De dam bao thoi gian thuc (real-time), hang doi dau vao co kich
       thuoc gioi han (maxsize=2): neu DetectionManager xu ly cham hon
       VideoManager doc frame, cac frame cu se bi loai bo (drop) thay vi
       tich tu - dung Queue thread-safe, KHONG dung time.sleep de dieu
       phoi.
"""

from __future__ import annotations

import os
import queue
from typing import List, Optional

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from core.logger_manager import LoggerManager
from models.vehicle import Detection, VehicleClass

# Mapping tu ten lop COCO (model yolov8n.pt pretrained) sang VehicleClass
# cua he thong. Cac lop khong co trong mapping se bi bo qua.
COCO_TO_SYSTEM_CLASS = {
    "person": VehicleClass.PERSON.value,
    "car": VehicleClass.CAR.value,
    "motorcycle": VehicleClass.MOTORCYCLE.value,
    "bus": VehicleClass.BUS.value,
    # Truck duoc xem nhu "car" o muc do giam sat lan duong (tuy chon).
    "truck": VehicleClass.CAR.value,
}

# Mapping ten lop cua model non bao hiem (custom). Gia su model duoc
# huan luyen voi 2 lop: "helmet" va "no_helmet" (hoac "no-helmet").
HELMET_CLASS_NORMALIZATION = {
    "helmet": VehicleClass.HELMET.value,
    "with_helmet": VehicleClass.HELMET.value,
    "no_helmet": VehicleClass.NO_HELMET.value,
    "no-helmet": VehicleClass.NO_HELMET.value,
    "without_helmet": VehicleClass.NO_HELMET.value,
}


class DetectionManager(QThread):
    """QThread chay YOLOv8 inference tren tung frame.

    Signals:
        detections_ready(object, object): Phat ra (frame_payload, detections)
            - frame_payload: dict tu VideoManager (chua "frame", "frame_number"...)
            - detections: List[Detection]
        detection_error(str): Phat ra khi co loi load model / inference.
        models_loaded(bool, bool): Phat ra (vehicle_model_ok, helmet_model_ok)
            sau khi qua trinh load model hoan tat.
    """

    detections_ready = pyqtSignal(object, object)
    detection_error = pyqtSignal(str)
    models_loaded = pyqtSignal(bool, bool)

    def __init__(
        self,
        vehicle_model_path: str,
        helmet_model_path: str,
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        device: str = "cpu",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.logger = LoggerManager.get_instance()
        self.vehicle_model_path = vehicle_model_path
        self.helmet_model_path = helmet_model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device

        self._input_queue: "queue.Queue[Optional[dict]]" = queue.Queue(maxsize=2)
        self._running = False

        self._vehicle_model = None
        self._helmet_model = None
        self.skip_every: int = 2
        self.infer_size: int = 960
        self._frame_counter: int = 0

    # ------------------------------------------------------------------
    # QThread lifecycle
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Vong lap chinh cua Detection Thread."""
        self._running = True
        self._load_models()
        self.logger.log_info("DetectionManager", "Detection thread da khoi dong.")

        while self._running:
            try:
                payload = self._input_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if payload is None:
                break

            try:
                self._frame_counter += 1
                if self._frame_counter % self.skip_every != 0:
                    continue
                h, w = payload["frame"].shape[:2]
                scale = self.infer_size / max(h, w)
                if scale < 1.0:
                    small = cv2.resize(payload["frame"], (int(w*scale), int(h*scale)))
                    sx, sy = 1.0/scale, 1.0/scale
                else:
                    small = payload["frame"]
                    sx, sy = 1.0, 1.0
                detections = self._run_inference(small, sx, sy)
                self.detections_ready.emit(payload, detections)
            except Exception as exc:  # noqa: BLE001
                self.logger.log_error("DetectionManager", f"Loi inference: {exc}")
                self.detection_error.emit(str(exc))

        self.logger.log_info("DetectionManager", "Detection thread da dung.")

    def stop(self) -> None:
        """Dung Detection Thread."""
        self._running = False
        try:
            self._input_queue.put_nowait(None)
        except queue.Full:
            pass

    # ------------------------------------------------------------------
    # Nap mo hinh
    # ------------------------------------------------------------------
    def _load_models(self) -> None:
        """Nap model YOLOv8 (phuong tien + non bao hiem).

        Neu file model khong ton tai, he thong van tiep tuc hoat dong
        voi model con lai (graceful degradation), tranh crash toan bo
        ung dung do thieu file weight.
        """
        vehicle_ok = False
        helmet_ok = False

        try:
            from ultralytics import YOLO

            if os.path.exists(self.vehicle_model_path):
                self._vehicle_model = YOLO(self.vehicle_model_path)
                vehicle_ok = True
                self.logger.log_info(
                    "DetectionManager",
                    f"Da nap vehicle model: {self.vehicle_model_path}",
                )
            else:
                self.logger.log_error(
                    "DetectionManager",
                    f"Khong tim thay vehicle model: {self.vehicle_model_path}",
                )

            if os.path.exists(self.helmet_model_path):
                self._helmet_model = YOLO(self.helmet_model_path)
                helmet_ok = True
                self.logger.log_info(
                    "DetectionManager",
                    f"Da nap helmet model: {self.helmet_model_path}",
                )
            else:
                self.logger.log_warning(
                    "DetectionManager",
                    f"Khong tim thay helmet model: {self.helmet_model_path}. "
                    f"Bo qua phat hien non bao hiem.",
                )
        except Exception as exc:  # noqa: BLE001
            self.logger.log_error("DetectionManager", f"Loi nap model YOLO: {exc}")

        self.models_loaded.emit(vehicle_ok, helmet_ok)

    # ------------------------------------------------------------------
    # API - dua frame vao hang doi (drop frame cu neu day)
    # ------------------------------------------------------------------
    def enqueue_frame(self, frame_payload: dict) -> None:
        """Dua mot frame moi vao hang doi xu ly.

        Neu hang doi day (detection cham hon video), frame cu nhat se
        bi loai bo de uu tien xu ly frame moi nhat - dam bao tinh "thoi
        gian thuc" cua he thong.
        """
        try:
            self._input_queue.put_nowait(frame_payload)
        except queue.Full:
            try:
                self._input_queue.get_nowait()  # bo frame cu
            except queue.Empty:
                pass
            try:
                self._input_queue.put_nowait(frame_payload)
            except queue.Full:
                pass

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _run_inference(self, frame: np.ndarray, sx: float = 1.0, sy: float = 1.0) -> List[Detection]:
        """Chay YOLOv8 (vehicle + helmet) tren mot frame va tra ve Detection list."""
        detections: List[Detection] = []

        if self._vehicle_model is not None:
            detections.extend(self._infer_vehicle(frame, sx, sy))

        if self._helmet_model is not None:
            detections.extend(self._infer_helmet(frame, sx, sy))

        return detections

    def _infer_vehicle(self, frame: np.ndarray, sx: float = 1.0, sy: float = 1.0) -> List[Detection]:
        results = self._vehicle_model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )
        detections: List[Detection] = []
        if not results:
            return detections

        result = results[0]
        names = result.names
        boxes = result.boxes
        if boxes is None:
            return detections

        for box in boxes:
            cls_id = int(box.cls[0].item())
            cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
            mapped_name = COCO_TO_SYSTEM_CLASS.get(cls_name.lower())
            if mapped_name is None:
                continue

            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            x1, x2 = x1*sx, x2*sx
            y1, y2 = y1*sy, y2*sy

            detections.append(
                Detection(
                    class_name=mapped_name,
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    class_id=cls_id,
                )
            )

        return detections

    def _infer_helmet(self, frame: np.ndarray, sx: float = 1.0, sy: float = 1.0) -> List[Detection]:
        results = self._helmet_model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )
        detections: List[Detection] = []
        if not results:
            return detections

        result = results[0]
        names = result.names
        boxes = result.boxes
        if boxes is None:
            return detections

        for box in boxes:
            cls_id = int(box.cls[0].item())
            cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
            mapped_name = HELMET_CLASS_NORMALIZATION.get(cls_name.lower())
            if mapped_name is None:
                continue

            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            x1, x2 = x1*sx, x2*sx
            y1, y2 = y1*sy, y2*sy

            detections.append(
                Detection(
                    class_name=mapped_name,
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    class_id=cls_id,
                )
            )

        return detections


