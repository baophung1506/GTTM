"""
===== FILE: core/config_manager.py =====

Module: Config Manager
Mo ta: Chiu trach nhiem doc/ghi toan bo cau hinh he thong tu cac file
       JSON trong thu muc config/:
           - settings.json          : cau hinh chung (model, tracking,
                                       OCR, email, database, UI...).
           - stop_line.json          : toa do vach dung.
           - lane_config.json        : danh sach lan duong hop le.
           - wrong_way.json          : vung kiem tra huong di cho phep.
           - forbidden_parking.json  : vung cam dung do.
           - forbidden_uturn.json    : vung cam quay dau.

       Neu file khong ton tai, ConfigManager se tu dong sinh file mac
       dinh de he thong van khoi dong duoc (tuan thu yeu cau "khong de
       trong ham, khong loi khi thieu cau hinh").
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from core.logger_manager import LoggerManager


class ConfigManager:
    """Quan ly toan bo cau hinh JSON cua he thong.

    Attributes:
        config_dir: Thu muc chua cac file JSON cau hinh.
        settings: Dict cau hinh chung (tu settings.json).
        stop_line: Dict chua toa do vach dung.
        lanes: Dict chua danh sach lan duong.
        wrong_way_zones: Dict chua cac vung kiem tra huong di.
        forbidden_parking_zones: Dict chua cac vung cam dung do.
        forbidden_uturn_zones: Dict chua cac vung cam quay dau.
    """

    def __init__(self, config_dir: str = "config") -> None:
        self.config_dir = config_dir
        self.logger = LoggerManager.get_instance()
        os.makedirs(self.config_dir, exist_ok=True)

        self.settings: Dict[str, Any] = {}
        self.stop_line: Dict[str, Any] = {}
        self.lanes: Dict[str, Any] = {}
        self.wrong_way_zones: Dict[str, Any] = {}
        self.forbidden_parking_zones: Dict[str, Any] = {}
        self.forbidden_uturn_zones: Dict[str, Any] = {}

        self.load_all()

    # ------------------------------------------------------------------
    # Tien ich doc/ghi file JSON
    # ------------------------------------------------------------------
    def _path(self, filename: str) -> str:
        return os.path.join(self.config_dir, filename)

    def _load_json(self, filename: str, default: Dict[str, Any]) -> Dict[str, Any]:
        """Doc file JSON. Neu khong ton tai hoac loi, tao file mac dinh."""
        path = self._path(filename)
        if not os.path.exists(path):
            self.logger.log_warning(
                "ConfigManager", f"Khong tim thay {path}, tao file mac dinh."
            )
            self._save_json(filename, default)
            return default

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except (json.JSONDecodeError, OSError) as exc:
            self.logger.log_error(
                "ConfigManager", f"Loi doc {path}: {exc}. Su dung gia tri mac dinh."
            )
            return default

    def _save_json(self, filename: str, data: Dict[str, Any]) -> None:
        path = self._path(filename)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        except OSError as exc:
            self.logger.log_error("ConfigManager", f"Loi ghi {path}: {exc}")

    # ------------------------------------------------------------------
    # Nap toan bo cau hinh
    # ------------------------------------------------------------------
    def load_all(self) -> None:
        """Nap tat ca file cau hinh. Sinh file mac dinh neu thieu."""
        self.settings = self._load_json("settings.json", self._default_settings())
        self.stop_line = self._load_json("stop_line.json", self._default_stop_line())
        self.lanes = self._load_json("lane_config.json", self._default_lanes())
        self.wrong_way_zones = self._load_json("wrong_way.json", self._default_wrong_way())
        self.forbidden_parking_zones = self._load_json(
            "forbidden_parking.json", self._default_forbidden_parking()
        )
        self.forbidden_uturn_zones = self._load_json(
            "forbidden_uturn.json", self._default_forbidden_uturn()
        )
        self.logger.log_info("ConfigManager", "Da nap toan bo cau hinh he thong.")

    def reload(self) -> None:
        """Nap lai toan bo cau hinh tu dia (vd: sau khi nguoi dung sua tay)."""
        self.load_all()

    # ------------------------------------------------------------------
    # Gia tri mac dinh
    # ------------------------------------------------------------------
    @staticmethod
    def _default_settings() -> Dict[str, Any]:
        return {
            "video": {"default_path": "", "loop": False, "target_fps": 25},
            "models": {
                "vehicle_model_path": "resources/yolov8n.pt",
                "helmet_model_path": "resources/helmet_model.pt",
                "confidence_threshold": 0.35,
                "iou_threshold": 0.45,
                "device": "cpu",
                "class_names": [
                    "person",
                    "car",
                    "motorcycle",
                    "bus",
                    "helmet",
                    "no_helmet",
                ],
            },
            "tracking": {
                "track_thresh": 0.5,
                "track_buffer": 30,
                "match_thresh": 0.8,
                "frame_rate": 25,
            },
            "violation": {
                "stationary_seconds_threshold": 5.0,
                "wrong_way_tolerance_degrees": 100.0,
                "min_track_age_seconds": 0.5,
                "min_history_for_direction": 5,
                "violation_cooldown_seconds": 5.0,
            },
            "ocr": {"languages": ["en"], "gpu": False, "min_confidence": 0.3},
            "email": {
                "sender": "traffic.ai.system@example.com",
                "recipient": "traffic.authority@example.com",
                "smtp_host": "smtp.example.com",
                "smtp_port": 587,
            },
            "database": {"path": "database/traffic.db"},
            "ui": {"theme": "dark", "overlay_font_scale": 0.6},
            "storage": {
                "violation_images_dir": "violations/images",
                "logs_dir": "violations/logs",
            },
        }

    @staticmethod
    def _default_stop_line() -> Dict[str, Any]:
        return {"stop_line": [[100, 300], [1000, 300]]}

    @staticmethod
    def _default_lanes() -> Dict[str, Any]:
        return {
            "lanes": [
                {
                    "id": "lane_1",
                    "polygon": [[0, 0], [640, 0], [640, 720], [0, 720]],
                    "allowed_classes": ["car", "motorcycle", "bus"],
                    "direction_degrees": 90.0,
                },
                {
                    "id": "lane_2",
                    "polygon": [[640, 0], [1280, 0], [1280, 720], [640, 720]],
                    "allowed_classes": ["car", "motorcycle", "bus"],
                    "direction_degrees": 90.0,
                },
            ]
        }

    @staticmethod
    def _default_wrong_way() -> Dict[str, Any]:
        return {
            "zones": [
                {
                    "id": "direction_zone_1",
                    "polygon": [[0, 0], [1280, 0], [1280, 720], [0, 720]],
                    "allowed_direction_degrees": 90.0,
                    "tolerance_degrees": 100.0,
                }
            ]
        }

    @staticmethod
    def _default_forbidden_parking() -> Dict[str, Any]:
        return {
            "zones": [
                {
                    "id": "no_parking_1",
                    "polygon": [[50, 50], [300, 50], [300, 250], [50, 250]],
                    "max_stationary_seconds": 5.0,
                }
            ]
        }

    @staticmethod
    def _default_forbidden_uturn() -> Dict[str, Any]:
        return {
            "zones": [
                {
                    "id": "no_uturn_1",
                    "polygon": [[400, 100], [800, 100], [800, 400], [400, 400]],
                }
            ]
        }

    # ------------------------------------------------------------------
    # Truy cap nhanh (getters) - su dung dot-notation key, vd "models.device"
    # ------------------------------------------------------------------
    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Lay gia tri tu settings.json bang khoa dang 'a.b.c'.

        Args:
            dotted_key: Khoa duoi dang "section.key", vd "ocr.languages".
            default: Gia tri tra ve neu khong tim thay.
        """
        node: Any = self.settings
        for part in dotted_key.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def get_stop_line(self) -> List[List[float]]:
        return self.stop_line.get("stop_line", [])

    def get_lanes(self) -> List[Dict[str, Any]]:
        return self.lanes.get("lanes", [])

    def get_wrong_way_zones(self) -> List[Dict[str, Any]]:
        return self.wrong_way_zones.get("zones", [])

    def get_forbidden_parking_zones(self) -> List[Dict[str, Any]]:
        return self.forbidden_parking_zones.get("zones", [])

    def get_forbidden_uturn_zones(self) -> List[Dict[str, Any]]:
        return self.forbidden_uturn_zones.get("zones", [])

    def get_database_path(self) -> str:
        return self.get("database.path", "database/traffic.db")

    def get_violation_images_dir(self) -> str:
        return self.get("storage.violation_images_dir", "violations/images")

    def get_vehicle_model_path(self) -> str:
        return self.get("models.vehicle_model_path", "resources/yolov8n.pt")

    def get_helmet_model_path(self) -> str:
        return self.get("models.helmet_model_path", "resources/helmet_model.pt")
