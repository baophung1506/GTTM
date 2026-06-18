"""
===== FILE: core/logger_manager.py =====

Module: Logger Manager
Mo ta: Cung cap he thong logging tap trung cho toan bo ung dung, sinh ra
       3 file log rieng biet:
           - logs/system.log    : log hoat dong chung cua he thong.
           - logs/error.log     : log loi (level ERROR / CRITICAL).
           - logs/violation.log : log moi vi pham giao thong duoc phat hien.

       Su dung mau Singleton de dam bao toan bo ung dung dung chung
       mot bo logger duy nhat, tranh xung dot khi nhieu QThread cung
       ghi log dong thoi (logging module cua Python da thread-safe o
       muc handler).
"""

from __future__ import annotations

import logging
import os
import threading
from logging.handlers import RotatingFileHandler
from typing import Optional


class LoggerManager:
    """Singleton quan ly toan bo logger cua he thong.

    Vi du su dung:
        logger = LoggerManager.get_instance()
        logger.log_info("VideoManager", "Da mo video thanh cong")
        logger.log_error("DetectionManager", "Khong tai duoc model YOLO")
        logger.log_violation("ViolationManager", "Track 12 vuot den do")
    """

    _instance: Optional["LoggerManager"] = None
    _lock = threading.Lock()

    LOG_DIR_DEFAULT = "violations/logs"

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, log_dir: Optional[str] = None) -> None:
        if self._initialized:
            return

        self.log_dir = log_dir or self.LOG_DIR_DEFAULT
        os.makedirs(self.log_dir, exist_ok=True)

        self._system_logger = self._build_logger(
            name="system",
            filename=os.path.join(self.log_dir, "system.log"),
            level=logging.DEBUG,
        )
        self._error_logger = self._build_logger(
            name="error",
            filename=os.path.join(self.log_dir, "error.log"),
            level=logging.ERROR,
        )
        self._violation_logger = self._build_logger(
            name="violation",
            filename=os.path.join(self.log_dir, "violation.log"),
            level=logging.INFO,
        )

        self._initialized = True
        self.log_info("LoggerManager", "He thong logging da duoc khoi tao.")

    @staticmethod
    def _build_logger(name: str, filename: str, level: int) -> logging.Logger:
        logger = logging.getLogger(f"traffic_ai.{name}")
        logger.setLevel(level)
        logger.propagate = False

        # Tranh them trung handler khi __init__ duoc goi nhieu lan (Singleton
        # van chay __init__ moi lan khoi tao doi tuong, nhung _initialized
        # flag se chan; day la lop bao ve thu hai).
        if logger.handlers:
            return logger

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(threadName)-15s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        file_handler = RotatingFileHandler(
            filename, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

        # Console handler chi cho system logger, giup debug khi chay tu CLI.
        if name == "system":
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            console_handler.setLevel(logging.INFO)
            logger.addHandler(console_handler)

        return logger

    @classmethod
    def get_instance(cls, log_dir: Optional[str] = None) -> "LoggerManager":
        """Tra ve instance Singleton duy nhat cua LoggerManager."""
        return cls(log_dir=log_dir)

    # ------------------------------------------------------------------
    # API ghi log
    # ------------------------------------------------------------------
    def log_info(self, source: str, message: str) -> None:
        """Ghi log muc INFO vao system.log."""
        self._system_logger.info("[%s] %s", source, message)

    def log_debug(self, source: str, message: str) -> None:
        """Ghi log muc DEBUG vao system.log."""
        self._system_logger.debug("[%s] %s", source, message)

    def log_warning(self, source: str, message: str) -> None:
        """Ghi log muc WARNING vao system.log."""
        self._system_logger.warning("[%s] %s", source, message)

    def log_error(self, source: str, message: str, exc_info: bool = False) -> None:
        """Ghi log muc ERROR vao system.log va error.log.

        Args:
            source: Ten module/thanh phan gay loi (vd: "DetectionManager").
            message: Noi dung loi.
            exc_info: Neu True, dinh kem traceback hien tai.
        """
        self._system_logger.error("[%s] %s", source, message, exc_info=exc_info)
        self._error_logger.error("[%s] %s", source, message, exc_info=exc_info)

    def log_critical(self, source: str, message: str, exc_info: bool = False) -> None:
        """Ghi log muc CRITICAL vao system.log va error.log."""
        self._system_logger.critical("[%s] %s", source, message, exc_info=exc_info)
        self._error_logger.critical("[%s] %s", source, message, exc_info=exc_info)

    def log_violation(self, source: str, message: str) -> None:
        """Ghi log mot vi pham vao violation.log (va system.log)."""
        self._violation_logger.info("[%s] %s", source, message)
        self._system_logger.info("[%s] %s", source, message)
