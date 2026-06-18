"""
===== FILE: core/video_manager.py =====

Module: Video Manager
Mo ta: QThread (Video Thread) chiu trach nhiem doc frame tu file video
       MP4 bang OpenCV (cv2.VideoCapture), tinh toan FPS thuc te va phat
       (emit) tin hieu frame_ready chua frame moi cho DetectionManager.

       Ho tro cac lenh: load video, start, pause, resume, stop, reset.

       Viec dieu phoi toc do phat (frame pacing) duoc thuc hien bang
       QThread.msleep() dua tren FPS goc cua video - day la co che mo
       phong "camera thoi gian thuc", KHONG phai dung sleep de dieu phoi
       giua cac luong xu ly (giao tiep giua cac luong van hoan toan dua
       tren signal/slot va Queue thread-safe).
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from core.logger_manager import LoggerManager


class VideoManager(QThread):
    """QThread doc va phat frame tu file video MP4.

    Signals:
        frame_ready(object): Phat ra dict {
            "frame": np.ndarray (BGR),
            "frame_number": int,
            "timestamp": float,
            "fps": float,          # FPS xu ly thuc te (cap nhat dinh ky)
            "total_frames": int,
            "video_fps": float,    # FPS goc cua file video
        }
        video_finished(): Phat ra khi video phat het (va khong loop).
        video_error(str): Phat ra khi co loi mo/doc video.
        playback_state_changed(str): Phat ra khi trang thai thay doi
            ("PLAYING", "PAUSED", "STOPPED").
    """

    frame_ready = pyqtSignal(object)
    video_finished = pyqtSignal()
    video_error = pyqtSignal(str)
    playback_state_changed = pyqtSignal(str)

    STATE_STOPPED = "STOPPED"
    STATE_PLAYING = "PLAYING"
    STATE_PAUSED = "PAUSED"

    def __init__(self, loop: bool = False, parent=None) -> None:
        super().__init__(parent)
        self.logger = LoggerManager.get_instance()
        self.loop = loop

        self._video_path: Optional[str] = None
        self._cap: Optional[cv2.VideoCapture] = None

        self._running = False
        self._paused = False
        self._state = self.STATE_STOPPED

        self._video_fps: float = 25.0
        self._total_frames: int = 0
        self._frame_number: int = 0

        # Bien dem FPS xu ly thuc te (cap nhat moi giay).
        self._fps_counter = 0
        self._fps_timer_start = time.time()
        self._current_processing_fps = 0.0

    # ------------------------------------------------------------------
    # Quan ly nguon video
    # ------------------------------------------------------------------
    def load_video(self, video_path: str) -> bool:
        """Mo file video MP4 va doc thong tin co ban (FPS, tong so frame).

        Args:
            video_path: Duong dan toi file .mp4.

        Returns:
            True neu mo thanh cong, False neu loi.
        """
        if self._cap is not None:
            self._cap.release()
            self._cap = None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            msg = f"Khong the mo file video: {video_path}"
            self.logger.log_error("VideoManager", msg)
            self.video_error.emit(msg)
            return False

        self._cap = cap
        self._video_path = video_path
        self._video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._frame_number = 0

        self.logger.log_info(
            "VideoManager",
            f"Da mo video '{video_path}' "
            f"(fps={self._video_fps:.2f}, total_frames={self._total_frames}).",
        )
        return True

    # ------------------------------------------------------------------
    # QThread lifecycle
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Vong lap chinh cua Video Thread."""
        if self._cap is None:
            msg = "Chua nap video truoc khi start()."
            self.logger.log_error("VideoManager", msg)
            self.video_error.emit(msg)
            return

        self._running = True
        self._paused = False
        self._set_state(self.STATE_PLAYING)
        self.logger.log_info("VideoManager", "Video thread da khoi dong.")

        frame_interval = 1.0 / self._video_fps if self._video_fps > 0 else 0.04
        next_frame_time = time.time()

        while self._running:
            if self._paused:
                self.msleep(20)
                next_frame_time = time.time()
                continue

            ret, frame = self._cap.read()
            if not ret:
                if self.loop:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self._frame_number = 0
                    continue
                self.logger.log_info("VideoManager", "Video da phat het.")
                self.video_finished.emit()
                break

            self._frame_number += 1
            now = time.time()
            self._update_fps_counter()

            payload = {
                "frame": frame,
                "frame_number": self._frame_number,
                "timestamp": now,
                "fps": self._current_processing_fps,
                "total_frames": self._total_frames,
                "video_fps": self._video_fps,
            }
            self.frame_ready.emit(payload)

            # Frame pacing de mo phong toc do phat thuc te cua camera.
            next_frame_time += frame_interval
            sleep_time = next_frame_time - time.time()
            if sleep_time > 0:
                self.msleep(int(sleep_time * 1000))
            else:
                next_frame_time = time.time()

        self._set_state(self.STATE_STOPPED)
        self.logger.log_info("VideoManager", "Video thread da dung.")

    def _update_fps_counter(self) -> None:
        self._fps_counter += 1
        elapsed = time.time() - self._fps_timer_start
        if elapsed >= 1.0:
            self._current_processing_fps = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_timer_start = time.time()

    # ------------------------------------------------------------------
    # Dieu khien playback
    # ------------------------------------------------------------------
    def pause(self) -> None:
        """Tam dung phat video (khong dung thread)."""
        self._paused = True
        self._set_state(self.STATE_PAUSED)
        self.logger.log_info("VideoManager", "Video da tam dung.")

    def resume(self) -> None:
        """Tiep tuc phat video sau khi pause."""
        self._paused = False
        self._set_state(self.STATE_PLAYING)
        self.logger.log_info("VideoManager", "Video tiep tuc phat.")

    def stop(self) -> None:
        """Dung han Video Thread (phai goi wait() sau do tu noi goi)."""
        self._running = False
        self._paused = False

    def reset(self) -> None:
        """Dat lai vi tri doc ve frame dau tien (chi co tac dung khi dang
        dung hoac pause)."""
        if self._cap is not None:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._frame_number = 0
            self.logger.log_info("VideoManager", "Da reset video ve frame dau tien.")

    def release(self) -> None:
        """Giai phong tai nguyen VideoCapture."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _set_state(self, state: str) -> None:
        self._state = state
        self.playback_state_changed.emit(state)

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------
    def get_state(self) -> str:
        return self._state

    def get_video_fps(self) -> float:
        return self._video_fps

    def get_total_frames(self) -> int:
        return self._total_frames

    def get_current_frame_number(self) -> int:
        return self._frame_number

    def get_video_path(self) -> Optional[str]:
        return self._video_path
