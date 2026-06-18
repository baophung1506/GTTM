"""
===== FILE: models/traffic_light_state.py =====

Module: Traffic Light State Model
Mo ta: Dinh nghia trang thai den giao thong (RED / YELLOW / GREEN) va
       bo dieu khien trang thai den (TrafficLightController) cho phep
       cac thanh phan khac (GUI, ViolationManager) doc/ghi trang thai
       mot cach an toan giua nhieu luong (thread-safe).

Tac gia: Sinh tu dac ta do an "Mo phong he thong giam sat giao thong
         thong minh su dung camera AI".
"""

from __future__ import annotations

import threading
import time
from enum import Enum
from typing import Callable, List


class TrafficLightState(Enum):
    """Enum bieu dien 3 trang thai co ban cua den giao thong."""

    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"

    @classmethod
    def from_string(cls, value: str) -> "TrafficLightState":
        """Chuyen chuoi (khong phan biet hoa/thuong) thanh TrafficLightState.

        Args:
            value: Chuoi dau vao, ví du "red", "Red", "RED".

        Returns:
            TrafficLightState tuong ung.

        Raises:
            ValueError: Neu chuoi khong khop voi bat ky trang thai nao.
        """
        normalized = value.strip().upper()
        for member in cls:
            if member.value == normalized:
                return member
        raise ValueError(f"Trang thai den khong hop le: {value}")

    def to_color_rgb(self) -> tuple:
        """Tra ve mau RGB tuong ung de hien thi tren GUI / overlay video."""
        mapping = {
            TrafficLightState.RED: (220, 20, 60),
            TrafficLightState.YELLOW: (255, 215, 0),
            TrafficLightState.GREEN: (50, 205, 50),
        }
        return mapping[self]


class TrafficLightController:
    """Bo dieu khien trang thai den giao thong, dung chung toan he thong.

    Lop nay duoc thiet ke theo nguyen tac thread-safe (su dung
    threading.Lock) vi trang thai den se duoc doc boi nhieu luong:
    - GUI thread: nguoi dung bam nut RED / YELLOW / GREEN.
    - Violation Manager thread: doc trang thai de kiem tra vi pham
      "Vuot den do" va "De vach dung".

    Ho tro co che Observer (callback) de thong bao khi trang thai
    thay doi, giup GUI cap nhat hien thi ngay lap tuc.
    """

    def __init__(self, initial_state: TrafficLightState = TrafficLightState.RED) -> None:
        self._state: TrafficLightState = initial_state
        self._lock = threading.RLock()
        self._observers: List[Callable[[TrafficLightState], None]] = []
        self._last_changed_timestamp: float = time.time()
        self._history: List[dict] = []

    def get_state(self) -> TrafficLightState:
        """Lay trang thai den hien tai (thread-safe)."""
        with self._lock:
            return self._state

    def set_state(self, new_state: TrafficLightState) -> None:
        """Thay doi trang thai den va thong bao cho cac observer.

        Args:
            new_state: Trang thai den moi (RED / YELLOW / GREEN).
        """
        with self._lock:
            if self._state == new_state:
                return
            old_state = self._state
            self._state = new_state
            self._last_changed_timestamp = time.time()
            self._history.append(
                {
                    "from": old_state.value,
                    "to": new_state.value,
                    "timestamp": self._last_changed_timestamp,
                }
            )
            if len(self._history) > 200:
                self._history = self._history[-200:]

        # Goi observer ngoai khoa de tranh deadlock neu observer
        # quay lai goi get_state()/set_state().
        for callback in list(self._observers):
            try:
                callback(new_state)
            except Exception:
                # Khong de loi tu observer lam crash he thong chinh.
                pass

    def register_observer(self, callback: Callable[[TrafficLightState], None]) -> None:
        """Dang ky ham callback se duoc goi khi trang thai den thay doi.

        Args:
            callback: Ham nhan vao 1 tham so TrafficLightState.
        """
        with self._lock:
            if callback not in self._observers:
                self._observers.append(callback)

    def unregister_observer(self, callback: Callable[[TrafficLightState], None]) -> None:
        """Huy dang ky observer."""
        with self._lock:
            if callback in self._observers:
                self._observers.remove(callback)

    def get_time_since_last_change(self) -> float:
        """Tra ve so giay da troi qua tu lan thay doi trang thai cuoi cung."""
        with self._lock:
            return time.time() - self._last_changed_timestamp

    def get_history(self) -> List[dict]:
        """Tra ve lich su thay doi trang thai (toi da 200 ban ghi gan nhat)."""
        with self._lock:
            return list(self._history)

    def is_red(self) -> bool:
        return self.get_state() == TrafficLightState.RED

    def is_yellow(self) -> bool:
        return self.get_state() == TrafficLightState.YELLOW

    def is_green(self) -> bool:
        return self.get_state() == TrafficLightState.GREEN
