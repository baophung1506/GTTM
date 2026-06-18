"""
===== FILE: core/email_manager.py =====

Module: Email Manager
Mo ta: Mo phong chuc nang gui email thong bao vi pham toi co quan chuc
       nang. KHONG gui email thuc te (khong ket noi SMTP). Moi vi pham
       duoc dua vao hang doi se duoc "gui" (mo phong) va ghi nhan trang
       thai vao database thong qua DatabaseManager.

       Chay duoi dang QThread rieng (Email Thread) de khong lam nghen
       ViolationManager hoac GUI thread.
"""

from __future__ import annotations

import queue
from typing import Optional

from PyQt5.QtCore import QThread, pyqtSignal

from core.database_manager import DatabaseManager
from core.logger_manager import LoggerManager
from models.violation import EmailStatus, Violation


class EmailManager(QThread):
    """QThread mo phong viec gui email thong bao vi pham.

    Signals:
        email_status_changed(int, str): Phat ra (violation_id, status_text)
            sau khi xu ly xong mot vi pham trong hang doi.
    """

    email_status_changed = pyqtSignal(int, str)

    def __init__(
        self,
        database_manager: DatabaseManager,
        sender: str = "traffic.ai.system@example.com",
        recipient: str = "traffic.authority@example.com",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.database_manager = database_manager
        self.sender = sender
        self.recipient = recipient
        self.logger = LoggerManager.get_instance()
        self._task_queue: "queue.Queue[Optional[Violation]]" = queue.Queue()
        self._running = False

    def run(self) -> None:
        """Vong lap chinh cua Email Thread."""
        self._running = True
        self.logger.log_info("EmailManager", "Email thread da khoi dong.")

        while self._running:
            try:
                violation = self._task_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if violation is None:
                break

            self._simulate_send(violation)

        self.logger.log_info("EmailManager", "Email thread da dung.")

    def stop(self) -> None:
        """Dung Email Thread."""
        self._running = False
        self._task_queue.put(None)

    def enqueue_violation(self, violation: Violation) -> None:
        """Dua mot vi pham vao hang doi de mo phong gui email."""
        self._task_queue.put(violation)

    # ------------------------------------------------------------------
    # Logic mo phong
    # ------------------------------------------------------------------
    def _simulate_send(self, violation: Violation) -> None:
        """Mo phong qua trinh gui email - khong ket noi mang thuc te.

        Tao noi dung email (subject/body) mang tinh mo ta day du, ghi log,
        va cap nhat trang thai 'Email sent successfully' vao database.
        """
        subject = self._build_subject(violation)
        body = self._build_body(violation)

        self.logger.log_info(
            "EmailManager",
            f"[MO PHONG] Gui email tu '{self.sender}' den '{self.recipient}' "
            f"- Subject: {subject}",
        )
        self.logger.log_debug("EmailManager", f"Noi dung email:\n{body}")

        # Mo phong gui thanh cong (luon thanh cong trong moi truong demo).
        status = EmailStatus.SENT

        if violation.violation_id is not None:
            self.database_manager.update_email_status_async(
                violation.violation_id, status
            )
            self.email_status_changed.emit(violation.violation_id, status.value)

        self.logger.log_info(
            "EmailManager",
            f"Email sent successfully cho vi pham #{violation.violation_id} "
            f"({violation.violation_type.value}).",
        )

    def _build_subject(self, violation: Violation) -> str:
        return (
            f"[CANH BAO VI PHAM] {violation.violation_type.value} - "
            f"Bien so: {violation.license_plate}"
        )

    def _build_body(self, violation: Violation) -> str:
        return (
            "He thong Camera Giam Sat Giao Thong AI - Bao cao vi pham\n"
            f"Thoi gian        : {violation.get_formatted_timestamp()}\n"
            f"Loai vi pham     : {violation.violation_type.value}\n"
            f"Loai phuong tien : {violation.vehicle_type}\n"
            f"Track ID         : {violation.track_id}\n"
            f"Bien so xe       : {violation.license_plate}\n"
            f"Anh minh chung   : {violation.image_path}\n"
            f"Thong tin them   : {violation.extra_info}\n"
            "----\n"
            "Day la email mo phong, khong duoc gui qua mang thuc te."
        )
