"""
===== FILE: core/database_manager.py =====

Module: Database Manager
Mo ta: Quan ly toan bo tuong tac voi co so du lieu SQLite (database/traffic.db).

       Theo yeu cau "Database Thread", lop DatabaseManager ke thua QThread
       va xu ly cac thao tac GHI (INSERT / UPDATE / DELETE) thong qua mot
       hang doi thread-safe (queue.Queue) de khong lam nghen ViolationManager
       hay GUI thread.

       Cac thao tac DOC (SELECT / SEARCH / FILTER / LOAD HISTORY) duoc thuc
       hien dong bo thong qua mot connection rieng (chi dung tu GUI thread),
       bao ve boi threading.Lock vi sqlite3 connection khong an toan khi
       dung dong thoi tu nhieu luong neu khong cau hinh check_same_thread.

Bang du lieu chinh: violations
    violation_id    INTEGER PRIMARY KEY AUTOINCREMENT
    timestamp       REAL
    vehicle_type    TEXT
    track_id        INTEGER
    license_plate   TEXT
    violation_type  TEXT
    image_path      TEXT
    plate_image_path TEXT
    email_status    TEXT
    frame_number    INTEGER
    confidence      REAL
    extra_info      TEXT
"""

from __future__ import annotations

import queue
import sqlite3
import threading
import time
from typing import List, Optional, Tuple

from PyQt5.QtCore import QThread, pyqtSignal

from core.logger_manager import LoggerManager
from models.violation import EmailStatus, Violation


class _WriteTask:
    """Mot tac vu ghi du lieu duoc dua vao hang doi cua DatabaseManager."""

    INSERT = "INSERT"
    UPDATE_EMAIL_STATUS = "UPDATE_EMAIL_STATUS"
    DELETE = "DELETE"
    UPDATE_PLATE = "UPDATE_PLATE"

    def __init__(self, task_type: str, payload: dict):
        self.task_type = task_type
        self.payload = payload


class DatabaseManager(QThread):
    """QThread quan ly ghi du lieu SQLite bat dong bo.

    Signals:
        violation_saved(Violation): Phat ra sau khi mot vi pham moi duoc
            INSERT thanh cong (violation_id da duoc gan tu DB).
        violation_updated(int): Phat ra sau khi mot ban ghi (violation_id)
            duoc UPDATE (vd: cap nhat email_status hoac bien so).
        violation_deleted(int): Phat ra sau khi mot ban ghi bi DELETE.
        db_error(str): Phat ra khi co loi xay ra trong qua trinh ghi DB.
    """

    violation_saved = pyqtSignal(object)
    violation_updated = pyqtSignal(int)
    violation_deleted = pyqtSignal(int)
    db_error = pyqtSignal(str)

    def __init__(self, db_path: str = "database/traffic.db", parent=None):
        super().__init__(parent)
        self.db_path = db_path
        self.logger = LoggerManager.get_instance()
        self._task_queue: "queue.Queue[Optional[_WriteTask]]" = queue.Queue()
        self._running = False

        # Connection rieng cho luong ghi (su dung trong run()).
        self._write_conn: Optional[sqlite3.Connection] = None

        # Connection rieng cho cac thao tac doc dong bo tu GUI thread.
        self._read_lock = threading.Lock()
        self._read_conn = sqlite3.connect(self.db_path, check_same_thread=False)

        self._create_tables(self._read_conn)

    # ------------------------------------------------------------------
    # Khoi tao schema
    # ------------------------------------------------------------------
    @staticmethod
    def _create_tables(conn: sqlite3.Connection) -> None:
        """CREATE TABLE cho bang violations (va vehicles_log phu tro)."""
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS violations (
                violation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                vehicle_type TEXT NOT NULL,
                track_id INTEGER NOT NULL,
                license_plate TEXT,
                violation_type TEXT NOT NULL,
                image_path TEXT,
                plate_image_path TEXT,
                email_status TEXT DEFAULT 'Pending',
                frame_number INTEGER DEFAULT 0,
                confidence REAL DEFAULT 0.0,
                extra_info TEXT DEFAULT ''
            )
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_violations_type
            ON violations (violation_type)
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_violations_plate
            ON violations (license_plate)
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS vehicle_log (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id INTEGER NOT NULL,
                class_name TEXT NOT NULL,
                first_seen REAL,
                last_seen REAL,
                session_start REAL
            )
            """
        )
        conn.commit()

    # ------------------------------------------------------------------
    # QThread lifecycle
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Vong lap chinh cua Database Thread - xu ly hang doi ghi du lieu."""
        self._running = True
        self._write_conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.logger.log_info("DatabaseManager", "Database thread da khoi dong.")

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
                self.logger.log_error("DatabaseManager", f"Loi xu ly task DB: {exc}")
                self.db_error.emit(str(exc))

        if self._write_conn:
            self._write_conn.close()
        self.logger.log_info("DatabaseManager", "Database thread da dung.")

    def stop(self) -> None:
        """Dung Database Thread mot cach an toan (xu ly het hang doi truoc)."""
        self._running = False
        self._task_queue.put(None)

    def _process_task(self, task: _WriteTask) -> None:
        assert self._write_conn is not None
        cursor = self._write_conn.cursor()

        if task.task_type == _WriteTask.INSERT:
            violation: Violation = task.payload["violation"]
            cursor.execute(
                """
                INSERT INTO violations
                (timestamp, vehicle_type, track_id, license_plate, violation_type,
                 image_path, plate_image_path, email_status, frame_number,
                 confidence, extra_info)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                violation.to_db_tuple(),
            )
            self._write_conn.commit()
            violation.violation_id = cursor.lastrowid
            self.logger.log_info(
                "DatabaseManager",
                f"Da luu vi pham #{violation.violation_id} "
                f"({violation.violation_type.value}, track {violation.track_id}).",
            )
            self.violation_saved.emit(violation)

        elif task.task_type == _WriteTask.UPDATE_EMAIL_STATUS:
            violation_id = task.payload["violation_id"]
            status: EmailStatus = task.payload["status"]
            cursor.execute(
                "UPDATE violations SET email_status = ? WHERE violation_id = ?",
                (status.value, violation_id),
            )
            self._write_conn.commit()
            self.violation_updated.emit(violation_id)

        elif task.task_type == _WriteTask.UPDATE_PLATE:
            violation_id = task.payload["violation_id"]
            plate_text = task.payload["plate_text"]
            plate_image_path = task.payload.get("plate_image_path", "")
            cursor.execute(
                "UPDATE violations SET license_plate = ?, plate_image_path = ? "
                "WHERE violation_id = ?",
                (plate_text, plate_image_path, violation_id),
            )
            self._write_conn.commit()
            self.violation_updated.emit(violation_id)

        elif task.task_type == _WriteTask.DELETE:
            violation_id = task.payload["violation_id"]
            cursor.execute(
                "DELETE FROM violations WHERE violation_id = ?", (violation_id,)
            )
            self._write_conn.commit()
            self.violation_deleted.emit(violation_id)

    # ------------------------------------------------------------------
    # API ghi (bat dong bo, dua vao hang doi)
    # ------------------------------------------------------------------
    def insert_violation_async(self, violation: Violation) -> None:
        """Dua mot vi pham moi vao hang doi de ghi vao DB (khong block)."""
        self._task_queue.put(_WriteTask(_WriteTask.INSERT, {"violation": violation}))

    def update_email_status_async(self, violation_id: int, status: EmailStatus) -> None:
        """Dua yeu cau cap nhat trang thai email vao hang doi."""
        self._task_queue.put(
            _WriteTask(
                _WriteTask.UPDATE_EMAIL_STATUS,
                {"violation_id": violation_id, "status": status},
            )
        )

    def update_plate_async(
        self, violation_id: int, plate_text: str, plate_image_path: str = ""
    ) -> None:
        """Dua yeu cau cap nhat bien so (sau khi OCR xu ly xong) vao hang doi."""
        self._task_queue.put(
            _WriteTask(
                _WriteTask.UPDATE_PLATE,
                {
                    "violation_id": violation_id,
                    "plate_text": plate_text,
                    "plate_image_path": plate_image_path,
                },
            )
        )

    def delete_violation_async(self, violation_id: int) -> None:
        """Dua yeu cau xoa mot ban ghi vi pham vao hang doi."""
        self._task_queue.put(
            _WriteTask(_WriteTask.DELETE, {"violation_id": violation_id})
        )

    # ------------------------------------------------------------------
    # API doc (dong bo - SELECT / SEARCH / FILTER / LOAD HISTORY)
    # ------------------------------------------------------------------
    _SELECT_COLUMNS = (
        "violation_id, timestamp, vehicle_type, track_id, license_plate, "
        "violation_type, image_path, plate_image_path, email_status, "
        "frame_number, confidence, extra_info"
    )

    def load_history(self, limit: int = 500) -> List[Violation]:
        """Tai lich su vi pham gan nhat (sap xep moi nhat truoc).

        Args:
            limit: So luong ban ghi toi da tra ve.
        """
        with self._read_lock:
            cursor = self._read_conn.cursor()
            cursor.execute(
                f"SELECT {self._SELECT_COLUMNS} FROM violations "
                f"ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )
            rows = cursor.fetchall()
        return [Violation.from_db_row(row) for row in rows]

    def select_all(self) -> List[Violation]:
        """SELECT * tu bang violations (sap xep theo violation_id)."""
        with self._read_lock:
            cursor = self._read_conn.cursor()
            cursor.execute(
                f"SELECT {self._SELECT_COLUMNS} FROM violations ORDER BY violation_id"
            )
            rows = cursor.fetchall()
        return [Violation.from_db_row(row) for row in rows]

    def select_by_id(self, violation_id: int) -> Optional[Violation]:
        """SELECT mot ban ghi theo violation_id."""
        with self._read_lock:
            cursor = self._read_conn.cursor()
            cursor.execute(
                f"SELECT {self._SELECT_COLUMNS} FROM violations WHERE violation_id = ?",
                (violation_id,),
            )
            row = cursor.fetchone()
        return Violation.from_db_row(row) if row else None

    def search_by_plate(self, plate_query: str) -> List[Violation]:
        """SEARCH cac vi pham theo bien so (tim kiem mo - LIKE).

        Args:
            plate_query: Mot phan hoac toan bo chuoi bien so can tim.
        """
        pattern = f"%{plate_query.strip().upper()}%"
        with self._read_lock:
            cursor = self._read_conn.cursor()
            cursor.execute(
                f"SELECT {self._SELECT_COLUMNS} FROM violations "
                f"WHERE UPPER(license_plate) LIKE ? ORDER BY timestamp DESC",
                (pattern,),
            )
            rows = cursor.fetchall()
        return [Violation.from_db_row(row) for row in rows]

    def filter_violations(
        self,
        violation_type: Optional[str] = None,
        vehicle_type: Optional[str] = None,
        start_timestamp: Optional[float] = None,
        end_timestamp: Optional[float] = None,
        email_status: Optional[str] = None,
    ) -> List[Violation]:
        """FILTER vi pham theo nhieu dieu kien (tat ca deu tuy chon).

        Args:
            violation_type: Gia tri ViolationType.value de loc (vd "Vuot den do").
            vehicle_type: Loai phuong tien (vd "car", "motorcycle").
            start_timestamp: Thoi gian bat dau (epoch seconds).
            end_timestamp: Thoi gian ket thuc (epoch seconds).
            email_status: Trang thai email can loc.

        Returns:
            Danh sach Violation thoa man tat ca dieu kien duoc cung cap.
        """
        clauses: List[str] = []
        params: List = []

        if violation_type:
            clauses.append("violation_type = ?")
            params.append(violation_type)
        if vehicle_type:
            clauses.append("vehicle_type = ?")
            params.append(vehicle_type)
        if start_timestamp is not None:
            clauses.append("timestamp >= ?")
            params.append(start_timestamp)
        if end_timestamp is not None:
            clauses.append("timestamp <= ?")
            params.append(end_timestamp)
        if email_status:
            clauses.append("email_status = ?")
            params.append(email_status)

        where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = (
            f"SELECT {self._SELECT_COLUMNS} FROM violations "
            f"{where_clause} ORDER BY timestamp DESC"
        )

        with self._read_lock:
            cursor = self._read_conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
        return [Violation.from_db_row(row) for row in rows]

    def get_statistics(self) -> dict:
        """Tra ve thong ke nhanh: tong so vi pham theo tung loai."""
        with self._read_lock:
            cursor = self._read_conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM violations")
            total = cursor.fetchone()[0]

            cursor.execute(
                "SELECT violation_type, COUNT(*) FROM violations "
                "GROUP BY violation_type"
            )
            by_type = dict(cursor.fetchall())

        return {"total": total, "by_type": by_type}

    def close(self) -> None:
        """Dong cac connection doc khi ung dung tat (goi tu GUI thread)."""
        with self._read_lock:
            try:
                self._read_conn.close()
            except sqlite3.Error:
                pass
