"""
===== FILE: core/tracking_manager.py =====

Tracking Thread - Su dung thuat toan ByteTrack (tu trien khai) de gan track_id
on dinh cho cac phuong tien duoc phat hien o tung frame.

Thuat toan ByteTrack duoc cai dat lai (self-contained) thay vi dung lop
BYTETracker noi bo cua ultralytics de:
    - Tranh phu thuoc phien ban giua cac version ultralytics khac nhau.
    - De dang tuy bien tham so track_thresh / track_buffer / match_thresh.

Luong xu ly:
    DetectionManager --> (frame_payload, detections) --> TrackingManager
    TrackingManager:
        1. Tach detections thanh nhom "vehicle" (motorcycle/car/bus) va
           nhom "helmet" (helmet/no_helmet).
        2. Cap nhat ByteTracker voi cac detections vehicle.
        3. Dong bo STrack -> Vehicle (tao moi hoac update Vehicle hien co).
        4. Gan trang thai helmet cho moi Vehicle la motorcycle dua tren
           overlap giua vung dau xe va detection helmet/no_helmet.
        5. Phat tin hieu tracking_ready(frame_payload, vehicles, helmet_detections)
           cho ViolationManager va GUI.
"""

import queue
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

try:
    from scipy.optimize import linear_sum_assignment
    _SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover - scipy phai co trong requirements.txt
    _SCIPY_AVAILABLE = False

from models.vehicle import Detection, Vehicle, VehicleClass


# =====================================================================
# Kalman Filter cho moi doi tuong duoc theo doi (constant velocity model)
# =====================================================================
class KalmanBoxTracker:
    """
    Bo loc Kalman 8 trang thai: [cx, cy, w, h, vx, vy, vw, vh]
    Mo hinh chuyen dong: van toc khong doi (constant velocity).

    Quan sat (measurement) la [cx, cy, w, h] tu bounding box phat hien duoc.
    """

    def __init__(self, bbox: Tuple[float, float, float, float]):
        # Ma tran chuyen trang thai F (8x8)
        self._dim_x = 8
        self._dim_z = 4

        self.F = np.eye(self._dim_x, dtype=np.float64)
        for i in range(4):
            self.F[i, i + 4] = 1.0  # vi tri += van toc * dt (dt = 1 frame)

        # Ma tran quan sat H (4x8) - chi quan sat duoc cx, cy, w, h
        self.H = np.zeros((self._dim_z, self._dim_x), dtype=np.float64)
        for i in range(4):
            self.H[i, i] = 1.0

        # Ma tran hiep phuong sai nhieu qua trinh Q
        self.Q = np.eye(self._dim_x, dtype=np.float64)
        self.Q[4:, 4:] *= 0.01  # van toc it bien dong hon vi tri

        # Ma tran hiep phuong sai nhieu do R
        self.R = np.eye(self._dim_z, dtype=np.float64) * 1.0

        # Hiep phuong sai uoc luong P
        self.P = np.eye(self._dim_x, dtype=np.float64) * 10.0
        self.P[4:, 4:] *= 1000.0  # van toc ban dau khong chac chan cao

        cx, cy, w, h = KalmanBoxTracker._xyxy_to_cxcywh(bbox)
        self.x = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    @staticmethod
    def _xyxy_to_cxcywh(bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = bbox
        w = max(x2 - x1, 1e-6)
        h = max(y2 - y1, 1e-6)
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        return cx, cy, w, h

    @staticmethod
    def _cxcywh_to_xyxy(cx: float, cy: float, w: float, h: float) -> Tuple[float, float, float, float]:
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return x1, y1, x2, y2

    def predict(self) -> Tuple[float, float, float, float]:
        """Du doan trang thai tiep theo va tra ve bbox (xyxy) du doan."""
        # Bao dam kich thuoc khong am
        if self.x[2] + self.x[6] <= 0:
            self.x[6] = 0.0
        if self.x[3] + self.x[7] <= 0:
            self.x[7] = 0.0

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        cx, cy, w, h = self.x[0], self.x[1], self.x[2], self.x[3]
        return KalmanBoxTracker._cxcywh_to_xyxy(cx, cy, w, h)

    def update(self, bbox: Tuple[float, float, float, float]) -> None:
        """Cap nhat bo loc voi mot quan sat (detection) moi."""
        z = np.array(KalmanBoxTracker._xyxy_to_cxcywh(bbox), dtype=np.float64)

        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I = np.eye(self._dim_x, dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P

    def get_state(self) -> Tuple[float, float, float, float]:
        """Tra ve bbox (xyxy) hien tai cua bo loc."""
        cx, cy, w, h = self.x[0], self.x[1], self.x[2], self.x[3]
        return KalmanBoxTracker._cxcywh_to_xyxy(cx, cy, w, h)


# =====================================================================
# STrack - mot doi tuong duoc theo doi boi ByteTracker
# =====================================================================
class TrackState:
    TRACKED = "tracked"
    LOST = "lost"
    REMOVED = "removed"


class STrack:
    """Bieu dien mot doi tuong duoc theo doi boi ByteTracker."""

    _next_id: int = 1
    _id_lock = threading.Lock()

    def __init__(self, bbox: Tuple[float, float, float, float], score: float, class_name: str):
        self.kalman_filter = KalmanBoxTracker(bbox)
        self.bbox = bbox
        self.score = score
        self.class_name = class_name

        self.track_id = STrack._allocate_id()
        self.state = TrackState.TRACKED

        self.time_since_update = 0
        self.hit_streak = 1
        self.is_activated = True

    @classmethod
    def _allocate_id(cls) -> int:
        with cls._id_lock:
            new_id = cls._next_id
            cls._next_id += 1
            return new_id

    @classmethod
    def reset_id_counter(cls) -> None:
        """Dat lai bo dem track_id (duoc goi khi he thong Reset)."""
        with cls._id_lock:
            cls._next_id = 1

    def predict(self) -> None:
        """Du doan vi tri moi cua track truoc khi gan voi detection."""
        predicted_bbox = self.kalman_filter.predict()
        self.bbox = predicted_bbox
        self.time_since_update += 1

    def update(self, bbox: Tuple[float, float, float, float], score: float, class_name: str) -> None:
        """Cap nhat track voi mot detection da duoc gan (matched)."""
        self.kalman_filter.update(bbox)
        self.bbox = self.kalman_filter.get_state()
        self.score = score
        self.class_name = class_name

        self.time_since_update = 0
        self.hit_streak += 1
        self.state = TrackState.TRACKED
        self.is_activated = True

    def mark_lost(self) -> None:
        self.state = TrackState.LOST

    def mark_removed(self) -> None:
        self.state = TrackState.REMOVED


# =====================================================================
# Cac ham tien ich IoU
# =====================================================================
def _iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    """Tinh Intersection over Union giua hai bounding box dang (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def _iou_matrix(tracks: List["STrack"], detections: List[Detection]) -> np.ndarray:
    """Tao ma tran IoU kich thuoc (len(tracks) x len(detections))."""
    if len(tracks) == 0 or len(detections) == 0:
        return np.zeros((len(tracks), len(detections)), dtype=np.float64)

    matrix = np.zeros((len(tracks), len(detections)), dtype=np.float64)
    for t_idx, trk in enumerate(tracks):
        for d_idx, det in enumerate(detections):
            matrix[t_idx, d_idx] = _iou(trk.bbox, det.bbox)
    return matrix


# =====================================================================
# ByteTracker
# =====================================================================
class ByteTracker:
    """
    Trien khai don gian hoa cua thuat toan ByteTrack:

    - Detections duoc chia thanh "high score" (>= track_thresh) va
      "low score" (< track_thresh, nhung > 0.1) de tang kha nang giu
      track khi phuong tien bi che khuat tam thoi.
    - Quy trinh ganh ket noi (association) gom 3 buoc dua tren IoU va
      thuat toan Hungarian (scipy.optimize.linear_sum_assignment).
    """

    def __init__(self, track_thresh: float = 0.5, track_buffer: int = 30,
                 match_thresh: float = 0.8, frame_rate: int = 30):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.frame_rate = max(frame_rate, 1)
        self.track_buffer = track_buffer

        # So frame toi da duoc phep "mat dau" truoc khi xoa track
        self.max_time_lost = int(self.frame_rate / 30.0 * self.track_buffer)
        if self.max_time_lost <= 0:
            self.max_time_lost = track_buffer

        self.tracked_tracks: List[STrack] = []
        self.lost_tracks: List[STrack] = []

        self._low_thresh = 0.1

    def update(self, detections: List[Detection]) -> List[STrack]:
        """
        Cap nhat ByteTracker voi danh sach Detection moi cua frame hien tai.
        Tra ve danh sach cac STrack dang o trang thai TRACKED.
        """
        # Buoc 1: Du doan vi tri moi cho tat ca cac track hien co
        for trk in self.tracked_tracks:
            trk.predict()
        for trk in self.lost_tracks:
            trk.predict()

        # Phan loai detections theo nguong tin cay
        high_dets: List[Detection] = []
        low_dets: List[Detection] = []
        for det in detections:
            if det.confidence >= self.track_thresh:
                high_dets.append(det)
            elif det.confidence >= self._low_thresh:
                low_dets.append(det)
            # detection co confidence qua thap (< 0.1) bi loai bo hoan toan

        # ----------------------------------------------------------------
        # Buoc 2: Ghep noi tracked_tracks <-> high_dets
        # ----------------------------------------------------------------
        matches_1, unmatched_trk_idx_1, unmatched_det_idx_1 = ByteTracker._associate(
            self.tracked_tracks, high_dets, self.match_thresh
        )

        for trk_idx, det_idx in matches_1:
            trk = self.tracked_tracks[trk_idx]
            det = high_dets[det_idx]
            trk.update(det.bbox, det.confidence, det.class_name)

        remaining_tracked = [self.tracked_tracks[i] for i in unmatched_trk_idx_1]
        remaining_high_dets = [high_dets[i] for i in unmatched_det_idx_1]

        # ----------------------------------------------------------------
        # Buoc 3: Ghep noi remaining_tracked (con lai sau buoc 2) <-> low_dets
        # ----------------------------------------------------------------
        matches_2, unmatched_trk_idx_2, _unmatched_low_idx_2 = ByteTracker._associate(
            remaining_tracked, low_dets, self.match_thresh
        )

        for trk_idx, det_idx in matches_2:
            trk = remaining_tracked[trk_idx]
            det = low_dets[det_idx]
            trk.update(det.bbox, det.confidence, det.class_name)

        still_unmatched_tracked = [remaining_tracked[i] for i in unmatched_trk_idx_2]

        # Cac track con khong duoc ghep -> chuyen sang trang thai LOST
        for trk in still_unmatched_tracked:
            trk.mark_lost()

        # ----------------------------------------------------------------
        # Buoc 4: Ghep noi lost_tracks <-> remaining_high_dets
        # (giup phuc hoi track sau khi bi che khuat tam thoi)
        # ----------------------------------------------------------------
        matches_3, _unmatched_lost_idx_3, unmatched_det_idx_3 = ByteTracker._associate(
            self.lost_tracks, remaining_high_dets, self.match_thresh
        )

        recovered_tracks: List[STrack] = []
        for trk_idx, det_idx in matches_3:
            trk = self.lost_tracks[trk_idx]
            det = remaining_high_dets[det_idx]
            trk.update(det.bbox, det.confidence, det.class_name)
            recovered_tracks.append(trk)

        recovered_ids = {id(t) for t in recovered_tracks}
        self.lost_tracks = [t for t in self.lost_tracks if id(t) not in recovered_ids]

        # ----------------------------------------------------------------
        # Buoc 5: Tao track moi cho cac detection high-score con lai
        # ----------------------------------------------------------------
        new_tracks: List[STrack] = []
        for det_idx in unmatched_det_idx_3:
            det = remaining_high_dets[det_idx]
            new_track = STrack(det.bbox, det.confidence, det.class_name)
            new_tracks.append(new_track)

        # ----------------------------------------------------------------
        # Buoc 6: Tong hop lai danh sach track theo trang thai
        # ----------------------------------------------------------------
        matched_tracked_ids = {id(self.tracked_tracks[i]) for i, _ in matches_1}
        for i, _ in matches_2:
            matched_tracked_ids.add(id(remaining_tracked[i]))

        all_tracked = []
        for trk in self.tracked_tracks:
            if id(trk) in matched_tracked_ids and trk.state != TrackState.LOST:
                all_tracked.append(trk)

        all_tracked.extend(recovered_tracks)
        all_tracked.extend(new_tracks)

        self.tracked_tracks = all_tracked

        # Cap nhat danh sach lost_tracks: them cac track vua bi mat
        for trk in still_unmatched_tracked:
            if trk.time_since_update <= self.max_time_lost:
                self.lost_tracks.append(trk)

        # Loai bo cac lost_tracks da qua thoi gian cho phep
        self.lost_tracks = [
            trk for trk in self.lost_tracks
            if trk.time_since_update <= self.max_time_lost
        ]
        for trk in self.lost_tracks:
            trk.mark_lost()

        return list(self.tracked_tracks)

    @staticmethod
    def _associate(
        tracks: List[STrack],
        detections: List[Detection],
        match_thresh: float,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Ghep noi tracks va detections dua tren IoU bang thuat toan Hungarian.

        Tra ve:
            matches: danh sach cac cap (track_index, detection_index) duoc ghep
            unmatched_track_indices: cac index trong `tracks` khong duoc ghep
            unmatched_det_indices: cac index trong `detections` khong duoc ghep
        """
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))

        iou_mat = _iou_matrix(tracks, detections)
        cost_mat = 1.0 - iou_mat

        if _SCIPY_AVAILABLE:
            row_idx, col_idx = linear_sum_assignment(cost_mat)
        else:
            # Fallback don gian: gan tham lam (greedy) theo IoU giam dan
            row_idx, col_idx = ByteTracker._greedy_assignment(iou_mat)

        matches: List[Tuple[int, int]] = []
        unmatched_tracks = set(range(len(tracks)))
        unmatched_dets = set(range(len(detections)))

        for r, c in zip(row_idx, col_idx):
            if iou_mat[r, c] >= (1.0 - match_thresh):
                matches.append((int(r), int(c)))
                unmatched_tracks.discard(int(r))
                unmatched_dets.discard(int(c))

        return matches, sorted(unmatched_tracks), sorted(unmatched_dets)

    @staticmethod
    def _greedy_assignment(iou_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback gan tham lam khi scipy khong co san."""
        rows, cols = [], []
        mat = iou_mat.copy()
        n_rows, n_cols = mat.shape
        used_rows = set()
        used_cols = set()

        flat_indices = np.dstack(np.unravel_index(np.argsort(-mat, axis=None), mat.shape))[0]
        for r, c in flat_indices:
            r, c = int(r), int(c)
            if r in used_rows or c in used_cols:
                continue
            if mat[r, c] <= 0:
                continue
            rows.append(r)
            cols.append(c)
            used_rows.add(r)
            used_cols.add(c)
            if len(used_rows) == n_rows or len(used_cols) == n_cols:
                break

        return np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64)


# =====================================================================
# TrackingManager (QThread) - "Tracking Thread"
# =====================================================================
class TrackingManager(QThread):
    """
    Tracking Thread: nhan (frame_payload, detections) tu DetectionManager,
    chay ByteTracker, duy tri Dict[track_id -> Vehicle] va gan trang thai
    helmet cho cac Vehicle la motorcycle.

    Tin hieu phat ra:
        tracking_ready(dict, list, list):
            frame_payload, danh sach Vehicle dang active, danh sach
            Detection helmet/no_helmet (de GUI co the ve overlay rieng)
        tracking_error(str)
    """

    tracking_ready = pyqtSignal(dict, list, list)
    tracking_error = pyqtSignal(str)

    # Nguong overlap toi thieu giua vung dau xe va detection helmet/no_helmet
    HELMET_OVERLAP_THRESHOLD = 0.05

    # Thoi gian (giay) khong thay vehicle truoc khi xoa khoi danh sach theo doi
    STALE_TIMEOUT_SECONDS = 5.0

    def __init__(self, track_thresh: float = 0.5, track_buffer: int = 30,
                 match_thresh: float = 0.8, frame_rate: int = 30, parent=None):
        super().__init__(parent)

        self._track_thresh = track_thresh
        self._track_buffer = track_buffer
        self._match_thresh = match_thresh
        self._frame_rate = frame_rate

        self._tracker = ByteTracker(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            frame_rate=frame_rate,
        )

        self._vehicles: Dict[int, Vehicle] = {}
        self._vehicles_lock = threading.Lock()

        self._input_queue: "queue.Queue" = queue.Queue(maxsize=2)
        self._running = False

        self._vehicle_classes = VehicleClass.vehicle_classes()
        self._helmet_classes = VehicleClass.helmet_related_classes()

    # ------------------------------------------------------------
    # API cong khai
    # ------------------------------------------------------------
    def enqueue_detections(self, frame_payload: dict, detections: List[Detection]) -> None:
        """
        Dua (frame_payload, detections) vao hang doi xu ly.
        Ap dung chinh sach drop-oldest neu hang doi day, dam bao
        Tracking Thread luon xu ly frame moi nhat (thoi gian thuc).
        """
        item = (frame_payload, detections)
        try:
            self._input_queue.put_nowait(item)
        except queue.Full:
            try:
                self._input_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._input_queue.put_nowait(item)
            except queue.Full:
                pass

    def get_vehicle(self, track_id: int) -> Optional[Vehicle]:
        """Lay Vehicle theo track_id (thread-safe)."""
        with self._vehicles_lock:
            return self._vehicles.get(track_id)

    def get_all_vehicles(self) -> List[Vehicle]:
        """Lay snapshot danh sach toan bo Vehicle dang theo doi."""
        with self._vehicles_lock:
            return list(self._vehicles.values())

    def reset(self) -> None:
        """
        Dat lai toan bo trang thai tracking: xoa Vehicle, tao lai
        ByteTracker va dat lai bo dem track_id ve 1.
        """
        with self._vehicles_lock:
            self._vehicles.clear()

        self._tracker = ByteTracker(
            track_thresh=self._track_thresh,
            track_buffer=self._track_buffer,
            match_thresh=self._match_thresh,
            frame_rate=self._frame_rate,
        )
        STrack.reset_id_counter()

        # Xoa cac item con trong hang doi
        while True:
            try:
                self._input_queue.get_nowait()
            except queue.Empty:
                break

    def stop(self) -> None:
        """Dung Tracking Thread mot cach an toan."""
        self._running = False
        # Day mot item "poison pill" de unblock get() dang cho
        try:
            self._input_queue.put_nowait((None, None))
        except queue.Full:
            pass
        self.wait(3000)

    # ------------------------------------------------------------
    # Vong lap chinh cua QThread
    # ------------------------------------------------------------
    def run(self) -> None:
        self._running = True

        while self._running:
            try:
                frame_payload, detections = self._input_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if frame_payload is None:
                # poison pill -> dung thread
                break

            try:
                vehicles, helmet_dets = self._process(detections, frame_payload)
                self.tracking_ready.emit(frame_payload, vehicles, helmet_dets)
            except Exception as exc:  # pylint: disable=broad-except
                self.tracking_error.emit(f"Loi xu ly tracking: {exc}")

    # ------------------------------------------------------------
    # Xu ly chinh
    # ------------------------------------------------------------
    def _process(self, detections: List[Detection], frame_payload: dict) -> Tuple[List[Vehicle], List[Detection]]:
        """
        Tach detections thanh nhom vehicle va helmet, chay ByteTracker,
        dong bo Vehicle dict va gan trang thai helmet.

        Tra ve (danh sach Vehicle active, danh sach Detection helmet/no_helmet)
        """
        timestamp = frame_payload.get("timestamp", 0.0)
        frame_number = frame_payload.get("frame_number", 0)

        vehicle_dets: List[Detection] = []
        helmet_dets: List[Detection] = []

        for det in detections:
            if det.class_name in self._vehicle_classes and det.class_name != VehicleClass.PERSON.value:
                vehicle_dets.append(det)
            elif det.class_name in self._helmet_classes:
                helmet_dets.append(det)
            elif det.class_name == VehicleClass.PERSON.value:
                # Person duoc theo doi nhu mot doi tuong rieng (de phat hien
                # nguoi di bo / nguoi dieu khien xe), nhung khong tinh vao
                # nhom xe co dong (motorcycle/car/bus).
                vehicle_dets.append(det)

        # Chay ByteTracker
        active_stracks = self._tracker.update(vehicle_dets)

        active_ids = set()
        with self._vehicles_lock:
            for strack in active_stracks:
                active_ids.add(strack.track_id)
                bbox = (
                    float(strack.bbox[0]), float(strack.bbox[1]),
                    float(strack.bbox[2]), float(strack.bbox[3]),
                )

                if strack.track_id not in self._vehicles:
                    vehicle = Vehicle(
                        track_id=strack.track_id,
                        class_name=strack.class_name,
                        bbox=bbox,
                        center_point=Detection.__new__(Detection) and (0.0, 0.0),
                        first_seen=timestamp,
                        last_seen=timestamp,
                    )
                    # center_point se duoc tinh chinh xac trong update()
                    vehicle.center_point = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
                    self._vehicles[strack.track_id] = vehicle

                vehicle = self._vehicles[strack.track_id]
                vehicle.update(bbox=bbox, class_name=strack.class_name,
                                timestamp=timestamp, frame_number=frame_number)

            # Xoa cac vehicle qua cu (khong xuat hien trong active_ids va
            # da lau khong duoc cap nhat)
            stale_ids = []
            for track_id, vehicle in self._vehicles.items():
                if track_id in active_ids:
                    continue
                if self._is_stale(vehicle, timestamp):
                    stale_ids.append(track_id)

            for track_id in stale_ids:
                del self._vehicles[track_id]

            active_vehicles = [self._vehicles[tid] for tid in active_ids if tid in self._vehicles]

        # Gan trang thai helmet cho cac vehicle la motorcycle
        self._assign_helmet_status(active_vehicles, helmet_dets)

        return active_vehicles, helmet_dets

    def _is_stale(self, vehicle: Vehicle, current_timestamp: float) -> bool:
        """Kiem tra mot Vehicle da qua lau khong duoc cap nhat hay chua."""
        return (current_timestamp - vehicle.last_seen) > TrackingManager.STALE_TIMEOUT_SECONDS

    def _assign_helmet_status(self, vehicles: List[Vehicle], helmet_dets: List[Detection]) -> None:
        """
        Voi moi Vehicle la motorcycle, xac dinh "vung dau xe" (head_region)
        la nua tren cua bounding box, sau do tim detection helmet/no_helmet
        co IoU lon nhat voi head_region. Neu IoU >= HELMET_OVERLAP_THRESHOLD:
            - class_name == "helmet"    -> has_helmet = True
            - class_name == "no_helmet" -> has_helmet = False
        Neu khong tim thay overlap nao dat nguong, giu nguyen gia tri cu
        (mac dinh None = chua xac dinh).
        """
        if not helmet_dets:
            return

        for vehicle in vehicles:
            if vehicle.class_name != VehicleClass.MOTORCYCLE.value:
                continue

            x1, y1, x2, y2 = vehicle.bbox
            height = y2 - y1
            head_region = (x1, y1, x2, y1 + height * 0.5)

            best_iou = 0.0
            best_det: Optional[Detection] = None

            for det in helmet_dets:
                iou_val = _iou(head_region, det.bbox)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_det = det

            if best_det is not None and best_iou >= TrackingManager.HELMET_OVERLAP_THRESHOLD:
                if best_det.class_name == VehicleClass.HELMET.value:
                    vehicle.has_helmet = True
                elif best_det.class_name == VehicleClass.NO_HELMET.value:
                    vehicle.has_helmet = False
