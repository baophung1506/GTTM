"""
===== FILE: gui/main_window.py =====

MainWindow - Cua so chinh cua he thong giam sat giao thong thong minh.
Ket noi tat ca cac QThread (Video / Detection / Tracking / Database /
OCR / Email) qua tin hieu PyQt5, hien thi GUI va dieu phoi hoat dong.

So do luong tin hieu:
    VideoManager     --frame_ready-->       DetectionManager
    DetectionManager --detections_ready-->  TrackingManager
    TrackingManager  --tracking_ready-->    ViolationManager (trong GUI thread)
                                        --> VideoWidget (cap nhat hien thi)
    ViolationManager --violation_detected-> ViolationPanel + ImagePreviewPanel
    DatabaseManager  --violation_saved-->   ViolationPanel (cap nhat ID thuc)
    EmailManager     --email_status_changed-> ViolationPanel

NGUON DAU VAO: Video file / Anh tinh / Webcam (camera truc tiep), chon
qua nut dropdown "Chọn Nguồn" tren toolbar. Ca 3 nguon deu dung chung
VideoManager.load_source() (anh tinh duoc xu ly nhu video 1 frame duy
nhat, webcam dung index camera thay vi duong dan file).
"""

import os

from PyQt5.QtCore import Qt, QTimer, pyqtSlot
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import (QAction, QApplication, QFileDialog, QGroupBox,
                              QHBoxLayout, QInputDialog, QLabel, QMainWindow,
                              QMenu, QMessageBox, QSizePolicy, QSplitter,
                              QStatusBar, QToolBar, QToolButton, QVBoxLayout,
                              QWidget)

from core.config_manager import ConfigManager
from core.database_manager import DatabaseManager
from core.detection_manager import DetectionManager
from core.email_manager import EmailManager
from core.logger_manager import LoggerManager
from core.ocr_manager import OCRManager
from core.tracking_manager import TrackingManager
from core.video_manager import VideoManager
from core.violation_manager import ViolationManager
from gui.image_preview_panel import ImagePreviewPanel
from gui.styles import DARK_STYLE
from gui.traffic_light_panel import TrafficLightPanel
from gui.video_widget import VideoWidget
from gui.violation_panel import ViolationPanel
from models.traffic_light_state import TrafficLightController, TrafficLightState
from models.vehicle import Vehicle
from models.violation import Violation


class MainWindow(QMainWindow):
    """Cua so chinh cua ung dung."""

    def __init__(self):
        super().__init__()

        # ----------------------------------------------------------------
        # Khoi tao cac manager (theo thu tu phu thuoc)
        # ----------------------------------------------------------------
        self._logger = LoggerManager()
        self._cfg = ConfigManager()

        self._db_manager = DatabaseManager(db_path=self._cfg.get_database_path())
        self._ocr_manager = OCRManager(
            languages=self._cfg.get("ocr.languages", ["vi", "en"]),
            use_gpu=self._cfg.get("ocr.use_gpu", False),
            min_confidence=self._cfg.get("ocr.min_confidence", 0.5),
            plate_model_path=self._cfg.get("ocr.plate_model_path", None),
            plate_confidence=self._cfg.get("ocr.plate_confidence", 0.4),
        )

        self._email_manager = EmailManager(
            database_manager=self._db_manager,
            sender=self._cfg.get("email.sender", default=""),
            recipient=self._cfg.get("email.recipient", default="")
        )
        self._video_manager = VideoManager()
        self._detection_manager = DetectionManager(
            vehicle_model_path=self._cfg.get_vehicle_model_path(),
            helmet_model_path=self._cfg.get_helmet_model_path(),
            confidence_threshold=self._cfg.get("models.confidence", default=0.45),
            iou_threshold=self._cfg.get("models.iou", default=0.5),
            device=self._cfg.get("models.device", "cpu"),
        )
        self._tracking_manager = TrackingManager(
            track_thresh=self._cfg.get("tracking.track_thresh", 0.5),
            track_buffer=self._cfg.get("tracking.track_buffer", 30),
            match_thresh=self._cfg.get("tracking.match_thresh", 0.8),
            frame_rate=self._cfg.get("tracking.frame_rate", 30),
        )

        self._traffic_light_ctrl = TrafficLightController(initial_state=TrafficLightState.RED)

        self._violation_manager = ViolationManager(
            config_manager=self._cfg,
            database_manager=self._db_manager,
            ocr_manager=self._ocr_manager,
            email_manager=self._email_manager,
            traffic_light_controller=self._traffic_light_ctrl,
            logger=self._logger,
        )

        # ----------------------------------------------------------------
        # Trang thai nguon dau vao hien tai ("video" | "image" | "camera")
        # ----------------------------------------------------------------
        self._current_source_type = "video"
        self._current_source_label = "Chưa chọn nguồn"

        # ----------------------------------------------------------------
        # Xay dung giao dien
        # ----------------------------------------------------------------
        self._setup_window()
        self._setup_toolbar()
        self._setup_central_widget()
        self._setup_status_bar()
        self._connect_signals()

        # ----------------------------------------------------------------
        # Khoi dong cac daemon thread
        # ----------------------------------------------------------------
        self._db_manager.start()
        self._ocr_manager.start()
        self._email_manager.start()
        self._detection_manager.start()
        self._tracking_manager.start()

        self._logger.log_info(source="MainWindow", message="He thong khoi dong thanh cong.")
        self._set_status("Sẵn sàng. Chọn nguồn (Video / Ảnh / Camera) để bắt đầu.")

    # ================================================================
    # Setup GUI
    # ================================================================

    def _setup_window(self) -> None:
        self.setWindowTitle("Hệ Thống Giám Sát Giao Thông Thông Minh  |  AI Camera")
        self.setMinimumSize(1280, 720)
        self.resize(1440, 860)
        self.setStyleSheet(DARK_STYLE)

    def _setup_toolbar(self) -> None:
        tb = QToolBar("Điều khiển", self)
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.addToolBar(tb)

        tb.addWidget(QLabel("  📂 "))

        # ---- Nut dropdown chon nguon: Video / Anh / Camera ----
        # (thay the nut "Chon Video" don le truoc day; giu nguyen vi tri
        # va phong cach hien thi tren toolbar)
        self._btn_source = QToolButton(self)
        self._btn_source.setText("Chọn Nguồn ▾")
        self._btn_source.setPopupMode(QToolButton.InstantPopup)
        self._btn_source.setToolButtonStyle(Qt.ToolButtonTextOnly)

        source_menu = QMenu(self._btn_source)

        act_open_video = QAction("🎬  Chọn Video...", self)
        act_open_video.triggered.connect(self._open_video)
        source_menu.addAction(act_open_video)

        act_open_image = QAction("🖼️  Chọn Ảnh...", self)
        act_open_image.triggered.connect(self._open_image)
        source_menu.addAction(act_open_image)

        act_open_camera = QAction("📷  Mở Camera (Webcam)...", self)
        act_open_camera.triggered.connect(self._open_camera)
        source_menu.addAction(act_open_camera)

        self._btn_source.setMenu(source_menu)
        tb.addWidget(self._btn_source)

        tb.addSeparator()

        self._act_start = QAction("▶ Bắt đầu", self)
        self._act_start.setEnabled(False)
        self._act_start.triggered.connect(self._start_processing)
        tb.addAction(self._act_start)

        self._act_pause = QAction("⏸ Tạm dừng", self)
        self._act_pause.setEnabled(False)
        self._act_pause.triggered.connect(self._pause_processing)
        tb.addAction(self._act_pause)

        self._act_stop = QAction("⏹ Dừng", self)
        self._act_stop.setEnabled(False)
        self._act_stop.triggered.connect(self._stop_processing)
        tb.addAction(self._act_stop)

        self._act_reset = QAction("🔄 Reset", self)
        self._act_reset.triggered.connect(self._reset_all)
        tb.addAction(self._act_reset)

        tb.addSeparator()

        self._lbl_video_path = QLabel("  Chưa chọn nguồn")
        self._lbl_video_path.setStyleSheet("color: #a0a0c0;")
        tb.addWidget(self._lbl_video_path)

    def _setup_central_widget(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        # ---- Chia trai / phai ----
        splitter = QSplitter(Qt.Horizontal)

        # === Cot trai: Video + Den giao thong ===
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        self._video_widget = VideoWidget()
        left_layout.addWidget(self._video_widget, 5)

        self._traffic_light_panel = TrafficLightPanel(self._traffic_light_ctrl)
        self._traffic_light_panel.setMaximumHeight(180)
        left_layout.addWidget(self._traffic_light_panel, 1)

        splitter.addWidget(left_widget)

        # === Cot phai: Bang vi pham + Xem anh ===
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        self._violation_panel = ViolationPanel(self._db_manager)
        right_layout.addWidget(self._violation_panel, 3)

        self._preview_panel = ImagePreviewPanel()
        self._preview_panel.setMaximumHeight(340)
        right_layout.addWidget(self._preview_panel, 2)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 6)
        splitter.setStretchFactor(1, 4)

        main_layout.addWidget(splitter)

    def _setup_status_bar(self) -> None:
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)

        self._lbl_fps = QLabel("FPS: —")
        self._lbl_vehicles = QLabel("Phương tiện: 0")
        self._lbl_violations = QLabel("Vi phạm: 0")
        self._lbl_model = QLabel("Model: —")

        for lbl in [self._lbl_fps, self._lbl_vehicles, self._lbl_violations, self._lbl_model]:
            lbl.setStyleSheet("padding: 0 12px;")
            self._status_bar.addPermanentWidget(lbl)

        self._violation_count = 0

    # ================================================================
    # Ket noi tin hieu
    # ================================================================

    def _connect_signals(self) -> None:
        # Video -> Detection
        self._video_manager.frame_ready.connect(self._on_frame_ready)
        self._video_manager.video_finished.connect(self._on_video_finished)
        self._video_manager.video_error.connect(self._on_error)

        # Detection -> Tracking
        self._detection_manager.detections_ready.connect(self._on_detections_ready)
        self._detection_manager.detection_error.connect(self._on_error)
        self._detection_manager.models_loaded.connect(self._on_models_loaded)

        # Tracking -> ViolationManager + GUI
        self._tracking_manager.tracking_ready.connect(self._on_tracking_ready)
        self._tracking_manager.tracking_error.connect(self._on_error)

        # ViolationManager -> ViolationPanel + ImagePreviewPanel
        self._violation_manager.violation_detected.connect(self._on_violation_detected)

        # DatabaseManager -> ViolationPanel
        self._db_manager.violation_saved.connect(self._on_violation_saved)

        # EmailManager -> ViolationPanel
        self._email_manager.email_status_changed.connect(self._on_email_status_changed)

        # Den giao thong
        self._traffic_light_panel.state_changed.connect(self._on_traffic_light_changed)

        # Click vao bang vi pham -> Xem anh
        self._violation_panel.violation_selected.connect(self._preview_panel.show_violation)

    # ================================================================
    # Slot xu ly tin hieu
    # ================================================================

    @pyqtSlot(dict)
    def _on_frame_ready(self, payload: dict) -> None:
        """Nhan frame moi tu VideoManager -> chuyen cho DetectionManager."""
        self._detection_manager.enqueue_frame(payload)
        fps = payload.get("fps", 0.0)
        self._lbl_fps.setText(f"FPS: {fps:.1f}")

    @pyqtSlot(object, object)
    def _on_detections_ready(self, frame_payload: dict, detections: list) -> None:
        """Nhan detections tu DetectionManager -> chuyen cho TrackingManager."""
        self._tracking_manager.enqueue_detections(frame_payload, detections)

    @pyqtSlot(dict, list, list)
    def _on_tracking_ready(self, frame_payload: dict, vehicles: list, helmet_dets: list) -> None:
        """
        Nhan ket qua tracking -> cap nhat VideoWidget + chay ViolationManager.
        """
        frame = frame_payload.get("frame")
        if frame is None:
            return

        tl_state = self._traffic_light_ctrl.get_state()
        fps = frame_payload.get("fps", 0.0)
        frame_number = frame_payload.get("frame_number", 0)

        # Cap nhat hien thi video
        self._video_widget.update_frame(
            frame=frame,
            vehicles=vehicles,
            helmet_dets=helmet_dets,
            fps=fps,
            frame_number=frame_number,
            traffic_light_state=tl_state,
            stop_line=self._cfg.get_stop_line(),
        )

        # Kiem tra vi pham
        self._violation_manager.process_frame(frame_payload, vehicles, helmet_dets, frame)

        # Cap nhat status bar
        self._lbl_vehicles.setText(f"Phương tiện: {len(vehicles)}")

    @pyqtSlot(object)
    def _on_violation_detected(self, violation: Violation) -> None:
        """Them vi pham moi vao bang."""
        self._violation_panel.add_violation(violation)
        self._violation_count += 1
        self._lbl_violations.setText(f"Vi phạm: {self._violation_count}")
        self._set_status(f"⚠ Vi phạm mới: {violation.violation_type} | Track#{violation.track_id}")

    @pyqtSlot(object)
    def _on_violation_saved(self, violation: Violation) -> None:
        """Database da luu thanh cong (co violation_id thuc)."""
        pass  # ViolationPanel da co du lieu, chi can log

    @pyqtSlot(int, str)
    def _on_email_status_changed(self, violation_id: int, status: str) -> None:
        self._violation_panel.update_email_status(violation_id, status)

    @pyqtSlot(TrafficLightState)
    def _on_traffic_light_changed(self, state: TrafficLightState) -> None:
        self._set_status(f"Đèn giao thông: {state.value.upper()}")

    @pyqtSlot(bool, bool)
    def _on_models_loaded(self, vehicle_ok: bool, helmet_ok: bool) -> None:
        v_txt = "✅ Vehicle" if vehicle_ok else "❌ Vehicle"
        h_txt = "✅ Helmet" if helmet_ok else "⚠ Helmet"
        self._lbl_model.setText(f"Model: {v_txt}  {h_txt}")
        if not vehicle_ok:
            QMessageBox.warning(
                self, "Cảnh báo Model",
                "Không tải được model phát hiện phương tiện (yolov8n.pt).\n"
                "Hãy đặt file model vào thư mục resources/ và khởi động lại.",
            )

    @pyqtSlot()
    def _on_video_finished(self) -> None:
        if self._current_source_type == "image":
            self._set_status("Đã xử lý xong ảnh.")
        else:
            self._set_status("Video kết thúc.")
        self._act_start.setEnabled(True)
        self._act_pause.setEnabled(False)
        self._act_stop.setEnabled(False)

    @pyqtSlot(str)
    def _on_error(self, message: str) -> None:
        self._logger.log_error("MainWindow", message)
        self._set_status(f"Lỗi: {message}")

    # ================================================================
    # Dieu khien phat lai - Chon nguon dau vao
    # ================================================================

    def _open_video(self) -> None:
        """Mo hop thoai chon file video MP4/AVI/MKV..."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Chọn file video",
            os.path.expanduser("~"),
            "Video files (*.mp4 *.avi *.mkv *.mov *.wmv);;All files (*)",
        )
        if not path:
            return

        self._load_source(source_path=path, source_type="video", display_name=os.path.basename(path))

    def _open_image(self) -> None:
        """
        Mo hop thoai chon 1 file anh tinh (jpg/png...).
        Anh duoc xu ly nhu video 1 frame duy nhat, chay qua dung pipeline
        Detection -> Tracking -> Violation -> OCR nhu video binh thuong.
        """
        path, _ = QFileDialog.getOpenFileName(
            self, "Chọn file ảnh",
            os.path.expanduser("~"),
            "Image files (*.jpg *.jpeg *.png *.bmp *.webp);;All files (*)",
        )
        if not path:
            return

        self._load_source(source_path=path, source_type="image", display_name=os.path.basename(path))

    def _open_camera(self) -> None:
        """
        Mo hop thoai nhap index camera (thuong la 0 cho webcam mac dinh),
        roi mo camera nhu mot nguon video truc tiep (VideoManager nhan
        index camera thay vi duong dan file, OpenCV xu ly giong nhau qua
        cv2.VideoCapture).
        """
        index, ok = QInputDialog.getInt(
            self, "Mở Camera",
            "Nhập chỉ số camera (thường là 0 cho webcam mặc định):",
            value=0, min=0, max=10,
        )
        if not ok:
            return

        self._load_source(
            source_path=index, source_type="camera",
            display_name=f"Webcam #{index} (trực tiếp)",
        )

    def _load_source(self, source_path, source_type: str, display_name: str) -> None:
        """
        Ham dung chung de nap nguon du lieu (video / anh / camera) vao
        VideoManager. Voi anh tinh, VideoManager se phat ra dung 1 frame
        roi tu dong ket thuc (video_finished), giong het luong xu ly video
        nhung chi co 1 frame duy nhat.
        """
        success = self._video_manager.load_source(source_path, source_type=source_type)
        if not success:
            QMessageBox.warning(
                self, "Lỗi nguồn đầu vào",
                f"Không thể mở: {display_name}\n"
                "Vui lòng kiểm tra lại đường dẫn file hoặc chỉ số camera.",
            )
            return

        self._current_source_type = source_type
        self._current_source_label = display_name

        icon = {"video": "🎬", "image": "🖼️", "camera": "📷"}.get(source_type, "📹")
        self._lbl_video_path.setText(f" {icon} {display_name}")
        self._act_start.setEnabled(True)
        self._set_status(f"Đã chọn {source_type}: {display_name}")

    def _start_processing(self) -> None:
        if not self._video_manager.isRunning():
            self._video_manager.start()
        else:
            self._video_manager.resume()
        self._act_start.setEnabled(False)
        self._act_pause.setEnabled(self._current_source_type != "image")
        self._act_stop.setEnabled(True)

        if self._current_source_type == "image":
            self._set_status("Đang xử lý ảnh...")
        elif self._current_source_type == "camera":
            self._set_status("Đang giám sát trực tiếp từ camera...")
        else:
            self._set_status("Đang giám sát...")

    def _pause_processing(self) -> None:
        self._video_manager.pause()
        self._act_start.setEnabled(True)
        self._act_pause.setEnabled(False)
        self._set_status("Đã tạm dừng.")

    def _stop_processing(self) -> None:
        self._video_manager.stop()
        self._act_start.setEnabled(True)
        self._act_pause.setEnabled(False)
        self._act_stop.setEnabled(False)
        self._set_status("Đã dừng.")

    def _reset_all(self) -> None:
        self._video_manager.stop()
        self._tracking_manager.reset()
        self._violation_manager.reset()
        self._video_widget.clear_display()
        self._violation_count = 0
        self._lbl_violations.setText("Vi phạm: 0")
        self._lbl_vehicles.setText("Phương tiện: 0")
        self._act_start.setEnabled(False)
        self._act_pause.setEnabled(False)
        self._act_stop.setEnabled(False)
        self._lbl_video_path.setText("  Chưa chọn nguồn")
        self._current_source_type = "video"
        self._current_source_label = "Chưa chọn nguồn"
        self._set_status("Đã reset. Chọn nguồn (Video / Ảnh / Camera) để bắt đầu lại.")

    def _set_status(self, msg: str) -> None:
        self._status_bar.showMessage(msg, 8000)

    # ================================================================
    # Dong ung dung
    # ================================================================

    def closeEvent(self, event) -> None:
        """Dung an toan tat ca cac thread truoc khi dong cua so."""
        self._logger.log_info("MainWindow", "Dang dong ung dung...")

        self._video_manager.stop()
        self._detection_manager.stop()
        self._tracking_manager.stop()
        self._ocr_manager.stop()
        self._email_manager.stop()
        self._db_manager.stop()
        self._traffic_light_panel.cleanup()

        self._logger.log_info("MainWindow", "Ung dung da dong an toan.")
        event.accept()