"""
===== FILE: main.py =====

Diem vao chinh cua ung dung Giam Sat Giao Thong Thong Minh.
Khoi tao QApplication, ap dung dark theme va mo MainWindow.
"""

import sys
import os

# Bao dam import duoc cac module trong du an
sys.path.insert(0, os.path.dirname(__file__))

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QSplashScreen, QLabel
from PyQt5.QtCore import QTimer

from gui.main_window import MainWindow
from gui.styles import DARK_STYLE


def main():
    # Cau hinh high DPI truoc khi tao QApplication
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    from core.config_manager import ConfigManager
    print("👉 ĐƯỜNG DẪN AI ĐANG TÌM LÀ:", ConfigManager().get_vehicle_model_path())
    app.setApplicationName("Smart Traffic Monitoring")
    app.setOrganizationName("TrafficAI")
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_STYLE)
    app.setFont(QFont("Segoe UI", 10))

    # Tao va hien thi cua so chinh
    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
