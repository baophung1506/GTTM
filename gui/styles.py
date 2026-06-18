"""
===== FILE: gui/styles.py =====
QSS Stylesheet - Giao dien toi mau cho he thong giam sat giao thong.
"""

DARK_STYLE = """
QMainWindow, QDialog {
    background-color: #1a1a2e;
    color: #e0e0e0;
}

QWidget {
    background-color: #1a1a2e;
    color: #e0e0e0;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 13px;
}

QGroupBox {
    border: 1px solid #3a3a5c;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 8px;
    font-weight: bold;
    color: #a0c4ff;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 6px;
}

QPushButton {
    background-color: #16213e;
    color: #e0e0e0;
    border: 1px solid #3a3a5c;
    border-radius: 5px;
    padding: 6px 16px;
    min-height: 28px;
}

QPushButton:hover {
    background-color: #0f3460;
    border-color: #a0c4ff;
}

QPushButton:pressed {
    background-color: #533483;
}

QPushButton:disabled {
    background-color: #2a2a3e;
    color: #555577;
    border-color: #2a2a3e;
}

QPushButton#btnStart {
    background-color: #1b5e20;
    border-color: #4caf50;
    font-weight: bold;
}
QPushButton#btnStart:hover { background-color: #2e7d32; }

QPushButton#btnStop {
    background-color: #b71c1c;
    border-color: #f44336;
    font-weight: bold;
}
QPushButton#btnStop:hover { background-color: #c62828; }

QPushButton#btnReset {
    background-color: #37474f;
    border-color: #78909c;
}
QPushButton#btnReset:hover { background-color: #455a64; }

/* Den tin hieu giao thong */
QPushButton#btnRed {
    background-color: #c62828;
    border: 3px solid #ff5252;
    border-radius: 40px;
    min-width: 70px;
    min-height: 70px;
    max-width: 70px;
    max-height: 70px;
}
QPushButton#btnYellow {
    background-color: #f57f17;
    border: 3px solid #ffeb3b;
    border-radius: 40px;
    min-width: 70px;
    min-height: 70px;
    max-width: 70px;
    max-height: 70px;
}
QPushButton#btnGreen {
    background-color: #1b5e20;
    border: 3px solid #69f0ae;
    border-radius: 40px;
    min-width: 70px;
    min-height: 70px;
    max-width: 70px;
    max-height: 70px;
}
QPushButton#btnRed:checked    { background-color: #ff1744; }
QPushButton#btnYellow:checked { background-color: #ffd600; }
QPushButton#btnGreen:checked  { background-color: #00e676; }

QTableWidget {
    background-color: #16213e;
    alternate-background-color: #1e2a4a;
    gridline-color: #3a3a5c;
    border: 1px solid #3a3a5c;
    border-radius: 4px;
    selection-background-color: #0f3460;
}

QTableWidget::item {
    padding: 4px 8px;
    border: none;
}

QHeaderView::section {
    background-color: #0f3460;
    color: #a0c4ff;
    padding: 6px 8px;
    border: none;
    border-right: 1px solid #3a3a5c;
    font-weight: bold;
}

QLineEdit, QComboBox, QSpinBox {
    background-color: #16213e;
    border: 1px solid #3a3a5c;
    border-radius: 4px;
    padding: 4px 8px;
    color: #e0e0e0;
    min-height: 26px;
}

QLineEdit:focus, QComboBox:focus {
    border-color: #a0c4ff;
}

QComboBox::drop-down {
    border: none;
    width: 24px;
}

QScrollBar:vertical {
    background: #16213e;
    width: 10px;
    border-radius: 5px;
}
QScrollBar::handle:vertical {
    background: #3a3a5c;
    border-radius: 5px;
    min-height: 30px;
}

QLabel {
    background-color: transparent;
    color: #e0e0e0;
}

QLabel#lblTitle {
    font-size: 16px;
    font-weight: bold;
    color: #a0c4ff;
}

QStatusBar {
    background-color: #0f3460;
    color: #a0c4ff;
    border-top: 1px solid #3a3a5c;
}

QTabWidget::pane {
    border: 1px solid #3a3a5c;
    border-radius: 4px;
}

QTabBar::tab {
    background-color: #16213e;
    color: #a0a0c0;
    padding: 8px 16px;
    border: 1px solid #3a3a5c;
    border-bottom: none;
    border-radius: 4px 4px 0 0;
}

QTabBar::tab:selected {
    background-color: #0f3460;
    color: #e0e0e0;
}

QSplitter::handle {
    background-color: #3a3a5c;
}

QToolBar {
    background-color: #16213e;
    border-bottom: 1px solid #3a3a5c;
    spacing: 4px;
    padding: 4px;
}

QToolBar QLabel {
    color: #a0c4ff;
    font-weight: bold;
}
"""

TRAFFIC_LIGHT_INACTIVE = {
    "RED":    "background-color: #4a1010; border: 3px solid #7a2020; border-radius: 40px;",
    "YELLOW": "background-color: #4a3a00; border: 3px solid #7a6000; border-radius: 40px;",
    "GREEN":  "background-color: #0a3010; border: 3px solid #1a5020; border-radius: 40px;",
}

TRAFFIC_LIGHT_ACTIVE = {
    "RED":    "background-color: #ff1744; border: 3px solid #ff6b6b; border-radius: 40px;",
    "YELLOW": "background-color: #ffd600; border: 3px solid #fff176; border-radius: 40px;",
    "GREEN":  "background-color: #00e676; border: 3px solid #69f0ae; border-radius: 40px;",
}
