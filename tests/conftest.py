import os
import sys

# Настройка переменных окружения для headless Qt (offscreen)
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["QT_API"] = "pyside6"

try:
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication(["-platform", "offscreen"])
except ImportError:
    app = None
