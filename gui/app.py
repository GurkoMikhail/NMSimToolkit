import logging
import multiprocessing
import os
import sys

# Настройка Qt API для совместимости с PyVista / QtInteractor
os.environ['QT_API'] = 'pyside6'

# Настройка потоков для Numba/OpenMP/MKL
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

# Базовая конфигурация логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

import vtk
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPalette, QColor

from gui.views.main_window import MainWindow
from gui.views.results_viewer import configure_pyqtgraph_theme

# Подавление спама внутренних предупреждений VTK OpenGL
vtk.vtkObject.GlobalWarningDisplayOff()


DARK_STYLE_SHEET = """
QMainWindow {
    background-color: #1e1e1e;
}
QWidget {
    background-color: #252526;
    color: #cccccc;
    font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
    font-size: 12px;
}
QDockWidget {
    color: #ffffff;
    font-weight: bold;
}
QDockWidget::title {
    background-color: #2d2d30;
    padding: 6px;
    border-bottom: 1px solid #3e3e42;
}
QTreeWidget, QTableView, QListView {
    background-color: #1e1e1e;
    border: 1px solid #3e3e42;
    selection-background-color: #094771;
    selection-color: #ffffff;
}
QHeaderView::section {
    background-color: #2d2d30;
    color: #cccccc;
    padding: 4px;
    border: 1px solid #3e3e42;
}
QGroupBox {
    border: 1px solid #3e3e42;
    margin-top: 8px;
    padding-top: 12px;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 4px;
    color: #3498db;
}
QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {
    background-color: #3c3c3c;
    border: 1px solid #555555;
    color: #ffffff;
    padding: 3px 6px;
    border-radius: 2px;
}
QLineEdit:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid #007acc;
}
QPushButton {
    background-color: #0e639c;
    color: #ffffff;
    border: none;
    padding: 6px 12px;
    border-radius: 2px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #1177bb;
}
QPushButton:pressed {
    background-color: #0d5c8f;
}
QTabWidget::pane {
    border: 1px solid #3e3e42;
    background-color: #1e1e1e;
}
QTabBar::tab {
    background-color: #2d2d30;
    color: #969696;
    padding: 6px 14px;
    border: 1px solid #3e3e42;
    border-bottom: none;
}
QTabBar::tab:selected {
    background-color: #1e1e1e;
    color: #ffffff;
    border-top: 2px solid #007acc;
}
QToolBar {
    background-color: #2d2d30;
    border-bottom: 1px solid #3e3e42;
    spacing: 4px;
    padding: 2px;
}
QStatusBar {
    background-color: #007acc;
    color: #ffffff;
}
"""


def main():
    """
    Главная точка входа для запуска графического интерфейса NMSimToolkit.
    Вызов freeze_support обязателен для предотвращения рекурсивного создания
    окон при запуске воркеров multiprocessing.Pool на Windows.
    """
    multiprocessing.freeze_support()

    configure_pyqtgraph_theme()
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setStyleSheet(DARK_STYLE_SHEET)

    # Настройка палитры темного оформления
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor('#1e1e1e'))
    palette.setColor(QPalette.WindowText, QColor('#cccccc'))
    palette.setColor(QPalette.Base, QColor('#252526'))
    palette.setColor(QPalette.AlternateBase, QColor('#2d2d30'))
    palette.setColor(QPalette.ToolTipBase, QColor('#252526'))
    palette.setColor(QPalette.ToolTipText, QColor('#ffffff'))
    palette.setColor(QPalette.Text, QColor('#ffffff'))
    palette.setColor(QPalette.Button, QColor('#2d2d30'))
    palette.setColor(QPalette.ButtonText, QColor('#ffffff'))
    palette.setColor(QPalette.Highlight, QColor('#094771'))
    palette.setColor(QPalette.HighlightedText, QColor('#ffffff'))
    app.setPalette(palette)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
