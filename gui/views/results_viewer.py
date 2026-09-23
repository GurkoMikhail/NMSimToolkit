import logging
from typing import Any, Optional

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QPushButton, QLabel, QSlider
)
import pyqtgraph as pg

_logger = logging.getLogger(__name__)


def configure_pyqtgraph_theme() -> None:
    """
    Настройка темной темы pyqtgraph.
    """
    pg.setConfigOption('background', '#181818')
    pg.setConfigOption('foreground', '#dcdcdc')
    pg.setConfigOption('antialias', True)


class ResultsViewer(QWidget):
    """
    Панель визуализации результатов моделирования и телеметрии в реальном времени.
    Отображает:
    1. Накопленные 2D-проекции детекторной матрицы (Image View).
    2. Энергетические спектры регистрируемых гамма-квантов (Energy Spectrum).
    3. Профили распределения счетов по осям X и Y (Profiles).
    """

    accumulation_cleared = Signal()

    def __init__(self, parent: Optional[Any] = None) -> None:
        super().__init__(parent)
        configure_pyqtgraph_theme()
        self.image_view: Any = None
        self.spectrum_plot: Any = None
        self.profile_plot: Any = None
        self.profile_legend: Any = None
        self._spectrum_curve: Any = None
        self._prof_x_curve: Any = None
        self._prof_y_curve: Any = None
        self._current_projection: Optional[np.ndarray] = None

        self._projections_stack: Optional[np.ndarray] = None
        self._current_view_idx: int = 0
        self._total_views: int = 1
        self._angular_range: float = 360.0
        self.proj_nav_widget: Optional[Any] = None

        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # Панель вкладок
        tabs = QTabWidget(self)

        # 1. Вкладка 2D-проекции
        proj_tab = QWidget()
        proj_layout = QVBoxLayout(proj_tab)
        proj_layout.setContentsMargins(2, 2, 2, 2)

        # Панель навигации по проекциям (ОФЭКТ)
        self.proj_nav_widget = QWidget(proj_tab)
        proj_nav_layout = QHBoxLayout(self.proj_nav_widget)
        proj_nav_layout.setContentsMargins(0, 0, 0, 2)

        self.lbl_projection_info = QLabel("Проекция: 1 / 1 (0.0°)")
        self.lbl_projection_info.setStyleSheet("font-weight: bold; color: #ecf0f1;")

        self.btn_prev_proj = QPushButton("◀")
        self.btn_prev_proj.setFixedWidth(32)
        self.btn_prev_proj.clicked.connect(self._on_prev_projection)

        self.slider_proj = QSlider(Qt.Horizontal)
        self.slider_proj.setRange(1, 1)
        self.slider_proj.setValue(1)
        self.slider_proj.valueChanged.connect(self._on_slider_proj_changed)

        self.btn_next_proj = QPushButton("▶")
        self.btn_next_proj.setFixedWidth(32)
        self.btn_next_proj.clicked.connect(self._on_next_projection)

        proj_nav_layout.addWidget(self.lbl_projection_info)
        proj_nav_layout.addWidget(self.btn_prev_proj)
        proj_nav_layout.addWidget(self.slider_proj)
        proj_nav_layout.addWidget(self.btn_next_proj)
        self.proj_nav_widget.setVisible(False)
        proj_layout.addWidget(self.proj_nav_widget)

        self.image_view = pg.ImageView(self)
        self.image_view.ui.menuBtn.hide()
        self.image_view.ui.roiBtn.hide()
        proj_layout.addWidget(self.image_view)

        tabs.addTab(proj_tab, "2D Проекция детектора")

        # 2. Вкладка энергетического спектра
        spec_tab = QWidget()
        spec_layout = QVBoxLayout(spec_tab)
        spec_layout.setContentsMargins(2, 2, 2, 2)

        self.spectrum_plot = pg.PlotWidget(self)
        self.spectrum_plot.setLabel('bottom', 'Энергия фотона (кэВ)')
        self.spectrum_plot.setLabel('left', 'Количество отсчетов (Counts)')
        self.spectrum_plot.showGrid(x=True, y=True, alpha=0.3)
        spec_layout.addWidget(self.spectrum_plot)

        tabs.addTab(spec_tab, "Энергетический спектр")

        # 3. Вкладка профилей счета
        prof_tab = QWidget()
        prof_layout = QVBoxLayout(prof_tab)
        prof_layout.setContentsMargins(2, 2, 2, 2)

        self.profile_plot = pg.PlotWidget(self)
        self.profile_plot.setLabel('bottom', 'Координата пикселя')
        self.profile_plot.setLabel('left', 'Интенсивность')
        self.profile_legend = self.profile_plot.addLegend()
        self.profile_plot.showGrid(x=True, y=True, alpha=0.3)
        prof_layout.addWidget(self.profile_plot)

        tabs.addTab(prof_tab, "Профиль счета (Profiles)")

        main_layout.addWidget(tabs)

        # Нижняя панель управления счетом
        ctrl_layout = QHBoxLayout()
        self.btn_clear = QPushButton("Сбросить накопление")
        self.btn_clear.clicked.connect(self.clear_results)
        self.lbl_stats = QLabel("Всего отсчетов: 0")

        ctrl_layout.addWidget(self.lbl_stats)
        ctrl_layout.addStretch()
        ctrl_layout.addWidget(self.btn_clear)
        main_layout.addLayout(ctrl_layout)

    def set_projection_data(self, image_data: np.ndarray) -> None:
        """
        Обновляет 2D матрицу проекции и пересчитывает линейные профили.
        """
        self._current_projection = image_data

        if self.image_view is not None:
            self.image_view.setImage(image_data.T, autoRange=False)

        total_counts = int(np.sum(image_data))
        self.lbl_stats.setText(f"Всего отсчетов: {total_counts:,}")

        self._update_profiles(image_data)

    def _update_profiles(self, image_data: np.ndarray) -> None:
        """
        Построение горизонтального и вертикального срезов через центр проекции.
        """
        if self.profile_plot is None:
            return

        h, w = image_data.shape
        mid_y = h // 2
        mid_x = w // 2

        prof_x = image_data[mid_y, :]
        prof_y = image_data[:, mid_x]

        if self._prof_x_curve is None or self._prof_y_curve is None:
            if self.profile_legend is not None:
                self.profile_legend.clear()
            self.profile_plot.clear()
            self._prof_x_curve = self.profile_plot.plot(prof_x, pen=pg.mkPen('#e74c3c', width=2), name="Горизонтальный X")
            self._prof_y_curve = self.profile_plot.plot(prof_y, pen=pg.mkPen('#3498db', width=2), name="Вертикальный Y")
        else:
            self._prof_x_curve.setData(prof_x)
            self._prof_y_curve.setData(prof_y)

    def set_spectrum_data(self, energy_array: np.ndarray, bins: int = 120, max_energy: float = 200.0) -> None:
        """
        Строит гистограмму энергетического спектра по зарегистрированным событиям.
        """
        if self.spectrum_plot is None or len(energy_array) == 0:
            return

        counts, edges = np.histogram(energy_array, bins=bins, range=(0, max_energy))
        bin_centers = (edges[:-1] + edges[1:]) / 2.0

        if self._spectrum_curve is None:
            self.spectrum_plot.clear()
            self._spectrum_curve = self.spectrum_plot.plot(
                bin_centers,
                counts,
                stepMode=False,
                fillLevel=0,
                fillBrush=(52, 152, 219, 100),
                pen=pg.mkPen('#3498db', width=2)
            )
        else:
            self._spectrum_curve.setData(bin_centers, counts)

    def set_projection_stack_data(self, stack: np.ndarray, current_idx: int, total_views: int, angle_deg: float) -> None:
        """
        Обновляет стек 3D-проекций ОФЭКТ сканирования и элементы навигации.
        """
        self._projections_stack = stack
        self._total_views = max(1, total_views)
        self._current_view_idx = current_idx

        if self._total_views > 1:
            self.proj_nav_widget.setVisible(True)
            self.slider_proj.blockSignals(True)
            self.slider_proj.setRange(1, self._total_views)
            self.slider_proj.setValue(current_idx + 1)
            self.slider_proj.blockSignals(False)
            self.lbl_projection_info.setText(f"Проекция: {current_idx + 1} / {self._total_views} ({angle_deg:.1f}°)")
        else:
            self.proj_nav_widget.setVisible(False)

        if 0 <= current_idx < stack.shape[0]:
            self.set_projection_data(stack[current_idx])

    def _on_slider_proj_changed(self, val: int) -> None:
        idx = val - 1
        if self._projections_stack is not None and 0 <= idx < self._projections_stack.shape[0]:
            self._current_view_idx = idx
            angle = (idx / self._total_views) * 360.0
            self.lbl_projection_info.setText(f"Проекция: {val} / {self._total_views} ({angle:.1f}°)")
            self.set_projection_data(self._projections_stack[idx])

    def _on_prev_projection(self) -> None:
        if self.slider_proj.value() > 1:
            self.slider_proj.setValue(self.slider_proj.value() - 1)

    def _on_next_projection(self) -> None:
        if self.slider_proj.value() < self._total_views:
            self.slider_proj.setValue(self.slider_proj.value() + 1)

    def clear_results(self) -> None:
        """
        Очищает все графики и накопленную матрицу детектора.
        """
        self._current_projection = None
        self._projections_stack = None
        self._total_views = 1
        self._current_view_idx = 0
        if self.proj_nav_widget is not None:
            self.proj_nav_widget.setVisible(False)

        if self.image_view is not None:
            self.image_view.clear()
        if self._spectrum_curve is not None:
            self._spectrum_curve.setData([], [])
        elif self.spectrum_plot is not None:
            self.spectrum_plot.clear()
        if self._prof_x_curve is not None:
            self._prof_x_curve.setData([])
        if self._prof_y_curve is not None:
            self._prof_y_curve.setData([])
        if self.profile_legend is not None:
            self.profile_legend.clear()
        if self.profile_plot is not None:
            self.profile_plot.clear()
            self._prof_x_curve = None
            self._prof_y_curve = None
        self.lbl_stats.setText("Всего отсчетов: 0")
        self.accumulation_cleared.emit()
