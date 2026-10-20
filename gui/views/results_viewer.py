"""
Панель визуализации результатов моделирования и телеметрии в реальном времени.
"""
import logging
from typing import Any, List, Optional, Sequence

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
import pyqtgraph as pg

_logger = logging.getLogger(__name__)

ALL_DETECTORS_LABEL = "Все детекторы (Сумма)"


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
    Одновременно отображает в едином окне:
    1. Накопленные 2D-проекции детекторной матрицы с выбором детектора и фильтрацией по энергии (левая область).
    2. Энергетические спектры регистрируемых гамма-квантов с интерактивным энергетическим окном
       и настройкой шкалы спектра / числа бинов (правая область).
    """

    accumulation_cleared = Signal()
    spectrum_settings_changed = Signal(float, int, bool)  # max_energy (кэВ), energy_bins, auto_mode
    auto_max_energy_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        configure_pyqtgraph_theme()
        self.splitter: Optional[QSplitter] = None
        self.projection_panel: Optional[QWidget] = None
        self.spectrum_panel: Optional[QWidget] = None
        self.combo_detector: Optional[QComboBox] = None
        self.image_view: Any = None
        self.spectrum_plot: Any = None
        self._spectrum_curve: Any = None
        self.energy_region: Any = None
        self.spin_emin: Optional[QDoubleSpinBox] = None
        self.spin_emax: Optional[QDoubleSpinBox] = None
        self.btn_reset_window: Optional[QPushButton] = None
        self.spin_max_energy: Optional[QDoubleSpinBox] = None
        self.spin_energy_bins: Optional[QSpinBox] = None
        self.btn_auto_max_energy: Optional[QPushButton] = None
        self.lbl_bin_step: Optional[QLabel] = None

        self._current_cube: Optional[np.ndarray] = None
        self._current_3d: Optional[np.ndarray] = None
        self._current_projection: Optional[np.ndarray] = None
        self._last_raw_energies: Optional[np.ndarray] = None

        self._detector_names: List[str] = []
        self._custom_detector_names_set: bool = False
        self._max_energy: float = 300.0
        self._energy_bins: int = 256
        self._auto_max_energy: bool = True
        self._syncing_energy_window: bool = False
        self._syncing_spectrum_settings: bool = False
        self._projection_aspect: float = 1.0
        self._target_splitter_sizes: List[int] = [340, 660]

        self._projections_stack: Optional[np.ndarray] = None
        self._current_view_idx: int = 0
        self._total_views: int = 1
        self._angular_range: float = 360.0
        self.proj_nav_widget: Optional[QWidget] = None

        self._init_ui()

    @property
    def max_energy(self) -> float:
        """Верхняя граница энергетической шкалы спектра (кэВ)."""
        return self._max_energy

    @property
    def energy_bins(self) -> int:
        """Количество энергетических каналов (бинов) спектра."""
        return self._energy_bins

    @property
    def auto_max_energy(self) -> bool:
        """Флаг автоматического подбора верхней границы спектра по источникам сцены."""
        return self._auto_max_energy

    @property
    def projection_aspect(self) -> float:
        """Соотношение сторон текущей матрицы 2D-проекции (Cols / Rows)."""
        return self._projection_aspect

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # Горизонтальный сплиттер для одновременного отображения 2D-проекции (слева) и спектра (справа) в одном окне
        self.splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.splitter.setChildrenCollapsible(False)

        # 1. Левая область: 2D-проекция детектора (ширина адаптируется под соотношение сторон проекции)
        self.projection_panel = QWidget(self.splitter)
        proj_layout = QVBoxLayout(self.projection_panel)
        proj_layout.setContentsMargins(2, 2, 2, 2)

        # Верхняя панель выбора детектора
        det_bar_layout = QHBoxLayout()
        det_bar_layout.setContentsMargins(0, 0, 0, 2)
        lbl_detector = QLabel("Детектор:")
        self.combo_detector = QComboBox(self.projection_panel)
        self._rebuild_detector_combo()
        self.combo_detector.currentIndexChanged.connect(self._on_detector_changed)
        det_bar_layout.addWidget(lbl_detector)
        det_bar_layout.addWidget(self.combo_detector, stretch=1)
        det_bar_layout.addStretch()
        proj_layout.addLayout(det_bar_layout)

        # Панель навигации по проекциям (ОФЭКТ)
        self.proj_nav_widget = QWidget(self.projection_panel)
        proj_nav_layout = QHBoxLayout(self.proj_nav_widget)
        proj_nav_layout.setContentsMargins(0, 0, 0, 2)

        self.lbl_projection_info = QLabel("Проекция: 1 / 1 (0.0°)")
        self.lbl_projection_info.setStyleSheet("font-weight: bold; color: #ecf0f1;")

        self.btn_prev_proj = QPushButton("◀")
        self.btn_prev_proj.setFixedWidth(32)
        self.btn_prev_proj.clicked.connect(self._on_prev_projection)

        self.slider_proj = QSlider(Qt.Orientation.Horizontal)
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

        self.image_view = pg.ImageView(self.projection_panel)
        self.image_view.ui.menuBtn.hide()
        self.image_view.ui.roiBtn.hide()
        proj_layout.addWidget(self.image_view, stretch=1)

        self.splitter.addWidget(self.projection_panel)

        # 2. Правая область: энергетический спектр с интерактивным энергетическим окном и настройками шкалы
        self.spectrum_panel = QWidget(self.splitter)
        self.spectrum_panel.setMinimumWidth(240)
        spec_layout = QVBoxLayout(self.spectrum_panel)
        spec_layout.setContentsMargins(2, 2, 2, 2)

        # Компактная шапка энергетического окна и параметров шкалы спектра (Ignored по горизонтали,
        # чтобы широкий набор спинбоксов не раздувал minimumSizeHint спектра и не сжимал проекцию в QSplitter)
        ewin_header = QWidget(self.spectrum_panel)
        ewin_header.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        ewin_layout = QHBoxLayout(ewin_header)
        ewin_layout.setContentsMargins(0, 0, 0, 2)
        ewin_layout.setSpacing(4)
        lbl_ewin = QLabel("Окно:", ewin_header)
        lbl_ewin.setToolTip("Энергетическое окно фильтрации 2D-проекции")

        self.spin_emin = QDoubleSpinBox(ewin_header)
        self.spin_emin.setRange(0.0, 10000.0)
        self.spin_emin.setDecimals(2)
        self.spin_emin.setSingleStep(1.0)
        self.spin_emin.setSuffix(" кэВ")
        self.spin_emin.setMinimumWidth(80)
        self.spin_emin.setValue(0.0)

        lbl_dash = QLabel("—", ewin_header)

        self.spin_emax = QDoubleSpinBox(ewin_header)
        self.spin_emax.setRange(0.0, 10000.0)
        self.spin_emax.setDecimals(2)
        self.spin_emax.setSingleStep(1.0)
        self.spin_emax.setSuffix(" кэВ")
        self.spin_emax.setMinimumWidth(80)
        self.spin_emax.setValue(self._max_energy)

        self.btn_reset_window = QPushButton("Весь спектр", ewin_header)
        self.btn_reset_window.setMinimumWidth(72)
        self.btn_reset_window.clicked.connect(self.reset_energy_window)

        self.spin_emin.valueChanged.connect(self._on_energy_spin_changed)
        self.spin_emax.valueChanged.connect(self._on_energy_spin_changed)

        lbl_scale = QLabel("Предел:", ewin_header)
        self.spin_max_energy = QDoubleSpinBox(ewin_header)
        self.spin_max_energy.setRange(1.0, 10000.0)
        self.spin_max_energy.setDecimals(2)
        self.spin_max_energy.setSingleStep(10.0)
        self.spin_max_energy.setSuffix(" кэВ")
        self.spin_max_energy.setMinimumWidth(80)
        self.spin_max_energy.setValue(self._max_energy)
        self.spin_max_energy.setToolTip("Верхняя граница шкалы энергетического спектра (max_energy)")
        self.spin_max_energy.valueChanged.connect(self._on_max_energy_spin_changed)

        self.btn_auto_max_energy = QPushButton("Авто", ewin_header)
        self.btn_auto_max_energy.setCheckable(True)
        self.btn_auto_max_energy.setChecked(self._auto_max_energy)
        self.btn_auto_max_energy.setFixedWidth(48)
        self.btn_auto_max_energy.setToolTip(
            "Автоматический подбор предела спектра по максимальной энергии источников в сцене (+20%)"
        )
        self.btn_auto_max_energy.clicked.connect(self._on_auto_max_energy_clicked)

        lbl_bins = QLabel("Бины:", ewin_header)
        self.spin_energy_bins = QSpinBox(ewin_header)
        self.spin_energy_bins.setRange(16, 4096)
        self.spin_energy_bins.setSingleStep(32)
        self.spin_energy_bins.setMinimumWidth(62)
        self.spin_energy_bins.setValue(self._energy_bins)
        self.spin_energy_bins.setToolTip("Число энергетических каналов (бинов) спектра")
        self.spin_energy_bins.valueChanged.connect(self._on_energy_bins_spin_changed)

        self.lbl_bin_step = QLabel(self._format_bin_step_text(), ewin_header)
        self.lbl_bin_step.setStyleSheet("color: #95a5a6; font-size: 11px;")
        self.lbl_bin_step.setToolTip("Ширина одного энергетического канала (шаг бина)")

        ewin_layout.addWidget(lbl_ewin)
        ewin_layout.addWidget(self.spin_emin)
        ewin_layout.addWidget(lbl_dash)
        ewin_layout.addWidget(self.spin_emax)
        ewin_layout.addWidget(self.btn_reset_window)
        ewin_layout.addSpacing(6)
        ewin_layout.addWidget(lbl_scale)
        ewin_layout.addWidget(self.spin_max_energy)
        ewin_layout.addWidget(self.btn_auto_max_energy)
        ewin_layout.addWidget(lbl_bins)
        ewin_layout.addWidget(self.spin_energy_bins)
        ewin_layout.addWidget(self.lbl_bin_step)
        ewin_layout.addStretch()
        spec_layout.addWidget(ewin_header)

        self.spectrum_plot = pg.PlotWidget(self.spectrum_panel)
        self.spectrum_plot.setLabel('bottom', 'Энергия фотона (кэВ)')
        self.spectrum_plot.setLabel('left', 'Количество отсчетов (Counts)')
        self.spectrum_plot.showGrid(x=True, y=True, alpha=0.3)
        self.spectrum_plot.setXRange(0.0, self._max_energy, padding=0.02)

        self._spectrum_curve = self.spectrum_plot.plot(
            [],
            [],
            stepMode=False,
            fillLevel=0,
            fillBrush=(52, 152, 219, 100),
            pen=pg.mkPen('#3498db', width=2),
        )

        self.energy_region = pg.LinearRegionItem(
            values=(0.0, self._max_energy),
            orientation='vertical',
            brush=pg.mkBrush(46, 204, 113, 50),
            pen=pg.mkPen('#2ecc71', width=1.5),
            bounds=(0.0, 10000.0),
            movable=True,
        )
        self.energy_region.sigRegionChanged.connect(self._on_energy_region_changed)
        self.spectrum_plot.addItem(self.energy_region)

        spec_layout.addWidget(self.spectrum_plot, stretch=1)

        self.splitter.addWidget(self.spectrum_panel)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)

        main_layout.addWidget(self.splitter, stretch=1)
        self._update_splitter_proportions()

        # Нижняя панель управления счетом
        ctrl_layout = QHBoxLayout()
        self.btn_clear = QPushButton("Сбросить накопление")
        self.btn_clear.clicked.connect(self.clear_results)
        self.lbl_stats = QLabel("Всего отсчетов: 0")

        ctrl_layout.addWidget(self.lbl_stats)
        ctrl_layout.addStretch()
        ctrl_layout.addWidget(self.btn_clear)
        main_layout.addLayout(ctrl_layout)

    def _format_bin_step_text(self) -> str:
        """Форматирует подпись ширины одного энергетического бина в кэВ."""
        step = self._max_energy / max(1, self._energy_bins)
        return f"({step:.2f} кэВ/бин)"

    def _update_bin_step_label(self) -> None:
        if self.lbl_bin_step is not None:
            self.lbl_bin_step.setText(self._format_bin_step_text())

    def _get_lut_histogram_width(self) -> int:
        """
        Возвращает ширину боковой шкалы Lookup Table (ui.histogram) в pg.ImageView,
        если она видима (не скрыта), чтобы шкала LUT не отъедала ширину у холста 2D-проекции.
        """
        if self.image_view is None:
            return 0
        hist = self.image_view.ui.histogram
        if hist is None or hist.isHidden():
            return 0
        return max(int(hist.width()), int(hist.sizeHint().width()), int(hist.minimumWidth()), 110)

    def _update_splitter_proportions(self) -> None:
        """
        Вычисляет и устанавливает размеры панелей горизонтального QSplitter так,
        чтобы сам холст 2D-проекции (image_view.view) имел ширину canvas_height * aspect_ratio,
        а панель projection_panel дополнительно включала ширину боковой шкалы Lookup Table
        (image_view.ui.histogram) и горизонтальные отступы компоновщика.
        Все остальное горизонтальное пространство отдается графику спектра.
        """
        if self.splitter is None:
            return
        root_layout = self.layout()
        if root_layout is not None:
            root_layout.activate()

        total_w = self.width() if self.width() > 200 else 1000
        total_h = (self.height() - 40) if self.height() > 140 else 280
        if self.splitter.width() > 200:
            total_w = self.splitter.width()
        if self.splitter.height() > 100:
            total_h = self.splitter.height()

        header_overhead = 38
        if self.proj_nav_widget is not None and not self.proj_nav_widget.isHidden():
            header_overhead += 30

        if (
            self.isVisible()
            and self.image_view is not None
            and 60 <= self.image_view.height() <= total_h
        ):
            canvas_height = int(self.image_view.height())
        else:
            canvas_height = max(80, total_h - header_overhead)

        lut_width = self._get_lut_histogram_width()
        layout_margins = 8
        if self.projection_panel is not None:
            proj_layout = self.projection_panel.layout()
            if proj_layout is not None:
                margins = proj_layout.contentsMargins()
                layout_margins = max(8, int(margins.left() + margins.right()) + 4)

        canvas_width = int(round(canvas_height * self._projection_aspect))
        desired_proj_w = canvas_width + lut_width + layout_margins
        max_proj_w = max(220, int(total_w * 0.45))
        proj_w = max(160, min(desired_proj_w, max_proj_w))
        spec_w = max(240, total_w - proj_w)
        if self.projection_panel is not None:
            self.projection_panel.setMinimumWidth(140)
            self.projection_panel.setMaximumWidth(max_proj_w)
        if self.spectrum_panel is not None:
            self.spectrum_panel.setMinimumWidth(240)
        self._target_splitter_sizes = [proj_w, spec_w]
        self.splitter.setSizes([proj_w, spec_w])

    @property
    def splitter_sizes(self) -> List[int]:
        """Целевые ширины панелей [projection_panel, spectrum_panel], рассчитанные по соотношению сторон."""
        self._update_splitter_proportions()
        if self.splitter is not None and self.isVisible():
            actual = self.splitter.sizes()
            if len(actual) == 2 and sum(actual) > 200:
                return list(actual)
        return list(self._target_splitter_sizes)

    def _update_projection_aspect(self, rows: int, cols: int) -> None:
        """Обновляет соотношение сторон матрицы проекции (Cols / Rows) и адаптирует сплиттер."""
        if rows <= 0 or cols <= 0:
            return
        new_aspect = float(cols) / float(rows)
        if abs(new_aspect - self._projection_aspect) > 1e-4:
            self._projection_aspect = new_aspect
            self._update_splitter_proportions()

    def showEvent(self, event: Any) -> None:
        super().showEvent(event)
        self._update_splitter_proportions()

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._update_splitter_proportions()

    def _on_max_energy_spin_changed(self, value: float) -> None:
        """Обработка ручного изменения верхней границы энергетической шкалы спектра."""
        if self._syncing_spectrum_settings:
            return
        new_max = max(1.0, float(value))
        old_max = self._max_energy
        self._max_energy = new_max
        self._auto_max_energy = False
        if self.btn_auto_max_energy is not None:
            self.btn_auto_max_energy.blockSignals(True)
            self.btn_auto_max_energy.setChecked(False)
            self.btn_auto_max_energy.blockSignals(False)
        self._update_bin_step_label()
        if self.spectrum_plot is not None:
            self.spectrum_plot.setXRange(0.0, self._max_energy, padding=0.02)
        if self.spin_emin is not None and self.spin_emax is not None:
            if abs(self.spin_emin.value()) < 1e-2 and abs(self.spin_emax.value() - old_max) < 1e-2:
                self.set_energy_window(0.0, self._max_energy)
            elif self.spin_emax.value() > self._max_energy:
                self.set_energy_window(min(self.spin_emin.value(), self._max_energy), self._max_energy)
        self._refresh_displayed_spectrum()
        self.spectrum_settings_changed.emit(self._max_energy, self._energy_bins, self._auto_max_energy)

    def _on_energy_bins_spin_changed(self, value: int) -> None:
        """Обработка изменения числа бинов спектра."""
        if self._syncing_spectrum_settings:
            return
        self._energy_bins = max(1, int(value))
        self._update_bin_step_label()
        self._refresh_displayed_spectrum()
        self.spectrum_settings_changed.emit(self._max_energy, self._energy_bins, self._auto_max_energy)

    def _on_auto_max_energy_clicked(self, checked: bool) -> None:
        """Переключение режима автоподбора верхней границы спектра по источникам сцены."""
        self._auto_max_energy = bool(checked)
        if self._auto_max_energy:
            self.auto_max_energy_requested.emit()
        self.spectrum_settings_changed.emit(self._max_energy, self._energy_bins, self._auto_max_energy)

    def set_spectrum_configuration(
        self,
        max_energy: Optional[float] = None,
        energy_bins: Optional[int] = None,
        auto_max_energy: Optional[bool] = None,
    ) -> None:
        """Программная синхронизация настроек шкалы спектра и числа бинов без обратной эмиссии сигнала."""
        old_max = self._max_energy
        self._syncing_spectrum_settings = True
        try:
            if auto_max_energy is not None:
                self._auto_max_energy = bool(auto_max_energy)
                if self.btn_auto_max_energy is not None:
                    self.btn_auto_max_energy.setChecked(self._auto_max_energy)
            if energy_bins is not None and int(energy_bins) > 0:
                self._energy_bins = int(energy_bins)
                if self.spin_energy_bins is not None:
                    self.spin_energy_bins.setValue(self._energy_bins)
            if max_energy is not None and float(max_energy) > 0.0:
                self._max_energy = float(max_energy)
                if self.spin_max_energy is not None:
                    self.spin_max_energy.setValue(self._max_energy)
        finally:
            self._syncing_spectrum_settings = False

        self._update_bin_step_label()
        if self.spectrum_plot is not None:
            self.spectrum_plot.setXRange(0.0, self._max_energy, padding=0.02)
        if self.spin_emin is not None and self.spin_emax is not None:
            if abs(self.spin_emin.value()) < 1e-2 and abs(self.spin_emax.value() - old_max) < 1e-2:
                self.set_energy_window(0.0, self._max_energy)
            elif self.spin_emax.value() > self._max_energy:
                self.set_energy_window(min(self.spin_emin.value(), self._max_energy), self._max_energy)
        self._refresh_displayed_spectrum()

    def _refresh_displayed_spectrum(self) -> None:
        """Перерисовывает текущий спектр из 4D-куба или кэшированного массива сырых энергий."""
        if self._current_cube is not None:
            self._refresh_from_cube()
        elif self._last_raw_energies is not None and self._last_raw_energies.size > 0:
            hist_counts, hist_edges = np.histogram(
                self._last_raw_energies,
                bins=self._energy_bins,
                range=(0.0, self._max_energy),
            )
            hist_centers = 0.5 * (hist_edges[:-1] + hist_edges[1:])
            self._set_spectrum_curve_data(hist_centers, hist_counts)

    def _rebuild_detector_combo(self) -> None:
        """Перестраивает выпадающий список детекторов с сохранением текущего выбора."""
        if self.combo_detector is None:
            return
        prev_count = self.combo_detector.count()
        prev_text = self.combo_detector.currentText()
        prev_idx = self.combo_detector.currentIndex()
        self.combo_detector.blockSignals(True)
        try:
            self.combo_detector.clear()
            for det_name in self._detector_names:
                self.combo_detector.addItem(det_name)
            if len(self._detector_names) != 1:
                self.combo_detector.addItem(ALL_DETECTORS_LABEL)

            if prev_text == ALL_DETECTORS_LABEL and prev_count > 1 and len(self._detector_names) != 1:
                self.combo_detector.setCurrentIndex(len(self._detector_names))
            elif prev_text in self._detector_names:
                self.combo_detector.setCurrentIndex(self._detector_names.index(prev_text))
            elif prev_text != ALL_DETECTORS_LABEL and 0 <= prev_idx < len(self._detector_names):
                self.combo_detector.setCurrentIndex(prev_idx)
            else:
                self.combo_detector.setCurrentIndex(0)
        finally:
            self.combo_detector.blockSignals(False)

    def set_detector_metadata(
        self,
        detector_names: Sequence[str],
        max_energy: Optional[float] = None,
        energy_bins: Optional[int] = None,
        auto_max_energy: Optional[bool] = None,
    ) -> None:
        """
        Устанавливает список имен детекторов и верхнюю границу энергетической шкалы (в кэВ).
        """
        old_max = self._max_energy
        self._syncing_spectrum_settings = True
        try:
            if auto_max_energy is not None:
                self._auto_max_energy = bool(auto_max_energy)
                if self.btn_auto_max_energy is not None:
                    self.btn_auto_max_energy.setChecked(self._auto_max_energy)
            if energy_bins is not None and int(energy_bins) > 0:
                self._energy_bins = int(energy_bins)
                if self.spin_energy_bins is not None:
                    self.spin_energy_bins.setValue(self._energy_bins)
            if max_energy is not None and float(max_energy) > 0.0:
                self._max_energy = float(max_energy)
                if self.spin_max_energy is not None:
                    self.spin_max_energy.setValue(self._max_energy)
        finally:
            self._syncing_spectrum_settings = False

        self._update_bin_step_label()
        if self.spectrum_plot is not None:
            self.spectrum_plot.setXRange(0.0, self._max_energy, padding=0.02)

        cleaned = [str(name).strip() for name in detector_names if str(name).strip()]
        if cleaned:
            self._detector_names = cleaned
            self._custom_detector_names_set = True
        else:
            if self._current_cube is not None and self._current_cube.ndim == 4:
                n_det = int(self._current_cube.shape[0])
            elif self._current_3d is not None and self._current_3d.ndim == 3:
                n_det = int(self._current_3d.shape[0])
            else:
                n_det = 0
            self._detector_names = [f"Detector_{idx + 1}" for idx in range(n_det)]
            self._custom_detector_names_set = False

        self._rebuild_detector_combo()

        if self.spin_emin is not None and self.spin_emax is not None:
            if abs(self.spin_emin.value()) < 1e-2 and abs(self.spin_emax.value() - old_max) < 1e-2:
                self.set_energy_window(0.0, self._max_energy)
            elif self.spin_emax.value() > self._max_energy:
                self.set_energy_window(min(self.spin_emin.value(), self._max_energy), self._max_energy)

        self._refresh_displayed_spectrum()

    def set_energy_window(self, e_min: float, e_max: float) -> None:
        """
        Программно устанавливает границы энергетического окна [e_min, e_max] (в кэВ)
        и синхронизирует спинбоксы, LinearRegionItem и 2D-проекцию.
        """
        low = max(0.0, min(float(e_min), float(e_max)))
        high = max(low, max(float(e_min), float(e_max)))
        self._syncing_energy_window = True
        try:
            if self.spin_emin is not None:
                self.spin_emin.blockSignals(True)
                self.spin_emin.setValue(low)
                self.spin_emin.blockSignals(False)
            if self.spin_emax is not None:
                self.spin_emax.blockSignals(True)
                self.spin_emax.setValue(high)
                self.spin_emax.blockSignals(False)
            if self.energy_region is not None:
                self.energy_region.blockSignals(True)
                self.energy_region.setRegion((low, high))
                self.energy_region.blockSignals(False)
        finally:
            self._syncing_energy_window = False

        if self._current_cube is not None:
            self._refresh_from_cube()

    def reset_energy_window(self) -> None:
        """Сбрасывает энергетическое окно к полному диапазону спектра [0.0, max_energy]."""
        self.set_energy_window(0.0, self._max_energy)

    def _on_energy_spin_changed(self, _val: float) -> None:
        if self._syncing_energy_window or self.spin_emin is None or self.spin_emax is None:
            return
        e_min = float(self.spin_emin.value())
        e_max = float(self.spin_emax.value())
        self._syncing_energy_window = True
        try:
            if self.energy_region is not None:
                self.energy_region.blockSignals(True)
                self.energy_region.setRegion((min(e_min, e_max), max(e_min, e_max)))
                self.energy_region.blockSignals(False)
        finally:
            self._syncing_energy_window = False

        if self._current_cube is not None:
            self._refresh_from_cube()

    def _on_energy_region_changed(self) -> None:
        if self._syncing_energy_window or self.energy_region is None:
            return
        r_min, r_max = self.energy_region.getRegion()
        low = max(0.0, float(min(r_min, r_max)))
        high = max(low, float(max(r_min, r_max)))
        self._syncing_energy_window = True
        try:
            if self.spin_emin is not None:
                self.spin_emin.blockSignals(True)
                self.spin_emin.setValue(low)
                self.spin_emin.blockSignals(False)
            if self.spin_emax is not None:
                self.spin_emax.blockSignals(True)
                self.spin_emax.setValue(high)
                self.spin_emax.blockSignals(False)
        finally:
            self._syncing_energy_window = False

        if self._current_cube is not None:
            self._refresh_from_cube()

    def _on_detector_changed(self, _index: int) -> None:
        """Пересчитывает проекцию и спектр при переключении выбранного детектора."""
        if self._current_cube is not None:
            self._refresh_from_cube()
        elif self._current_3d is not None:
            self._refresh_from_3d()

    def _ensure_detector_names_match(self, n_det: int) -> None:
        """Синхронизирует количество имен детекторов в селекторе с первой размерностью массива."""
        if n_det <= 0:
            return
        expected_combo_count = 1 if n_det == 1 else n_det + 1
        if len(self._detector_names) != n_det:
            if self._custom_detector_names_set:
                if len(self._detector_names) < n_det:
                    for idx in range(len(self._detector_names), n_det):
                        self._detector_names.append(f"Detector_{idx + 1}")
                else:
                    self._detector_names = self._detector_names[:n_det]
            else:
                self._detector_names = [f"Detector_{idx + 1}" for idx in range(n_det)]
            self._rebuild_detector_combo()
        elif self.combo_detector is not None and self.combo_detector.count() != expected_combo_count:
            self._rebuild_detector_combo()

    def _set_spectrum_curve_data(self, x_vals: Any, y_vals: Any) -> None:
        """Обновляет кривую спектра, гарантируя сохранность интерактивного LinearRegionItem."""
        if self.spectrum_plot is None:
            return
        if self._spectrum_curve is None:
            self._spectrum_curve = self.spectrum_plot.plot(
                x_vals,
                y_vals,
                stepMode=False,
                fillLevel=0,
                fillBrush=(52, 152, 219, 100),
                pen=pg.mkPen('#3498db', width=2),
            )
        else:
            self._spectrum_curve.setData(x_vals, y_vals)

        if self.energy_region is not None and self.energy_region not in self.spectrum_plot.items():
            self.spectrum_plot.addItem(self.energy_region)

    def _refresh_from_cube(self) -> None:
        """
        Пересчитывает 1D энергетический спектр и 2D проекцию в окне [e_min, e_max]
        для выбранного детектора (или суммы всех детекторов) из 4D-куба.
        """
        if self._current_cube is None or self._current_cube.ndim != 4:
            return

        n_det, e_bins, rows, cols = self._current_cube.shape
        if n_det == 0 or e_bins == 0:
            return

        self._update_projection_aspect(rows, cols)

        det_idx = self.combo_detector.currentIndex() if self.combo_detector is not None else 0
        if 0 <= det_idx < n_det:
            cube_det = self._current_cube[det_idx]
        else:
            cube_det = np.sum(self._current_cube, axis=0, dtype=np.float32)

        # 1. Вычисление и отрисовка 1D спектра
        spec_counts = np.sum(cube_det, axis=(1, 2), dtype=np.float32)
        max_e = max(1e-6, float(self._max_energy))
        edges = np.linspace(0.0, max_e, e_bins + 1, dtype=np.float32)
        bin_centers = 0.5 * (edges[:-1] + edges[1:])
        self._set_spectrum_curve_data(bin_centers, spec_counts)

        # 2. Вычисление 2D-среза в текущем энергетическом окне [e_min, e_max]
        raw_emin = float(self.spin_emin.value()) if self.spin_emin is not None else 0.0
        raw_emax = float(self.spin_emax.value()) if self.spin_emax is not None else max_e
        e_min = min(raw_emin, raw_emax)
        e_max = max(raw_emin, raw_emax)

        b_min = int(np.clip(np.floor((e_min / max_e) * e_bins), 0, e_bins))
        b_max = int(np.clip(np.ceil((e_max / max_e) * e_bins), 0, e_bins))

        if b_max > b_min:
            proj_2d = np.sum(cube_det[b_min:b_max, :, :], axis=0, dtype=np.float32)
        else:
            proj_2d = np.zeros((rows, cols), dtype=np.float32)

        if self._current_projection is None or self._current_projection.shape != proj_2d.shape:
            self._current_projection = np.array(proj_2d, dtype=np.float32, copy=True)
        else:
            np.copyto(self._current_projection, proj_2d)

        if self.image_view is not None:
            # Передача proj_2d.T формы (Cols, Rows): ось 0 -> X (col, слева направо), ось 1 -> Y (row, сверху вниз)
            self.image_view.setImage(self._current_projection.T, autoRange=False)

        window_counts = int(np.sum(self._current_projection))
        total_counts = int(np.sum(cube_det))
        self.lbl_stats.setText(f"Отсчетов в окне: {window_counts:,} (Всего: {total_counts:,})")

    def _refresh_from_3d(self) -> None:
        """Обновляет 2D-проекцию из 3D-массива детекторов (N_det, Rows, Cols)."""
        if self._current_3d is None or self._current_3d.ndim != 3:
            return
        n_det, rows, cols = self._current_3d.shape
        if n_det == 0:
            return
        self._update_projection_aspect(rows, cols)
        det_idx = self.combo_detector.currentIndex() if self.combo_detector is not None else 0
        if 0 <= det_idx < n_det:
            proj_2d = self._current_3d[det_idx]
        else:
            proj_2d = np.sum(self._current_3d, axis=0, dtype=np.float32)

        if self._current_projection is None or self._current_projection.shape != proj_2d.shape:
            self._current_projection = np.array(proj_2d, dtype=np.float32, copy=True)
        else:
            np.copyto(self._current_projection, proj_2d)

        if self.image_view is not None:
            self.image_view.setImage(self._current_projection.T, autoRange=False)

        total_counts = int(np.sum(self._current_projection))
        self.lbl_stats.setText(f"Всего отсчетов: {total_counts:,}")

    def set_projection_data(self, image_data: np.ndarray) -> None:
        """
        Обновляет данные детектора (поддерживает 4D гиперспектральный куб (N_det, E_bins, Rows, Cols),
        3D стек детекторов (N_det, Rows, Cols) и классическую 2D матрицу (Rows, Cols)).
        Использует локальную копию буфера, не удерживая прямую ссылку на SharedMemory.buf.
        """
        if image_data is None:
            return

        arr = np.asarray(image_data, dtype=np.float32)
        if arr.ndim == 4:
            self._current_3d = None
            if self._current_cube is None or self._current_cube.shape != arr.shape:
                self._current_cube = np.array(arr, dtype=np.float32, copy=True)
            else:
                np.copyto(self._current_cube, arr)
            self._ensure_detector_names_match(int(arr.shape[0]))
            self._refresh_from_cube()
            return

        if arr.ndim == 3:
            self._current_cube = None
            if self._current_3d is None or self._current_3d.shape != arr.shape:
                self._current_3d = np.array(arr, dtype=np.float32, copy=True)
            else:
                np.copyto(self._current_3d, arr)
            self._ensure_detector_names_match(int(arr.shape[0]))
            self._refresh_from_3d()
            return

        self._current_cube = None
        self._current_3d = None
        if arr.ndim == 2:
            self._update_projection_aspect(int(arr.shape[0]), int(arr.shape[1]))
        if self._current_projection is None or self._current_projection.shape != arr.shape:
            self._current_projection = np.array(arr, dtype=np.float32, copy=True)
        else:
            np.copyto(self._current_projection, arr)

        if self.image_view is not None:
            self.image_view.setImage(self._current_projection.T, autoRange=False)

        total_counts = int(np.sum(self._current_projection))
        self.lbl_stats.setText(f"Всего отсчетов: {total_counts:,}")

    def update_spectrum(
        self,
        energies: np.ndarray,
        bins: Optional[int] = None,
        max_energy: Optional[float] = None,
    ) -> None:
        """
        Обновляет гистограмму энергетического спектра (с очисткой при пустом массиве).
        """
        self.set_spectrum_data(energies, bins=bins, max_energy=max_energy)

    def set_spectrum_data(
        self,
        energy_array: np.ndarray,
        bins: Optional[int] = None,
        max_energy: Optional[float] = None,
    ) -> None:
        """
        Строит гистограмму энергетического спектра по зарегистрированным событиям
        или предрассчитанным бинам куба, не удаляя LinearRegionItem с графика.
        Если активен 4D гиперспектральный куб, спектр всегда берется из него как из единого источника истины.
        """
        if self.spectrum_plot is None or energy_array is None:
            return

        if self._current_cube is not None:
            self._refresh_from_cube()
            return

        arr = np.asarray(energy_array, dtype=np.float32)
        if arr.size == 0:
            self._last_raw_energies = None
            self._set_spectrum_curve_data([], [])
            return

        if arr.ndim == 2:
            n_det, e_bins = arr.shape
            det_idx = self.combo_detector.currentIndex() if self.combo_detector is not None else 0
            if 0 <= det_idx < n_det:
                counts = arr[det_idx]
            else:
                counts = np.sum(arr, axis=0, dtype=np.float32)
            max_e = max(1e-6, float(max_energy if max_energy is not None else self._max_energy))
            edges = np.linspace(0.0, max_e, e_bins + 1, dtype=np.float32)
            bin_centers = 0.5 * (edges[:-1] + edges[1:])
            self._set_spectrum_curve_data(bin_centers, counts)
            return

        self._last_raw_energies = np.array(arr, dtype=np.float32, copy=True)
        effective_bins = max(1, int(bins if bins is not None else self._energy_bins))
        effective_max = max(1e-6, float(max_energy if max_energy is not None else self._max_energy))
        hist_counts, hist_edges = np.histogram(arr, bins=effective_bins, range=(0.0, effective_max))
        hist_centers = (hist_edges[:-1] + hist_edges[1:]) / 2.0
        self._set_spectrum_curve_data(hist_centers, hist_counts)

    def set_projection_stack_data(self, stack: np.ndarray, current_idx: int, total_views: int, angle: float) -> None:
        """
        Обновляет стек проекций ОФЭКТ сканирования и элементы навигации.
        """
        self._projections_stack = stack
        self._total_views = max(1, total_views)
        self._current_view_idx = current_idx

        if self._total_views > 1:
            if self.proj_nav_widget is not None:
                self.proj_nav_widget.setVisible(True)
            self.slider_proj.blockSignals(True)
            self.slider_proj.setRange(1, self._total_views)
            self.slider_proj.setValue(current_idx + 1)
            self.slider_proj.blockSignals(False)
            self.lbl_projection_info.setText(f"Проекция: {current_idx + 1} / {self._total_views} ({angle:.1f}°)")
        else:
            if self.proj_nav_widget is not None:
                self.proj_nav_widget.setVisible(False)

        self._update_splitter_proportions()
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
        self._current_cube = None
        self._current_3d = None
        self._current_projection = None
        self._last_raw_energies = None
        self._projections_stack = None
        self._total_views = 1
        self._current_view_idx = 0
        if self.proj_nav_widget is not None:
            self.proj_nav_widget.setVisible(False)

        if self.image_view is not None:
            self.image_view.clear()
        self._set_spectrum_curve_data([], [])
        self.lbl_stats.setText("Всего отсчетов: 0")
        self.accumulation_cleared.emit()


ResultsViewerWidget = ResultsViewer

__all__ = [
    "ALL_DETECTORS_LABEL",
    "configure_pyqtgraph_theme",
    "ResultsViewer",
    "ResultsViewerWidget",
]


