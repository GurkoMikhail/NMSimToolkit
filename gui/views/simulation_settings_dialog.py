import os
from typing import Any, Dict, Optional
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QGroupBox, QFormLayout,
    QSpinBox, QDoubleSpinBox, QCheckBox, QDialogButtonBox, QWidget
)


class SimulationSettingsDialog(QDialog):
    """
    Диалоговое окно настройки общих параметров симуляции, пула воркеров, буфера данных
    и параметров отрисовки 3D-треков частиц.
    """

    def __init__(self, current_settings: Dict[str, Any], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Параметры расчета симуляции")
        self.resize(450, 420)
        self.setModal(True)

        self._settings = dict(current_settings)
        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)

        # 1. Секция общих параметров физического расчета
        grp_calc = QGroupBox("Параметры расчета (SimulationManager)", self)
        form_calc = QFormLayout(grp_calc)

        self.spin_particles = QSpinBox(grp_calc)
        self.spin_particles.setRange(1, 1_000_000_000)
        self.spin_particles.setSingleStep(1000)
        self.spin_particles.setValue(int(self._settings.get("particles_number", 5000)))
        form_calc.addRow("Число частиц на задачу:", self.spin_particles)

        self.spin_pool = QSpinBox(grp_calc)
        cpu_cnt = os.cpu_count() or 4
        self.spin_pool.setRange(1, max(1, cpu_cnt * 2))
        self.spin_pool.setValue(int(self._settings.get("pool_size", 1)))
        form_calc.addRow("Размер пула воркеров:", self.spin_pool)

        self.spin_stop_time = QDoubleSpinBox(grp_calc)
        self.spin_stop_time.setRange(0.001, 100_000.0)
        self.spin_stop_time.setSingleStep(0.5)
        self.spin_stop_time.setSuffix(" с")
        self.spin_stop_time.setValue(float(self._settings.get("stop_time", 1.0)))
        form_calc.addRow("Время счета (Stop Time):", self.spin_stop_time)

        self.spin_min_energy = QDoubleSpinBox(grp_calc)
        self.spin_min_energy.setRange(0.01, 100_000.0)
        self.spin_min_energy.setSingleStep(1.0)
        self.spin_min_energy.setSuffix(" кэВ")
        self.spin_min_energy.setValue(float(self._settings.get("min_energy", 1.0)))
        form_calc.addRow("Минимальная энергия:", self.spin_min_energy)

        main_layout.addWidget(grp_calc)

        # 2. Секция параметров диспетчера данных
        grp_data = QGroupBox("Параметры диспетчера данных (DataManager)", self)
        form_data = QFormLayout(grp_data)

        self.spin_buffer = QSpinBox(grp_data)
        self.spin_buffer.setRange(1000, 100_000_000)
        self.spin_buffer.setSingleStep(50000)
        self.spin_buffer.setValue(int(self._settings.get("buffer_capacity", 10000)))
        form_data.addRow("Емкость буфера частиц:", self.spin_buffer)

        main_layout.addWidget(grp_data)

        # 3. Секция визуализации треков во вьюпорте
        grp_tracks = QGroupBox("Параметры отображения треков (3D Viewport)", self)
        form_tracks = QFormLayout(grp_tracks)

        self.spin_max_batch = QSpinBox(grp_tracks)
        self.spin_max_batch.setRange(10, 100_000)
        self.spin_max_batch.setSingleStep(500)
        self.spin_max_batch.setValue(int(self._settings.get("max_tracks_per_batch", 2000)))
        form_tracks.addRow("Максимум треков в пачке:", self.spin_max_batch)

        self.spin_max_points = QSpinBox(grp_tracks)
        self.spin_max_points.setRange(100, 1_000_000)
        self.spin_max_points.setSingleStep(5000)
        self.spin_max_points.setValue(int(self._settings.get("max_tracks_points", 50000)))
        form_tracks.addRow("Максимум точек треков:", self.spin_max_points)

        self.chk_render_lines = QCheckBox("Отрисовывать треки сплошными линиями", grp_tracks)
        self.chk_render_lines.setChecked(bool(self._settings.get("render_as_lines", True)))
        form_tracks.addRow(self.chk_render_lines)

        self.chk_show_escaped = QCheckBox("Отображать вылетевшие за пределы треки", grp_tracks)
        self.chk_show_escaped.setChecked(bool(self._settings.get("show_escaped_tracks", False)))
        form_tracks.addRow(self.chk_show_escaped)

        main_layout.addWidget(grp_tracks)

        # 4. Стандартные кнопки диалога
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        main_layout.addWidget(buttons)

    def get_settings(self) -> Dict[str, Any]:
        """
        Возвращает обновленный словарь параметров симуляции.
        """
        return {
            "particles_number": self.spin_particles.value(),
            "pool_size": self.spin_pool.value(),
            "stop_time": self.spin_stop_time.value(),
            "min_energy": self.spin_min_energy.value(),
            "buffer_capacity": self.spin_buffer.value(),
            "max_tracks_per_batch": self.spin_max_batch.value(),
            "max_tracks_points": self.spin_max_points.value(),
            "render_as_lines": self.chk_render_lines.isChecked(),
            "show_escaped_tracks": self.chk_show_escaped.isChecked(),
        }
