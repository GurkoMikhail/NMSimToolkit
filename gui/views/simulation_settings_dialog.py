import os
from typing import Any, Callable, Dict, List, Optional
import hepunits as units
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QGroupBox, QFormLayout,
    QSpinBox, QDoubleSpinBox, QCheckBox, QDialogButtonBox, QWidget, QLabel
)

from gui.models.gui_settings import GuiSimulationSettings
from gui.views.widgets import BindableDoubleField


class SimulationSettingsDialog(QDialog):
    """
    Диалоговое окно настройки общих параметров симуляции, пула воркеров,
    параметров отрисовки 3D-треков частиц и дискретной привязки 3D-манипулятора.
    Поддерживает Property Binding параметров расчета к переменным протокола.
    """

    def __init__(
        self,
        current_settings: GuiSimulationSettings,
        parent: Optional[QWidget] = None,
        variable_provider: Optional[Callable[[], List[str]]] = None,
        bindings: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Параметры расчета и визуализации")
        self.resize(460, 440)
        self.setModal(True)

        if not isinstance(current_settings, GuiSimulationSettings):
            raise TypeError("current_settings должен быть экземпляром GuiSimulationSettings")
        self._settings: GuiSimulationSettings = current_settings
        self._variable_provider = variable_provider
        self._initial_bindings = bindings or {}
        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)

        # Информационная сноска согласно SSOT
        info_label = QLabel(
            "ℹ Параметры времени экспозиции и ракурсов определяются "
            "активным протоколом исследования (ОФЭКТ / ПЭТ / CustomSweep).",
            self
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #7fba00; font-size: 11px; background-color: #252525; padding: 6px; border-radius: 4px;")
        main_layout.addWidget(info_label)

        # 1. Секция вычислительных ресурсов
        grp_calc = QGroupBox("Вычислительные ресурсы (SimulationManager)", self)
        form_calc = QFormLayout(grp_calc)

        self.spin_particles = BindableDoubleField(
            min_val=1.0,
            max_val=1_000_000_000.0,
            step=1000.0,
            decimals=0,
            parent=grp_calc,
        )
        self.spin_particles.setValue(float(self._settings.particles_number))
        if self._variable_provider is not None:
            self.spin_particles.set_variable_provider(self._variable_provider)
        if "particles_number" in self._initial_bindings:
            self.spin_particles.bind_variable(self._initial_bindings["particles_number"])
        form_calc.addRow("Число частиц на задачу:", self.spin_particles)

        self.spin_pool = QSpinBox(grp_calc)
        cpu_cnt = os.cpu_count() or 4
        self.spin_pool.setRange(1, max(1, cpu_cnt * 2))
        self.spin_pool.setValue(int(self._settings.pool_size))
        form_calc.addRow("Размер пула воркеров:", self.spin_pool)

        self.spin_min_energy = BindableDoubleField(
            min_val=0.01,
            max_val=100_000.0,
            step=1.0,
            decimals=2,
            suffix=" кэВ",
            parent=grp_calc,
        )
        self.spin_min_energy.setValue(float(self._settings.min_energy))
        if self._variable_provider is not None:
            self.spin_min_energy.set_variable_provider(self._variable_provider)
        if "min_energy" in self._initial_bindings:
            self.spin_min_energy.bind_variable(self._initial_bindings["min_energy"])
        form_calc.addRow("Минимальная энергия:", self.spin_min_energy)

        main_layout.addWidget(grp_calc)

        # 2. Секция визуализации треков и параметров Gizmo
        grp_tracks = QGroupBox("Параметры 3D-визуализации и манипулятора (Gizmo)", self)
        form_tracks = QFormLayout(grp_tracks)

        self.spin_max_batch = QSpinBox(grp_tracks)
        self.spin_max_batch.setRange(10, 100_000)
        self.spin_max_batch.setSingleStep(500)
        self.spin_max_batch.setValue(int(self._settings.max_tracks_per_batch))
        form_tracks.addRow("Максимум треков в пачке:", self.spin_max_batch)

        self.spin_max_points = QSpinBox(grp_tracks)
        self.spin_max_points.setRange(100, 1_000_000)
        self.spin_max_points.setSingleStep(5000)
        self.spin_max_points.setValue(int(self._settings.max_tracks_points))
        form_tracks.addRow("Максимум точек треков:", self.spin_max_points)

        self.chk_render_lines = QCheckBox("Отрисовывать треки сплошными линиями", grp_tracks)
        self.chk_render_lines.setChecked(bool(self._settings.render_as_lines))
        form_tracks.addRow(self.chk_render_lines)

        self.spin_grid_snap = QDoubleSpinBox(grp_tracks)
        self.spin_grid_snap.setRange(0.1, 1000.0)
        self.spin_grid_snap.setSingleStep(5.0)
        self.spin_grid_snap.setSuffix(" мм")
        self.spin_grid_snap.setValue(float(self._settings.grid_snap_step))
        form_tracks.addRow("Шаг сетки перемещения (Grid Snap):", self.spin_grid_snap)

        self.spin_angle_snap = QDoubleSpinBox(grp_tracks)
        self.spin_angle_snap.setRange(1.0, 180.0)
        self.spin_angle_snap.setSingleStep(5.0)
        self.spin_angle_snap.setSuffix(" °")
        self.spin_angle_snap.setValue(float(self._settings.angle_snap_step))
        form_tracks.addRow("Шаг угловой привязки (Angle Snap):", self.spin_angle_snap)

        self.spin_scale_snap = QDoubleSpinBox(grp_tracks)
        self.spin_scale_snap.setRange(0.1, 100.0)
        self.spin_scale_snap.setSingleStep(1.0)
        self.spin_scale_snap.setSuffix(" мм")
        self.spin_scale_snap.setValue(float(self._settings.scale_snap_step))
        form_tracks.addRow("Шаг привязки масштаба (Scale Snap):", self.spin_scale_snap)

        self.spin_xray_energy = QDoubleSpinBox(grp_tracks)
        self.spin_xray_energy.setRange(1.0, 10000.0)
        self.spin_xray_energy.setSingleStep(5.0)
        self.spin_xray_energy.setSuffix(" кэВ")
        self.spin_xray_energy.setValue(float(self._settings.xray_energy / units.keV))
        form_tracks.addRow("Энергия расчета ослабления (X-Ray):", self.spin_xray_energy)

        self.chk_pseudo_xray = QCheckBox("Режим отображения в псевдорентгене", grp_tracks)
        self.chk_pseudo_xray.setChecked(bool(self._settings.pseudo_xray_mode))
        form_tracks.addRow(self.chk_pseudo_xray)

        main_layout.addWidget(grp_tracks)

        # 3. Стандартные кнопки диалога
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        main_layout.addWidget(buttons)

    def get_bindings(self) -> Dict[str, str]:
        """
        Возвращает словарь активных привязок параметров расчета к переменным протокола.
        """
        bindings_map: Dict[str, str] = {}
        if self.spin_particles.bound_variable:
            bindings_map["particles_number"] = self.spin_particles.bound_variable
        if self.spin_min_energy.bound_variable:
            bindings_map["min_energy"] = self.spin_min_energy.bound_variable
        return bindings_map

    def get_settings(self) -> GuiSimulationSettings:
        """
        Возвращает обновленную типизированную модель конфигурации GuiSimulationSettings.
        """
        updated_dict = self._settings.to_dict()
        updated_dict.update({
            "particles_number": int(self.spin_particles.value()),
            "pool_size": self.spin_pool.value(),
            "min_energy": float(self.spin_min_energy.value()),
            "max_tracks_per_batch": self.spin_max_batch.value(),
            "max_tracks_points": self.spin_max_points.value(),
            "render_as_lines": self.chk_render_lines.isChecked(),
            "grid_snap_step": self.spin_grid_snap.value(),
            "angle_snap_step": self.spin_angle_snap.value(),
            "scale_snap_step": self.spin_scale_snap.value(),
            "xray_energy": float(self.spin_xray_energy.value()) * units.keV,
            "pseudo_xray_mode": self.chk_pseudo_xray.isChecked(),
        })
        return GuiSimulationSettings(**updated_dict)
