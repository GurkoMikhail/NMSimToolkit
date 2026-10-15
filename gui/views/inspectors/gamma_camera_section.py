"""
Секция параметров гамма-камеры (GammaCameraNode / GammaCameraViewModel).
"""
from typing import Any, Optional

import numpy as np
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QWidget,
)

from gui.viewmodels.nodes.gamma_camera_vm import GammaCameraViewModel
from gui.views.inspectors.base_section import BaseInspectorSection


class GammaCameraInspectorSection(BaseInspectorSection):
    """Секция настройки геометрии детекторной головки, кристалла, защитного экрана Pb и корпуса."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Инициализирует секцию параметров гамма-камеры."""
        super().__init__("Параметры гамма-камеры", parent)

    def _init_ui(self) -> None:
        """Создает элементы управления габаритами детекторной головки, кристалла и защиты."""
        form_layout = QFormLayout(self)

        self.spin_detector_size_x = QDoubleSpinBox()
        self.spin_detector_size_x.setRange(10.0, 2000.0)
        self.spin_detector_size_x.setSingleStep(10.0)
        self.spin_detector_size_x.setSuffix(" мм")
        self.spin_detector_size_x.valueChanged.connect(self._on_spect_detector_size_changed)

        self.spin_detector_size_y = QDoubleSpinBox()
        self.spin_detector_size_y.setRange(10.0, 2000.0)
        self.spin_detector_size_y.setSingleStep(10.0)
        self.spin_detector_size_y.setSuffix(" мм")
        self.spin_detector_size_y.valueChanged.connect(self._on_spect_detector_size_changed)

        det_size_layout = QHBoxLayout()
        det_size_layout.addWidget(QLabel("X:"))
        det_size_layout.addWidget(self.spin_detector_size_x)
        det_size_layout.addWidget(QLabel("Y:"))
        det_size_layout.addWidget(self.spin_detector_size_y)

        self.spin_detector_thickness = QDoubleSpinBox()
        self.spin_detector_thickness.setRange(0.1, 200.0)
        self.spin_detector_thickness.setSingleStep(0.5)
        self.spin_detector_thickness.setSuffix(" мм")
        self.spin_detector_thickness.valueChanged.connect(self._on_spect_detector_thickness_changed)

        self.spin_collimator_thickness = QDoubleSpinBox()
        self.spin_collimator_thickness.setRange(1.0, 300.0)
        self.spin_collimator_thickness.setSingleStep(1.0)
        self.spin_collimator_thickness.setSuffix(" мм")
        self.spin_collimator_thickness.valueChanged.connect(self._on_spect_collimator_thickness_changed)

        self.spin_cam_gap = QDoubleSpinBox()
        self.spin_cam_gap.setRange(0.0, 100.0)
        self.spin_cam_gap.setSingleStep(0.5)
        self.spin_cam_gap.setValue(1.0)
        self.spin_cam_gap.setSuffix(" мм")
        self.spin_cam_gap.valueChanged.connect(self._on_spect_cam_gap_changed)

        self.spin_cam_shielding = QDoubleSpinBox()
        self.spin_cam_shielding.setRange(0.0, 200.0)
        self.spin_cam_shielding.setSingleStep(1.0)
        self.spin_cam_shielding.setValue(20.0)
        self.spin_cam_shielding.setSuffix(" мм")
        self.spin_cam_shielding.valueChanged.connect(self._on_spect_cam_shielding_changed)

        self.spin_cam_glass = QDoubleSpinBox()
        self.spin_cam_glass.setRange(0.0, 200.0)
        self.spin_cam_glass.setSingleStep(1.0)
        self.spin_cam_glass.setValue(50.0)
        self.spin_cam_glass.setSuffix(" мм")
        self.spin_cam_glass.valueChanged.connect(self._on_spect_cam_glass_changed)

        self.lbl_housing_size = QLabel("-")

        form_layout.addRow("Активное поле детектора (X×Y):", det_size_layout)
        form_layout.addRow("Толщина кристалла (Z):", self.spin_detector_thickness)
        form_layout.addRow("Толщина коллиматора (Z):", self.spin_collimator_thickness)
        form_layout.addRow("Зазор детектор-коллиматор:", self.spin_cam_gap)
        form_layout.addRow("Толщина свинцовой защиты Pb:", self.spin_cam_shielding)
        form_layout.addRow("Толщина оптического стекла:", self.spin_cam_glass)
        form_layout.addRow("Габариты корпуса (X×Y×Z):", self.lbl_housing_size)

    def can_handle(self, view_model: Any) -> bool:
        """Проверяет, является ли модель представления гамма-камерой GammaCameraViewModel."""
        return isinstance(view_model, GammaCameraViewModel)

    def update_ui_from_vm(self) -> None:
        """Синхронизирует все поля секции с текущей моделью GammaCameraViewModel."""
        if not isinstance(self._current_vm, GammaCameraViewModel):
            return

        old_state = self._is_updating_ui
        self._is_updating_ui = True
        try:
            detector_dimensions = self._current_vm.detector_size
            self.spin_detector_size_x.setValue(float(detector_dimensions[0]))
            self.spin_detector_size_y.setValue(float(detector_dimensions[1]))
            self.spin_detector_thickness.setValue(float(self._current_vm.detector_thickness))
            self.spin_collimator_thickness.setValue(float(self._current_vm.collimator_thickness))
            self.spin_cam_gap.setValue(float(self._current_vm.gap))
            self.spin_cam_shielding.setValue(float(self._current_vm.shielding_thickness))
            self.spin_cam_glass.setValue(float(self._current_vm.glass_backend_thickness))
            self._update_housing_size_label()
        finally:
            self._is_updating_ui = old_state

    def on_property_changed_externally(self, prop_name: str, new_val: Any) -> None:
        """Обновляет поля размеров и толщин гамма-камеры при внешнем изменении модели."""
        if not isinstance(self._current_vm, GammaCameraViewModel):
            return

        if prop_name in (
            "detector_size", "detector_thickness", "collimator_thickness",
            "gap", "shielding_thickness", "glass_backend_thickness", "housing_size"
        ):
            self._is_updating_ui = True
            try:
                if prop_name == "detector_size" and isinstance(new_val, (list, tuple, np.ndarray)) and len(new_val) >= 2:
                    self.spin_detector_size_x.setValue(float(new_val[0]))
                    self.spin_detector_size_y.setValue(float(new_val[1]))
                elif prop_name == "detector_thickness":
                    self.spin_detector_thickness.setValue(float(new_val))
                elif prop_name == "collimator_thickness":
                    self.spin_collimator_thickness.setValue(float(new_val))
                elif prop_name == "gap":
                    self.spin_cam_gap.setValue(float(new_val))
                elif prop_name == "shielding_thickness":
                    self.spin_cam_shielding.setValue(float(new_val))
                elif prop_name == "glass_backend_thickness":
                    self.spin_cam_glass.setValue(float(new_val))
                self._update_housing_size_label()
            finally:
                self._is_updating_ui = False

    def _update_housing_size_label(self) -> None:
        """Обновляет информационную метку внешних габаритов корпуса гамма-камеры."""
        if isinstance(self._current_vm, GammaCameraViewModel):
            housing_dims = self._current_vm.housing_size
            self.lbl_housing_size.setText(f"{housing_dims[0]:.1f} × {housing_dims[1]:.1f} × {housing_dims[2]:.1f} мм")

    def _on_spect_detector_size_changed(self) -> None:
        """Применяет новые размеры активного поля детектора (X, Y)."""
        if self._is_updating_ui or not isinstance(self._current_vm, GammaCameraViewModel):
            return
        size_x = float(self.spin_detector_size_x.value())
        size_y = float(self.spin_detector_size_y.value())
        self._current_vm.detector_size = (size_x, size_y)
        self._update_housing_size_label()

    def _on_spect_detector_thickness_changed(self, value: float) -> None:
        """Применяет новую толщину сцинтилляционного кристалла."""
        if self._is_updating_ui or not isinstance(self._current_vm, GammaCameraViewModel):
            return
        self._current_vm.detector_thickness = float(value)
        self._update_housing_size_label()

    def _on_spect_collimator_thickness_changed(self, value: float) -> None:
        """Применяет новую толщину коллиматора."""
        if self._is_updating_ui or not isinstance(self._current_vm, GammaCameraViewModel):
            return
        self._current_vm.collimator_thickness = float(value)
        self._update_housing_size_label()

    def _on_spect_cam_gap_changed(self, value: float) -> None:
        """Применяет новое значение воздушного зазора между детектором и коллиматором."""
        if self._is_updating_ui or not isinstance(self._current_vm, GammaCameraViewModel):
            return
        self._current_vm.gap = float(value)
        self._update_housing_size_label()

    def _on_spect_cam_shielding_changed(self, value: float) -> None:
        """Применяет новую толщину свинцового защитного экрана."""
        if self._is_updating_ui or not isinstance(self._current_vm, GammaCameraViewModel):
            return
        self._current_vm.shielding_thickness = float(value)
        self._update_housing_size_label()

    def _on_spect_cam_glass_changed(self, value: float) -> None:
        """Применяет новую толщину заднего оптического световода."""
        if self._is_updating_ui or not isinstance(self._current_vm, GammaCameraViewModel):
            return
        self._current_vm.glass_backend_thickness = float(value)
        self._update_housing_size_label()


__all__ = ["GammaCameraInspectorSection"]
