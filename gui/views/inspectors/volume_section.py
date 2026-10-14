"""
Секция геометрических размеров и материала объема (Volume / Box).
"""
from typing import Any, Optional

import numpy as np
from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QFormLayout, QWidget

import settings.database_setting as database_setting
from gui.viewmodels.nodes.base_node_vm import NodeViewModel
from gui.viewmodels.nodes.collimator_vm import CollimatorViewModel
from gui.viewmodels.nodes.gamma_camera_vm import GammaCameraViewModel
from gui.viewmodels.nodes.volume_vm import VolumeViewModel
from gui.viewmodels.nodes.voxel_volume_vm import VoxelVolumeViewModel
from gui.views.inspectors.base_section import BaseInspectorSection


class VolumeInspectorSection(BaseInspectorSection):
    """Секция редактирования размеров Box и выбора материала NIST."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Инициализирует секцию редактирования геометрии и материала объема."""
        super().__init__("Геометрия и материал", parent)

    def _init_ui(self) -> None:
        """Создает спинбоксы размеров и выпадающий список материалов."""
        form_layout = QFormLayout(self)

        self.spin_size_x = self._create_size_spinbox(self._on_volume_size_changed)
        self.spin_size_y = self._create_size_spinbox(self._on_volume_size_changed)
        self.spin_size_z = self._create_size_spinbox(self._on_volume_size_changed)

        form_layout.addRow("Размер X (мм):", self.spin_size_x)
        form_layout.addRow("Размер Y (мм):", self.spin_size_y)
        form_layout.addRow("Размер Z (мм):", self.spin_size_z)

        self.combo_material = QComboBox()
        self._populate_materials()
        self.combo_material.currentTextChanged.connect(self._on_material_changed)
        form_layout.addRow("Материал:", self.combo_material)

    def _create_size_spinbox(self, callback: Any) -> QDoubleSpinBox:
        """Создает спинбокс для габаритного размера объема."""
        spin = QDoubleSpinBox()
        spin.setRange(0.1, 5000.0)
        spin.setSingleStep(5.0)
        spin.valueChanged.connect(callback)
        return spin

    def _populate_materials(self) -> None:
        """Заполняет выпадающий список доступными материалами из базы NIST."""
        if database_setting.material_database:
            for name in sorted(database_setting.material_database.keys()):
                self.combo_material.addItem(name)
        else:
            self.combo_material.addItems(["Water", "Air", "Lead", "Vacuum"])

    def can_handle(self, view_model: Any) -> bool:
        """Проверяет применимость секции к моделям объема и коллиматора."""
        return (
            (isinstance(view_model, VolumeViewModel) or isinstance(view_model, CollimatorViewModel))
            and not isinstance(view_model, (GammaCameraViewModel, VoxelVolumeViewModel))
        )

    def update_constraint_state(self, view_model: Any) -> None:
        """Обновление доступности спинбоксов масштабирования по кинематическим ограничениям узла."""
        if not isinstance(view_model, NodeViewModel):
            return
        constraint = view_model.get_effective_kinematic_constraint()
        if constraint is not None:
            scale_allowed = constraint.is_scale_allowed()
            is_root_volume = False
            if isinstance(view_model, VolumeViewModel):
                if self._scene_vm is not None:
                    is_root_volume = self._scene_vm.is_root_volume(view_model)
                else:
                    is_root_volume = view_model.is_root_volume

            if is_root_volume:
                self.spin_size_x.setEnabled(True)
                self.spin_size_y.setEnabled(True)
                self.spin_size_z.setEnabled(True)
            else:
                self.spin_size_x.setEnabled(scale_allowed)
                self.spin_size_y.setEnabled(scale_allowed)
                self.spin_size_z.setEnabled(scale_allowed)
        else:
            self.spin_size_x.setEnabled(True)
            self.spin_size_y.setEnabled(True)
            self.spin_size_z.setEnabled(True)

    def update_ui_from_vm(self) -> None:
        """Синхронизирует размеры и материал объема из текущей модели представления."""
        if not isinstance(self._current_vm, (VolumeViewModel, CollimatorViewModel)):
            return

        self.update_constraint_state(self._current_vm)

        old_state = self._is_updating_ui
        self._is_updating_ui = True
        try:
            volume_size = self._current_vm.size
            self.spin_size_x.setValue(float(volume_size[0]))
            self.spin_size_y.setValue(float(volume_size[1]))
            self.spin_size_z.setValue(float(volume_size[2]))
            material_index = self.combo_material.findText(self._current_vm.material_name)
            if material_index >= 0:
                self.combo_material.setCurrentIndex(material_index)
        finally:
            self._is_updating_ui = old_state

    def on_property_changed_externally(self, prop_name: str, new_val: Any) -> None:
        """Обновляет поля размеров или материала при внешнем изменении свойств модели."""
        if not isinstance(self._current_vm, (VolumeViewModel, CollimatorViewModel)):
            return
        if prop_name == "size":
            volume_size = new_val if isinstance(new_val, (list, tuple, np.ndarray)) else self._current_vm.size
            self._is_updating_ui = True
            try:
                self.spin_size_x.setValue(float(volume_size[0]))
                self.spin_size_y.setValue(float(volume_size[1]))
                self.spin_size_z.setValue(float(volume_size[2]))
            finally:
                self._is_updating_ui = False
        elif prop_name == "material_name":
            material_index = self.combo_material.findText(str(new_val))
            if material_index >= 0:
                self._is_updating_ui = True
                try:
                    self.combo_material.setCurrentIndex(material_index)
                finally:
                    self._is_updating_ui = False

    def _on_volume_size_changed(self) -> None:
        """Применяет новые габаритные размеры к модели представления объема."""
        if self._is_updating_ui or not isinstance(self._current_vm, (VolumeViewModel, CollimatorViewModel)):
            return
        self._current_vm.size = [
            self.spin_size_x.value(),
            self.spin_size_y.value(),
            self.spin_size_z.value(),
        ]

    def _on_material_changed(self, mat_name: str) -> None:
        """Применяет выбранный материал NIST к модели представления объема."""
        if self._is_updating_ui or not isinstance(self._current_vm, (VolumeViewModel, CollimatorViewModel)):
            return
        self._current_vm.material_name = mat_name


__all__ = ["VolumeInspectorSection"]
