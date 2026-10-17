"""
Секция параметров сетки поглощенной дозы (DoseGridNode / DoseGridViewModel).
"""
from typing import Any, Optional

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from gui.viewmodels.nodes.dose_grid_vm import DoseGridViewModel
from gui.viewmodels.nodes.gamma_camera_vm import GammaCameraViewModel
from gui.viewmodels.nodes.volume_vm import VolumeViewModel
from gui.views.inspectors.base_section import BaseInspectorSection


class DoseGridInspectorSection(BaseInspectorSection):
    """Секция настройки геометрии воксельной сетки дозы и адаптации под родительский узел."""

    dose_voxel_size_changed = Signal(float)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Инициализирует секцию настройки воксельной сетки дозы."""
        super().__init__("Параметры сетки дозы (Dose Scorer)", parent)

    def _init_ui(self) -> None:
        """Создает элементы управления геометрией и разрешением сетки дозы."""
        form_layout = QFormLayout(self)

        self.btn_fit_dose_grid_to_parent = QPushButton("⇲ Подогнать размер под родителя")
        self.btn_fit_dose_grid_to_parent.setToolTip(
            "Автоматически адаптирует размеры сетки дозы под BoundingBox родительского узла сцены"
        )
        self.btn_fit_dose_grid_to_parent.clicked.connect(self._on_fit_dose_grid_to_parent)
        form_layout.addRow(self.btn_fit_dose_grid_to_parent)

        self.spin_dose_grid_size_x = self._create_size_spinbox(self._on_dose_grid_size_changed)
        self.spin_dose_grid_size_y = self._create_size_spinbox(self._on_dose_grid_size_changed)
        self.spin_dose_grid_size_z = self._create_size_spinbox(self._on_dose_grid_size_changed)

        form_layout.addRow("Размер X (мм):", self.spin_dose_grid_size_x)
        form_layout.addRow("Размер Y (мм):", self.spin_dose_grid_size_y)
        form_layout.addRow("Размер Z (мм):", self.spin_dose_grid_size_z)

        self.spin_dose_grid_voxel = QDoubleSpinBox()
        self.spin_dose_grid_voxel.setRange(0.1, 100.0)
        self.spin_dose_grid_voxel.setSingleStep(0.5)
        self.spin_dose_grid_voxel.setValue(5.0)
        self.spin_dose_grid_voxel.setSuffix(" мм")
        self.spin_dose_grid_voxel.setToolTip("Размер стороны вокселя сетки дозы")
        self.spin_dose_grid_voxel.valueChanged.connect(self._on_dose_grid_voxel_step_changed)
        form_layout.addRow("Шаг вокселя:", self.spin_dose_grid_voxel)

        self.combo_dose_grid_dtype = QComboBox()
        self.combo_dose_grid_dtype.addItems(["float32", "float64"])
        self.combo_dose_grid_dtype.setToolTip(
            "Точность представления вокселей карты дозы (float32 — 4 байта, float64 — 8 байт)"
        )
        self.combo_dose_grid_dtype.currentTextChanged.connect(self._on_dose_grid_dtype_changed)
        form_layout.addRow("Точность (dtype):", self.combo_dose_grid_dtype)

        self.lbl_dose_grid_shape = QLabel("–")
        self.lbl_dose_grid_shape.setToolTip("Разрешение воксельной сетки (Nx × Ny × Nz)")
        form_layout.addRow("Разрешение сетки:", self.lbl_dose_grid_shape)

        self.lbl_dose_grid_memory = QLabel("–")
        self.lbl_dose_grid_memory.setToolTip("Оценка расхода оперативной памяти RAM для сетки")
        form_layout.addRow("Расход памяти RAM:", self.lbl_dose_grid_memory)

        self.chk_dose_grid_active = QCheckBox("Активна для накопления дозы")
        self.chk_dose_grid_active.setChecked(True)
        self.chk_dose_grid_active.toggled.connect(self._on_dose_grid_active_toggled)
        form_layout.addRow(self.chk_dose_grid_active)

    def _create_size_spinbox(self, callback: Any) -> QDoubleSpinBox:
        """Создает поле ввода размера сетки дозы по одной из осей."""
        spin = QDoubleSpinBox()
        spin.setRange(0.1, 5000.0)
        spin.setSingleStep(5.0)
        spin.valueChanged.connect(callback)
        return spin

    def can_handle(self, view_model: Any) -> bool:
        """Проверяет, является ли модель представления узлом сетки дозы."""
        return isinstance(view_model, DoseGridViewModel)

    def update_ui_from_vm(self) -> None:
        """Синхронизирует виджеты размеров, шага вокселя и метрик с текущей DoseGridViewModel."""
        if not isinstance(self._current_vm, DoseGridViewModel):
            return
        old_state = self._is_updating_ui
        self._is_updating_ui = True
        try:
            grid_size = self._current_vm.size
            self.spin_dose_grid_size_x.setValue(float(grid_size[0]))
            self.spin_dose_grid_size_y.setValue(float(grid_size[1]))
            self.spin_dose_grid_size_z.setValue(float(grid_size[2]))
            self.spin_dose_grid_voxel.setValue(float(self._current_vm.dose_voxel_size))
            self.combo_dose_grid_dtype.setCurrentText(str(self._current_vm.dtype))
            self.chk_dose_grid_active.setChecked(bool(self._current_vm.is_active))
            self._update_dose_grid_metrics()

            parent_vm = self._current_vm.parent_vm
            if isinstance(parent_vm, (VolumeViewModel, GammaCameraViewModel)):
                self.btn_fit_dose_grid_to_parent.setEnabled(True)
                self.btn_fit_dose_grid_to_parent.setText(f"⇲ Подогнать под {parent_vm.name}")
            else:
                self.btn_fit_dose_grid_to_parent.setEnabled(False)
                self.btn_fit_dose_grid_to_parent.setText("⇲ Подогнать размер под родителя")
        finally:
            self._is_updating_ui = old_state

    def on_property_changed_externally(self, prop_name: str, new_val: Any) -> None:
        """Обновляет поля UI при внешнем изменении свойств сетки дозы."""
        if not isinstance(self._current_vm, DoseGridViewModel):
            return
        if prop_name == "size":
            grid_size = new_val if isinstance(new_val, (list, tuple, np.ndarray)) else self._current_vm.size
            self._is_updating_ui = True
            try:
                self.spin_dose_grid_size_x.setValue(float(grid_size[0]))
                self.spin_dose_grid_size_y.setValue(float(grid_size[1]))
                self.spin_dose_grid_size_z.setValue(float(grid_size[2]))
                self._update_dose_grid_metrics()
            finally:
                self._is_updating_ui = False
        elif prop_name == "dose_voxel_size":
            self._is_updating_ui = True
            try:
                self.spin_dose_grid_voxel.setValue(float(new_val))
                self._update_dose_grid_metrics()
            finally:
                self._is_updating_ui = False
        elif prop_name == "dtype":
            self._is_updating_ui = True
            try:
                self.combo_dose_grid_dtype.setCurrentText(str(new_val))
                self._update_dose_grid_metrics()
            finally:
                self._is_updating_ui = False
        elif prop_name in ("grid_shape", "memory_mb"):
            self._update_dose_grid_metrics()
        elif prop_name == "is_active":
            self._is_updating_ui = True
            try:
                self.chk_dose_grid_active.setChecked(bool(new_val))
            finally:
                self._is_updating_ui = False

    def _update_dose_grid_metrics(self) -> None:
        """Расчет разрешения сетки вокселей и объема RAM для узла DoseGridViewModel."""
        if not isinstance(self._current_vm, DoseGridViewModel):
            return
        shape = self._current_vm.grid_shape
        n_voxels = int(shape[0] * shape[1] * shape[2])
        memory_megabytes = self._current_vm.memory_mb
        dtype_str = self._current_vm.dtype
        self.lbl_dose_grid_shape.setText(f"{shape[0]} × {shape[1]} × {shape[2]} ({n_voxels:,} вокс.)")
        self.lbl_dose_grid_memory.setText(f"{memory_megabytes:.2f} МБ ({dtype_str})")

    def _on_dose_grid_dtype_changed(self, dtype_text: str) -> None:
        """Обновляет тип данных (float32 / float64) сетки дозы в ViewModel."""
        if self._is_updating_ui or not isinstance(self._current_vm, DoseGridViewModel):
            return
        if dtype_text in ("float32", "float64"):
            self._current_vm.dtype = dtype_text
            self._update_dose_grid_metrics()

    def _on_dose_grid_size_changed(self) -> None:
        """Обновляет размеры сетки дозы в ViewModel при редактировании в UI."""
        if self._is_updating_ui or not isinstance(self._current_vm, DoseGridViewModel):
            return
        self._current_vm.size = (
            self.spin_dose_grid_size_x.value(),
            self.spin_dose_grid_size_y.value(),
            self.spin_dose_grid_size_z.value(),
        )
        self._update_dose_grid_metrics()

    def _on_dose_grid_voxel_step_changed(self) -> None:
        """Обновляет шаг вокселя сетки дозы и испускает сигнал перестройки визуализации."""
        if self._is_updating_ui or not isinstance(self._current_vm, DoseGridViewModel):
            return
        self._current_vm.dose_voxel_size = self.spin_dose_grid_voxel.value()
        self._update_dose_grid_metrics()
        self.dose_voxel_size_changed.emit(self.spin_dose_grid_voxel.value())

    def _on_dose_grid_active_toggled(self, checked: bool) -> None:
        """Переключает активность накопления дозы в ViewModel."""
        if self._is_updating_ui or not isinstance(self._current_vm, DoseGridViewModel):
            return
        self._current_vm.is_active = checked

    def _on_fit_dose_grid_to_parent(self) -> None:
        """Автоматически подгоняет размеры сетки дозы под BoundingBox родительского узла."""
        if self._is_updating_ui or not isinstance(self._current_vm, DoseGridViewModel):
            return
        parent_vm = self._current_vm.parent_vm
        if not isinstance(parent_vm, (VolumeViewModel, GammaCameraViewModel)):
            return

        bounding_box_size = parent_vm.local_bound
        new_size = (float(bounding_box_size[0]), float(bounding_box_size[1]), float(bounding_box_size[2]))
        self._current_vm.size = new_size
        self._is_updating_ui = True
        try:
            self.spin_dose_grid_size_x.setValue(new_size[0])
            self.spin_dose_grid_size_y.setValue(new_size[1])
            self.spin_dose_grid_size_z.setValue(new_size[2])
            self._update_dose_grid_metrics()
        finally:
            self._is_updating_ui = False


__all__ = ["DoseGridInspectorSection"]
