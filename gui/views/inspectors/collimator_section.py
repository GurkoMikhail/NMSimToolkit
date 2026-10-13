"""
Секция параметров коллиматора (прямой Direct и параметрический Parametric).
"""
import logging
from typing import Any, Optional

import numpy as np
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QWidget,
)

import settings.database_setting as database_setting
from core.geometry.direct_collimators import CollimatorHoleShape, DirectParallelCollimator
from core.geometry.parametric_collimators import ParametricParallelCollimator
from core.materials.materials import Material
from gui.viewmodels.nodes.collimator_vm import CollimatorViewModel
from gui.views.inspectors.base_section import BaseInspectorSection

_logger = logging.getLogger(__name__)


class CollimatorInspectorSection(BaseInspectorSection):
    """Секция настройки геометрии каналов коллиматора, септ и переключения типов."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Параметры коллиматора (Collimator)", parent)

    def _init_ui(self) -> None:
        form_layout = QFormLayout(self)

        self.combo_collimator_type = QComboBox()
        for title, val in (("Детерминированный (Сквозные каналы)", "direct"), ("Параметрический (RayCasting)", "parametric")):
            self.combo_collimator_type.addItem(title, val)
        self.combo_collimator_type.currentIndexChanged.connect(self._on_collimator_type_changed)
        form_layout.addRow("Тип коллиматора:", self.combo_collimator_type)

        self.lbl_collimator_shape = QLabel("Форма каналов:")
        self.combo_collimator_shape = QComboBox()
        for title, shape in (("Гексагональный", CollimatorHoleShape.HEXAGONAL), ("Квадратный", CollimatorHoleShape.SQUARE), ("Круглый", CollimatorHoleShape.ROUND)):
            self.combo_collimator_shape.addItem(title, shape)
        self.combo_collimator_shape.currentIndexChanged.connect(self._on_collimator_shape_changed)
        form_layout.addRow(self.lbl_collimator_shape, self.combo_collimator_shape)

        self.lbl_hole_material = QLabel("Материал каналов:")
        self.combo_hole_material = QComboBox()
        self._populate_hole_materials()
        self.combo_hole_material.currentIndexChanged.connect(self._on_hole_material_changed)
        form_layout.addRow(self.lbl_hole_material, self.combo_hole_material)

        self.lbl_collimator_hole = QLabel("Диаметр отверстий:")
        self.spin_collimator_hole = QDoubleSpinBox()
        self.spin_collimator_hole.setRange(0.01, 50.0)
        self.spin_collimator_hole.setSingleStep(0.1)
        self.spin_collimator_hole.setValue(1.5)
        self.spin_collimator_hole.setSuffix(" мм")
        self.spin_collimator_hole.valueChanged.connect(self._on_collimator_hole_changed)
        form_layout.addRow(self.lbl_collimator_hole, self.spin_collimator_hole)

        self.spin_collimator_septa = QDoubleSpinBox()
        self.spin_collimator_septa.setRange(0.001, 20.0)
        self.spin_collimator_septa.setSingleStep(0.05)
        self.spin_collimator_septa.setValue(0.2)
        self.spin_collimator_septa.setSuffix(" мм")
        self.spin_collimator_septa.valueChanged.connect(self._on_collimator_septa_changed)
        form_layout.addRow("Толщина септ:", self.spin_collimator_septa)

    def _populate_hole_materials(self) -> None:
        self.combo_hole_material.clear()
        self.combo_hole_material.addItem("По умолчанию (от родителя)", None)
        if database_setting.material_database:
            for name in sorted(database_setting.material_database.keys()):
                self.combo_hole_material.addItem(name, name)
        else:
            for name in ["Air", "Vacuum", "Water"]:
                self.combo_hole_material.addItem(name, name)

    def can_handle(self, view_model: Any) -> bool:
        return isinstance(view_model, CollimatorViewModel)

    def update_ui_from_vm(self) -> None:
        if not isinstance(self._current_vm, CollimatorViewModel):
            return

        old_state = self._is_updating_ui
        self._is_updating_ui = True
        try:
            col_kind = self._current_vm.collimator_kind
            type_index = self.combo_collimator_type.findData(col_kind)
            if type_index >= 0:
                self.combo_collimator_type.setCurrentIndex(type_index)

            is_direct = (col_kind == "direct")
            self.lbl_collimator_shape.setVisible(True)
            self.combo_collimator_shape.setVisible(True)
            self.lbl_hole_material.setVisible(is_direct)
            self.combo_hole_material.setVisible(is_direct)

            current_shape = self._current_vm.hole_shape
            if current_shape == CollimatorHoleShape.SQUARE:
                self.lbl_collimator_hole.setText("Ширина отверстий:")
            else:
                self.lbl_collimator_hole.setText("Диаметр отверстий:")
            self.spin_collimator_hole.setValue(float(self._current_vm.hole_diameter))
            self.spin_collimator_septa.setValue(float(self._current_vm.septa))

            current_shape_val = current_shape.value if isinstance(current_shape, CollimatorHoleShape) else current_shape
            for shape_index in range(self.combo_collimator_shape.count()):
                item_shape = self.combo_collimator_shape.itemData(shape_index)
                if item_shape == current_shape or item_shape == current_shape_val:
                    self.combo_collimator_shape.setCurrentIndex(shape_index)
                    break

            if is_direct:
                hole_mat_name = self._current_vm.hole_material_name
                if hole_mat_name is None:
                    self.combo_hole_material.setCurrentIndex(0)
                else:
                    material_index = self.combo_hole_material.findData(hole_mat_name)
                    if material_index >= 0:
                        self.combo_hole_material.setCurrentIndex(material_index)
                    else:
                        self.combo_hole_material.setCurrentIndex(0)
        finally:
            self._is_updating_ui = old_state

    def _run_ui_update(self, callback: Any) -> None:
        self._is_updating_ui = True
        try:
            callback()
        finally:
            self._is_updating_ui = False

    def on_property_changed_externally(self, prop_name: str, new_val: Any) -> None:
        if not isinstance(self._current_vm, CollimatorViewModel):
            return
        if prop_name == "hole_material_name":
            found_idx = self.combo_hole_material.findData(str(new_val)) if new_val is not None else 0
            self._run_ui_update(lambda: self.combo_hole_material.setCurrentIndex(max(0, found_idx)))
        elif prop_name in ("hole_diameter", "hole_width"):
            self._run_ui_update(lambda: self.spin_collimator_hole.setValue(float(new_val)))
        elif prop_name == "septa":
            self._run_ui_update(lambda: self.spin_collimator_septa.setValue(float(new_val)))
        elif prop_name == "hole_shape":
            new_shape_val = new_val.value if isinstance(new_val, CollimatorHoleShape) else new_val
            for shape_index in range(self.combo_collimator_shape.count()):
                item_shape = self.combo_collimator_shape.itemData(shape_index)
                if item_shape in (new_val, new_shape_val):
                    self._run_ui_update(lambda idx=shape_index: self.combo_collimator_shape.setCurrentIndex(idx))
                    break

    def _on_collimator_hole_changed(self, value: float) -> None:
        if self._is_updating_ui or not isinstance(self._current_vm, CollimatorViewModel):
            return
        self._current_vm.hole_diameter = float(value)

    def _on_collimator_septa_changed(self, value: float) -> None:
        if self._is_updating_ui or not isinstance(self._current_vm, CollimatorViewModel):
            return
        self._current_vm.septa = float(value)

    def _on_collimator_shape_changed(self, index: int) -> None:
        if self._is_updating_ui or not isinstance(self._current_vm, CollimatorViewModel):
            return
        shape_data = self.combo_collimator_shape.itemData(index)
        if shape_data is None:
            return
        shape_enum = CollimatorHoleShape(shape_data) if isinstance(shape_data, str) else shape_data
        if isinstance(self._current_vm.core_node, DirectParallelCollimator) and shape_enum != CollimatorHoleShape.HEXAGONAL:
            shape_name = shape_enum.value if isinstance(shape_enum, CollimatorHoleShape) else str(shape_enum)
            _logger.warning("Форма каналов '%s' еще не реализована в прямом параллельном коллиматоре.", shape_name)
            self._is_updating_ui = True
            try:
                current_shape = self._current_vm.hole_shape
                current_shape_val = current_shape.value if isinstance(current_shape, CollimatorHoleShape) else current_shape
                for shape_index in range(self.combo_collimator_shape.count()):
                    item_shape = self.combo_collimator_shape.itemData(shape_index)
                    if item_shape == current_shape or item_shape == current_shape_val:
                        self.combo_collimator_shape.setCurrentIndex(shape_index)
                        break
            finally:
                self._is_updating_ui = False
            return
        self._current_vm.hole_shape = shape_data

    def _on_hole_material_changed(self, index: int) -> None:
        if self._is_updating_ui or not isinstance(self._current_vm, CollimatorViewModel):
            return
        selected_hole_mat_name = self.combo_hole_material.itemData(index)
        self._current_vm.hole_material_name = selected_hole_mat_name

    def _on_collimator_type_changed(self, index: int) -> None:
        if self._is_updating_ui or not isinstance(self._current_vm, CollimatorViewModel):
            return
        collimator_vm = self._current_vm
        target_kind = self.combo_collimator_type.itemData(index)
        if target_kind == collimator_vm.collimator_kind:
            return

        col_size = np.copy(collimator_vm.size)
        hole_diameter = float(collimator_vm.hole_diameter)
        septa = float(collimator_vm.septa)
        hole_shape = collimator_vm.hole_shape
        material_name = collimator_vm.material_name
        lead_mat = database_setting.material_database.get(material_name, Material(name=material_name))

        if target_kind == "direct":
            new_core = DirectParallelCollimator(
                size=col_size,
                hole_diameter=hole_diameter,
                septa=septa,
                material=lead_mat,
                hole_material=None,
                hole_shape=hole_shape,
                name=collimator_vm.name,
            )
        elif target_kind == "parametric":
            new_core = ParametricParallelCollimator(
                size=col_size,
                hole_diameter=hole_diameter,
                septa=septa,
                material=lead_mat,
                hole_shape=hole_shape,
                name=collimator_vm.name,
            )
        else:
            return

        new_core.local_matrix = np.copy(collimator_vm.local_matrix)
        new_col_vm = CollimatorViewModel(new_core)

        if self._scene_vm is not None:
            self._scene_vm.replace_node(collimator_vm, new_col_vm)
            self.target_viewmodel_requested.emit(new_col_vm)
