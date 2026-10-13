"""
Секция пространственных трансформаций узла сцены (положение и ориентация).
"""
from typing import Any, Optional

import numpy as np
from scipy.spatial.transform import Rotation
from PySide6.QtWidgets import QFormLayout, QWidget

from gui.viewmodels.nodes.base_node_vm import NodeViewModel
from gui.viewmodels.nodes.gamma_camera_vm import GammaCameraViewModel
from gui.viewmodels.nodes.gantry_vm import GantryViewModel
from gui.viewmodels.nodes.volume_vm import VolumeViewModel
from gui.viewmodels.procedure_viewmodel import (
    CustomSweepProcedureViewModel,
    StepAndShootProcedureViewModel,
)
from gui.viewport_3d.transform_gizmo import GizmoAxis, GizmoMode
from gui.views.inspectors.base_section import BaseInspectorSection
from gui.views.widgets import BindableDoubleField


class TransformInspectorSection(BaseInspectorSection):
    """Секция редактирования пространственных координат и углов поворота."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        self._is_transform_connected: bool = False
        super().__init__("Положение и ориентация", parent)

    def _init_ui(self) -> None:
        form_layout = QFormLayout(self)

        self.spin_x = self._create_coord_spinbox(self._on_transform_changed, "translation.x")
        self.spin_y = self._create_coord_spinbox(self._on_transform_changed, "translation.y")
        self.spin_z = self._create_coord_spinbox(self._on_transform_changed, "translation.z")

        self.spin_rot_x = self._create_rot_spinbox(self._on_transform_changed, "rotation.alpha")
        self.spin_rot_y = self._create_rot_spinbox(self._on_transform_changed, "rotation.beta")
        self.spin_rot_z = self._create_rot_spinbox(self._on_transform_changed, "rotation.gamma")

        form_layout.addRow("Позиция X (мм):", self.spin_x)
        form_layout.addRow("Позиция Y (мм):", self.spin_y)
        form_layout.addRow("Позиция Z (мм):", self.spin_z)
        form_layout.addRow("Поворот Roll (°):", self.spin_rot_x)
        form_layout.addRow("Поворот Pitch (°):", self.spin_rot_y)
        form_layout.addRow("Поворот Yaw (°):", self.spin_rot_z)

    def _create_coord_spinbox(self, callback: Any, prop_name: str) -> BindableDoubleField:
        field = BindableDoubleField(
            min_val=-5000.0,
            max_val=5000.0,
            step=1.0,
            decimals=2,
        )
        field.set_variable_provider(self._get_protocol_variables)
        field.set_variable_created_callback(self._on_variable_created)
        field.valueChanged.connect(callback)
        field.bindingChanged.connect(
            lambda var_name, property_key=prop_name: self._on_field_binding_changed(property_key, var_name)
        )
        return field

    def _create_rot_spinbox(self, callback: Any, prop_name: str) -> BindableDoubleField:
        field = BindableDoubleField(
            min_val=-180.0,
            max_val=180.0,
            step=1.0,
            decimals=2,
        )
        field.set_variable_provider(self._get_protocol_variables)
        field.set_variable_created_callback(self._on_variable_created)
        field.valueChanged.connect(callback)
        field.bindingChanged.connect(
            lambda var_name, property_key=prop_name: self._on_field_binding_changed(property_key, var_name)
        )
        return field

    def can_handle(self, view_model: Any) -> bool:
        return isinstance(view_model, NodeViewModel)

    def bind_viewmodel(self, view_model: Any) -> None:
        super().bind_viewmodel(view_model)
        if isinstance(self._current_vm, NodeViewModel) and not self._is_transform_connected:
            self._current_vm.transform_changed.connect(self.update_transform_fields)
            self._is_transform_connected = True

    def unbind(self) -> None:
        if self._is_transform_connected and isinstance(self._current_vm, NodeViewModel):
            self._current_vm.transform_changed.disconnect(self.update_transform_fields)
            self._is_transform_connected = False
        super().unbind()
        self.spin_x.unbind()
        self.spin_y.unbind()
        self.spin_z.unbind()
        self.spin_rot_x.unbind()
        self.spin_rot_y.unbind()
        self.spin_rot_z.unbind()

    def update_ui_from_vm(self) -> None:
        self.update_transform_fields()

    def on_property_changed_externally(self, prop_name: str, new_val: Any) -> None:
        if prop_name == "bindings":
            self.update_transform_fields()

    def update_transform_fields(self) -> None:
        """Синхронизация спинбоксов с матрицей трансформации и кинематическими ограничениями."""
        if not isinstance(self._current_vm, NodeViewModel):
            return

        constraint = self._current_vm.get_effective_kinematic_constraint()
        if constraint is not None:
            allowed_trans = constraint.get_allowed_axes(GizmoMode.TRANSLATE)
            allowed_rot = constraint.get_allowed_axes(GizmoMode.ROTATE)

            self.spin_x.setEnabled(GizmoAxis.X in allowed_trans)
            self.spin_y.setEnabled(GizmoAxis.Y in allowed_trans)
            self.spin_z.setEnabled(GizmoAxis.Z in allowed_trans)

            self.spin_rot_x.setEnabled(GizmoAxis.X in allowed_rot)
            self.spin_rot_y.setEnabled(GizmoAxis.Y in allowed_rot)
            self.spin_rot_z.setEnabled(GizmoAxis.Z in allowed_rot)
        else:
            self.spin_x.setEnabled(True)
            self.spin_y.setEnabled(True)
            self.spin_z.setEnabled(True)

            self.spin_rot_x.setEnabled(True)
            self.spin_rot_y.setEnabled(True)
            self.spin_rot_z.setEnabled(True)

        old_state = self._is_updating_ui
        self._is_updating_ui = True
        try:
            mat = self._current_vm.local_matrix
            if isinstance(mat, np.ndarray) and mat.shape == (4, 4) and np.all(np.isfinite(mat)):
                self.spin_x.setValue(float(mat[0, 3]))
                self.spin_y.setValue(float(mat[1, 3]))
                self.spin_z.setValue(float(mat[2, 3]))

                rot_mat = mat[:3, :3]
                det = float(np.linalg.det(rot_mat))
                if abs(det - 1.0) < 1e-3 and np.allclose(rot_mat @ rot_mat.T, np.eye(3), atol=1e-3):
                    rotation_obj = Rotation.from_matrix(rot_mat)
                    euler = rotation_obj.as_euler("xyz", degrees=True)
                    if np.all(np.isfinite(euler)):
                        self.spin_rot_x.setValue(float(euler[0]))
                        self.spin_rot_y.setValue(float(euler[1]))
                        self.spin_rot_z.setValue(float(euler[2]))

            # Синхронизация биндингов пространственных трансформаций
            bindings = self._current_vm.bindings
            if not bindings and self._scene_vm is not None and self._current_vm.core_node in self._scene_vm.bindings_registry:
                bindings = self._scene_vm.bindings_registry[self._current_vm.core_node]

            bx = bindings.get("translation.x") or bindings.get("x")
            if bx:
                self.spin_x.bind_variable(bx)
            else:
                self.spin_x.unbind()

            by = bindings.get("translation.y") or bindings.get("y") or bindings.get("radius")
            if not by and isinstance(self._current_vm, GammaCameraViewModel) and self._procedure_vm is not None:
                if isinstance(self._procedure_vm, StepAndShootProcedureViewModel):
                    by = "radius"
                elif isinstance(self._procedure_vm, CustomSweepProcedureViewModel):
                    if "radius" in self._procedure_vm.grid_variables or "radius" in self._procedure_vm.zipped_variables:
                        by = "radius"
            if by:
                self.spin_y.bind_variable(by)
            else:
                self.spin_y.unbind()

            bz = bindings.get("translation.z") or bindings.get("z")
            if bz:
                self.spin_z.bind_variable(bz)
            else:
                self.spin_z.unbind()

            b_alpha = bindings.get("rotation.alpha") or bindings.get("alpha") or bindings.get("gantry_angle")
            if not b_alpha and isinstance(self._current_vm, GantryViewModel) and self._procedure_vm is not None:
                if isinstance(self._procedure_vm, StepAndShootProcedureViewModel):
                    b_alpha = "gantry_angle"
                elif isinstance(self._procedure_vm, CustomSweepProcedureViewModel):
                    if "gantry_angle" in self._procedure_vm.grid_variables or "gantry_angle" in self._procedure_vm.zipped_variables:
                        b_alpha = "gantry_angle"
            if b_alpha:
                self.spin_rot_x.bind_variable(b_alpha)
            else:
                self.spin_rot_x.unbind()

            b_beta = bindings.get("rotation.beta") or bindings.get("beta")
            if b_beta:
                self.spin_rot_y.bind_variable(b_beta)
            else:
                self.spin_rot_y.unbind()

            b_gamma = bindings.get("rotation.gamma") or bindings.get("gamma")
            if b_gamma:
                self.spin_rot_z.bind_variable(b_gamma)
            else:
                self.spin_rot_z.unbind()
        finally:
            self._is_updating_ui = old_state

    def _on_transform_changed(self) -> None:
        if self._is_updating_ui or not isinstance(self._current_vm, NodeViewModel):
            return
        mat = np.eye(4, dtype=float)
        rot_angles = [
            float(self.spin_rot_x.value()),
            float(self.spin_rot_y.value()),
            float(self.spin_rot_z.value()),
        ]
        if all(np.isfinite(val) for val in rot_angles):
            rot = Rotation.from_euler("xyz", rot_angles, degrees=True)
            mat[:3, :3] = rot.as_matrix()
        else:
            mat[:3, :3] = self._current_vm.local_matrix[:3, :3]

        mat[0, 3] = self.spin_x.value()
        mat[1, 3] = self.spin_y.value()
        mat[2, 3] = self.spin_z.value()
        self._current_vm.local_matrix = mat
