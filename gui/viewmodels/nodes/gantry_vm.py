"""
Модель представления поворотной станины томографа GantryNode (паттерн MVVM).
Управляет вращением ротора вокруг продольной оси Z в изоцентре (0, 0, 0)
и накладывает кинематические ограничения на себя и дочерние детекторы.
"""

import logging
import math
import weakref
from typing import Any, Optional, Sequence
import numpy as np

from core.scene.gamma_camera_node import GammaCameraNode
from core.scene.gantry_node import GantryNode
from gui.viewmodels.decorators import gui_field
from gui.viewmodels.nodes.base_node_vm import NodeViewModel
from gui.viewport_3d.kinematic_constraints import (
    CameraMountKinematicConstraint,
    GantryKinematicConstraint,
    IKinematicConstraint,
)

_logger = logging.getLogger(__name__)


class GantryViewModel(NodeViewModel):
    """
    ViewModel для станины томографа (ротора).
    """

    wireframe_visible = gui_field(default=True)

    def __init__(self, core_node: GantryNode, parent_vm: Optional[NodeViewModel] = None) -> None:
        super().__init__(core_node, parent_vm)
        self._procedure_ref: Optional[weakref.ref] = None

    @property
    def procedure_vm(self) -> Optional[Any]:
        """Возвращает связанную процедуру исследования или None."""
        return self._procedure_ref() if self._procedure_ref is not None else None

    @procedure_vm.setter
    def procedure_vm(self, procedure: Optional[Any]) -> None:
        """Устанавливает процедуру исследования и обновляет связанные ограничения."""
        self._procedure_ref = weakref.ref(procedure) if procedure is not None else None
        if self._self_kinematic_constraint is not None:
            if isinstance(self._self_kinematic_constraint, GantryKinematicConstraint):
                self._self_kinematic_constraint.procedure_vm = procedure
        if self._default_child_kinematic_constraint is not None:
            if isinstance(self._default_child_kinematic_constraint, CameraMountKinematicConstraint):
                self._default_child_kinematic_constraint.procedure_vm = procedure
        for child_constraint in self._child_kinematic_constraints.values():
            if isinstance(child_constraint, CameraMountKinematicConstraint):
                child_constraint.procedure_vm = procedure

    def get_self_kinematic_constraint(self) -> Optional[IKinematicConstraint]:
        """
        Возвращает кинематическое ограничение ротора станины (1-DOF вращение вокруг оси Z).
        """
        if self._self_kinematic_constraint is None:
            self._self_kinematic_constraint = GantryKinematicConstraint(
                procedure_vm=self.procedure_vm,
                gantry_vm=self,
            )
        return self._self_kinematic_constraint

    def get_child_kinematic_constraint(self, child_vm: NodeViewModel) -> Optional[IKinematicConstraint]:
        """
        Возвращает кинематическое ограничение для дочернего узла на станине.
        Для гамма-камер и детекторного оборудования возвращает CameraMountKinematicConstraint,
        обеспечивающий перемещение по рельсам станины и тангенциальный поворот ротора.
        """
        child_identifier = id(child_vm)
        if child_identifier in self._child_kinematic_constraints:
            return self._child_kinematic_constraints[child_identifier]
        if self._default_child_kinematic_constraint is not None:
            return self._default_child_kinematic_constraint

        # Ограничение каретки станины для детекторов и оборудования
        mount_constraint = CameraMountKinematicConstraint(
            procedure_vm=self.procedure_vm,
            camera_vm=child_vm if isinstance(child_vm.core_node, GammaCameraNode) else None,
            gantry_vm=self,
        )
        self._child_kinematic_constraints[child_identifier] = mount_constraint
        return mount_constraint

    @property
    def gantry_angle_deg(self) -> float:
        """Угол поворота станины вокруг оси Z в градусах."""
        if isinstance(self.core_node, GantryNode):
            return float(math.degrees(self.core_node.gantry_angle) % 360.0)
        return 0.0

    @gantry_angle_deg.setter
    def gantry_angle_deg(self, angle_deg: float) -> None:
        """Установка угла поворота станины вокруг оси Z в градусах."""
        if isinstance(self.core_node, GantryNode):
            angle_value = float(angle_deg)
            self.core_node.set_rotation_angle(math.radians(angle_value))
            self._notify_transform_changed()
            self.property_changed.emit("gantry_angle_deg", self.gantry_angle_deg)
            self.property_changed.emit("local_matrix", self.core_node.local_matrix)

    @NodeViewModel.local_matrix.setter
    def local_matrix(self, matrix: np.ndarray) -> None:
        NodeViewModel.local_matrix.fset(self, matrix)
        self.property_changed.emit("gantry_angle_deg", self.gantry_angle_deg)

    def rotate(
        self,
        alpha: float = 0.0,
        beta: float = 0.0,
        gamma: float = 0.0,
        rotation_center: Sequence[float] = (0.0, 0.0, 0.0),
        in_local: bool = False,
    ) -> None:
        """Вращение узла станины с гарантированным испусканием сигнала угла станины."""
        super().rotate(alpha=alpha, beta=beta, gamma=gamma, rotation_center=rotation_center, in_local=in_local)
        self.property_changed.emit("gantry_angle_deg", self.gantry_angle_deg)

    def set_gantry_angle(self, angle_deg: float) -> None:
        """Метод установки угла поворота станины."""
        self.gantry_angle_deg = angle_deg


__all__ = ["GantryViewModel"]
