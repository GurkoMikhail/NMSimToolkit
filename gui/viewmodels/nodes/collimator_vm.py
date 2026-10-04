"""
Единая универсальная модель представления (ViewModel) для всех типов коллиматоров:
детерминированных (DirectParallelCollimator) и параметрических (ParametricParallelCollimator).
"""

import logging
from typing import Optional, Sequence, Union
import numpy as np

from core.geometry.direct_collimators import CollimatorHoleShape, DirectParallelCollimator
from core.geometry.parametric_collimators import (
    ParametricParallelCollimator,
)
from core.materials.materials import Material
from core.other.typing_definitions import Float
import settings.database_setting as database_setting
from gui.viewmodels.decorators import gui_field
from gui.viewmodels.nodes.base_node_vm import NodeViewModel
from gui.viewport_3d.kinematic_constraints import FixedSubcomponentKinematicConstraint

_logger = logging.getLogger(__name__)

CollimatorCoreNode = Union[
    DirectParallelCollimator,
    ParametricParallelCollimator,
]


class CollimatorViewModel(NodeViewModel):
    """
    Единая универсальная модель представления для коллиматоров гамма-камер и томографов.
    Обеспечивает унифицированный интерфейс геометрических параметров, формы каналов,
    материалов и реактивных уведомлений независимо от физической реализации ядра
    (детерминированный CompositeNode или параметрический Volume).
    """

    color = gui_field(default=(0.35, 0.35, 0.35, 0.7))

    def __init__(
        self,
        core_node: CollimatorCoreNode,
        parent_vm: Optional[NodeViewModel] = None,
    ) -> None:
        super().__init__(core_node, parent_vm)
        if isinstance(self.core_node, DirectParallelCollimator):
            self._apply_fixed_constraints_to_subcomponents()

    def _apply_fixed_constraints_to_subcomponents(self) -> None:
        """
        Назначает кинематическое ограничение FixedSubcomponentKinematicConstraint
        дочерним объемам корпуса и каналов (lead_body, channels), предотвращая их взаимное смещение.
        """
        fixed_constraint = FixedSubcomponentKinematicConstraint()
        self.set_child_kinematic_constraint(fixed_constraint)
        search_stack = list(self.children)
        while search_stack:
            current_vm = search_stack.pop()
            current_vm.set_self_kinematic_constraint(fixed_constraint)
            current_vm.set_child_kinematic_constraint(fixed_constraint)
            search_stack.extend(current_vm.children)

    def sync_children_from_core(self) -> None:
        """Синхронизирует дочерние узлы с расчетным ядром и накладывает кинематические ограничения."""
        super().sync_children_from_core()
        if isinstance(self.core_node, DirectParallelCollimator):
            self._apply_fixed_constraints_to_subcomponents()

    def _notify_children_geometry_changed(self) -> None:
        """
        Уведомляет дочерние узлы (lead_body и channels) об обновлении геометрических параметров.
        """
        for child_vm in self.children:
            child_vm.property_changed.emit('size', child_vm.size)
            if isinstance(self.core_node, DirectParallelCollimator):
                if child_vm.core_node is self.core_node.channels:
                    child_vm.property_changed.emit('hole_diameter', self.hole_diameter)
                    child_vm.property_changed.emit('hole_width', self.hole_width)
                    child_vm.property_changed.emit('septa', self.septa)
                    child_vm.property_changed.emit('hole_shape', self.hole_shape)
            child_vm._notify_transform_changed()

    @property
    def collimator_kind(self) -> str:
        """
        Строковый идентификатор типа коллиматора:
        'direct' (детерминированный) или 'parametric' (параметрический RayCasting).
        """
        if isinstance(self.core_node, DirectParallelCollimator):
            return "direct"
        return "parametric"

    @property
    def collimator_type(self) -> str:
        """Человекочитаемое наименование типа коллиматора для интерфейса."""
        if isinstance(self.core_node, DirectParallelCollimator):
            return "Детерминированный (Сквозные каналы)"
        return "Параметрический (RayCasting)"

    @property
    def size(self) -> np.ndarray:
        """Габаритные размеры коллиматора [Lx, Ly, Lz] в мм."""
        return np.asarray(self.core_node.size, dtype=float)

    @size.setter
    def size(self, new_size: Sequence[float]) -> None:
        new_size_array = np.asarray(new_size, dtype=float)
        self.core_node.size = new_size_array
        if isinstance(self.core_node, DirectParallelCollimator):
            self._notify_children_geometry_changed()
        else:
            self.core_node.invalidate_geometry()
        self.property_changed.emit('size', new_size_array)

    @property
    def local_bound(self) -> np.ndarray:
        """Локальные габариты геометрии объема [Lx, Ly, Lz]."""
        return np.asarray(self.core_node.size, dtype=float)

    @property
    def hole_diameter(self) -> float:
        """Диаметр отверстий канала в мм."""
        return float(self.core_node.hole_diameter)

    @hole_diameter.setter
    def hole_diameter(self, diameter_value: float) -> None:
        numeric_diameter = float(diameter_value)
        self.core_node.hole_diameter = Float(numeric_diameter)
        if isinstance(self.core_node, DirectParallelCollimator):
            self._notify_children_geometry_changed()
        else:
            self.core_node.invalidate_geometry()
        self.property_changed.emit('hole_diameter', numeric_diameter)
        self.property_changed.emit('hole_width', numeric_diameter)

    @property
    def hole_width(self) -> float:
        """Ширина отверстий канала в мм."""
        return float(self.core_node.hole_width)

    @hole_width.setter
    def hole_width(self, width_value: float) -> None:
        numeric_width = float(width_value)
        self.core_node.hole_width = Float(numeric_width)
        if isinstance(self.core_node, DirectParallelCollimator):
            self._notify_children_geometry_changed()
        else:
            self.core_node.invalidate_geometry()
        self.property_changed.emit('hole_width', numeric_width)
        self.property_changed.emit('hole_diameter', numeric_width)

    @property
    def septa(self) -> float:
        """Толщина септ (перегородок между каналами) в мм."""
        return float(self.core_node.septa)

    @septa.setter
    def septa(self, septa_value: float) -> None:
        numeric_septa = float(septa_value)
        self.core_node.septa = Float(numeric_septa)
        if isinstance(self.core_node, DirectParallelCollimator):
            self._notify_children_geometry_changed()
        else:
            self.core_node.invalidate_geometry()
        self.property_changed.emit('septa', numeric_septa)

    @property
    def hole_shape(self) -> CollimatorHoleShape:
        """Форма поперечного сечения каналов коллиматора."""
        return self.core_node.hole_shape

    @hole_shape.setter
    def hole_shape(self, shape_value: Union[CollimatorHoleShape, str]) -> None:
        self.core_node.hole_shape = shape_value
        if isinstance(self.core_node, DirectParallelCollimator):
            self._notify_children_geometry_changed()
        else:
            self.core_node.invalidate_geometry()
        self.property_changed.emit('hole_shape', self.core_node.hole_shape)

    @property
    def material_name(self) -> str:
        """Наименование материала корпуса коллиматора (Pb)."""
        material_obj = self.core_node.material
        return material_obj.name if material_obj is not None else "Pb"

    @material_name.setter
    def material_name(self, new_material_name: str) -> None:
        if new_material_name == "Vacuum":
            material_instance = Material(name="Vacuum")
        elif new_material_name in database_setting.material_database:
            material_instance = database_setting.material_database[new_material_name]
        else:
            raise KeyError(f"Материал '{new_material_name}' не найден в базе данных материалов.")

        self.core_node.material = material_instance
        if isinstance(self.core_node, DirectParallelCollimator):
            for child_vm in self.children:
                if child_vm.core_node is self.core_node.lead_body:
                    child_vm.property_changed.emit('material_name', new_material_name)
        else:
            self.core_node.invalidate_geometry()
        self.property_changed.emit('material_name', new_material_name)

    @property
    def hole_material_name(self) -> Optional[str]:
        """
        Наименование материала внутри каналов.
        Возвращает None, если материал наследуется от родительского объема.
        """
        if isinstance(self.core_node, DirectParallelCollimator):
            if self.core_node.explicit_hole_material is None:
                return None
            return self.core_node.explicit_hole_material.name
        return None

    @hole_material_name.setter
    def hole_material_name(self, new_material_name: Optional[str]) -> None:
        if isinstance(self.core_node, DirectParallelCollimator):
            if new_material_name is None:
                self.core_node.hole_material = None
            elif new_material_name == "Vacuum":
                self.core_node.hole_material = Material(name="Vacuum")
            elif new_material_name in database_setting.material_database:
                self.core_node.hole_material = database_setting.material_database[new_material_name]
            else:
                raise KeyError(f"Материал '{new_material_name}' не найден в базе данных материалов.")

            for child_vm in self.children:
                if child_vm.core_node is self.core_node.channels:
                    child_vm.property_changed.emit('material_name', self.core_node.channels.material.name)
            self.property_changed.emit('hole_material_name', new_material_name)


__all__ = [
    "CollimatorViewModel",
]
