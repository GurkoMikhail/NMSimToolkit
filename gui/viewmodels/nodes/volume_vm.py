import logging
from typing import ClassVar, Optional, Sequence, Set
import weakref
import numpy as np

from core.geometry.volumes import Volume
from core.materials.materials import Material
from core.geometry.parametric_collimators import (
    ParametricParallelCollimator,
    ParametricParallelSquareCollimator,
)
from core.other.typing_definitions import Float
import settings.database_setting as database_setting
from gui.viewmodels.decorators import gui_field
from gui.viewmodels.nodes.base_node_vm import NodeViewModel
from gui.viewport_3d.material_palette import get_material_rgba, normalize_material_name

_logger = logging.getLogger(__name__)


class VolumeViewModel(NodeViewModel):
    """
    ViewModel для геометрического объема Volume.
    """
    color = gui_field(default=(0.5, 0.7, 1.0, 0.4))
    _sensitive_core_volumes: ClassVar[weakref.WeakSet[Volume]] = weakref.WeakSet()

    def __init__(self, core_node: Volume, parent_vm: Optional[NodeViewModel] = None) -> None:
        super().__init__(core_node, parent_vm)
        self._is_sensitive_detector: bool = False
        initial_material_name = self.material_name
        self.color = get_material_rgba(initial_material_name)

    @classmethod
    def get_sensitive_volumes(cls) -> Set[Volume]:
        """Возвращает набор узлов Volume расчетного ядра, помеченных в GUI как чувствительные детекторы."""
        return set(cls._sensitive_core_volumes)

    @classmethod
    def clear_sensitive_volumes(cls) -> None:
        """Очищает глобальный реестр чувствительных объемов ядра."""
        cls._sensitive_core_volumes.clear()

    @property
    def local_bound(self) -> np.ndarray:
        """Локальные габариты геометрии объема [Lx, Ly, Lz]."""
        return np.asarray(self.core_node.local_bound, dtype=float)

    @property
    def is_sensitive_detector(self) -> bool:
        return self._is_sensitive_detector

    @is_sensitive_detector.setter
    def is_sensitive_detector(self, is_detector_flag: bool) -> None:
        flag_bool = bool(is_detector_flag)
        if self._is_sensitive_detector == flag_bool:
            return
        self._is_sensitive_detector = flag_bool
        if flag_bool and isinstance(self.core_node, Volume):
            VolumeViewModel._sensitive_core_volumes.add(self.core_node)
        elif isinstance(self.core_node, Volume):
            VolumeViewModel._sensitive_core_volumes.discard(self.core_node)
        self.property_changed.emit('is_sensitive_detector', flag_bool)

    @property
    def material_name(self) -> str:
        current_material = self.core_node.material if isinstance(self.core_node, Volume) else None
        return current_material.name if current_material is not None else "Vacuum"

    @material_name.setter
    def material_name(self, new_material_name: str) -> None:
        canonical_name = normalize_material_name(new_material_name)
        if canonical_name == "Vacuum":
            self.core_node.material = Material(name="Vacuum")
        elif new_material_name in database_setting.material_database:
            self.core_node.material = database_setting.material_database[new_material_name]
        elif canonical_name in database_setting.material_database:
            self.core_node.material = database_setting.material_database[canonical_name]
        else:
            raise KeyError(f"Material '{new_material_name}' is not found in the material database.")
        self.core_node.invalidate_geometry()
        self.color = get_material_rgba(new_material_name)
        self.property_changed.emit('material_name', new_material_name)

    @property
    def size(self) -> np.ndarray:
        return np.asarray(self.core_node.size, dtype=float)

    @size.setter
    def size(self, new_size: Sequence[float]) -> None:
        new_size_arr = np.asarray(new_size, dtype=float)
        self.core_node.size = new_size_arr
        self.property_changed.emit('size', new_size_arr)


class CollimatorViewModel(VolumeViewModel):
    """
    Базовая модель представления для коллиматоров гамма-камер.
    """
    def __init__(self, core_node: Volume, parent_vm: Optional[NodeViewModel] = None) -> None:
        super().__init__(core_node, parent_vm)
        self.color = (0.35, 0.35, 0.35, 0.7)

    @property
    def collimator_type(self) -> str:
        """Человекочитаемый тип геометрии каналов коллиматора."""
        if isinstance(self.core_node, ParametricParallelSquareCollimator):
            return "Квадратный (CZT)"
        elif isinstance(self.core_node, ParametricParallelCollimator):
            return "Гексагональный (LEHR/LEGP)"
        return "Параллельный"

    @property
    def septa_thickness(self) -> float:
        """Толщина септ коллиматора в мм."""
        if isinstance(self.core_node, (ParametricParallelCollimator, ParametricParallelSquareCollimator)):
            return float(self.core_node.septa)
        return 0.2

    @septa_thickness.setter
    def septa_thickness(self, thickness_value: float) -> None:
        numeric_thickness = float(thickness_value)
        if isinstance(self.core_node, (ParametricParallelCollimator, ParametricParallelSquareCollimator)):
            self.core_node.septa = Float(numeric_thickness)
            self.property_changed.emit('septa_thickness', numeric_thickness)


class ParametricParallelCollimatorViewModel(CollimatorViewModel):
    """
    Модель представления для коллиматора с круглыми/гексагональными каналами (LEHR, LEGP).
    """
    def __init__(self, core_node: ParametricParallelCollimator, parent_vm: Optional[NodeViewModel] = None) -> None:
        super().__init__(core_node, parent_vm)

    @property
    def hole_diameter(self) -> float:
        """Диаметр отверстий коллиматора в мм."""
        if isinstance(self.core_node, ParametricParallelCollimator):
            return float(self.core_node.hole_diameter)
        return 1.5

    @hole_diameter.setter
    def hole_diameter(self, diameter_value: float) -> None:
        if isinstance(self.core_node, ParametricParallelCollimator):
            numeric_diameter = float(diameter_value)
            self.core_node.hole_diameter = Float(numeric_diameter)
            self.property_changed.emit('hole_diameter', numeric_diameter)


class ParametricParallelSquareCollimatorViewModel(CollimatorViewModel):
    """
    Модель представления для коллиматора с квадратными каналами (CZT).
    """
    def __init__(self, core_node: ParametricParallelSquareCollimator, parent_vm: Optional[NodeViewModel] = None) -> None:
        super().__init__(core_node, parent_vm)

    @property
    def hole_width(self) -> float:
        """Ширина квадратного отверстия коллиматора в мм."""
        if isinstance(self.core_node, ParametricParallelSquareCollimator):
            return float(self.core_node.hole_width)
        return 1.5

    @hole_width.setter
    def hole_width(self, width_value: float) -> None:
        if isinstance(self.core_node, ParametricParallelSquareCollimator):
            numeric_width = float(width_value)
            self.core_node.hole_width = Float(numeric_width)
            self.property_changed.emit('hole_width', numeric_width)
