"""
Модель представления физической детекторной головки гамма-камеры (паттерн MVVM).
Управляет открытой декларативной сборкой компонентов со слотами (Slots / Sockets):
casing, detector_box, collimator, crystal, glass_backend.
Динамически рассчитывает габариты и смещения слоев и фиксирует внутренние
компоненты через FixedSubcomponentKinematicConstraint.
"""

import logging
from typing import Dict, Optional, Sequence, Tuple, Union
import numpy as np

from core.geometry.volumes import Volume
from core.materials.materials import Material
from core.other.typing_definitions import Float
from core.scene.gamma_camera_node import GammaCameraNode
from core.config.models import GammaCameraSlotsConfig
from gui.factories.gamma_camera_factory import create_default_gamma_camera
from gui.viewmodels.nodes.base_node_vm import NodeViewModel
from gui.viewmodels.nodes.volume_vm import VolumeViewModel
from gui.viewmodels.nodes.collimator_vm import CollimatorViewModel
from gui.viewport_3d.kinematic_constraints import FixedSubcomponentKinematicConstraint

_logger = logging.getLogger(__name__)


class GammaCameraViewModel(NodeViewModel):
    """
    ViewModel для физической детекторной головки гамма-камеры с архитектурой слотов.
    """

    def __init__(
        self,
        core_node: GammaCameraNode,
        parent_vm: Optional[NodeViewModel] = None,
        slots: Optional[Union[GammaCameraSlotsConfig, Dict[str, str]]] = None,
    ) -> None:
        super().__init__(core_node, parent_vm)
        if slots is not None:
            if isinstance(slots, GammaCameraSlotsConfig):
                self.slots: GammaCameraSlotsConfig = slots
            else:
                self.slots = GammaCameraSlotsConfig(**slots)
            for slot_key, slot_val in self.slots.model_dump().items():
                if slot_val is not None:
                    self.core_node.slots[slot_key] = slot_val
        else:
            self.slots = GammaCameraSlotsConfig(
                casing=core_node.slots.get("casing"),
                detector_box=core_node.slots.get("detector_box"),
                collimator=core_node.slots.get("collimator"),
                crystal=core_node.slots.get("crystal"),
                glass_backend=core_node.slots.get("glass_backend"),
            )

        box = self.detector_box_vm
        col = self.collimator_vm
        det = self.crystal_vm
        glass = self.glass_backend_vm
        if box is not None and col is not None and det is not None and glass is not None:
            self._stored_gap = max(0.0, float(box.size[2] - col.size[2] - det.size[2] - glass.size[2]))
        else:
            self._stored_gap = 1.0

        casing = self.casing_vm
        if casing is not None and box is not None:
            diff_x = casing.size[0] - box.size[0]
            self._stored_shielding_thickness = float(diff_x / 2.0) if diff_x > 0 else 20.0
        else:
            self._stored_shielding_thickness = 20.0

        self._apply_fixed_constraints_to_subcomponents()

    def _find_volume_by_name(self, target_name: Optional[str]) -> Optional[VolumeViewModel]:
        """
        Ищет дочерний узел типа VolumeViewModel по точному имени в иерархии потомков.
        """
        if not target_name:
            return None
        search_stack = list(self.children)
        while search_stack:
            current_vm = search_stack.pop()
            if current_vm.name == target_name and isinstance(current_vm, VolumeViewModel):
                return current_vm
            search_stack.extend(current_vm.children)
        return None

    def _find_collimator_by_name(
        self,
        target_name: Optional[str]
    ) -> Optional[CollimatorViewModel]:
        """
        Ищет узел коллиматора (CollimatorViewModel) по имени в иерархии потомков.
        """
        if not target_name:
            return None
        search_stack = list(self.children)
        while search_stack:
            current_vm = search_stack.pop()
            if current_vm.name == target_name and isinstance(current_vm, CollimatorViewModel):
                return current_vm
            search_stack.extend(current_vm.children)
        return None

    # -------------------------------------------------------------------------
    # Типизированный доступ к слотам (Sockets)
    # -------------------------------------------------------------------------
    @property
    def crystal_vm(self) -> Optional[VolumeViewModel]:
        """ViewModel сцинтилляционного кристалла (слот crystal)."""
        return self._find_volume_by_name(self.slots.crystal)

    @property
    def detector_vm(self) -> Optional[VolumeViewModel]:
        """Алиас для доступа к сцинтилляционному кристаллу (слот crystal)."""
        return self.crystal_vm

    @property
    def collimator_vm(self) -> Optional[CollimatorViewModel]:
        """ViewModel коллиматора (слот collimator)."""
        return self._find_collimator_by_name(self.slots.collimator)

    @property
    def casing_vm(self) -> Optional[VolumeViewModel]:
        """ViewModel защитного корпуса (слот casing)."""
        return self._find_volume_by_name(self.slots.casing)

    @property
    def detector_box_vm(self) -> Optional[VolumeViewModel]:
        """ViewModel внутренней воздушной полости детектора (слот detector_box)."""
        return self._find_volume_by_name(self.slots.detector_box)

    @property
    def glass_backend_vm(self) -> Optional[VolumeViewModel]:
        """ViewModel оптической подложки (слот glass_backend)."""
        return self._find_volume_by_name(self.slots.glass_backend)

    # -------------------------------------------------------------------------
    # Геометрические параметры и их динамический пересчет
    # -------------------------------------------------------------------------
    @property
    def size(self) -> np.ndarray:
        """Габариты внешнего корпуса гамма-камеры [Lx, Ly, Lz] (мм)."""
        casing = self.casing_vm
        if casing is not None:
            return np.asarray(casing.size, dtype=float)
        return np.array([440.0, 440.0, 110.5], dtype=float)

    @size.setter
    def size(self, new_size: Sequence[float]) -> None:
        casing = self.casing_vm
        if casing is not None:
            casing.size = new_size
            self.property_changed.emit('size', self.size)
            self.property_changed.emit('housing_size', self.housing_size)
            self._notify_transform_changed()

    @property
    def local_bound(self) -> np.ndarray:
        """Локальные габариты камеры."""
        return self.size

    @property
    def housing_size(self) -> Tuple[float, float, float]:
        """Габариты внешнего корпуса гамма-камеры [X, Y, Z] (мм)."""
        camera_size = self.size
        return (float(camera_size[0]), float(camera_size[1]), float(camera_size[2]))

    @property
    def half_thickness(self) -> float:
        """
        Половина толщины гамма-камеры вдоль оси Z (мм).
        Лицевая поверхность коллиматора/камеры находится на расстоянии half_thickness
        от геометрического центра камеры в направлении нормали (+Z, к центру орбиты).
        """
        camera_size = self.size
        return float(camera_size[2]) / 2.0 if len(camera_size) >= 3 and camera_size[2] > 0 else 0.0

    @property
    def detector_size(self) -> Tuple[float, float]:
        """Размер активной чувствительной области детектора [Lx, Ly] (мм)."""
        crystal = self.crystal_vm
        if crystal is not None:
            return (float(crystal.size[0]), float(crystal.size[1]))
        return (400.0, 400.0)

    @detector_size.setter
    def detector_size(self, size_xy: Sequence[float]) -> None:
        if len(size_xy) != 2:
            raise ValueError(f"detector_size должен содержать ровно 2 элемента [Lx, Ly], получено {len(size_xy)}")
        size_x = float(size_xy[0])
        size_y = float(size_xy[1])
        if size_x <= 0 or size_y <= 0:
            raise ValueError(f"Элементы detector_size должны быть положительными: ({size_x}, {size_y})")

        crystal = self.crystal_vm
        if crystal is not None:
            crystal.size = [size_x, size_y, crystal.size[2]]

        collimator = self.collimator_vm
        if collimator is not None:
            collimator.size = [size_x, size_y, collimator.size[2]]

        self._rebuild_geometry()

    @property
    def detector_thickness(self) -> float:
        """Толщина кристалла детектора по оси Z (мм)."""
        crystal = self.crystal_vm
        if crystal is not None:
            return float(crystal.size[2])
        return 10.0

    @detector_thickness.setter
    def detector_thickness(self, thickness_val: float) -> None:
        numeric_val = float(thickness_val)
        if numeric_val <= 0:
            raise ValueError(f"detector_thickness должен быть строго больше 0, получено {numeric_val}")
        crystal = self.crystal_vm
        if crystal is not None:
            crystal.size = [crystal.size[0], crystal.size[1], numeric_val]
            self._rebuild_geometry()

    @property
    def collimator_thickness(self) -> float:
        """Толщина коллиматора по оси Z (мм)."""
        collimator = self.collimator_vm
        if collimator is not None:
            return float(collimator.size[2])
        return 30.0

    @collimator_thickness.setter
    def collimator_thickness(self, thickness_val: float) -> None:
        numeric_val = float(thickness_val)
        if numeric_val <= 0:
            raise ValueError(f"collimator_thickness должен быть строго больше 0, получено {numeric_val}")
        collimator = self.collimator_vm
        if collimator is not None:
            collimator.size = [collimator.size[0], collimator.size[1], numeric_val]
            self._rebuild_geometry()

    @property
    def gap(self) -> float:
        """Внутренний зазор между коллиматором и детектором (мм)."""
        return self._stored_gap

    @gap.setter
    def gap(self, gap_value: float) -> None:
        numeric_gap = float(gap_value)
        if numeric_gap < 0:
            raise ValueError(f"gap не может быть отрицательным, получено {numeric_gap}")
        self._stored_gap = numeric_gap
        self._rebuild_geometry(gap=numeric_gap)

    @property
    def shielding_thickness(self) -> float:
        """Толщина свинцовой защиты корпуса гамма-камеры (мм)."""
        return self._stored_shielding_thickness

    @shielding_thickness.setter
    def shielding_thickness(self, thickness_value: float) -> None:
        numeric_thickness = float(thickness_value)
        if numeric_thickness <= 0:
            raise ValueError(f"shielding_thickness должен быть положительным, получено {numeric_thickness}")
        self._stored_shielding_thickness = numeric_thickness
        self._rebuild_geometry(shielding_thickness=numeric_thickness)

    @property
    def glass_backend_thickness(self) -> float:
        """Толщина подложки оптического стекла (мм)."""
        glass = self.glass_backend_vm
        if glass is not None:
            return float(glass.size[2])
        return 50.0

    @glass_backend_thickness.setter
    def glass_backend_thickness(self, thickness_value: float) -> None:
        numeric_thickness = float(thickness_value)
        if numeric_thickness <= 0:
            raise ValueError(f"glass_backend_thickness должен быть положительным, получено {numeric_thickness}")
        glass = self.glass_backend_vm
        if glass is not None:
            glass.size = [glass.size[0], glass.size[1], numeric_thickness]
            self._rebuild_geometry()

    def _rebuild_geometry(
        self,
        gap: Optional[float] = None,
        shielding_thickness: Optional[float] = None
    ) -> None:
        """
        Пересчитывает геометрические размеры корпуса и взаимное расположение компонентов сборки.
        """
        crystal = self.crystal_vm
        collimator = self.collimator_vm
        casing = self.casing_vm
        detector_box = self.detector_box_vm
        glass_backend = self.glass_backend_vm

        if not (crystal and collimator and casing and detector_box and glass_backend):
            return

        gap_value = gap if gap is not None else self.gap
        shielding_value = shielding_thickness if shielding_thickness is not None else self.shielding_thickness

        det_size_x = max(collimator.size[0], crystal.size[0])
        det_size_y = max(collimator.size[1], crystal.size[1])
        det_box_z = collimator.size[2] + gap_value + crystal.size[2] + glass_backend.size[2]

        detector_box.size = [det_size_x, det_size_y, det_box_z]

        casing_x = det_size_x + 2.0 * shielding_value
        casing_y = det_size_y + 2.0 * shielding_value
        casing_z = det_box_z + shielding_value
        casing.size = [casing_x, casing_y, casing_z]

        detector_box.local_matrix = np.eye(4, dtype=Float)
        detector_box.translate(z=shielding_value / 2.0)

        collimator.local_matrix = np.eye(4, dtype=Float)
        collimator.translate(z=(det_box_z / 2.0 - collimator.size[2] / 2.0))

        crystal.local_matrix = np.eye(4, dtype=Float)
        crystal.translate(z=(det_box_z / 2.0 - collimator.size[2] - crystal.size[2] / 2.0 - gap_value))

        glass_size = [det_size_x, det_size_y, glass_backend.size[2]]
        glass_backend.size = glass_size
        glass_backend.local_matrix = np.eye(4, dtype=Float)
        glass_backend.translate(z=(glass_backend.size[2] / 2.0 - det_box_z / 2.0))

        casing.core_node.invalidate_geometry()

        self.property_changed.emit('detector_size', self.detector_size)
        self.property_changed.emit('detector_thickness', self.detector_thickness)
        self.property_changed.emit('collimator_thickness', self.collimator_thickness)
        self.property_changed.emit('gap', self.gap)
        self.property_changed.emit('shielding_thickness', self.shielding_thickness)
        self.property_changed.emit('glass_backend_thickness', self.glass_backend_thickness)
        self.property_changed.emit('size', self.size)
        self.property_changed.emit('housing_size', self.housing_size)
        self.property_changed.emit('half_thickness', self.half_thickness)
        self._notify_transform_changed()

    def _apply_fixed_constraints_to_subcomponents(self) -> None:
        """
        Назначает кинематическое ограничение FixedSubcomponentKinematicConstraint
        всем внутренним компонентам гамма-камеры (коллиматор, сцинтиллятор, оптическое стекло).
        """
        fixed_constraint = FixedSubcomponentKinematicConstraint()
        self.set_child_kinematic_constraint(fixed_constraint)
        stack = list(self.children)
        while stack:
            current_vm = stack.pop()
            current_vm.set_self_kinematic_constraint(fixed_constraint)
            current_vm.set_child_kinematic_constraint(fixed_constraint)
            stack.extend(current_vm.children)

    def sync_children_from_core(self) -> None:
        super().sync_children_from_core()
        self._apply_fixed_constraints_to_subcomponents()
        crystal = self.crystal_vm
        if crystal is not None:
            crystal.is_sensitive_detector = True


def create_default_gamma_camera_vm(
    name: Optional[str] = None,
    detector_size: Sequence[float] = (400.0, 400.0),
    detector_thickness: float = 10.0,
    collimator_thickness: float = 30.0,
    gap: float = 1.0,
    shielding_thickness: float = 20.0,
    glass_backend_thickness: float = 50.0,
    collimator_material_name: str = "Pb",
    detector_material_name: str = "Sodium Iodide",
    casing_material_name: str = "Pb",
    internal_material_name: str = "Air, Dry (near sea level)",
    glass_material_name: str = "Glass, Borosilicate (Pyrex)",
    collimator: Optional[Volume] = None,
    detector: Optional[Volume] = None,
    shielding_material: Optional[Material] = None,
    internal_medium: Optional[Material] = None,
    glass_material: Optional[Material] = None,
) -> GammaCameraViewModel:
    """
    Удобная вспомогательная функция для создания полностью инициализированной
    GammaCameraViewModel со стандартной 5-узловой геометрией и слотами.
    """
    camera_node, slots_config = create_default_gamma_camera(
        name=name,
        detector_size=detector_size,
        detector_thickness=detector_thickness,
        collimator_thickness=collimator_thickness,
        gap=gap,
        shielding_thickness=shielding_thickness,
        glass_backend_thickness=glass_backend_thickness,
        collimator_material_name=collimator_material_name,
        detector_material_name=detector_material_name,
        casing_material_name=casing_material_name,
        internal_material_name=internal_material_name,
        glass_material_name=glass_material_name,
        collimator=collimator,
        detector=detector,
        shielding_material=shielding_material,
        internal_medium=internal_medium,
        glass_material=glass_material,
    )
    return GammaCameraViewModel(core_node=camera_node, slots=slots_config)


__all__ = [
    "GammaCameraViewModel",
    "create_default_gamma_camera_vm",
]
