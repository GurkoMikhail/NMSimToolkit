"""
Модуль альтернативного параметрического коллиматора с параллельными шестиугольными каналами
на основе детерминированного RayCasting-а (аналитического пересечения с периодической решеткой).
"""

from typing import List, Optional, Sequence, Union

import numpy as np

import settings.database_setting as settings
from core.geometry.geometries import Box, PeriodicHexPrism
from core.geometry.volumes import Volume
from core.materials.materials import Material
from core.other.typing_definitions import Float, Length, Vector3D
from core.scene.nodes import CompositeNode


class DirectParallelCollimator(CompositeNode):
    """
    Альтернативный параметрический коллиматор с параллельными шестиугольными каналами
    на базе детерминированного аналитического RayCasting'а.

    Является семантическим составным узлом (CompositeNode), инкапсулирующим иерархию физических объемов:
    - Свинцовый корпус коллиматора (Volume с геометрией Box)
    - Решетка каналов (дочерний Volume с геометрией PeriodicHexPrism и материалом Vacuum/Air)
    """

    lead_body: Volume
    channels: Volume
    _hole_diameter: Float
    _septa: Float
    _size: Vector3D

    def __init__(
        self,
        size: Union[Sequence[Length], Vector3D],
        hole_diameter: Float,
        septa: Float,
        material: Optional[Material] = None,
        hole_material: Optional[Material] = None,
        name: Optional[str] = None
    ) -> None:
        collimator_name = name if name is not None else "DirectParallelCollimator"
        super().__init__(name=collimator_name)

        self._hole_diameter = Float(hole_diameter)
        self._septa = Float(septa)
        self._size = np.array(size, dtype=Float)

        if any(dimension <= 0.0 for dimension in self._size):
            raise ValueError(f"Габаритные размеры коллиматора должны быть строго положительными (> 0), получено {size}")
        if self._hole_diameter <= 0.0:
            raise ValueError(f"Диаметр отверстия должен быть строго положительным (> 0), получено {hole_diameter}")
        if self._septa <= 0.0:
            raise ValueError(f"Толщина септы должна быть строго положительной (> 0), получено {septa}")

        lead_material = settings.material_database['Pb'] if material is None else material
        vacuum_material = settings.material_database['Vacuum'] if hole_material is None else hole_material

        self.lead_body = Volume(
            geometry=Box(*self._size),
            material=lead_material,
            name=f"{collimator_name}_body"
        )

        self.channels = Volume(
            geometry=PeriodicHexPrism(
                size=self._size,
                hole_diameter=self._hole_diameter,
                septa=self._septa
            ),
            material=vacuum_material,
            name=f"{collimator_name}_channels"
        )

        self.lead_body.add_child(self.channels)
        self.add_child(self.lead_body)

    @property
    def hole_diameter(self) -> Float:
        """Диаметр гексагонального канала между параллельными гранями."""
        return self._hole_diameter

    @hole_diameter.setter
    def hole_diameter(self, value: Float) -> None:
        float_value = Float(value)
        if float_value <= 0.0:
            raise ValueError(f"Диаметр отверстия должен быть строго положительным (> 0), получено {value}")
        self._hole_diameter = float_value
        prism_geometry: PeriodicHexPrism = self.channels.geometry  # type: ignore
        prism_geometry.hole_diameter = self._hole_diameter
        self.invalidate_geometry()

    @property
    def septa(self) -> Float:
        """Толщина перегородки (септы) между каналами."""
        return self._septa

    @septa.setter
    def septa(self, value: Float) -> None:
        float_value = Float(value)
        if float_value <= 0.0:
            raise ValueError(f"Толщина септы должна быть строго положительной (> 0), получено {value}")
        self._septa = float_value
        prism_geometry: PeriodicHexPrism = self.channels.geometry  # type: ignore
        prism_geometry.septa = self._septa
        self.invalidate_geometry()

    @property
    def size(self) -> Vector3D:
        """Габаритные размеры коллиматора [Lx, Ly, Lz]."""
        return self._size

    @size.setter
    def size(self, value: Union[Sequence[Length], Vector3D]) -> None:
        parsed_size = np.array(value, dtype=Float)
        if any(dimension <= 0.0 for dimension in parsed_size):
            raise ValueError(f"Габаритные размеры коллиматора должны быть строго положительными (> 0), получено {value}")
        self._size = parsed_size
        self.lead_body.size = self._size
        self.channels.size = self._size
        self.invalidate_geometry()

    @property
    def material(self) -> Material:
        """Основной материал корпуса коллиматора (Pb)."""
        return self.lead_body.material

    @material.setter
    def material(self, value: Material) -> None:
        self.lead_body.material = value

    @property
    def hole_material(self) -> Material:
        """Материал внутри каналов (Vacuum или Air)."""
        return self.channels.material

    @hole_material.setter
    def hole_material(self, value: Material) -> None:
        self.channels.material = value

    @property
    def material_list(self) -> List[Material]:
        """Список материалов, используемых в коллиматоре."""
        return [self.lead_body.material, self.channels.material]

    def invalidate_geometry(self) -> None:
        """Сбрасывает кэш геометрии дочерних физических объемов."""
        self.lead_body.invalidate_geometry()
        self.channels.invalidate_geometry()


__all__ = ["DirectParallelCollimator"]
