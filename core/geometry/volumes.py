"""
Модуль базовых физических объемов (Volume) и индексированных массивов объемов (VolumeArray).
"""
from __future__ import annotations

from itertools import chain, count
from typing import Any, Callable, Iterator, List, Optional, Sequence
import numpy as np
from numpy.typing import NDArray

from core.geometry.geometries import Geometry
import core.geometry.geometry_compiler as geom_compiler
from core.geometry.geometry_compiler import GeometryBufferDType
from core.materials.materials import Material, MaterialArray
from core.other.nonunique_array import NonuniqueArray
from core.other.typing_definitions import Float, Index, Vector3D
from core.scene.nodes import CompositeNode, SpatialNode


class Volume(CompositeNode):
    """
    Базовый класс элементарного физического объема с геометрией и материалом,
    наследующий CompositeNode для построения иерархического графа сцены.
    """

    _counter = count(1)

    geometry: Geometry
    material: Material
    name: str

    def __init__(
        self,
        geometry: Geometry,
        material: Material,
        name: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(name=name, tags=tags)
        self.geometry = geometry
        self.material = material
        self.name = f'{self.__class__.__name__}{next(self._counter)}' if name is None else name
        self._geometry_buffer: Optional[NDArray[Any]] = None

    def __init_subclass__(cls) -> None:
        cls._counter = count(1)

    def __repr__(self) -> str:
        return f'{self.name}'

    @property
    def material_cfunc(self) -> Optional[Callable[[Float, Float, Float], Index]]:
        """Указатель CFUNCTYPE на @cfunc для параметрических объемов Вудкока (None для стандартных объемов)."""
        return None

    @property
    def majorant_material(self) -> Material:
        """Возвращает мажорантный материал объема (по умолчанию self.material)."""
        return self.material

    @property
    def material_list(self) -> List[Material]:
        """
        Возвращает список всех уникальных материалов, используемых в данном объеме и всем его поддереве,
        включая объемы внутри промежуточных CompositeNode и дочерние параметрические объемы Вудкока.
        """
        def collect_from_children(parent_node: CompositeNode) -> Iterator[Material]:
            for child in parent_node.childs:
                if isinstance(child, Volume):
                    yield from child.material_list
                elif isinstance(child, CompositeNode):
                    yield from collect_from_children(child)

        return list(dict.fromkeys(chain((self.material,), collect_from_children(self))))

    @property
    def size(self) -> Vector3D:
        """Габаритные размеры объема [Lx, Ly, Lz] в мм."""
        return self.geometry.size

    @size.setter
    def size(self, value: Vector3D) -> None:
        self.geometry.size = value
        self.invalidate_geometry()

    @property
    def local_bound(self) -> Vector3D:
        """Локальные габариты объема (размеры геометрии [Lx, Ly, Lz])."""
        return self.geometry.size

    @property
    def geometry_buffer(self) -> NDArray[Any]:
        """Ленивая компиляция буфера геометрии GeometryBuffer (структурированный массив AoS)."""
        if self._geometry_buffer is None:
            self._geometry_buffer = geom_compiler.GeometryCompiler().compile_scene(self)
        return self._geometry_buffer

    def invalidate_geometry(self) -> None:
        """Сбрасывает кэш скомпилированной геометрии у этого объёма и всех родительских Volume."""
        self._geometry_buffer = None
        curr: Optional[CompositeNode] = self.parent
        while curr is not None:
            if isinstance(curr, Volume):
                curr._geometry_buffer = None
            curr = curr.parent

    def invalidate_matrix_cache(self) -> None:
        """Инвалидирует кэш матриц и буфер геометрии у текущего узла и его предков/потомков."""
        super().invalidate_matrix_cache()
        self.invalidate_geometry()

    def add_child(self, child: SpatialNode) -> None:
        """Добавление дочернего узла с инвалидацией буферов геометрии."""
        super().add_child(child)
        self.invalidate_geometry()

    def remove_child(self, child: SpatialNode) -> None:
        """Удаление дочернего узла с инвалидацией буферов геометрии."""
        super().remove_child(child)
        self.invalidate_geometry()

    @property
    def top_volume(self) -> Volume:
        """Возвращает наивысший узел Volume в текущей ветви иерархии сцены."""
        current: SpatialNode = self
        top_vol: Volume = self
        while current.parent is not None:
            current = current.parent
            if isinstance(current, Volume):
                top_vol = current
        return top_vol

    def set_parent(self, parent: CompositeNode) -> None:
        """Устанавливает родительский составной узел."""
        parent.add_child(self)


class VolumeArray(NonuniqueArray):
    """Индексированный массив физических объемов."""

    element_list: List[Optional[Volume]]

    @property
    def material(self) -> MaterialArray:
        """Массив материалов, соответствующих объемам в ячейках."""
        material = MaterialArray(self.shape)
        for volume, indices in self.inverse_indices.items():
            if volume is None:
                continue
            material[indices] = volume.material
        return material


__all__ = [
    "GeometryBufferDType",
    "Volume",
    "VolumeArray",
]

