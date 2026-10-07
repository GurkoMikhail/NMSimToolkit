from itertools import count
from typing import Any, List, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from core.scene.nodes import CompositeNode
from core.geometry.geometries import Geometry
from core.materials.materials import Material, MaterialArray
from core.other.nonunique_array import NonuniqueArray
from core.other.typing_definitions import Float, Vector3D, Index, CMaterialFunc
from core.other.transform import TransformDType
from core.geometry.geometries import ShapeDataDType
GeometryBufferDType = np.dtype([
    ('shape_data', ShapeDataDType),
    ('transform', TransformDType),
    ('miss_index', Index),
    ('parent_index', Index),
    ('volume_index', Index)
])

import core.geometry.geometry_compiler as geom_compiler


class Volume(CompositeNode):
    """
    Базовый класс элементарного физического объема с геометрией и материалом,
    наследующий CompositeNode для построения иерархического графа сцены.
    """

    _counter = count(1)

    geometry: Geometry
    material: Material
    name: str

    def __init__(self, geometry: Geometry, material: Material, name: Optional[str] = None, tags: Optional[Sequence[str]] = None) -> None:
        super().__init__(name=name, tags=tags)
        self.geometry = geometry
        self.material = material
        self.name = f'{self.__class__.__name__}{next(self._counter)}' if name is None else name
        self._geometry_buffer: Optional[NDArray[Any]] = None

    def __init_subclass__(cls):
        cls._counter = count(1)

    def __repr__(self):
        return f'{self.name}'

    @property
    def material_cfunc(self) -> CMaterialFunc:
        """Указатель CFUNCTYPE на @cfunc для параметрических объемов Вудкока (None для стандартных объемов)."""
        return None

    @property
    def majorant_material(self) -> Material:
        """Возвращает мажорантный материал объема (по умолчанию self.material)."""
        return self.material

    @property
    def material_list(self) -> List[Material]:
        """Возвращает список всех уникальных материалов, используемых в данном объеме и его потомках."""
        def recursive_generator(node):
            if isinstance(node, Volume):
                yield node.material
            for child in node.childs:
                if isinstance(child, Volume):
                    yield from recursive_generator(child)

        return list(dict.fromkeys(recursive_generator(self)))

    @property
    def size(self) -> Vector3D:
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
        curr = self.parent
        while curr is not None:
            if isinstance(curr, Volume):
                curr._geometry_buffer = None
            curr = curr.parent

    def invalidate_matrix_cache(self) -> None:
        """Инвалидирует кэш матриц и буфер геометрии у текущего узла и его предков/потомков."""
        super().invalidate_matrix_cache()
        self.invalidate_geometry()

    def add_child(self, child: 'SpatialNode') -> None:
        """Добавление дочернего узла с инвалидацией буферов геометрии."""
        super().add_child(child)
        self.invalidate_geometry()

    def remove_child(self, child: 'SpatialNode') -> None:
        """Удаление дочернего узла с инвалидацией буферов геометрии."""
        super().remove_child(child)
        self.invalidate_geometry()

    @property
    def top_volume(self) -> 'Volume':
        """Возвращает наивысший узел Volume в текущей ветви иерархии сцены."""
        current = self
        top_vol = self
        while current.parent is not None:
            current = current.parent
            if isinstance(current, Volume):
                top_vol = current
        return top_vol

    def set_parent(self, parent: 'CompositeNode') -> None:
        parent.add_child(self)



class VolumeArray(NonuniqueArray):
    """ Класс списка объёмов """
    element_list: List[Optional[Volume]]

    @property
    def material(self) -> MaterialArray:
        """ Список материалов """
        material = MaterialArray(self.shape)
        for volume, indices in self.inverse_indices.items():
            if volume is None:
                continue
            material[indices] = volume.material
        return material
