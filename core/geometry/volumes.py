from copy import deepcopy
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
from core.geometry.flattened_scene import FlattenedScene

GeometryBufferDType = np.dtype([
    ('shape_data', ShapeDataDType),
    ('transform', TransformDType),
    ('miss_index', Index),
    ('parent_index', Index),
    ('volume_index', Index)
])

class Volume(CompositeNode):
    """ Base class for an elementary volume, inheriting from CompositeNode for scene graph hierarchy. """

    _counter = count(1)

    geometry: Geometry
    material: Material
    name: str

    def __init__(self, geometry: Geometry, material: Material, name: Optional[str] = None) -> None:
        super().__init__()
        self.geometry = geometry
        self.material = material
        self.name = f'{self.__class__.__name__}{next(self._counter)}' if name is None else name
        self._dublicate_counter = count(1)
        self._geometry_buffer: Optional[NDArray[Any]] = None
        self._flattened_scene: Optional[FlattenedScene] = None

    def __init_subclass__(cls):
        cls._counter = count(1)

    def __repr__(self):
        return f'{self.name}'

    @property
    def flattened_scene(self) -> FlattenedScene:
        """ Lazy evaluation of the flattened scene starting from this volume. """
        if self._flattened_scene is None:
            self._flattened_scene = FlattenedScene(self)
        return self._flattened_scene

    @property
    def material_cfunc(self) -> CMaterialFunc:
        """ CFUNCTYPE pointer of the @cfunc for Woodcock paramteric volumes. Defaults to None for normal volumes. """
        return None

    @property
    def majorant_material(self) -> Material:
        """ Returns the majorant material. Defaults to self.material for normal volumes. """
        return self.material

    @property
    def material_list(self) -> List[Material]:
        """ Returns a list of all materials used in this volume (and its children). """
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
        self.invalidate_scene()

    @property
    def geometry_buffer(self) -> NDArray[Any]:
        """ Lazy compilation of GeometryBuffer (AoS Structured Array) """
        if self._geometry_buffer is None:
            from core.geometry.geometry_compiler import GeometryCompiler
            self._geometry_buffer = GeometryCompiler().compile_scene(self)
        return self._geometry_buffer

    def invalidate_scene(self) -> None:
        """ Инвалидация кэша геометрии у этого объекта и его родителей/детей. """
        self._geometry_buffer = None
        self._flattened_scene = None
        # We need to invalidate matrix cache as well
        self.invalidate_matrix_cache()


    @property
    def top_volume(self) -> 'Volume':
        """ Returns the top-most Volume in the current branch. """
        current = self
        top_vol = self
        while current.parent is not None:
            current = current.parent
            if isinstance(current, Volume):
                top_vol = current
        return top_vol

    def dublicate(self):
        result = deepcopy(self)
        result.name = f'{self.name}.{next(self._dublicate_counter)}'
        result.parent = None
        # Dublicate children
        childs = result.childs
        result.childs = []
        for child in childs:
            if hasattr(child, 'dublicate'):
                child_copy = child.dublicate()
                result.add_child(child_copy)
        return result



    def set_parent(self, parent: 'CompositeNode') -> None:
        parent.add_child(self)
        self.invalidate_scene()



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
