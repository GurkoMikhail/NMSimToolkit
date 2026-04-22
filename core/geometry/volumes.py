from copy import deepcopy
from itertools import count
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

import core.other.utils as utils
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

class Volume:
    """ Базовый класс элементарного объёма """

    _counter = count(1)

    geometry: Geometry
    material: Material
    name: str

    def __init__(self, geometry: Geometry, material: Material, name: Optional[str] = None) -> None:
        """ Конструктор объёма """
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
        return [self.material]

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

    def dublicate(self):
        result = deepcopy(self)
        result.name = f'{self.name}.{next(self._dublicate_counter)}'
        return result

class VolumeWithChilds(Volume):
    """ Базовый класс объёма с детьми """    
    childs: List['TransformableVolume']

    def __init__(self, geometry: Geometry, material: Material, name: Optional[str] = None) -> None:
        super().__init__(geometry, material, name)
        self.childs = []

    @property
    def material_list(self) -> List[Material]:
        materials = [self.material]
        for child in self.childs:
            materials.extend(child.material_list)
        return list(set(materials))

    def dublicate(self):
        result = super().dublicate()
        childs = result.childs
        result.childs = []
        for child in childs:
            child.dublicate()
        return result

    def invalidate_scene(self) -> None:
        if self._geometry_buffer is not None or self._flattened_scene is not None:
            self._geometry_buffer = None
            self._flattened_scene = None
            for child in self.childs:
                child.invalidate_scene()
        # Если есть родитель (для TransformableVolumeWithChild), он тоже должен быть инвалидирован,
        # но это будет решаться в TransformableVolume

    def add_child(self, child: 'TransformableVolume') -> None:
        """ Добавить дочерний объём """
        assert isinstance(child, TransformableVolume), 'Только трансформируемый объём может быть дочерним'
        if child.parent is None:
            self.childs.append(child)
        elif child in self.childs:
            print('Добавляемый объём уже является дочерним данному объёму')
        else:
            print('Внимение! Добавляемый объём уже является дочерним. Новый родитель установлен')
            child.parent.childs.remove(child)
        child.parent = self
        self.invalidate_scene()
        child.invalidate_scene()

class TransformableVolume(Volume):
    """ Базовый класс трансформируемого объёма """
    transformation_matrix: NDArray[Float]
    parent: Optional[VolumeWithChilds]

    def __init__(self, geometry: Geometry, material: Material, name: Optional[str] = None) -> None:
        super().__init__(geometry, material, name)
        self.transformation_matrix = np.array([
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]
        ], dtype=Float)
        self.parent = None

    def dublicate(self):
        result = super().dublicate()
        result.parent = None
        if self.parent is not None:
            result.set_parent(self.parent)
        return result

    @property
    def total_transformation_matrix(self) -> NDArray[Float]:
        if isinstance(self.parent, TransformableVolume):
            return self.transformation_matrix@self.parent.total_transformation_matrix
        return self.transformation_matrix

    def convert_to_local_position(self, position: Vector3D, as_parent: bool = True) -> Vector3D:
        """ Преобразовать в локальные координаты """
        # transformation_matrix = self.transformation_matrix if as_parent else self.total_transformation_matrix
        if not as_parent and isinstance(self.parent, TransformableVolume):
            position = self.parent.convert_to_local_position(position, as_parent)
        transformation_matrix = self.transformation_matrix
        local_position = np.ones((position.shape[0], 4), dtype=position.dtype)
        local_position[:, :3] = position
        np.matmul(local_position, transformation_matrix.T.astype(position.dtype), out=local_position)
        position = local_position[:, :3]
        return position

    def convert_to_local_direction(self, direction: Vector3D, as_parent: bool = True) -> Vector3D:
        """ Преобразовать в локальное направление """
        # transformation_matrix = self.transformation_matrix if as_parent else self.total_transformation_matrix
        if not as_parent and isinstance(self.parent, TransformableVolume):
            direction = self.parent.convert_to_local_direction(direction, as_parent)
        transformation_matrix = self.transformation_matrix
        direction = np.copy(direction)
        np.matmul(direction, transformation_matrix[:3, :3].T.astype(direction.dtype), out=direction)
        return direction

    def translate(self, x: Float = Float(0.), y: Float = Float(0.), z: Float = Float(0.), inLocal: bool = False) -> None:
        """ Переместить объём """
        translation = np.asarray([x, y, z])
        translation_matrix = utils.compute_translation_matrix(-translation)
        if inLocal:
            self.transformation_matrix = translation_matrix@self.transformation_matrix
        else:
            self.transformation_matrix = self.transformation_matrix@translation_matrix
        self.invalidate_scene()

    def rotate(self, alpha: Float = Float(0.), beta: Float = Float(0.), gamma: Float = Float(0.), rotation_center: Sequence[Float] = (Float(0), Float(0), Float(0)), inLocal: bool = False) -> None:
        """ Повернуть объём вокруг координатных осей """
        rotation_angles = np.asarray([alpha, beta, gamma])
        rot_center = np.asarray(rotation_center)
        rotation_matrix = utils.compute_translation_matrix(-rot_center)
        rotation_matrix = rotation_matrix@utils.compute_rotation_matrix(-rotation_angles)
        rotation_matrix = rotation_matrix@utils.compute_translation_matrix(rot_center)
        if inLocal:
            self.transformation_matrix = rotation_matrix@self.transformation_matrix
        else:
            self.transformation_matrix = self.transformation_matrix@rotation_matrix
        self.invalidate_scene()

    def invalidate_scene(self) -> None:
        super().invalidate_scene()
        if self.parent is not None:
            self.parent.invalidate_scene()

    def set_parent(self, parent: VolumeWithChilds) -> None:
        assert isinstance(parent, VolumeWithChilds), 'Этот объём не может быть родителем'
        parent.add_child(self)

    @property
    def root_volume(self) -> Volume:
        """ Возвращает корневой объём дерева сцены """
        current = self
        while isinstance(current, TransformableVolume) and current.parent is not None:
            current = current.parent
        return current

class TransformableVolumeWithChild(TransformableVolume, VolumeWithChilds):
    """ Базовый класс трансформируемого объёма с детьми """  

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
