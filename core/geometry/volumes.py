from copy import deepcopy
from itertools import count
import numpy as np
from numpy import matmul, inf
from typing import List, Optional, Tuple, Union, Any, Type, Sequence, Generic
from numpy.typing import NDArray
from core.materials.materials import MaterialArray, Material
from core.geometry.geometries import Geometry
from core.other.nonunique_array import NonuniqueArray
import core.other.utils as utils
from core.other.typing_definitions import Vector3D, Precision


class ElementaryVolume(Generic[Precision]):
    """ Базовый класс элементарного объёма """

    _counter = count(1)

    geometry: Geometry[Precision]
    material: Material
    name: str

    def __init__(self, geometry: Geometry[Precision], material: Material, name: Optional[str] = None) -> None:
        """ Конструктор объёма """
        self.geometry = geometry
        self.material = material
        self.name = f'{self.__class__.__name__}{next(self._counter)}' if name is None else name
        self._dublicate_counter = count(1)

    def __init_subclass__(cls):
        cls._counter = count(1)

    def __repr__(self):
        return f'{self.name}'

    @property
    def size(self) -> Vector3D[Precision]: # type: ignore
        return self.geometry.size

    @size.setter
    def size(self, value: Vector3D[Precision]) -> None: # type: ignore
        self.geometry.size = value

    def dublicate(self):
        result = deepcopy(self)
        result.name = f'{self.name}.{next(self._dublicate_counter)}'
        return result

    def check_inside(self, position: Vector3D[Precision]) -> Union[bool, NDArray[np.bool_]]: # type: ignore
        """ Проверка на попадание в объём """
        return self.geometry.check_inside(position)

    def check_outside(self, position: Vector3D[Precision]) -> Union[bool, NDArray[np.bool_]]: # type: ignore
        """ Проверка на непопадание в объём """
        return self.geometry.check_outside(position)

    def cast_path(self, position: Vector3D[Precision], direction: Vector3D[Precision]) -> Tuple[NDArray[Precision], 'VolumeArray']: # type: ignore
        """ Определение объекта местонахождения и длины пути частицы """
        current_volume = VolumeArray(position.shape[0])
        distance, inside = self.geometry.cast_path(position, direction)
        current_volume[inside] = self
        return distance, current_volume

    def get_material_by_position(self, position: Vector3D[Precision]) -> MaterialArray: # type: ignore
        """ Получить материал по координаты """
        material = MaterialArray(position.shape[0])
        inside = self.geometry.check_inside(position)
        material[inside] = self.material
        return material


class VolumeWithChilds(ElementaryVolume[Precision]):
    """ Базовый класс объёма с детьми """    
    childs: List['TransformableVolume[Precision]']

    def __init__(self, geometry: Geometry[Precision], material: Material, name: Optional[str] = None) -> None:
        super().__init__(geometry, material, name)
        self.childs = []

    def dublicate(self):
        result = super().dublicate()
        childs = result.childs
        result.childs = []
        for child in childs:
            child.dublicate()
        return result

    def cast_path(self, position: Vector3D[Precision], direction: Vector3D[Precision]) -> Tuple[NDArray[Precision], 'VolumeArray']: # type: ignore
        distance, current_volume = super().cast_path(position, direction)
        if len(self.childs) > 0:
            inside = current_volume != 0
            position_inside = position[inside]
            direction_inside = direction[inside]
            distance_inside = distance[inside]
            current_volume_inside = current_volume[inside]
            distance_to_child = np.full((len(self.childs), position_inside.shape[0]), inf)
            for i, child in enumerate(self.childs):
                _distance_to_child, child_volume = child.cast_path(position_inside, direction_inside)
                inside_child = child_volume != 0
                current_volume_inside[inside_child] = child_volume[inside_child]
                distance_to_child[i] = _distance_to_child
            distance_to_child_min = distance_to_child.min(axis=0)
            current_volume[inside] = current_volume_inside
            distance[inside] = np.where(distance_inside < distance_to_child_min, distance_inside, distance_to_child_min)
        return distance, current_volume

    def get_material_by_position(self, position: Vector3D[Precision]) -> MaterialArray: # type: ignore
        material = super().get_material_by_position(position)
        if len(self.childs) > 0:
            inside = material != 0
            position_inside = position[inside]
            material_inside = material[inside]
            for child in self.childs:
                child_material = child.get_material_by_position(position_inside)
                inside_child = child_material != 0
                material_inside[inside_child] = child_material[inside_child]
            material[inside] = material_inside
        return material

    def add_child(self, child: 'TransformableVolume[Precision]') -> None: # type: ignore
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


class TransformableVolume(ElementaryVolume[Precision]):
    """ Базовый класс трансформируемого объёма """
    transformation_matrix: NDArray[np.float64]
    parent: Optional[VolumeWithChilds[Precision]] # type: ignore

    def __init__(self, geometry: Geometry[Precision], material: Material, name: Optional[str] = None) -> None:
        super().__init__(geometry, material, name)
        self.transformation_matrix = np.array([
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]
        ], dtype=np.float64)
        self.parent = None

    def dublicate(self):
        result = super().dublicate()
        result.parent = None
        if self.parent is not None:
            result.set_parent(self.parent)
        return result

    @property
    def total_transformation_matrix(self) -> NDArray[np.float64]:
        if isinstance(self.parent, TransformableVolume):
            return self.parent.total_transformation_matrix@self.transformation_matrix
        return self.transformation_matrix

    def convert_to_local_position(self, position: Vector3D[Precision], as_parent: bool = True) -> Vector3D[Precision]: # type: ignore
        """ Преобразовать в локальные координаты """
        # transformation_matrix = self.transformation_matrix if as_parent else self.total_transformation_matrix
        if not as_parent and isinstance(self.parent, TransformableVolume):
            position = self.parent.convert_to_local_position(position, as_parent)
        transformation_matrix = self.transformation_matrix
        local_position = np.ones((position.shape[0], 4), dtype=position.dtype)
        local_position[:, :3] = position
        matmul(local_position, transformation_matrix.T.astype(position.dtype), out=local_position)
        position = local_position[:, :3]
        return position

    def convert_to_local_direction(self, direction: Vector3D[Precision], as_parent: bool = True) -> Vector3D[Precision]: # type: ignore
        """ Преобразовать в локальное направление """
        # transformation_matrix = self.transformation_matrix if as_parent else self.total_transformation_matrix
        if not as_parent and isinstance(self.parent, TransformableVolume):
            direction = self.parent.convert_to_local_direction(direction, as_parent)
        transformation_matrix = self.transformation_matrix
        direction = np.copy(direction)
        matmul(direction, transformation_matrix[:3, :3].T.astype(direction.dtype), out=direction)
        return direction

    def check_inside(self, position: Vector3D[Precision], local: bool = False, as_parent: bool = False) -> Union[bool, NDArray[np.bool_]]: # type: ignore
        if not local:
            position = self.convert_to_local_position(position, as_parent)
        return super().check_inside(position)

    def check_outside(self, position: Vector3D[Precision], local: bool = False, as_parent: bool = False) -> Union[bool, NDArray[np.bool_]]: # type: ignore
        if not local:
            position = self.convert_to_local_position(position, as_parent)
        return super().check_outside(position)

    def translate(self, x: float = 0., y: float = 0., z: float = 0., inLocal: bool = False) -> None:
        """ Переместить объём """
        translation = np.asarray([x, y, z])
        translation_matrix = utils.compute_translation_matrix(-translation)
        if inLocal:
            self.transformation_matrix = translation_matrix@self.transformation_matrix
        else:
            self.transformation_matrix = self.transformation_matrix@translation_matrix

    def rotate(self, alpha: float = 0., beta: float = 0., gamma: float = 0., rotation_center: Sequence[float] = [0., 0., 0.], inLocal: bool = False) -> None:
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

    def cast_path(self, position: Vector3D[Precision], direction: Vector3D[Precision], local: bool = False, as_parent: bool = True) -> Tuple[NDArray[Precision], 'VolumeArray']: # type: ignore
        if not local:
            position = self.convert_to_local_position(position, as_parent)
            direction = self.convert_to_local_direction(direction, as_parent)
        return super().cast_path(position, direction)

    def get_material_by_position(self, position: Vector3D[Precision], local: bool = False, as_parent: bool = True) -> MaterialArray: # type: ignore
        if not local:
            position = self.convert_to_local_position(position, as_parent)
        material = super().get_material_by_position(position)
        return material

    def set_parent(self, parent: VolumeWithChilds[Precision]) -> None: # type: ignore
        assert isinstance(parent, VolumeWithChilds), 'Этот объём не может быть родителем'
        parent.add_child(self)


class TransformableVolumeWithChild(TransformableVolume[Precision], VolumeWithChilds[Precision]): # type: ignore
    """ Базовый класс трансформируемого объёма с детьми """  


class VolumeArray(NonuniqueArray):
    """ Класс списка объёмов """
    element_list: List[Optional[ElementaryVolume]]

    @property
    def material(self) -> MaterialArray:
        """ Список материалов """
        material = MaterialArray(self.shape)
        for volume, indices in self.inverse_indices.items():
            if volume is None:
                continue
            material[indices] = volume.material
        return material

