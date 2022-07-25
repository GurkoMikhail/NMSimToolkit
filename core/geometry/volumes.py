from copy import deepcopy
from itertools import count
import numpy as np
from numpy import matmul, inf
from core.materials.materials import MaterialArray
import core.other.utils as utils


class ElementaryVolume:
    """ Базовый класс элементарного объёма """

    _counter = count(1)

    def __init__(self, geometry, material, name=None):
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
    def size(self):
        return self.geometry.size

    @size.setter
    def size(self, value):
        self.geometry.size = value

    def dublicate(self):
        result = deepcopy(self)
        result.name = f'{self.name}.{next(self._dublicate_counter)}'
        return result

    def check_inside(self, position):
        """ Проверка на попадание в объём """
        return self.geometry.check_inside(position)

    def check_outside(self, position):
        """ Проверка на непопадание в объём """
        return self.geometry.check_outside(position)

    def cast_path(self, position, direction):
        """ Определение объекта местонахождения и длины пути частицы """
        current_volume = VolumeArray(position.shape[0])
        distance, inside = self.geometry.cast_path(position, direction)
        current_volume[inside] = self
        return distance, current_volume

    def get_material_by_position(self, position):
        """ Получить материал по координаты """
        material = MaterialArray(position.shape[0])
        inside = self.geometry.check_inside(position)
        material[inside] = self.material
        return material


class VolumeWithChilds(ElementaryVolume):
    """ Базовый класс объёма с детьми """    

    def __init__(self, geometry, material, name=None):
        super().__init__(geometry, material, name)
        self.childs = []

    def dublicate(self):
        result = super().dublicate()
        childs = result.childs
        result.childs = []
        for child in childs:
            child.dublicate()
        return result

    def cast_path(self, position, direction):
        distance, current_volume = super().cast_path(position, direction)
        if len(self.childs) > 0:
            inside = np.array([i for i, volume in enumerate(current_volume) if volume is not None], dtype=int)
            position = position[inside]
            direction = direction[inside]
            distance_inside = distance[inside]
            current_volume_inside = current_volume[inside]
            distance_to_child = np.full((len(self.childs), position.shape[0]), inf)
            for i, child in enumerate(self.childs):
                _distance_to_child, child_volume = child.cast_path(position, direction)
                inside_child = np.array([i for i, volume in enumerate(child_volume) if volume is not None], dtype=int)
                current_volume_inside[inside_child] = child_volume[inside_child]
                distance_to_child[i] = _distance_to_child
            distance_to_child = distance_to_child.min(axis=0)
            current_volume[inside] = current_volume_inside
            distance[inside] = np.where(distance_inside < distance_to_child, distance_inside, distance_to_child)
        return distance, current_volume

    def get_material_by_position(self, position):
        material = super().get_material_by_position(position)
        if len(self.childs) > 0:
            inside = material.ID != 0
            position = position[inside]
            material_inside = material[inside]
            for child in self.childs:
                child_material = child.get_material_by_position(position)
                inside_child = child_material.ID != 0
                material_inside[inside_child] = child_material[inside_child]
            material[inside] = material_inside
        return material

    def add_child(self, child):
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


class TransformableVolume(ElementaryVolume):
    """ Базовый класс трансформируемого объёма """

    def __init__(self, geometry, material, name=None):
        super().__init__(geometry, material, name)
        self.transformation_matrix = np.array([
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]
        ])
        self.parent = None

    def dublicate(self):
        result = super().dublicate()
        result.parent = None
        if self.parent is not None:
            result.set_parent(self.parent)
        return result

    @property
    def total_transformation_matrix(self):
        if isinstance(self.parent, TransformableVolume):
            return self.parent.total_transformation_matrix@self.transformation_matrix
        return self.transformation_matrix

    def convert_to_local_position(self, position, as_parent=True):
        """ Преобразовать в локальные координаты """
        # transformation_matrix = self.transformation_matrix if as_parent else self.total_transformation_matrix
        if not as_parent and isinstance(self.parent, TransformableVolume):
            position = self.parent.convert_to_local_position(position, as_parent)
        transformation_matrix = self.transformation_matrix
        local_position = np.ones((position.shape[0], 4), dtype=float)
        local_position[:, :3] = position
        matmul(local_position, transformation_matrix.T, out=local_position)
        position = local_position[:, :3]
        return position

    def convert_to_local_direction(self, direction, as_parent=True):
        """ Преобразовать в локальное направление """
        # transformation_matrix = self.transformation_matrix if as_parent else self.total_transformation_matrix
        if not as_parent and isinstance(self.parent, TransformableVolume):
            direction = self.parent.convert_to_local_direction(direction, as_parent)
        transformation_matrix = self.transformation_matrix
        direction = np.copy(direction)
        matmul(direction, transformation_matrix[:3, :3].T, out=direction)
        return direction

    def check_inside(self, position, local=False, as_parent=False):
        if not local:
            position = self.convert_to_local_position(position, as_parent)
        return super().check_inside(position)

    def check_outside(self, position, local=False, as_parent=False):
        if not local:
            position = self.convert_to_local_position(position, as_parent)
        return super().check_outside(position)

    def translate(self, x=0., y=0., z=0., inLocal=False):
        """ Переместить объём """
        translation = np.asarray([x, y, z])
        translation_matrix = utils.compute_translation_matrix(-translation)
        if inLocal:
            self.transformation_matrix = translation_matrix@self.transformation_matrix
        else:
            self.transformation_matrix = self.transformation_matrix@translation_matrix

    def rotate(self, alpha=0., beta=0., gamma=0., rotation_center=[0., 0., 0.], inLocal=False):
        """ Повернуть объём вокруг координатных осей """
        rotation_angles = np.asarray([alpha, beta, gamma])
        rotation_center = np.asarray(rotation_center)
        rotation_matrix = utils.compute_translation_matrix(-rotation_center)
        rotation_matrix = rotation_matrix@utils.compute_rotation_matrix(-rotation_angles)
        rotation_matrix = rotation_matrix@utils.compute_translation_matrix(rotation_center)
        if inLocal:
            self.transformation_matrix = rotation_matrix@self.transformation_matrix
        else:
            self.transformation_matrix = self.transformation_matrix@rotation_matrix

    def cast_path(self, position, direction, local=False, as_parent=True):
        if not local:
            position = self.convert_to_local_position(position, as_parent)
            direction = self.convert_to_local_direction(direction, as_parent)
        return super().cast_path(position, direction)

    def get_material_by_position(self, position, local=False, as_parent=True):
        if not local:
            position = self.convert_to_local_position(position, as_parent)
        material = super().get_material_by_position(position)
        return material

    def set_parent(self, parent):
        assert isinstance(parent, VolumeWithChilds), 'Этот объём не может быть родителем'
        parent.add_child(self)


class TransformableVolumeWithChild(TransformableVolume, VolumeWithChilds):
    """ Базовый класс трансформируемого объёма с детьми """  


class VolumeArray(np.ndarray):
    """ Класс списка объёмов """

    def __new__(subtype, shape):
        return super().__new__(subtype, shape, object)

    @property
    def material(self):
        """ Список материалов """
        material = MaterialArray(self.shape)
        for i, volume in enumerate(self):
            material[i] = volume.material
        return material

