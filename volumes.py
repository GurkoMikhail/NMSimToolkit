from copy import copy, deepcopy
from itertools import count
import numpy as np
from numpy import matmul, inf
import utils


class ElementaryVolume:
    """ Базовый класс элементарного объёма """

    _counter = count(1)

    def __init__(self, geometry, material, name=None):
        """ Конструктор объёма """
        self.geometry = geometry
        self.material = material
        self.name = f'{self.__class__.__name__}{next(self._counter)}' if name is None else name
        self._dublicateCounter = count(1)

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
        result.name = f'{self.name}.{next(self._dublicateCounter)}'
        return result

    def checkInside(self, position):
        """ Проверка на попадание в объём """
        return self.geometry.checkInside(position)

    def checkOutside(self, position):
        """ Проверка на непопадание в объём """
        return self.geometry.checkOutside(position)

    def castPath(self, position, direction):
        """ Определение объекта местонахождения и длины пути частицы """
        currentVolume = VolumesList(position.shape[0])
        distance, inside = self.geometry.castPath(position, direction)
        currentVolume[inside] = self
        return distance, currentVolume

    def getMaterialByPosition(self, position):
        """ Получить материал по координаты """
        material = np.ndarray(position.shape[0], dtype=object)
        inside = self.geometry.checkInside(position)
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

    def castPath(self, position, direction):
        distance, currentVolume = super().castPath(position, direction)
        if len(self.childs) > 0:
            inside = np.array([i for i, volume in enumerate(currentVolume) if volume is not None], dtype=int)
            position = position[inside]
            direction = direction[inside]
            distanceInside = distance[inside]
            currentVolumeInside = currentVolume[inside]
            distanceToChild = np.full((len(self.childs), position.shape[0]), inf)
            for i, child in enumerate(self.childs):
                _distanceToChild, childVolume = child.castPath(position, direction)
                insideChild = np.array([i for i, volume in enumerate(childVolume) if volume is not None], dtype=int)
                currentVolumeInside[insideChild] = childVolume[insideChild]
                distanceToChild[i] = _distanceToChild
            distanceToChild = distanceToChild.min(axis=0)
            currentVolume[inside] = currentVolumeInside
            distance[inside] = np.where(distanceInside < distanceToChild, distanceInside, distanceToChild)
        return distance, currentVolume

    def getMaterialByPosition(self, position):
        materials = super().getMaterialByPosition(position)
        if len(self.childs) > 0:
            inside = np.array([i for i, material in enumerate(materials) if material is not None], dtype=int)
            position = position[inside]
            materialsInside = materials[inside]
            for child in self.childs:
                childMaterial = child.getMaterialByPosition(position)
                insideChild = np.array([i for i, material in enumerate(childMaterial) if material is not None], dtype=int)
                materialsInside[insideChild] = childMaterial[insideChild]
            materials[inside] = materialsInside
        return materials

    def addChild(self, child):
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
        self.transformationMatrix = np.array([
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
            result.setParent(self.parent)
        return result

    @property
    def totalTransformationMatrix(self):
        if isinstance(self.parent, TransformableVolume):
            return self.parent.totalTransformationMatrix@self.transformationMatrix
        return self.transformationMatrix

    def convertToLocalPosition(self, position, asParent=True):
        """ Преобразовать в локальные координаты """
        # transformationMatrix = self.transformationMatrix if asParent else self.totalTransformationMatrix
        if not asParent and isinstance(self.parent, TransformableVolume):
            position = self.parent.convertToLocalPosition(position, asParent)
        transformationMatrix = self.transformationMatrix
        localPosition = np.ones((position.shape[0], 4), dtype=float)
        localPosition[:, :3] = position
        matmul(localPosition, transformationMatrix.T, out=localPosition)
        position = localPosition[:, :3]
        return position

    def convertToLocalDirection(self, direction, asParent=True):
        """ Преобразовать в локальное направление """
        # transformationMatrix = self.transformationMatrix if asParent else self.totalTransformationMatrix
        if not asParent and isinstance(self.parent, TransformableVolume):
            direction = self.parent.convertToLocalDirection(direction, asParent)
        transformationMatrix = self.transformationMatrix
        direction = np.copy(direction)
        matmul(direction, transformationMatrix[:3, :3].T, out=direction)
        return direction

    def checkInside(self, position, local=False, asParent=False):
        if not local:
            position = self.convertToLocalPosition(position, asParent)
        return super().checkInside(position)

    def checkOutside(self, position, local=False, asParent=False):
        if not local:
            position = self.convertToLocalPosition(position, asParent)
        return super().checkOutside(position)

    def translate(self, x=0., y=0., z=0., inLocal=False):
        """ Переместить объём """
        translation = np.asarray([x, y, z])
        translationMatrix = utils.computeTranslationMatrix(-translation)
        if inLocal:
            self.transformationMatrix = translationMatrix@self.transformationMatrix
        else:
            self.transformationMatrix = self.transformationMatrix@translationMatrix

    def rotate(self, alpha=0., beta=0., gamma=0., rotationCenter=[0., 0., 0.], inLocal=False):
        """ Повернуть объём вокруг координатных осей """
        rotationAngles = np.asarray([alpha, beta, gamma])
        rotationCenter = np.asarray(rotationCenter)
        rotationMatrix = utils.computeTranslationMatrix(-rotationCenter)
        rotationMatrix = rotationMatrix@utils.computeRotationMatrix(-rotationAngles)
        rotationMatrix = rotationMatrix@utils.computeTranslationMatrix(rotationCenter)
        if inLocal:
            self.transformationMatrix = rotationMatrix@self.transformationMatrix
        else:
            self.transformationMatrix = self.transformationMatrix@rotationMatrix

    def castPath(self, position, direction, local=False, asParent=True):
        if not local:
            position = self.convertToLocalPosition(position, asParent)
            direction = self.convertToLocalDirection(direction, asParent)
        return super().castPath(position, direction)

    def getMaterialByPosition(self, position, local=False, asParent=True):
        if not local:
            position = self.convertToLocalPosition(position, asParent)
        material = super().getMaterialByPosition(position)
        return material

    def setParent(self, parent):
        assert isinstance(parent, VolumeWithChilds), 'Этот объём не может быть родителем'
        parent.addChild(self)


class TransformableVolumeWithChild(TransformableVolume, VolumeWithChilds):
    """ Базовый класс трансформируемого объёма с детьми """  


class VolumesList(np.ndarray):
    """ Класс списка объёмов """

    def __new__(subtype, shape):
        return super().__new__(subtype, shape, object)

    @property
    def materials(self):
        """ Список материалов """
        materials = [volume.material if volume is not None else None for volume in self]
        return np.asarray(materials, dtype=object)

