import numpy as np
from typing import Optional, List, Sequence, Tuple
from numpy.typing import NDArray

from core.other.typing_definitions import Float
import core.other.utils as utils

class SpatialNode:
    """
    Базовый узел графа сцены, отвечающий за пространственные преобразования (матрицы 4x4).

    Управляет локальными трансформациями, вычисляет и кэширует глобальную
    и обратную глобальную матрицы преобразования.
    """
    def __init__(self, name: Optional[str] = None, tags: Optional[Sequence[str]] = None):
        self.name = name if name is not None else self.__class__.__name__
        self.tags: List[str] = [str(t) for t in tags] if tags is not None else []
        self.local_matrix = np.eye(4, dtype=Float)
        self._parent: Optional['CompositeNode'] = None
        self._global_matrix_cache: Optional[NDArray[Float]] = None
        self._inverse_global_matrix_cache: Optional[NDArray[Float]] = None

    @property
    def parent(self) -> Optional['CompositeNode']:
        return self._parent

    @parent.setter
    def parent(self, value: Optional['CompositeNode']) -> None:
        self._parent = value
        self.invalidate_matrix_cache()

    @property
    def root(self) -> 'SpatialNode':
        current = self
        while current.parent is not None:
            current = current.parent
        return current

    def invalidate_matrix_cache(self) -> None:
        """Сбрасывает кэш матриц трансформации узла и всех его потомков."""
        self._global_matrix_cache = None
        self._inverse_global_matrix_cache = None

    @property
    def global_matrix(self) -> NDArray[Float]:
        """Вычисляет и кэширует прямую глобальную матрицу трансформации."""
        if self._global_matrix_cache is None:
            if self.parent is not None:
                self._global_matrix_cache = self.parent.global_matrix @ self.local_matrix
            else:
                self._global_matrix_cache = self.local_matrix
        return self._global_matrix_cache

    @property
    def inverse_global_matrix(self) -> NDArray[Float]:
        """Вычисляет и кэширует обратную глобальную матрицу трансформации."""
        if self._inverse_global_matrix_cache is None:
            self._inverse_global_matrix_cache = np.linalg.inv(self.global_matrix)
        return self._inverse_global_matrix_cache

    def translate(self, x: Float = Float(0.), y: Float = Float(0.), z: Float = Float(0.), in_local: bool = False) -> None:
        """Переместить узел. Модифицирует local_matrix и сбрасывает кэш."""
        translation = np.asarray([float(x), float(y), float(z)], dtype=float)
        translation_matrix = utils.compute_translation_matrix(translation)
        if in_local:
            self.local_matrix = self.local_matrix @ translation_matrix
        else:
            self.local_matrix = translation_matrix @ self.local_matrix
        self.invalidate_matrix_cache()

    def rotate(self, alpha: Float = Float(0.), beta: Float = Float(0.), gamma: Float = Float(0.), rotation_center: Sequence[Float] = (Float(0), Float(0), Float(0)), in_local: bool = False) -> None:
        """Повернуть узел. Модифицирует local_matrix и сбрасывает кэш."""
        rotation_angles = np.asarray([float(alpha), float(beta), float(gamma)], dtype=float)
        rot_center = np.asarray([float(c) for c in rotation_center], dtype=float)
        rotation_matrix = utils.compute_translation_matrix(rot_center)
        rotation_matrix = rotation_matrix @ utils.compute_rotation_matrix(rotation_angles)
        rotation_matrix = rotation_matrix @ utils.compute_translation_matrix(-rot_center)
        if in_local:
            self.local_matrix = self.local_matrix @ rotation_matrix
        else:
            self.local_matrix = rotation_matrix @ self.local_matrix
        self.invalidate_matrix_cache()

    def convert_to_local_position(self, position: NDArray[Float]) -> NDArray[Float]:
        """ Преобразовать в локальные координаты. Use inverse_global_matrix. """
        local_position = np.ones((position.shape[0], 4), dtype=position.dtype)
        local_position[:, :3] = position
        np.matmul(local_position, self.inverse_global_matrix.T.astype(position.dtype), out=local_position)
        return local_position[:, :3]

    def convert_to_local_direction(self, direction: NDArray[Float]) -> NDArray[Float]:
        """ Преобразовать в локальное направление. Use inverse_global_matrix rotation part. """
        direction_copy = np.copy(direction)
        np.matmul(direction_copy, self.inverse_global_matrix[:3, :3].T.astype(direction_copy.dtype), out=direction_copy)
        return direction_copy

    def convert_to_global_position(self, position: NDArray[Float]) -> NDArray[Float]:
        """ Преобразовать в глобальные координаты. Use global_matrix. """
        global_position = np.ones((position.shape[0], 4), dtype=position.dtype)
        global_position[:, :3] = position
        np.matmul(global_position, self.global_matrix.T.astype(position.dtype), out=global_position)
        return global_position[:, :3]

    def convert_to_global_direction(self, direction: NDArray[Float]) -> NDArray[Float]:
        """ Преобразовать в глобальное направление. Use global_matrix rotation part. """
        direction_copy = np.copy(direction)
        np.matmul(direction_copy, self.global_matrix[:3, :3].T.astype(direction_copy.dtype), out=direction_copy)
        return direction_copy


class CompositeNode(SpatialNode):
    """
    Составной узел для управления древовидной гетерогенной иерархией SpatialNode.
    """
    def __init__(self, name: Optional[str] = None, tags: Optional[Sequence[str]] = None):
        super().__init__(name=name, tags=tags)
        self.childs: List['SpatialNode'] = []

    def invalidate_matrix_cache(self) -> None:
        """Рекурсивно сбрасывает кэш матриц трансформации вниз по дереву дочерних узлов."""
        super().invalidate_matrix_cache()
        for child in self.childs:
            child.invalidate_matrix_cache()

    def add_child(self, child: 'SpatialNode') -> None:
        """Добавляет дочерний узел с корректным обновлением ссылки на родительский узел."""
        if child.parent is self and child in self.childs:
            return
        if child.parent is not None:
            if child in child.parent.childs:
                child.parent.childs.remove(child)
        if child not in self.childs:
            self.childs.append(child)
        child.parent = self
        child.invalidate_matrix_cache()

    def remove_child(self, child: 'SpatialNode') -> None:
        """Удаляет дочерний узел и сбрасывает ссылку на родителя."""
        if child in self.childs:
            self.childs.remove(child)
            child.parent = None
            child.invalidate_matrix_cache()



