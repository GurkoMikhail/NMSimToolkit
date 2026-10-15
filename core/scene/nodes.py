"""
Модуль базовых узлов графа сцены (SpatialNode, CompositeNode),
отвечающих за иерархию и пространственные матрицы трансформации 4x4.
"""
from __future__ import annotations

from typing import List, Optional, Sequence
import numpy as np
from numpy.typing import NDArray

from core.other.typing_definitions import Float
import core.other.utils as utils


class SpatialNode:
    """
    Базовый узел графа сцены, отвечающий за пространственные преобразования (матрицы 4x4).

    Управляет локальными трансформациями, вычисляет и кэширует глобальную
    и обратную глобальную матрицы преобразования.
    """

    def __init__(self, name: Optional[str] = None, tags: Optional[Sequence[str]] = None) -> None:
        self.name = name if name is not None else self.__class__.__name__
        self.tags: List[str] = [str(tag) for tag in tags] if tags is not None else []
        self.local_matrix: NDArray[Float] = np.eye(4, dtype=Float)
        self._parent: Optional[CompositeNode] = None
        self._global_matrix_cache: Optional[NDArray[Float]] = None
        self._inverse_global_matrix_cache: Optional[NDArray[Float]] = None

    @property
    def parent(self) -> Optional[CompositeNode]:
        """Родительский составной узел в иерархии сцены."""
        return self._parent

    @parent.setter
    def parent(self, value: Optional[CompositeNode]) -> None:
        self._parent = value
        self.invalidate_matrix_cache()

    @property
    def root(self) -> SpatialNode:
        """Корневой узел текущего дерева сцены."""
        current: SpatialNode = self
        while current.parent is not None:
            current = current.parent
        return current

    def invalidate_geometry(self) -> None:
        """Инвалидирует кэш скомпилированной геометрии вверх по иерархии родительских узлов."""
        if self._parent is not None:
            self._parent.invalidate_geometry()

    def invalidate_matrix_cache(self) -> None:
        """Сбрасывает кэш матриц трансформации узла."""
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

    def translate(
        self,
        x: Float | float = Float(0.0),
        y: Float | float = Float(0.0),
        z: Float | float = Float(0.0),
        in_local: bool = False,
    ) -> None:
        """Перемещает узел, модифицируя local_matrix и сбрасывая кэш матриц."""
        translation = np.asarray([Float(x), Float(y), Float(z)], dtype=Float)
        translation_matrix = utils.compute_translation_matrix(translation)
        if in_local:
            self.local_matrix = self.local_matrix @ translation_matrix
        else:
            self.local_matrix = translation_matrix @ self.local_matrix
        self.invalidate_matrix_cache()

    def rotate(
        self,
        alpha: Float | float = Float(0.0),
        beta: Float | float = Float(0.0),
        gamma: Float | float = Float(0.0),
        rotation_center: Sequence[Float | float] = (Float(0.0), Float(0.0), Float(0.0)),
        in_local: bool = False,
    ) -> None:
        """Поворачивает узел вокруг заданного центра, модифицируя local_matrix и сбрасывая кэш матриц."""
        rotation_angles = np.asarray([Float(alpha), Float(beta), Float(gamma)], dtype=Float)
        rot_center = np.asarray([Float(coord) for coord in rotation_center], dtype=Float)
        rotation_matrix = utils.compute_translation_matrix(rot_center)
        rotation_matrix = rotation_matrix @ utils.compute_rotation_matrix(rotation_angles)
        rotation_matrix = rotation_matrix @ utils.compute_translation_matrix(-rot_center)
        if in_local:
            self.local_matrix = self.local_matrix @ rotation_matrix
        else:
            self.local_matrix = rotation_matrix @ self.local_matrix
        self.invalidate_matrix_cache()

    def convert_to_local_position(self, position: NDArray[Float]) -> NDArray[Float]:
        """Преобразует мировые координаты точек в локальную систему координат через inverse_global_matrix."""
        local_position = np.ones((position.shape[0], 4), dtype=position.dtype)
        local_position[:, :3] = position
        np.matmul(local_position, self.inverse_global_matrix.T.astype(position.dtype), out=local_position)
        return local_position[:, :3]

    def convert_to_local_direction(self, direction: NDArray[Float]) -> NDArray[Float]:
        """Преобразует мировые векторы направлений в локальную систему координат."""
        direction_copy = np.copy(direction)
        np.matmul(direction_copy, self.inverse_global_matrix[:3, :3].T.astype(direction_copy.dtype), out=direction_copy)
        return direction_copy

    def convert_to_global_position(self, position: NDArray[Float]) -> NDArray[Float]:
        """Преобразует локальные координаты точек в мировую систему координат через global_matrix."""
        global_position = np.ones((position.shape[0], 4), dtype=position.dtype)
        global_position[:, :3] = position
        np.matmul(global_position, self.global_matrix.T.astype(position.dtype), out=global_position)
        return global_position[:, :3]

    def convert_to_global_direction(self, direction: NDArray[Float]) -> NDArray[Float]:
        """Преобразует локальные векторы направлений в мировую систему координат."""
        direction_copy = np.copy(direction)
        np.matmul(direction_copy, self.global_matrix[:3, :3].T.astype(direction_copy.dtype), out=direction_copy)
        return direction_copy


class CompositeNode(SpatialNode):
    """
    Составной узел для управления древовидной гетерогенной иерархией SpatialNode.
    """

    def __init__(self, name: Optional[str] = None, tags: Optional[Sequence[str]] = None) -> None:
        super().__init__(name=name, tags=tags)
        self.childs: List[SpatialNode] = []

    def invalidate_matrix_cache(self) -> None:
        """Рекурсивно сбрасывает кэш матриц трансформации вниз по дереву дочерних узлов."""
        super().invalidate_matrix_cache()
        for child in self.childs:
            child.invalidate_matrix_cache()

    def add_child(self, child: SpatialNode) -> None:
        """Добавляет дочерний узел с корректным переносом от прежнего родителя и инвалидацией кэшей."""
        if child.parent is self and child in self.childs:
            return
        if child.parent is not None:
            child.parent.remove_child(child)
        if child not in self.childs:
            self.childs.append(child)
        child.parent = self
        self.invalidate_geometry()

    def remove_child(self, child: SpatialNode) -> None:
        """Удаляет дочерний узел, сбрасывает ссылку на родителя и инвалидирует кэши геометрии и матриц."""
        if child in self.childs:
            self.childs.remove(child)
            child.parent = None
            self.invalidate_geometry()


__all__ = [
    "SpatialNode",
    "CompositeNode",
]




