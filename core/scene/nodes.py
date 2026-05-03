import numpy as np
from typing import Optional, List, Sequence
from numpy.typing import NDArray

from core.other.typing_definitions import Float
import core.other.utils as utils

class SpatialNode:
    """
    Base node that is responsible only for spatial mathematics (4x4 matrices).
    It manages local transformations and computes global and inverse global matrices.
    """
    def __init__(self):
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
        """Invalidates the matrix cache for this node and all of its descendants."""
        self._global_matrix_cache = None
        self._inverse_global_matrix_cache = None

    @property
    def global_matrix(self) -> NDArray[Float]:
        """Calculates and caches the direct global transformation matrix."""
        if self._global_matrix_cache is None:
            if self.parent is not None:
                self._global_matrix_cache = self.parent.global_matrix @ self.local_matrix
            else:
                self._global_matrix_cache = self.local_matrix
        return self._global_matrix_cache

    @property
    def inverse_global_matrix(self) -> NDArray[Float]:
        """Calculates and caches the inverse global transformation matrix."""
        if self._inverse_global_matrix_cache is None:
            self._inverse_global_matrix_cache = np.linalg.inv(self.global_matrix)
        return self._inverse_global_matrix_cache

    def translate(self, x: Float = Float(0.), y: Float = Float(0.), z: Float = Float(0.), in_local: bool = False) -> None:
        """Translates the node. Modifies local_matrix and invalidates cache."""
        translation = np.asarray([x, y, z])
        translation_matrix = utils.compute_translation_matrix(translation)
        if in_local:
            self.local_matrix = self.local_matrix @ translation_matrix
        else:
            self.local_matrix = translation_matrix @ self.local_matrix
        self.invalidate_matrix_cache()

    def rotate(self, alpha: Float = Float(0.), beta: Float = Float(0.), gamma: Float = Float(0.), rotation_center: Sequence[Float] = (Float(0), Float(0), Float(0)), in_local: bool = False) -> None:
        """Rotates the node. Modifies local_matrix and invalidates cache."""
        rotation_angles = np.asarray([alpha, beta, gamma])
        rot_center = np.asarray(rotation_center)
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


class CompositeNode(SpatialNode):
    """
    Composite Node to manage a heterogeneous hierarchy of SpatialNodes.
    """
    def __init__(self):
        super().__init__()
        self.childs: List['SpatialNode'] = []


    def invalidate_matrix_cache(self) -> None:
        """Invalidates matrix cache recursively down the tree."""
        super().invalidate_matrix_cache()
        for child in self.childs:
            child.invalidate_matrix_cache()

    def add_child(self, child: 'SpatialNode') -> None:
        """Adds a child node and correctly handles parent reassignment."""
        if child.parent is not None:
            if child in child.parent.childs:
                child.parent.childs.remove(child)
        self.childs.append(child)
        child.parent = self
        child.invalidate_matrix_cache()
