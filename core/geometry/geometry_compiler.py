"""
Модуль компилятора геометрии сцены (GeometryCompiler) в структурированный массив NumPy (AoS).
"""
from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Tuple
import numpy as np
from numpy.typing import NDArray

from core.geometry.geometries import ShapeDataDType
from core.other.transform import TransformDType
from core.other.typing_definitions import Float, Index
from core.scene.nodes import CompositeNode

GeometryBufferDType = np.dtype([
    ('shape_data', ShapeDataDType),
    ('transform', TransformDType),
    ('miss_index', Index),
    ('parent_index', Index),
    ('volume_index', Index),
])

import core.geometry.flattened_scene as flattened_scene_mod

if TYPE_CHECKING:
    from core.geometry.volumes import Volume


class GeometryCompiler:
    """
    Компилирует ООП-граф сцены в структурированный NumPy массив Array of Structures (AoS),
    оптимизированный для быстрого вычисления пересечений лучей в кернелах Numba.
    """

    def compile_scene(self, root_node: CompositeNode) -> NDArray[np.void]:
        """
        Главная точка входа компиляции геометрии сцены.
        Преобразует иерархию узлов в плоскую AoS-структуру NumPy.
        """
        flat_list = flattened_scene_mod.FlattenedScene(root_node).flat_list
        capacity = len(flat_list)
        buffer: NDArray[np.void] = np.zeros(capacity, dtype=GeometryBufferDType)

        if capacity > 0:
            self._compute_miss_indices(flat_list, buffer)
            self._populate_buffer(flat_list, buffer)

        return buffer

    def _compute_miss_indices(
        self,
        flat_list: List[Tuple[Volume, NDArray[Float], Index]],
        buffer: NDArray[np.void],
    ) -> None:
        """
        Вычисляет и назначает miss_index для отсечения непересекаемых ветвей (Frustum Culling).
        miss_index указывает на индекс узла, следующего непосредственно за поддеревом текущего узла.
        """
        capacity = len(flat_list)
        if capacity == 0:
            return

        # Шаг 1: O(N) построение списка смежности
        children_map: Dict[int, List[int]] = defaultdict(list)
        for i in range(capacity):
            _, _, parent_idx = flat_list[i]
            if int(parent_idx) != -1:
                children_map[int(parent_idx)].append(i)

        # Шаг 2: O(N) вычисление размера поддерева через DFS
        def subtree_size(node_idx: int) -> int:
            count = 1
            for child_idx in children_map[node_idx]:
                count += subtree_size(child_idx)
            buffer[node_idx]['miss_index'] = node_idx + count
            return count

        # Вызываем DFS для корней леса (узлов с parent_index == -1)
        for i in range(capacity):
            _, _, parent_idx = flat_list[i]
            if int(parent_idx) == -1:
                subtree_size(i)

    def _populate_buffer(
        self,
        flat_list: List[Tuple[Volume, NDArray[Float], Index]],
        buffer: NDArray[np.void],
    ) -> None:
        """
        Заполняет буфер структурированного массива геометрическими формами, параметрами, индексами и трансформациями.
        """
        for i, (vol, inv_matrix, parent_idx) in enumerate(flat_list):
            vol.geometry.write_shape_data(buffer['shape_data'], i)

            buffer[i]['volume_index'] = i
            buffer[i]['parent_index'] = parent_idx

            rotation = buffer[i]['transform']['rotation']
            rotation['m00'] = inv_matrix[0, 0]
            rotation['m01'] = inv_matrix[0, 1]
            rotation['m02'] = inv_matrix[0, 2]

            rotation['m10'] = inv_matrix[1, 0]
            rotation['m11'] = inv_matrix[1, 1]
            rotation['m12'] = inv_matrix[1, 2]

            rotation['m20'] = inv_matrix[2, 0]
            rotation['m21'] = inv_matrix[2, 1]
            rotation['m22'] = inv_matrix[2, 2]

            translation = buffer[i]['transform']['translation']
            translation['x'] = inv_matrix[0, 3]
            translation['y'] = inv_matrix[1, 3]
            translation['z'] = inv_matrix[2, 3]


__all__ = ["GeometryCompiler", "GeometryBufferDType"]

