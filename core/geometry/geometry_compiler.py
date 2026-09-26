from collections import defaultdict
import numpy as np
from numpy.typing import NDArray

import core.geometry.flattened_scene as flattened_scene_mod
from core.geometry.volumes import GeometryBufferDType
from core.scene.nodes import CompositeNode


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
        buffer = np.zeros(capacity, dtype=GeometryBufferDType)

        if capacity > 0:
            self._compute_miss_indices(flat_list, buffer)
            self._populate_buffer(flat_list, buffer)

        return buffer

    def _compute_miss_indices(self, flat_list: list, buffer: NDArray[np.void]) -> None:
        """
        Вычисляет и назначает miss_index для отсечения непересекаемых ветвей (Frustum Culling).
        miss_index указывает на индекс узла, следующего непосредственно за поддеревом текущего узла.
        """
        capacity = len(flat_list)
        if capacity == 0:
            return

        # Шаг 1: O(N) построение списка смежности
        children_map = defaultdict(list)
        for i in range(capacity):
            _, _, p_idx = flat_list[i]
            if p_idx != -1:
                children_map[p_idx].append(i)

        # Шаг 2: O(N) вычисление размера поддерева через DFS
        def subtree_size(node_idx: int) -> int:
            count = 1
            for child_idx in children_map[node_idx]:
                count += subtree_size(child_idx)
            buffer[node_idx]['miss_index'] = node_idx + count
            return count

        # Вызываем DFS для корней леса (узлов с parent_index == -1)
        # Обычно это только нулевой индекс
        for i in range(capacity):
            _, _, p_idx = flat_list[i]
            if p_idx == -1:
                subtree_size(i)

    def _populate_buffer(self, flat_list: list, buffer: NDArray[np.void]) -> None:
        """
        Заполняет буфер структурированного массива геометрическими формами, параметрами, индексами и трансформациями.
        """
        for i, (vol, mat, p_idx) in enumerate(flat_list):
            # Polymorphic delegation to shape-specific data writing
            vol.geometry.write_shape_data(buffer['shape_data'], i)

            buffer[i]['volume_index'] = i
            buffer[i]['parent_index'] = p_idx

            # Matrix: World -> Local (Unrolled explicitly)
            rotation = buffer[i]['transform']['rotation']
            rotation['m00'] = mat[0, 0]
            rotation['m01'] = mat[0, 1]
            rotation['m02'] = mat[0, 2]

            rotation['m10'] = mat[1, 0]
            rotation['m11'] = mat[1, 1]
            rotation['m12'] = mat[1, 2]

            rotation['m20'] = mat[2, 0]
            rotation['m21'] = mat[2, 1]
            rotation['m22'] = mat[2, 2]

            translation = buffer[i]['transform']['translation']
            translation['x'] = mat[0, 3]
            translation['y'] = mat[1, 3]
            translation['z'] = mat[2, 3]
