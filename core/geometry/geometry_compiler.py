import numpy as np
from numpy.typing import NDArray

from core.geometry.volumes import Volume, GeometryBufferDType
from core.scene.nodes import CompositeNode


class GeometryCompiler:
    """
    Compiles an OOP Scene Graph into an Array of Structures (AoS) numpy structured array
    optimized for fast Numba raycasting.
    """

    def compile_scene(self, root_node: CompositeNode) -> NDArray[np.void]:
        """
        Main entry point for scene compilation.
        Converts the OOP hierarchy into a flat numpy AoS structure.
        """
        from core.geometry.flattened_scene import FlattenedScene
        flat_list = FlattenedScene(root_node).flat_list
        capacity = len(flat_list)
        buffer = np.zeros(capacity, dtype=GeometryBufferDType)

        if capacity > 0:
            self._compute_miss_indices(flat_list, buffer)
            self._populate_buffer(flat_list, buffer)

        return buffer

    def _compute_miss_indices(self, flat_list: list, buffer: NDArray[np.void]) -> None:
        """
        Calculates and assigns `miss_index` for Frustum Culling.
        The miss_index points to the node directly after the current node's subtree.
        """
        capacity = len(flat_list)
        if capacity == 0:
            return

        # Шаг 1: O(N) построение списка смежности
        from collections import defaultdict
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
        Populates the structured array buffer with shapes, parameters, indices, and transforms.
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
