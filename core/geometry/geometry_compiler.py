import numpy as np
from numpy.typing import NDArray

from core.other.typing_definitions import Float
from core.geometry.volumes import Volume, VolumeWithChilds, TransformableVolume, GeometryBufferDType


class GeometryCompiler:
    """
    Compiles an OOP Scene Graph into an Array of Structures (AoS) numpy structured array
    optimized for fast Numba raycasting.
    """

    def compile_scene(self, root_volume: Volume) -> NDArray[np.void]:
        """
        Main entry point for scene compilation.
        Converts the OOP hierarchy into a flat numpy AoS structure.
        """
        flat_list = self._flatten_scene_graph(root_volume)
        capacity = len(flat_list)
        buffer = np.zeros(capacity, dtype=GeometryBufferDType)

        if capacity > 0:
            self._compute_miss_indices(flat_list, buffer)
            self._populate_buffer(flat_list, buffer)

        return buffer

    def _flatten_scene_graph(self, root_volume: Volume) -> list:
        """
        Traverses the OOP Scene Graph via DFS and returns a flattened list.
        Each element is a tuple: (Volume, total_transformation_matrix, parent_index).
        """
        flat_list = []

        def dfs(volume: Volume, parent_matrix: NDArray[Float], parent_index: int) -> int:
            if isinstance(volume, TransformableVolume):
                total_matrix = volume.total_transformation_matrix
            else:
                total_matrix = parent_matrix

            current_index = len(flat_list)
            flat_list.append((volume, total_matrix, parent_index))

            child_count = 0
            if isinstance(volume, VolumeWithChilds):
                for child in volume.childs:
                    child_count += dfs(child, total_matrix, current_index)

            return child_count + 1

        identity_matrix = np.eye(4, dtype=Float)
        dfs(root_volume, identity_matrix, -1)

        return flat_list

    def _compute_miss_indices(self, flat_list: list, buffer: NDArray[np.void]) -> None:
        """
        Calculates and assigns `miss_index` for Frustum Culling.
        The miss_index points to the node directly after the current node's subtree.
        """
        capacity = len(flat_list)

        def calc_miss(node_idx: int) -> int:
            vol, _, _ = flat_list[node_idx]
            count = 1
            if isinstance(vol, VolumeWithChilds):
                for child in vol.childs:
                    for c_idx in range(node_idx + 1, capacity):
                        if flat_list[c_idx][0] is child:
                            count += calc_miss(c_idx)
                            break
            buffer[node_idx]['miss_index'] = node_idx + count
            return count

        calc_miss(0)

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
