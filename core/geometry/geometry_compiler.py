import numpy as np
from typing import Any
from numpy.typing import NDArray

from core.other.typing_definitions import Float, ShapeID, Index


class GeometryCompiler:
    """
    Compiles an OOP Scene Graph into an Array of Structures (AoS) numpy structured array
    optimized for fast Numba raycasting.
    """

    def compile_scene(self, root_volume: 'core.geometry.volumes.Volume') -> NDArray[np.void]:
        import core.geometry.volumes
        from core.geometry.volumes import VolumeWithChilds, TransformableVolume
        from core.geometry.geometries import Box

        flat_list = []

        def dfs(volume, parent_matrix):
            if isinstance(volume, TransformableVolume):
                total_matrix = volume.total_transformation_matrix
            else:
                total_matrix = parent_matrix

            current_index = len(flat_list)
            flat_list.append((volume, total_matrix))

            child_count = 0
            if isinstance(volume, VolumeWithChilds):
                for child in volume.childs:
                    child_count += dfs(child, total_matrix)

            return child_count + 1

        identity_matrix = np.eye(4, dtype=Float)
        dfs(root_volume, identity_matrix)

        capacity = len(flat_list)
        from core.geometry.volumes import GeometryBufferDType
        buffer = np.zeros(capacity, dtype=GeometryBufferDType)

        def calc_miss(node_idx):
            vol, mat = flat_list[node_idx]
            count = 1
            if isinstance(vol, VolumeWithChilds):
                for child in vol.childs:
                    for c_idx in range(node_idx + 1, capacity):
                        if flat_list[c_idx][0] is child:
                            count += calc_miss(c_idx)
                            break
            buffer[node_idx]['miss_index'] = node_idx + count
            return count

        if capacity > 0:
            calc_miss(0)

        for i, (vol, mat) in enumerate(flat_list):
            # Shape Data
            if hasattr(vol.geometry, 'write_shape_data'):
                vol.geometry.write_shape_data(buffer['shape_data'], i)
            else:
                buffer[i]['shape_data']['shape'] = -1

            buffer[i]['volume_index'] = i

            # Matrix: World -> Local
            # Rotation
            buffer[i]['transform']['rotation']['m00'] = mat[0, 0]
            buffer[i]['transform']['rotation']['m01'] = mat[0, 1]
            buffer[i]['transform']['rotation']['m02'] = mat[0, 2]

            buffer[i]['transform']['rotation']['m10'] = mat[1, 0]
            buffer[i]['transform']['rotation']['m11'] = mat[1, 1]
            buffer[i]['transform']['rotation']['m12'] = mat[1, 2]

            buffer[i]['transform']['rotation']['m20'] = mat[2, 0]
            buffer[i]['transform']['rotation']['m21'] = mat[2, 1]
            buffer[i]['transform']['rotation']['m22'] = mat[2, 2]

            # Translation
            buffer[i]['transform']['translation']['x'] = mat[0, 3]
            buffer[i]['transform']['translation']['y'] = mat[1, 3]
            buffer[i]['transform']['translation']['z'] = mat[2, 3]

        return buffer
