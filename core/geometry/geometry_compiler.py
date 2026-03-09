import numpy as np
from typing import Any
from numpy.typing import NDArray

from core.other.typing_definitions import Float, ShapeID, Index


# Define Structured Array Types (AoS)

# 1. Transform Data
TransformDType = np.dtype([
    ('m00', Float), ('m01', Float), ('m02', Float),
    ('m10', Float), ('m11', Float), ('m12', Float),
    ('m20', Float), ('m21', Float), ('m22', Float),
    ('tx', Float), ('ty', Float), ('tz', Float)
])

# 2. Shape Data
# "В ShapeParameters включить индекс формы геометрии (shape) и дать более исчерпывающее название."
ShapeDataDType = np.dtype([
    ('shape', ShapeID),
    ('param_0', Float),
    ('param_1', Float),
    ('param_2', Float),
    ('param_3', Float)
])

# 3. Geometry Buffer Element
GeometryBufferDType = np.dtype([
    ('shape_data', ShapeDataDType),
    ('transform', TransformDType),
    ('miss_index', Index),
    ('volume_index', Index)
])


class GeometryCompiler:
    """
    Compiles an OOP Scene Graph into an Array of Structures (AoS) numpy structured array
    optimized for fast Numba raycasting.
    """

    def compile_scene(self, root_volume: Any) -> NDArray[Any]:
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
            if isinstance(vol.geometry, Box):
                buffer[i]['shape_data']['shape'] = 0
                buffer[i]['shape_data']['param_0'] = vol.geometry.half_size[0]
                buffer[i]['shape_data']['param_1'] = vol.geometry.half_size[1]
                buffer[i]['shape_data']['param_2'] = vol.geometry.half_size[2]
                buffer[i]['shape_data']['param_3'] = getattr(vol.geometry, 'distance_epsilon', Float(1e-3))
            else:
                buffer[i]['shape_data']['shape'] = -1

            buffer[i]['volume_index'] = i

            # Matrix: World -> Local
            # Rotation
            buffer[i]['transform']['m00'] = mat[0, 0]
            buffer[i]['transform']['m01'] = mat[0, 1]
            buffer[i]['transform']['m02'] = mat[0, 2]

            buffer[i]['transform']['m10'] = mat[1, 0]
            buffer[i]['transform']['m11'] = mat[1, 1]
            buffer[i]['transform']['m12'] = mat[1, 2]

            buffer[i]['transform']['m20'] = mat[2, 0]
            buffer[i]['transform']['m21'] = mat[2, 1]
            buffer[i]['transform']['m22'] = mat[2, 2]

            # Translation
            buffer[i]['transform']['tx'] = mat[0, 3]
            buffer[i]['transform']['ty'] = mat[1, 3]
            buffer[i]['transform']['tz'] = mat[2, 3]

        return buffer
