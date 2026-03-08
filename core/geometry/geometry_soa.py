from typing import NamedTuple, Any, List, Optional
import numpy as np
from numpy.typing import NDArray

from core.other.typing_definitions import Float, Length
from core.other.transform_soa import TransformSoA


class GeometryBuffer(NamedTuple):
    """
    Structure of Arrays (SoA) representation for the entire Scene Graph.
    Contains flat 1D C-contiguous numpy arrays.
    """
    # 0 for Box, 1 for Cylinder (future), etc.
    geom_type: NDArray[np.int32]

    # Universal parameters for geometry dimensions
    param_0: NDArray[Float]
    param_1: NDArray[Float]
    param_2: NDArray[Float]
    param_3: NDArray[Float]

    # Inverse transformation parameters (World -> Local)
    transform: TransformSoA

    # Index of the next geometry to jump to if ray misses this volume (Frustum Culling)
    miss_index: NDArray[np.int32]

    # Material index for mapping or ID to map back to OOP Volumes
    material_index: NDArray[np.int32]

    def validate(self) -> None:
        """
        Validates that all arrays within GeometryBuffer have matching capacities
        and are 1-dimensional.
        """
        self.transform.validate()

        arrays = [
            self.geom_type,
            self.param_0,
            self.param_1,
            self.param_2,
            self.param_3,
            self.miss_index,
            self.material_index
        ]

        for arr in arrays:
            if arr.ndim != 1:
                raise ValueError("All arrays in GeometryBuffer must be 1-dimensional.")

        capacity = self.geom_type.shape[0]
        for arr in arrays:
            if arr.shape[0] != capacity:
                raise ValueError("All arrays in GeometryBuffer must have the same length (capacity).")

        if self.transform.capacity != capacity:
            raise ValueError("TransformSoA arrays in GeometryBuffer must have the same length as base arrays.")


def compile_scene(root_volume: Any) -> GeometryBuffer:
    """
    Compiles an OOP Scene Graph starting from root_volume into a flat GeometryBuffer.
    Performs DFS traversal, extracting inverse matrices, geometry parameters,
    and calculating miss_indices for Frustum Culling.
    """
    from core.geometry.volumes import VolumeWithChilds, TransformableVolume
    from core.geometry.geometries import Box

    # List to hold tuples of (volume, inverse_matrix, miss_index_placeholder)
    flat_list = []

    def dfs(volume, parent_matrix):
        if isinstance(volume, TransformableVolume):
            # volume.transformation_matrix is Local -> Parent
            # To get World -> Local, we need the inverse of the Total Transformation Matrix.
            # But wait, total_transformation_matrix is World -> Local! (Memory states: total_transformation_matrix stored in OOP volumes.py objects is already configured as a World -> Local transformation matrix. Do not invert it)
            total_matrix = volume.total_transformation_matrix
        else:
            total_matrix = parent_matrix

        current_index = len(flat_list)
        flat_list.append((volume, total_matrix))

        child_count = 0
        if isinstance(volume, VolumeWithChilds):
            for child in volume.childs:
                child_count += dfs(child, total_matrix)

        # The miss_index is the index of the node right after this subtree
        miss_index = current_index + child_count + 1
        return child_count + 1

    # Start DFS
    # Root volume usually doesn't have a transformation, so we pass identity
    identity_matrix = np.eye(4, dtype=Float)
    dfs(root_volume, identity_matrix)

    capacity = len(flat_list)

    # Initialize SoA arrays
    geom_type = np.zeros(capacity, dtype=np.int32)
    param_0 = np.zeros(capacity, dtype=Float)
    param_1 = np.zeros(capacity, dtype=Float)
    param_2 = np.zeros(capacity, dtype=Float)
    param_3 = np.zeros(capacity, dtype=Float)
    miss_index = np.zeros(capacity, dtype=np.int32)
    material_index = np.zeros(capacity, dtype=np.int32)

    # Transform arrays
    rot_00 = np.zeros(capacity, dtype=Float)
    rot_01 = np.zeros(capacity, dtype=Float)
    rot_02 = np.zeros(capacity, dtype=Float)
    rot_10 = np.zeros(capacity, dtype=Float)
    rot_11 = np.zeros(capacity, dtype=Float)
    rot_12 = np.zeros(capacity, dtype=Float)
    rot_20 = np.zeros(capacity, dtype=Float)
    rot_21 = np.zeros(capacity, dtype=Float)
    rot_22 = np.zeros(capacity, dtype=Float)
    tr_x = np.zeros(capacity, dtype=Float)
    tr_y = np.zeros(capacity, dtype=Float)
    tr_z = np.zeros(capacity, dtype=Float)

    # Re-run a simpler index calculation
    def calc_miss(node_idx):
        vol, mat = flat_list[node_idx]
        count = 1
        if isinstance(vol, VolumeWithChilds):
            for child in vol.childs:
                # Find child index
                for c_idx in range(node_idx + 1, capacity):
                    if flat_list[c_idx][0] is child:
                        count += calc_miss(c_idx)
                        break
        miss_index[node_idx] = node_idx + count
        return count

    if capacity > 0:
        calc_miss(0)

    for i, (vol, mat) in enumerate(flat_list):
        # Determine geometry type
        if isinstance(vol.geometry, Box):
            geom_type[i] = 0
            # half_size
            param_0[i] = vol.geometry.half_size[0]
            param_1[i] = vol.geometry.half_size[1]
            param_2[i] = vol.geometry.half_size[2]
            param_3[i] = getattr(vol.geometry, 'distance_epsilon', Float(1e-3))
        else:
            # Fallback for unsupported geometries
            geom_type[i] = -1

        # Material index
        material_index[i] = i # temporary: mapping to flat_list index for backward compatibility

        # Matrix is World -> Local. The total_transformation_matrix is defined as 4x4.
        # [ R R R Tx ]
        # [ R R R Ty ]
        # [ R R R Tz ]
        # [ 0 0 0 1  ]
        # In `volumes.py`:
        # "np.matmul(local_position, transformation_matrix.T.astype(position.dtype), out=local_position)"
        # Note: if local_position is [x, y, z, 1] row vector, local_position @ mat.T = (mat @ local_position.T).T
        # This means mat is applied as M * v column vector.
        # So mat[0:3, 0:3] is rotation, mat[0:3, 3] is translation.
        rot_00[i] = mat[0, 0]
        rot_01[i] = mat[0, 1]
        rot_02[i] = mat[0, 2]
        rot_10[i] = mat[1, 0]
        rot_11[i] = mat[1, 1]
        rot_12[i] = mat[1, 2]
        rot_20[i] = mat[2, 0]
        rot_21[i] = mat[2, 1]
        rot_22[i] = mat[2, 2]

        tr_x[i] = mat[0, 3]
        tr_y[i] = mat[1, 3]
        tr_z[i] = mat[2, 3]

    transform_soa = TransformSoA(
        rot_00=rot_00, rot_01=rot_01, rot_02=rot_02,
        rot_10=rot_10, rot_11=rot_11, rot_12=rot_12,
        rot_20=rot_20, rot_21=rot_21, rot_22=rot_22,
        tr_x=tr_x, tr_y=tr_y, tr_z=tr_z
    )

    buffer = GeometryBuffer(
        geom_type=geom_type,
        param_0=param_0,
        param_1=param_1,
        param_2=param_2,
        param_3=param_3,
        transform=transform_soa,
        miss_index=miss_index,
        material_index=material_index
    )
    buffer.validate()

    return buffer, flat_list
