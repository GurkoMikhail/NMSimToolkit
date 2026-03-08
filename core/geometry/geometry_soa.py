from typing import NamedTuple, Any, List, Optional
import numpy as np
from numpy.typing import NDArray

from core.other.typing_definitions import Float, Length, ShapeID, Index
from core.other.transform_soa import TransformSoA, Matrix3x3SoA
from core.other.vectors_soa import Vector3DSoA


class ShapeParameters(NamedTuple):
    """
    Universal parameters for geometry dimensions
    """
    param_0: NDArray[Float]
    param_1: NDArray[Float]
    param_2: NDArray[Float]
    param_3: NDArray[Float]

    def validate(self) -> None:
        arrays = [self.param_0, self.param_1, self.param_2, self.param_3]
        for arr in arrays:
            if arr.ndim != 1:
                raise ValueError("ShapeParameters arrays must be 1-dimensional.")
        length = arrays[0].shape[0]
        for arr in arrays:
            if arr.shape[0] != length:
                raise ValueError("ShapeParameters arrays must have the same length.")

    @property
    def capacity(self) -> int:
        return self.param_0.shape[0]

class ShapeBuffer(NamedTuple):
    """
    Structure of Arrays (SoA) representation for the shape.
    """
    shape: NDArray[ShapeID]
    shape_parameters: ShapeParameters

    def validate(self) -> None:
        self.shape_parameters.validate()
        if self.shape.ndim != 1:
            raise ValueError("ShapeBuffer arrays must be 1-dimensional.")
        if self.shape.shape[0] != self.shape_parameters.capacity:
            raise ValueError("ShapeBuffer arrays must have the same length.")

    @property
    def capacity(self) -> int:
        return self.shape.shape[0]

class GeometryBuffer(NamedTuple):
    """
    Structure of Arrays (SoA) representation for the entire Scene Graph.
    Contains flat 1D C-contiguous numpy arrays.
    """
    shape_buffer: ShapeBuffer

    # Inverse transformation parameters (World -> Local)
    transform: TransformSoA

    # Index of the next geometry to jump to if ray misses this volume (Frustum Culling)
    miss_index: NDArray[Index]

    # Index for mapping back to OOP Volumes
    volume_index: NDArray[Index]

    def validate(self) -> None:
        """
        Validates that all arrays within GeometryBuffer have matching capacities
        and are 1-dimensional.
        """
        self.transform.validate()
        self.shape_buffer.validate()

        arrays = [
            self.miss_index,
            self.volume_index
        ]

        for arr in arrays:
            if arr.ndim != 1:
                raise ValueError("All arrays in GeometryBuffer must be 1-dimensional.")

        capacity = self.shape_buffer.capacity
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
    shape = np.zeros(capacity, dtype=ShapeID)
    param_0 = np.zeros(capacity, dtype=Float)
    param_1 = np.zeros(capacity, dtype=Float)
    param_2 = np.zeros(capacity, dtype=Float)
    param_3 = np.zeros(capacity, dtype=Float)
    miss_index = np.zeros(capacity, dtype=Index)
    volume_index = np.zeros(capacity, dtype=Index)

    # Transform arrays
    m00 = np.zeros(capacity, dtype=Float)
    m01 = np.zeros(capacity, dtype=Float)
    m02 = np.zeros(capacity, dtype=Float)
    m10 = np.zeros(capacity, dtype=Float)
    m11 = np.zeros(capacity, dtype=Float)
    m12 = np.zeros(capacity, dtype=Float)
    m20 = np.zeros(capacity, dtype=Float)
    m21 = np.zeros(capacity, dtype=Float)
    m22 = np.zeros(capacity, dtype=Float)
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
            shape[i] = 0
            # half_size
            param_0[i] = vol.geometry.half_size[0]
            param_1[i] = vol.geometry.half_size[1]
            param_2[i] = vol.geometry.half_size[2]
            param_3[i] = getattr(vol.geometry, 'distance_epsilon', Float(1e-3))
        else:
            # Fallback for unsupported geometries
            shape[i] = -1

        # Volume index
        volume_index[i] = i # temporary: mapping to flat_list index for backward compatibility

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
        m00[i] = mat[0, 0]
        m01[i] = mat[0, 1]
        m02[i] = mat[0, 2]
        m10[i] = mat[1, 0]
        m11[i] = mat[1, 1]
        m12[i] = mat[1, 2]
        m20[i] = mat[2, 0]
        m21[i] = mat[2, 1]
        m22[i] = mat[2, 2]

        tr_x[i] = mat[0, 3]
        tr_y[i] = mat[1, 3]
        tr_z[i] = mat[2, 3]

    matrix_soa = Matrix3x3SoA(
        m00=m00, m01=m01, m02=m02,
        m10=m10, m11=m11, m12=m12,
        m20=m20, m21=m21, m22=m22
    )
    vector_soa = Vector3DSoA(x=tr_x, y=tr_y, z=tr_z)

    transform_soa = TransformSoA(
        rotation=matrix_soa,
        translation=vector_soa
    )

    shape_parameters = ShapeParameters(
        param_0=param_0,
        param_1=param_1,
        param_2=param_2,
        param_3=param_3
    )

    shape_buffer = ShapeBuffer(
        shape=shape,
        shape_parameters=shape_parameters
    )

    buffer = GeometryBuffer(
        shape_buffer=shape_buffer,
        transform=transform_soa,
        miss_index=miss_index,
        volume_index=volume_index
    )
    buffer.validate()

    return buffer
