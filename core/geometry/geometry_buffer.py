from typing import NamedTuple
import numpy as np
from numpy.typing import NDArray

from core.other.typing_definitions import Float, Index
from core.other.vectors_soa import Vector3DSoA


class TransformSoA(NamedTuple):
    """
    Structure of Arrays representation for World -> Local Transformations.
    Stores flattened rotation matrix elements (r00 to r22) and translation
    vector elements (tx, ty, tz) for N objects to explicitly unroll
    3x3 and 3x1 matrix operations inside scalar Numba kernels.
    """
    r00: NDArray[Float]
    r01: NDArray[Float]
    r02: NDArray[Float]
    r10: NDArray[Float]
    r11: NDArray[Float]
    r12: NDArray[Float]
    r20: NDArray[Float]
    r21: NDArray[Float]
    r22: NDArray[Float]

    tx: NDArray[Float]
    ty: NDArray[Float]
    tz: NDArray[Float]

    @property
    def capacity(self) -> int:
        return self.tx.shape[0]

    def validate(self) -> None:
        arrays = [
            self.r00, self.r01, self.r02,
            self.r10, self.r11, self.r12,
            self.r20, self.r21, self.r22,
            self.tx, self.ty, self.tz
        ]
        cap = self.capacity
        for arr in arrays:
            if arr.ndim != 1:
                raise ValueError("Transform arrays must be 1-dimensional.")
            if arr.shape[0] != cap:
                raise ValueError("Transform arrays must have identical capacities.")


class GeometryBuffer(NamedTuple):
    """
    Structure of Arrays flat representation of the Scene Graph.
    Enables Frustum Culling through Numba `while` loop jumping.
    """
    # Unique IDs assigned during Depth-First Search compilation
    volume_id: NDArray[Index]

    # 0 = Box, 1 = Cylinder, 2 = Sphere, etc.
    shape_type: NDArray[np.int32]

    # Primitive Dimensions. For Box: x=half_x, y=half_y, z=half_z
    sizes: Vector3DSoA

    # World -> Local Transformation components
    transform: TransformSoA

    # ID of the material filling this volume
    material_id: NDArray[Index]

    # Frustum Culling Jump Link:
    # If a ray misses this volume entirely, the while-loop index jumps to miss_index[i]
    miss_index: NDArray[Index]

    @property
    def capacity(self) -> int:
        return self.volume_id.shape[0]

    def validate(self) -> None:
        self.sizes.validate()
        self.transform.validate()

        arrays = [
            self.volume_id,
            self.shape_type,
            self.material_id,
            self.miss_index
        ]

        cap = self.capacity
        for arr in arrays:
            if arr.ndim != 1:
                raise ValueError("GeometryBuffer arrays must be 1-dimensional.")
            if arr.shape[0] != cap:
                raise ValueError("GeometryBuffer arrays must have identical capacities.")

        if self.sizes.x.shape[0] != cap or self.transform.capacity != cap:
            raise ValueError("GeometryBuffer sizes and transforms must match the base capacity.")
