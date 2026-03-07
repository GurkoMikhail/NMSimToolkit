import numpy as np
from typing import NamedTuple, Any
from numpy.typing import NDArray

from core.other.typing_definitions import Float, Index

class GeometryBuffer(NamedTuple):
    """
    Structure of Arrays (SoA) representation of a flattened Scene Graph.
    """
    node_type: NDArray[np.uint8]        # e.g., 0: Box, ...
    material_id: NDArray[np.uint32]     # Index linking to a Material list
    miss_index: NDArray[Index]          # Jump index if ray misses this bounding box (-1 if no more nodes)

    half_size_x: NDArray[Float]         # Precomputed geometry parameters
    half_size_y: NDArray[Float]
    half_size_z: NDArray[Float]

    # Pre-calculated INVERSE transformation matrix elements (World -> Local)
    # R_xx, R_xy, R_xz
    # R_yx, R_yy, R_yz
    # R_zx, R_zy, R_zz
    # T_x,  T_y,  T_z
    # Formula: local_p = R @ p + T
    # (Since transformations in OOP are often P = P @ T.T, we pre-invert them)
    R_xx: NDArray[Float]
    R_xy: NDArray[Float]
    R_xz: NDArray[Float]

    R_yx: NDArray[Float]
    R_yy: NDArray[Float]
    R_yz: NDArray[Float]

    R_zx: NDArray[Float]
    R_zy: NDArray[Float]
    R_zz: NDArray[Float]

    T_x: NDArray[Float]
    T_y: NDArray[Float]
    T_z: NDArray[Float]

    def _validate(self) -> None:
        sizes = [getattr(self, field).size for field in self._fields]
        if len(set(sizes)) > 1:
            raise ValueError(f"All arrays in GeometryBuffer must have identical sizes. Got sizes: {sizes}")
