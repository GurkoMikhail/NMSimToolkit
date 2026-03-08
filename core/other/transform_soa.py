from typing import NamedTuple
import numpy as np
from numpy.typing import NDArray

from core.other.typing_definitions import Float


class TransformSoA(NamedTuple):
    """
    Structure of Arrays (SoA) representation for 3D affine transformations.
    Contains flat 1D C-contiguous numpy arrays for a 3x3 rotation matrix and a 3D translation vector.
    Used for World -> Local coordinate conversions without np.matmul.
    """
    # Rotation Matrix 3x3 (row-major)
    rot_00: NDArray[Float]
    rot_01: NDArray[Float]
    rot_02: NDArray[Float]
    rot_10: NDArray[Float]
    rot_11: NDArray[Float]
    rot_12: NDArray[Float]
    rot_20: NDArray[Float]
    rot_21: NDArray[Float]
    rot_22: NDArray[Float]

    # Translation Vector 3D
    tr_x: NDArray[Float]
    tr_y: NDArray[Float]
    tr_z: NDArray[Float]

    def validate(self) -> None:
        """
        Validates that the TransformSoA contains 1-dimensional arrays
        of equal length.
        """
        arrays = [
            self.rot_00, self.rot_01, self.rot_02,
            self.rot_10, self.rot_11, self.rot_12,
            self.rot_20, self.rot_21, self.rot_22,
            self.tr_x, self.tr_y, self.tr_z
        ]

        for arr in arrays:
            if arr.ndim != 1:
                raise ValueError("TransformSoA arrays must be 1-dimensional.")

        length = arrays[0].shape[0]
        for arr in arrays[1:]:
            if arr.shape[0] != length:
                raise ValueError("TransformSoA arrays must have the same length.")

    @property
    def capacity(self) -> int:
        return self.rot_00.shape[0]
