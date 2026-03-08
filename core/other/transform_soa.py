from typing import NamedTuple
import numpy as np
from numpy.typing import NDArray

from core.other.typing_definitions import Float
from core.other.vectors_soa import Vector3DSoA


class Matrix3x3SoA(NamedTuple):
    """
    Structure of Arrays (SoA) representation for a 3x3 matrix.
    Contains flat 1D C-contiguous numpy arrays.
    """
    m00: NDArray[Float]
    m01: NDArray[Float]
    m02: NDArray[Float]
    m10: NDArray[Float]
    m11: NDArray[Float]
    m12: NDArray[Float]
    m20: NDArray[Float]
    m21: NDArray[Float]
    m22: NDArray[Float]

    def validate(self) -> None:
        arrays = [
            self.m00, self.m01, self.m02,
            self.m10, self.m11, self.m12,
            self.m20, self.m21, self.m22
        ]
        for arr in arrays:
            if arr.ndim != 1:
                raise ValueError("Matrix3x3SoA arrays must be 1-dimensional.")

        length = arrays[0].shape[0]
        for arr in arrays[1:]:
            if arr.shape[0] != length:
                raise ValueError("Matrix3x3SoA arrays must have the same length.")

    @property
    def capacity(self) -> int:
        return self.m00.shape[0]


class TransformSoA(NamedTuple):
    """
    Structure of Arrays (SoA) representation for 3D affine transformations.
    Contains flat 1D C-contiguous numpy arrays for a 3x3 rotation matrix and a 3D translation vector.
    Used for World -> Local coordinate conversions without np.matmul.
    """
    rotation: Matrix3x3SoA
    translation: Vector3DSoA

    def validate(self) -> None:
        """
        Validates that the TransformSoA contains 1-dimensional arrays
        of equal length.
        """
        self.rotation.validate()
        self.translation.validate()

        if self.rotation.capacity != self.translation.x.shape[0]:
            raise ValueError("TransformSoA rotation and translation must have the same length.")

    @property
    def capacity(self) -> int:
        return self.rotation.capacity
