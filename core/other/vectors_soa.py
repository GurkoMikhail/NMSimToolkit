from typing import NamedTuple
import numpy as np
from numpy.typing import NDArray

from core.other.typing_definitions import Float


from numba import njit


@njit(inline='always')
def _rotate_direction_scalar(dir_x: Float, dir_y: Float, dir_z: Float, theta: Float, phi: Float) -> tuple[Float, Float, Float]:
    """
    Applies a theta and phi rotation to a 3D unit direction vector.
    Calculations strictly inline without allocations.
    """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    delta1 = sin_theta * np.cos(phi)
    delta2 = sin_theta * np.sin(phi)

    delta = 1.0
    if dir_z < 0.0:
        delta = -1.0

    b = dir_x * delta1 + dir_y * delta2
    abs_z = np.abs(dir_z)
    tmp = cos_theta - b / (1.0 + abs_z)

    new_dir_x = dir_x * tmp + delta1
    new_dir_y = dir_y * tmp + delta2
    new_dir_z = dir_z * cos_theta - delta * b

    return new_dir_x, new_dir_y, new_dir_z


class Vector3DSoA(NamedTuple):
    """
    Structure of Arrays (SoA) representation for 3D vectors.
    Contains flat 1D C-contiguous numpy arrays for X, Y, and Z components.
    """
    x: NDArray[Float]
    y: NDArray[Float]
    z: NDArray[Float]

    def validate(self) -> None:
        """
        Validates that the Vector3DSoA contains 1-dimensional arrays
        of equal length.
        """
        if self.x.ndim != 1 or self.y.ndim != 1 or self.z.ndim != 1:
            raise ValueError("Vector3DSoA arrays must be 1-dimensional.")

        length = self.x.shape[0]
        if self.y.shape[0] != length or self.z.shape[0] != length:
            raise ValueError("Vector3DSoA arrays must have the same length.")

    @classmethod
    def allocate(cls, capacity: int, dtype: np.dtype = Float) -> 'Vector3DSoA':
        """
        Allocates a Vector3DSoA with uninitialized memory for the given capacity.
        """
        buffer = cls(
            x=np.empty(capacity, dtype=dtype),
            y=np.empty(capacity, dtype=dtype),
            z=np.empty(capacity, dtype=dtype)
        )
        buffer.validate()
        return buffer
