from typing import NamedTuple
import numpy as np
from numpy.typing import NDArray

from core.other.typing_definitions import Float


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
