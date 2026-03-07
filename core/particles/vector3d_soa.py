import numpy as np
from typing import NamedTuple
from numpy.typing import NDArray
from core.other.typing_definitions import Float

class Vector3DSoA(NamedTuple):
    x: NDArray[Float]
    y: NDArray[Float]
    z: NDArray[Float]

    @classmethod
    def create(cls, x: NDArray[Float], y: NDArray[Float], z: NDArray[Float]) -> 'Vector3DSoA':
        if not (x.size == y.size == z.size):
            raise ValueError(f"Arrays must have identical sizes. Got sizes: x={x.size}, y={y.size}, z={z.size}")
        return cls(x, y, z)
