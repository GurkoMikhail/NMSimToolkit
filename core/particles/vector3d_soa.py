import numpy as np
from typing import NamedTuple
from numpy.typing import NDArray
from core.other.typing_definitions import Float

class Vector3DSoA(NamedTuple):
    x: NDArray[Float]
    y: NDArray[Float]
    z: NDArray[Float]

    def _validate(self) -> None:
        if not (self.x.size == self.y.size == self.z.size):
            raise ValueError(f"Arrays must have identical sizes. Got sizes: x={self.x.size}, y={self.y.size}, z={self.z.size}")
