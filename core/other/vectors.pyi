import numpy as np
from typing import NamedTuple
from numpy.typing import NDArray
from core.other.typing_definitions import Float


class Vector3D(NamedTuple):
    x: NDArray[Float]
    y: NDArray[Float]
    z: NDArray[Float]

    def validate(self) -> None: ...
    @classmethod
    def allocate(cls, capacity: int, dtype: np.dtype = ...) -> 'Vector3D': ...
