from typing import NamedTuple
from numpy.typing import NDArray
from core.other.typing_definitions import Float

class Vector3DSoA(NamedTuple):
    x: NDArray[Float]
    y: NDArray[Float]
    z: NDArray[Float]
    @classmethod
    def create(cls, x: NDArray[Float], y: NDArray[Float], z: NDArray[Float]) -> 'Vector3DSoA': ...
