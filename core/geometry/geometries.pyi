import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Sequence, Any
from numpy.typing import NDArray
from core.other.typing_definitions import Length, Vector3D, Float

class Geometry(ABC):
    size: Vector3D
    def __init__(self, size: Union[Sequence[Length], Vector3D]) -> None: ...
    @property
    def half_size(self) -> Vector3D: ...
    @property
    def quarter_size(self) -> Vector3D: ...
    @abstractmethod
    def write_shape_data(self, shape_data_array: NDArray[np.void], index: int) -> None: ...

class Box(Geometry):
    distance_method: str
    distance_epsilon: Length
    def __init__(self, x: Length, y: Length, z: Length, **kwds: Any) -> None: ...
    def write_shape_data(self, shape_data_array: NDArray[np.void], index: int) -> None: ...
