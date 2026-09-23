from abc import ABC, abstractmethod
from typing import Any, Sequence, Union

import numpy as np
import hepunits as units
from numpy.typing import NDArray

from core.other.typing_definitions import Float, Length, Vector3D, ShapeID

ShapeDataDType = np.dtype([
    ('shape', ShapeID),
    ('param_0', Float),
    ('param_1', Float),
    ('param_2', Float)
])

class Geometry(ABC):
    size: Vector3D
    
    def __init__(self, size: Union[Sequence[Length], Vector3D]) -> None:
        self.size = np.array(size)

    @property
    def half_size(self) -> Vector3D:
        return self.size/2

    @property
    def quarter_size(self) -> Vector3D:
        return self.size/4

    @abstractmethod
    def write_shape_data(self, shape_data_array: NDArray[np.void], index: int) -> None:
        pass

class Box(Geometry):
    distance_method: str
    distance_epsilon: Length

    def __init__(
        self,
        x: Length,
        y: Length,
        z: Length,
        distance_method: str = 'ray_casting',
        distance_epsilon: Float = Float(1. * units.micron),
        **kwds: Any
    ) -> None:
        super().__init__([x, y, z])
        self.distance_method = str(kwds.get('distance_method', distance_method))
        self.distance_epsilon = Float(kwds.get('distance_epsilon', distance_epsilon))

    @property
    def x(self) -> Length:
        return self.size[0]

    @property
    def y(self) -> Length:
        return self.size[1]

    @property
    def z(self) -> Length:
        return self.size[2]

    def write_shape_data(self, shape_data_array: NDArray[np.void], index: int) -> None:
        shape_data_array[index]['shape'] = 0
        shape_data_array[index]['param_0'] = self.half_size[0]
        shape_data_array[index]['param_1'] = self.half_size[1]
        shape_data_array[index]['param_2'] = self.half_size[2]

