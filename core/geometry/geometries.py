from abc import ABC, abstractmethod
from typing import Any, Sequence, Tuple, Union

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

    @abstractmethod

    @abstractmethod

    @abstractmethod
    def write_shape_data(self, shape_data_array: NDArray[np.void], index: int) -> None:
        pass

class Box(Geometry):
    distance_method: str
    distance_epsilon: Length

    def __init__(self, x: Length, y: Length, z: Length, **kwds: Any) -> None:
        super().__init__([x, y, z])
        self.distance_method = 'ray_casting'
        self.distance_epsilon = Float(1. * units.micron)
        args = [
            'distance_method',
            'distance_epsilon'
        ]

        for arg in args:
            if arg in kwds:
                setattr(self, arg, kwds[arg])




    def write_shape_data(self, shape_data_array: NDArray[np.void], index: int) -> None:
        shape_data_array[index]['shape'] = 0
        shape_data_array[index]['param_0'] = self.half_size[0]
        shape_data_array[index]['param_1'] = self.half_size[1]
        shape_data_array[index]['param_2'] = self.half_size[2]

