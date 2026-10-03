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
    ('param_2', Float),
    ('param_3', Float),
    ('param_4', Float),
    ('param_5', Float),
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


class PeriodicHexPrism(Geometry):
    """
    Геометрия бесконечной периодической гексагональной решетки шестиугольных каналов (призм),
    ограниченная по оси Z габаритами коллиматора.
    """
    _hole_diameter: Float
    _septa: Float
    _x_period: Float
    _y_period: Float
    _channel_half_width: Float
    _channel_side_limit: Float
    _cell_half_width: Float

    def __init__(
        self,
        size: Union[Sequence[Length], Vector3D],
        hole_diameter: Float,
        septa: Float,
    ) -> None:
        super().__init__(size)
        if any(dimension <= 0.0 for dimension in self.size):
            raise ValueError(f"Габаритные размеры должны быть строго положительными (> 0), получено {size}")
        if Float(hole_diameter) <= 0.0:
            raise ValueError(f"Диаметр отверстия должен быть строго положительным (> 0), получено {hole_diameter}")
        if Float(septa) <= 0.0:
            raise ValueError(f"Толщина септы должна быть строго положительной (> 0), получено {septa}")
        self._hole_diameter = Float(hole_diameter)
        self._septa = Float(septa)
        self._compute_parameters()

    def _compute_parameters(self) -> None:
        self._x_period = Float(self._hole_diameter + self._septa)
        self._y_period = Float(np.sqrt(3.0) * self._x_period)
        self._channel_half_width = Float(0.5 * self._hole_diameter)
        self._channel_side_limit = Float(self._hole_diameter)
        self._cell_half_width = Float(0.5 * self._x_period)

    @property
    def hole_diameter(self) -> Float:
        """Диаметр гексагонального канала между противоположными параллельными гранями."""
        return self._hole_diameter

    @hole_diameter.setter
    def hole_diameter(self, value: Float) -> None:
        float_value = Float(value)
        if float_value <= 0.0:
            raise ValueError(f"Диаметр отверстия должен быть строго положительным (> 0), получено {value}")
        self._hole_diameter = float_value
        self._compute_parameters()

    @property
    def septa(self) -> Float:
        """Толщина перегородки (септы) между каналами."""
        return self._septa

    @septa.setter
    def septa(self, value: Float) -> None:
        float_value = Float(value)
        if float_value <= 0.0:
            raise ValueError(f"Толщина септы должна быть строго положительной (> 0), получено {value}")
        self._septa = float_value
        self._compute_parameters()

    def write_shape_data(self, shape_data_array: NDArray[np.void], index: int) -> None:
        shape_data_array[index]['shape'] = 1
        shape_data_array[index]['param_0'] = self._x_period
        shape_data_array[index]['param_1'] = self._y_period
        shape_data_array[index]['param_2'] = self._channel_half_width
        shape_data_array[index]['param_3'] = self._channel_side_limit
        shape_data_array[index]['param_4'] = Float(self.half_size[2])
        shape_data_array[index]['param_5'] = self._cell_half_width

