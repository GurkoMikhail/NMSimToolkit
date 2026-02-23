from abc import ABC, abstractmethod
from typing import Any, Sequence, Tuple, Union

import numpy as np
from hepunits import micron
from numpy.typing import NDArray

from core.other.typing_definitions import Float, Length, Vector3D


class Geometry(ABC):
    size: Vector3D # type: ignore
    
    def __init__(self, size: Union[Sequence[Length], Vector3D]) -> None: # type: ignore
        self.size = np.array(size)

    @property
    def half_size(self) -> Vector3D: # type: ignore
        return self.size/2

    @property
    def quarter_size(self) -> Vector3D: # type: ignore
        return self.size/4

    @abstractmethod
    def check_outside(self, position: Vector3D) -> Union[bool, NDArray[np.bool_]]: # type: ignore
        pass

    @abstractmethod
    def check_inside(self, position: Vector3D) -> Union[bool, NDArray[np.bool_]]: # type: ignore
        pass

    @abstractmethod
    def cast_path(self, position: Vector3D, direction: Vector3D) -> Tuple[NDArray[Float], Union[bool, NDArray[np.bool_]]]: # type: ignore
        pass
    

class Box(Geometry):
    distance_method: str
    distance_epsilon: Length

    def __init__(self, x: Length, y: Length, z: Length, **kwds: Any) -> None:
        super().__init__([x, y, z])
        self.distance_method = 'ray_casting'
        self.distance_epsilon = 1.*micron
        args = [
            'distance_method',
            'distance_epsilon'
        ]

        for arg in args:
            if arg in kwds:
                setattr(self, arg, kwds[arg])

    def check_outside(self, position: Vector3D) -> Union[bool, NDArray[np.bool_]]: # type: ignore
        return np.max(np.abs(position) - self.half_size, axis=1) > 0

    def check_inside(self, position: Vector3D) -> Union[bool, NDArray[np.bool_]]: # type: ignore
        return np.max(np.abs(position) - self.half_size, axis=1) <= 0

    def cast_path(self, position: Vector3D, direction: Vector3D) -> Tuple[NDArray[Float], Union[bool, NDArray[np.bool_]]]: # type: ignore
        return getattr(self, self.distance_method)(position, direction)

    def ray_marching(self, position: Vector3D, *args: Any) -> Tuple[NDArray[Float], Union[bool, NDArray[np.bool_]]]: # type: ignore
        q = np.abs(position) - self.half_size
        maxXYZ = q.max(axis=1)
        lengthq = np.linalg.norm(np.where(q > 0, q, 0.), axis=1)
        distance = lengthq + np.where(maxXYZ < 0, maxXYZ, 0.)
        inside = distance < 0
        distance = np.abs(distance) + self.distance_epsilon
        return distance, inside

    def ray_casting(self, position: Vector3D, direction: Vector3D) -> Tuple[NDArray[Float], Union[bool, NDArray[np.bool_]]]:
        inside = self.check_inside(position)
        norm_pos = -position / direction
        norm_size = np.abs(self.half_size / direction)
        tmin = np.max(norm_pos - norm_size, axis=1)
        tmax = np.min(norm_pos + norm_size, axis=1)
        distance = np.where(tmax > tmin, tmin, np.inf)
        distance[inside] = tmax[inside]
        distance[distance < 0] = np.inf
        distance += self.distance_epsilon
        return distance, inside
