import numpy as np
from numba import njit
from abc import ABC, abstractmethod
from numpy import inf
from hepunits import*
from core.numba_backend.ray_cast import numba_ray_casting
from core.numba_backend.check_inside import numba_check_inside


class Geometry(ABC):
    
    def __init__(self, size):
        self.size = np.array(size)

    @property
    def half_size(self):
        return self.size/2

    @property
    def quarter_size(self):
        return self.size/4

    @abstractmethod
    def check_outside(self, position):
        pass

    @abstractmethod
    def check_inside(self, position):
        pass

    @abstractmethod
    def cast_path(self, position, direction):
        pass
    

class Box(Geometry):

    def __init__(self, x, y, z, **kwds):
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

    def check_outside(self, position):
        return np.max(np.abs(position) - self.half_size, axis=1) > 0

    def check_inside(self, position):
        return numba_check_inside(position, self.half_size)

    def cast_path(self, position, direction):
        return getattr(self, self.distance_method)(position, direction)

    def ray_marching(self, position, *args):
        q = np.abs(position) - self.half_size
        maxXYZ = q.max(axis=1)
        lengthq = np.linalg.norm(np.where(q > 0, q, 0.), axis=1)
        distance = lengthq + np.where(maxXYZ < 0, maxXYZ, 0.)
        inside = distance < 0
        distance = np.abs(distance) + self.distance_epsilon
        return distance, inside

    def ray_casting(self, position, direction):
        return numba_ray_casting(position, direction, np.array(self.half_size), self.distance_epsilon)
