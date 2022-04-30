import numpy as np
from numpy import inf
from hepunits import*

class Geometry:
    
    def __init__(self, size):
        self.size = np.array(size)

    @property
    def halfSize(self):
        return self.size/2

    @property
    def quarterSize(self):
        return self.size/4

    def checkOutside(self, position):
        pass

    def checkInside(self, position):
        pass

    def castPath(self, position, direction):
        pass
    

class Box(Geometry):

    def __init__(self, x, y, z, **kwds):
        super().__init__([x, y, z])
        self.distanceMethod = 'rayCasting'
        self.distanceEpsilon = 1.*micron
        args = [
            'distanceMethod',
            'distanceEpsilon'
        ]

        for arg in args:
            if arg in kwds:
                setattr(self, arg, kwds[arg])

    def checkOutside(self, position):
        return np.max(np.abs(position) - self.halfSize, axis=1) > 0

    def checkInside(self, position):
        return np.max(np.abs(position) - self.halfSize, axis=1) <= 0

    def castPath(self, position, direction):
        return getattr(self, self.distanceMethod)(position, direction)

    def rayMarching(self, position, *args):
        q = np.abs(position) - self.halfSize
        maxXYZ = q.max(axis=1)
        lengthq = np.linalg.norm(np.where(q > 0, q, 0.), axis=1)
        distance = lengthq + np.where(maxXYZ < 0, maxXYZ, 0.)
        inside = distance < 0
        distance = np.abs(distance) + self.distanceEpsilon
        return distance, inside

    def rayCasting(self, position, direction):
        inside = self.checkInside(position)
        normPos = -position/direction
        normSize = np.abs(self.halfSize/direction)
        tmin = np.max(normPos - normSize, axis=1)
        tmax = np.min(normPos + normSize, axis=1)
        distance = np.where(tmax > tmin, tmin, inf)
        distance[inside] = tmax[inside]
        distance[distance < 0] = inf
        distance += self.distanceEpsilon
        return distance, inside
        
