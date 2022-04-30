from woodcoockVolumes import WoodcockParameticVolume
from materials import Material
from geometries import Box
from numpy import sqrt, mod, abs, stack


class ParametricParallelCollimator(WoodcockParameticVolume):
    """
    Класс параметрического коллиматора с параллельными каналами

    [origin = (x, y, z)] = mm\n
    [size = (dx, dy, dz)] = mm\n
    [holeDiameter] = mm\n
    [septa] = mm\n
    """

    def __init__(self, size, holeDiameter, septa, material=Material.load('Pb'), name=None):
        super().__init__(
            geometry=Box(*size),
            material=material,
            name=name
            )
        self._holeDiameter = holeDiameter
        self._septa = septa
        self._computeConstants()
    
    def _computeConstants(self):
        xPeriod = self._holeDiameter + self._septa
        yPeriod = sqrt(3)*xPeriod
        self._period = stack((xPeriod, yPeriod))
        self._a = sqrt(3)/4
        d = self._holeDiameter*2/sqrt(3)
        self._corner = self._period/2
        self._ad = self._a*d
        self._ad_2 = self._ad/2

    def _parametricFunction(self, position):
        position = mod(position[:, :2], self._period)
        position = abs(position - self._corner)
        collimated = (position[:, 0] <= self._ad)*(self._a*position[:, 1] + position[:, 0]/4 <= self._ad_2)
        position = abs(position[~collimated] - self._corner)
        collimated[~collimated] = (position[:, 0] <= self._ad)*(self._a*position[:, 1] + position[:, 0]/4 <= self._ad_2)
        return collimated, None
