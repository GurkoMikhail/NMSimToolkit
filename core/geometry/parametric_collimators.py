from core.geometry.woodcoock_volumes import WoodcockParameticVolume
import settings.database_setting as settings
from core.geometry.geometries import Box
from numpy import sqrt, mod, abs, stack


class ParametricParallelCollimator(WoodcockParameticVolume):
    """
    Класс параметрического коллиматора с параллельными каналами

    [origin = (x, y, z)] = mm\n
    [size = (dx, dy, dz)] = mm\n
    [hole_diameter] = mm\n
    [septa] = mm\n
    """

    def __init__(self, size, hole_diameter, septa, material=None, name=None):
        material = settings.material_database['Pb'] if material is None else material
        super().__init__(
            geometry=Box(*size),
            material=material,
            name=name
            )
        self._hole_diameter = hole_diameter
        self._septa = septa
        self._vacuum = settings.material_database['Vacuum']
        self._compute_constants()
    
    def _compute_constants(self):
        x_period = self._hole_diameter + self._septa
        y_period = sqrt(3)*x_period
        self._period = stack((x_period, y_period))
        self._a = sqrt(3)/4
        d = self._hole_diameter*2/sqrt(3)
        self._corner = self._period/2
        self._ad = self._a*d
        self._ad_2 = self._ad/2

    def _parametric_function(self, position):
        position = mod(position[:, :2], self._period)
        position = abs(position - self._corner)
        collimated = (position[:, 0] <= self._ad)*(self._a*position[:, 1] + position[:, 0]/4 <= self._ad_2)
        position = abs(position[~collimated] - self._corner)
        collimated[~collimated] = (position[:, 0] <= self._ad)*(self._a*position[:, 1] + position[:, 0]/4 <= self._ad_2)
        return collimated, self._vacuum
