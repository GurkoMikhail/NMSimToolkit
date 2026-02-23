from typing import Any, Optional, Tuple, Union

import numpy as np

import settings.database_setting as settings
from core.geometry.geometries import Box
from core.geometry.woodcoock_volumes import WoodcockParameticVolume
from core.materials.materials import Material
from core.other.typing_definitions import Vector3D


class ParametricParallelCollimator(WoodcockParameticVolume):
    """
    Класс параметрического коллиматора с параллельными каналами

    [origin = (x, y, z)] = mm\n
    [size = (dx, dy, dz)] = mm\n
    [hole_diameter] = mm\n
    [septa] = mm\n
    """

    def __init__(self, size: Union[np.ndarray, list, tuple], hole_diameter: float, septa: float, material: Optional[Material] = None, name: Optional[str] = None) -> None:
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
    
    def _compute_constants(self) -> None:
        x_period = self._hole_diameter + self._septa
        y_period = np.sqrt(3) * x_period
        self._period = np.stack((x_period, y_period))
        self._a = np.sqrt(3) / 4
        d = self._hole_diameter * 2 / np.sqrt(3)
        self._corner = self._period / 2
        self._ad = self._a * d
        self._ad_2 = self._ad / 2

    def _parametric_function(self, position: Vector3D) -> Tuple[np.ndarray, Material]:
        position_2d = np.mod(position[:, :2], self._period)
        position_2d = np.abs(position_2d - self._corner)
        collimated = (position_2d[:, 0] <= self._ad) * (self._a * position_2d[:, 1] + position_2d[:, 0] / 4 <= self._ad_2)
        position_remaining = np.abs(position_2d[~collimated] - self._corner)
        collimated[~collimated] = (position_remaining[:, 0] <= self._ad) * (self._a * position_remaining[:, 1] + position_remaining[:, 0] / 4 <= self._ad_2)
        return collimated, self._vacuum
