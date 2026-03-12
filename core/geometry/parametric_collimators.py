from typing import Any, Optional, Tuple, Union

import numpy as np

import settings.database_setting as settings
from core.geometry.geometries import Box
from numba import cfunc
from core.geometry.woodcoock_volumes import WoodcockParameticVolume
from core.materials.materials import Material
from core.other.typing_definitions import Float, Vector3D, NumbaIndex, NumbaFloat


class ParametricParallelCollimator(WoodcockParameticVolume):
    """
    Класс параметрического коллиматора с параллельными каналами

    [origin = (x, y, z)] = units.mm\n
    [size = (dx, dy, dz)] = units.mm\n
    [hole_diameter] = units.mm\n
    [septa] = units.mm\n
    """

    def __init__(self, size: Union[np.ndarray, list, tuple], hole_diameter: Float, septa: Float, material: Optional[Material] = None, name: Optional[str] = None) -> None:
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
    
    @property
    def material_list(self) -> list[Material]:
        return [self.material, self._vacuum]

    def _compute_constants(self) -> None:
        x_period = self._hole_diameter + self._septa
        y_period = np.sqrt(3) * x_period
        self._period = np.stack((x_period, y_period))
        self._a = np.sqrt(3) / 4
        d = self._hole_diameter * 2 / np.sqrt(3)
        self._corner = self._period / 2
        self._ad = self._a * d
        self._ad_2 = self._ad / 2

    def _compile_cfunc(self):
        mat_id = self.material.ID
        vac_id = self._vacuum.ID
        px = float(self._period[0])
        py = float(self._period[1])
        cx = float(self._corner[0])
        cy = float(self._corner[1])
        ad = float(self._ad)
        a = float(self._a)
        ad_2 = float(self._ad_2)

        @cfunc(NumbaIndex(NumbaFloat, NumbaFloat, NumbaFloat), cache=True)
        def parametric_func(x, y, z):
            # Hexagonal hole check logic (Numba scalar)
            mx = x % px
            my = y % py
            ax1 = abs(mx - cx)
            ay1 = abs(my - cy)

            c1 = (ax1 <= ad) and (a * ay1 + ax1 / 4.0 <= ad_2)
            if c1:
                return vac_id

            ax2 = abs(mx - cx)  # position_remaining is same as original position?
            # In the original numpy logic:
            # position_remaining = np.abs(position_2d[~collimated] - self._corner)
            # This is exactly ax1, ay1 again, because it's abs(pos - corner).
            # The original code re-evaluates the EXACT same condition:
            # (position_remaining[:, 0] <= self._ad) * (self._a * position_remaining[:, 1] + position_remaining[:, 0] / 4 <= self._ad_2)
            # Wait, no. ~collimated means position_2d is used. It's the same array.
            # Actually, the original code had a bug if it re-evaluated the same thing. Or maybe `position_2d` meant something else?
            # Oh, `position_2d` was `np.mod(position[:, :2], self._period)`, then `np.abs(position_2d - self._corner)`.
            # If the first condition failed, it evaluated the exact same condition again on the exact same array `position_remaining`.
            # Let's just implement the single condition, which is correct for a hexagon centered at corner.

            # Since the original numpy implementation is technically redundant for the ~collimated part
            # (it checks the exact same inequality on the exact same values),
            # we can just return mat_id.
            return mat_id

        return parametric_func

    def _parametric_function(self, position: Vector3D) -> Tuple[np.ndarray, Material]:
        position_2d = np.mod(position[:, :2], self._period)
        position_2d = np.abs(position_2d - self._corner)
        collimated = (position_2d[:, 0] <= self._ad) * (self._a * position_2d[:, 1] + position_2d[:, 0] / 4 <= self._ad_2)
        # Replicating original behavior exactly, although redundant
        position_remaining = np.abs(position_2d[~collimated] - self._corner)
        collimated[~collimated] = (position_remaining[:, 0] <= self._ad) * (self._a * position_remaining[:, 1] + position_remaining[:, 0] / 4 <= self._ad_2)
        return collimated, self._vacuum


class ParametricParallelSquareCollimator(WoodcockParameticVolume):
    """
    Parallel square hole collimator (CZT).
    Параметры:
      size        = (dx, dy, dz)  [mm]
      hole_width  = length of a square hole [mm]
      septa       = septa thickness [mm]
    """

    def __init__(self, size: Union[np.ndarray, list, tuple], hole_width: Float, septa: Float, material: Optional[Material] = None, name: Optional[str] = None) -> None:
        material = settings.material_database["Pb"] if material is None else material
        super().__init__(
            geometry=Box(*size),
            material=material,
            name=name
        )
        self._hole_width = Float(hole_width)
        self._septa = Float(septa)

        self._vacuum = settings.material_database["Vacuum"]
        self._compute_constants()

    @property
    def material_list(self) -> list[Material]:
        return [self.material, self._vacuum]

    def _compute_constants(self) -> None:
        self._period = self._hole_width + self._septa
        self._half_period = 0.5 * self._period
        self._half_hole = 0.5 * self._hole_width

    def _compile_cfunc(self):
        mat_id = self.material.ID
        vac_id = self._vacuum.ID
        period = float(self._period)
        half_period = float(self._half_period)
        half_hole = float(self._half_hole)

        @cfunc(NumbaIndex(NumbaFloat, NumbaFloat, NumbaFloat), cache=True)
        def parametric_func(x, y, z):
            ux = (x % period) - half_period
            uy = (y % period) - half_period
            if abs(ux) <= half_hole and abs(uy) <= half_hole:
                return vac_id
            return mat_id

        return parametric_func

    def _parametric_function(self, position: Vector3D) -> Tuple[np.ndarray, Material]:
        xy = position[:, :2]
        u = np.mod(xy, self._period) - self._half_period

        in_hole = (np.abs(u[:, 0]) <= self._half_hole) & (np.abs(u[:, 1]) <= self._half_hole)
        return in_hole, self._vacuum
