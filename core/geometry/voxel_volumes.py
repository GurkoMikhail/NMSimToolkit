from functools import cache
from typing import Optional, Tuple, cast

import numpy as np

from core.geometry.geometries import Box
from numba import cfunc
from core.geometry.woodcoock_volumes import WoodcockParameticVolume
from core.materials.materials import Material, MaterialArray
from core.other.typing_definitions import Float, Length, Vector3D, NumbaFloat, NumbaIndex


class WoodcockVoxelVolume(WoodcockParameticVolume):
    """
    Класс воксельного Woodcock объёма
    
    [coordinates = (x, y, z)] = units.cm\n
    [material] = uint[:,:,:]\n
    [voxel_size] = units.cm
    """

    material_distribution: MaterialArray
    _voxel_size_ratio: Vector3D

    def __init__(self, voxel_size: Length, material_distribution: MaterialArray, name: Optional[str] = None) -> None:
        size = np.asarray(material_distribution.shape)*voxel_size
        super().__init__(
            geometry=Box(size[0], size[1], size[2]),
            material=Material(),
            name=name
            )
        self.material_distribution = material_distribution
        self._voxel_size_ratio = voxel_size/self.size

    @property
    def voxel_size(self) -> Vector3D:
        return self.size*self._voxel_size_ratio

    @voxel_size.setter
    def voxel_size(self, value: Vector3D) -> None:
        self._voxel_size_ratio = value/self.size

    @property
    @cache
    def material(self) -> Material:
        material_list = self.material_distribution.material_list
        return max(material_list)

    @material.setter
    def material(self, value: Material) -> None:
        pass

    @property
    def material_list(self) -> list[Material]:
        return self.material_distribution.material_list

    def _compile_cfunc(self):
        # Flatten the 3D material IDs into a 1D numba array for fast lookup
        mat_dist_3d = self.material_distribution.ID
        shape_x, shape_y, shape_z = mat_dist_3d.shape
        mat_dist_1d = mat_dist_3d.flatten()

        size_x = float(self.size[0])
        size_y = float(self.size[1])
        size_z = float(self.size[2])

        vox_size_x = float(self.voxel_size[0])
        vox_size_y = float(self.voxel_size[1])
        vox_size_z = float(self.voxel_size[2])

        @cfunc(NumbaIndex(NumbaFloat, NumbaFloat, NumbaFloat), cache=True)
        def parametric_func(x, y, z):
            # Compute 3D indices (replicating numpy logic)
            ix = int((x + (size_x / 2.0 - vox_size_x / 2.0)) / vox_size_x)
            iy = int((y + (size_y / 2.0 - vox_size_y / 2.0)) / vox_size_y)
            iz = int((z + (size_z / 2.0 - vox_size_z / 2.0)) / vox_size_z)

            # Bounds checking (clamping to valid voxel indices)
            if ix < 0: ix = 0
            if ix >= shape_x: ix = shape_x - 1
            if iy < 0: iy = 0
            if iy >= shape_y: iy = shape_y - 1
            if iz < 0: iz = 0
            if iz >= shape_z: iz = shape_z - 1

            # Flat 3D lookup: index = ix * (shape_y * shape_z) + iy * shape_z + iz
            flat_idx = ix * shape_y * shape_z + iy * shape_z + iz
            return mat_dist_1d[flat_idx]

        return parametric_func

    def _parametric_function(self, position: Vector3D) -> Tuple[np.ndarray, MaterialArray]:
        indices = ((position + (self.size / 2 - self.voxel_size / 2)) / self.voxel_size).astype(int)
        material = self.material_distribution[indices[:, 0], indices[:, 1], indices[:, 2]]
        return np.ones_like(material, dtype=bool), material
