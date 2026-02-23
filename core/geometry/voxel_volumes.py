from functools import cache
from core.geometry.woodcoock_volumes import WoodcockParameticVolume
from core.geometry.geometries import Box
import numpy as np


class WoodcockVoxelVolume(WoodcockParameticVolume):
    """
    Класс воксельного Woodcock объёма
    
    [coordinates = (x, y, z)] = cm\n
    [material] = uint[:,:,:]\n
    [voxel_size] = cm
    """

    def __init__(self, voxel_size, material_distribution, name=None):
        size = np.asarray(material_distribution.shape)*voxel_size
        super().__init__(
            geometry=Box(*size),
            material=None,
            name=name
            )
        self.material_distribution = material_distribution
        self._voxel_size_ratio = voxel_size/self.size

    @property
    def voxel_size(self):
        return self.size*self._voxel_size_ratio

    @voxel_size.setter
    def voxel_size(self, value):
        self._voxel_size_ratio = value/self.size

    @property
    @cache
    def material(self):
        material_list = self.material_distribution.material_list
        return max(material_list)

    @material.setter
    def material(self, value):
        pass

    def _parametric_function(self, position):
        indices = ((position + (self.size/2 - self.voxel_size/2))/self.voxel_size).astype(int)
        material = self.material_distribution[indices[:, 0], indices[:, 1], indices[:, 2]]
        return np.ones_like(material, dtype=bool), material

