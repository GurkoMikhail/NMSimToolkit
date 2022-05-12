from functools import cache
from woodcoockVolumes import WoodcockParameticVolume
from geometries import Box
import numpy as np


class WoodcockVoxelVolume(WoodcockParameticVolume):
    """
    Класс воксельного Woodcock объёма
    
    [coordinates = (x, y, z)] = cm\n
    [material] = uint[:,:,:]\n
    [voxelSize] = cm
    """

    def __init__(self, voxelSize, materialDistribution, name=None):
        size = np.asarray(materialDistribution.shape)*voxelSize
        super().__init__(
            geometry=Box(*size),
            material=None,
            name=name
            )
        self.materialDistribution = materialDistribution
        self._voxelSizeRatio = voxelSize/self.size

    @property
    def voxelSize(self):
        return self.size*self._voxelSizeRatio

    @voxelSize.setter
    def voxelSize(self, value):
        self._voxelSizeRatio = value/self.size

    @property
    @cache
    def material(self):
        return max(set(self.materialDistribution.ravel()))

    @material.setter
    def material(self, value):
        pass

    def _parametricFunction(self, position):
        indices = ((position + (self.size/2 - self.voxelSize/2))/self.voxelSize).astype(int)
        materials = self.materialDistribution[indices[:, 0], indices[:, 1], indices[:, 2]]
        return np.ones_like(materials, dtype=bool), materials

