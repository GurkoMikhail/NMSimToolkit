from functools import cache
from core.geometry.woodcoock_volumes import WoodcockParameticVolume
from core.geometry.geometries import Box
import numpy as np
from typing import Optional, Tuple, Any
from core.materials.materials import Material, MaterialArray
from core.other.typing_definitions import Length, Vector3D, Precision


class WoodcockVoxelVolume(WoodcockParameticVolume):
    """
    Класс воксельного Woodcock объёма
    
    [coordinates = (x, y, z)] = cm\n
    [material] = uint[:,:,:]\n
    [voxel_size] = cm
    """

    material_distribution: MaterialArray
    _voxel_size_ratio: Length

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
    def voxel_size(self) -> Vector3D: # type: ignore
        return self.size*self._voxel_size_ratio

    @voxel_size.setter
    def voxel_size(self, value: Vector3D) -> None: # type: ignore
        self._voxel_size_ratio = value/self.size

    @property
    @cache
    def material(self) -> Material:
        material_list = self.material_distribution.material_list
        return max(material_list)

    @material.setter
    def material(self, value: Material) -> None:
        pass

    def _parametric_function(self, position: Vector3D) -> Tuple[np.ndarray, Any]: # type: ignore
        indices = ((position + (self.size/2 - self.voxel_size/2))/self.voxel_size).astype(int)
        material = self.material_distribution[indices[:, 0], indices[:, 1], indices[:, 2]]
        return np.ones_like(material, dtype=bool), material
