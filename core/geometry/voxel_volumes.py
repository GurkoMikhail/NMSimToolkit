from functools import cache
from typing import Optional, Tuple, cast
import math

import numpy as np

from core.geometry.geometries import Box
from numba import cfunc, types, carray
from numba.extending import intrinsic
from core.geometry.woodcoock_volumes import WoodcockParameticVolume
from core.materials.materials import Material, MaterialArray
from core.other.typing_definitions import Float, Index, Length, Vector3D, NumbaFloat, NumbaIndex


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
        mat_dist_3d = self.material_distribution.ID
        shape_x = Index(mat_dist_3d.shape[0])
        shape_y = Index(mat_dist_3d.shape[1])
        shape_z = Index(mat_dist_3d.shape[2])

        # Save a reference to prevent garbage collection and get a raw pointer
        self._mat_dist_1d = np.ascontiguousarray(mat_dist_3d.flatten(), dtype=np.int64)
        mat_dist_ptr = self._mat_dist_1d.ctypes.data
        mat_dist_size = self._mat_dist_1d.size

        size_x = Float(self.size[0])
        size_y = Float(self.size[1])
        size_z = Float(self.size[2])

        vox_size_x = Float(self.voxel_size[0])
        vox_size_y = Float(self.voxel_size[1])
        vox_size_z = Float(self.voxel_size[2])

        @intrinsic
        def ptr_from_int(typingctx, ptr_val):
            sig = types.CPointer(types.int64)(types.int64)
            def codegen(context, builder, signature, args):
                ptr = args[0]
                ptr_type = context.get_value_type(signature.return_type)
                return builder.inttoptr(ptr, ptr_type)
            return sig, codegen

        @cfunc(NumbaIndex(NumbaFloat, NumbaFloat, NumbaFloat))
        def parametric_func(x, y, z):
            c_ptr = ptr_from_int(mat_dist_ptr)
            c_arr = carray(c_ptr, (mat_dist_size,))
            # Compute 3D indices
            ix = Index(math.floor((x + (size_x / 2.0 - vox_size_x / 2.0)) / vox_size_x))
            iy = Index(math.floor((y + (size_y / 2.0 - vox_size_y / 2.0)) / vox_size_y))
            iz = Index(math.floor((z + (size_z / 2.0 - vox_size_z / 2.0)) / vox_size_z))

            # Bounds checking (clamping to valid voxel indices)
            ix = max(0, min(ix, shape_x - 1))
            iy = max(0, min(iy, shape_y - 1))
            iz = max(0, min(iz, shape_z - 1))

            # Flat 3D lookup: index = ix * (shape_y * shape_z) + iy * shape_z + iz
            flat_idx = ix * shape_y * shape_z + iy * shape_z + iz
            return c_arr[flat_idx]

        return parametric_func

    def _parametric_function(self, position: Vector3D) -> Tuple[np.ndarray, MaterialArray]:
        indices = ((position + (self.size / 2 - self.voxel_size / 2)) / self.voxel_size).astype(int)
        material = self.material_distribution[indices[:, 0], indices[:, 1], indices[:, 2]]
        return np.ones_like(material, dtype=bool), material
