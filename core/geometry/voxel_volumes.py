"""
Модуль воксельных объемов с трекингом Вудкока (WoodcockVoxelVolume).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from numba import cfunc

from core.geometry.geometries import Box
from core.geometry.volumes import Volume
from core.geometry.woodcock_volumes import WoodcockParametricVolume
from core.materials.materials import Material, MaterialArray
from core.other.typing_definitions import (
    Float,
    Index,
    Length,
    NumbaFloat,
    NumbaIndex,
    Vector3D,
)

if TYPE_CHECKING:
    from core.scene.nodes import SpatialNode


class WoodcockVoxelVolume(WoodcockParametricVolume):
    """
    Класс воксельного объема с навигацией методом Вудкока (Woodcock tracking).

    Центр воксельной сетки совпадает с началом локальной системы координат (0, 0, 0),
    а габаритные размеры вычисляются как shape * voxel_size (в мм).
    """

    _material_distribution: MaterialArray
    _voxel_size_ratio: Vector3D
    _cached_majorant_material: Optional[Material]

    def __init__(
        self,
        voxel_size: Length | Vector3D,
        material_distribution: MaterialArray,
        name: Optional[str] = None,
    ) -> None:
        self._cached_majorant_material = None
        self._material_distribution = material_distribution
        size = np.asarray(material_distribution.shape, dtype=Float) * np.asarray(voxel_size, dtype=Float)
        super().__init__(
            geometry=Box(Float(size[0]), Float(size[1]), Float(size[2])),
            material=Material(),
            name=name,
        )
        self._voxel_size_ratio = np.asarray(voxel_size, dtype=Float) / self.size

    @property
    def material_distribution(self) -> MaterialArray:
        """Трехмерное распределение материалов по вокселям."""
        return self._material_distribution

    @material_distribution.setter
    def material_distribution(self, value: MaterialArray) -> None:
        current_voxel_size = self.voxel_size
        self._material_distribution = value
        self._cached_majorant_material = None
        new_size = np.asarray(value.shape, dtype=Float) * current_voxel_size
        self.geometry.size = new_size
        self._voxel_size_ratio = current_voxel_size / self.size
        self.invalidate_geometry()

    @property
    def voxel_size(self) -> Vector3D:
        """Размер одного вокселя по осям (dx, dy, dz) в мм."""
        return self.size * self._voxel_size_ratio

    @voxel_size.setter
    def voxel_size(self, value: Vector3D | Length) -> None:
        vox_arr = np.broadcast_to(np.asarray(value, dtype=Float), (3,)).copy()
        new_size = np.asarray(self._material_distribution.shape, dtype=Float) * vox_arr
        self.geometry.size = new_size
        self._voxel_size_ratio = vox_arr / self.size
        self.invalidate_geometry()

    @property
    def material(self) -> Material:
        """Мажорантный материал воксельного объема (кэшируется на экземпляре)."""
        if self._cached_majorant_material is None:
            self._cached_majorant_material = max(self._material_distribution.material_list)
        return self._cached_majorant_material

    @material.setter
    def material(self, value: Material) -> None:
        pass

    @property
    def material_list(self) -> list[Material]:
        """Список всех материалов, присутствующих в воксельном распределении."""
        return self._material_distribution.material_list

    def invalidate_geometry(self) -> None:
        """Инвалидирует геометрический буфер, скомпилированную функцию и кэш мажорантного материала."""
        super().invalidate_geometry()
        self._cached_majorant_material = None

    def _compile_cfunc(self) -> Any:
        """Компилирует Numba @cfunc для быстрого поиска ID материала по локальным координатам."""
        mat_dist_3d = self._material_distribution.ID
        shape_x = Index(mat_dist_3d.shape[0])
        shape_y = Index(mat_dist_3d.shape[1])
        shape_z = Index(mat_dist_3d.shape[2])
        mat_dist_1d = mat_dist_3d.flatten()

        size_x = Float(self.size[0])
        size_y = Float(self.size[1])
        size_z = Float(self.size[2])

        vox_size_x = Float(self.voxel_size[0])
        vox_size_y = Float(self.voxel_size[1])
        vox_size_z = Float(self.voxel_size[2])

        @cfunc(NumbaIndex(NumbaFloat, NumbaFloat, NumbaFloat))
        def parametric_func(pos_x: Float, pos_y: Float, pos_z: Float) -> Index:
            ix = Index(np.floor((pos_x + 0.5 * size_x) / vox_size_x))
            iy = Index(np.floor((pos_y + 0.5 * size_y) / vox_size_y))
            iz = Index(np.floor((pos_z + 0.5 * size_z) / vox_size_z))

            ix = max(Index(0), min(ix, shape_x - Index(1)))
            iy = max(Index(0), min(iy, shape_y - Index(1)))
            iz = max(Index(0), min(iz, shape_z - Index(1)))

            flat_idx = ix * shape_y * shape_z + iy * shape_z + iz
            return Index(mat_dist_1d[flat_idx])

        return parametric_func

    def _parametric_function(self, position: Vector3D) -> Tuple[NDArray[np.bool_], Any]:
        """Векторизованное определение материалов воксельной сетки по массиву локальных координат."""
        raw_indices = np.floor((position + 0.5 * self.size) / self.voxel_size).astype(Index)
        max_bounds = np.asarray(self._material_distribution.shape, dtype=Index) - 1
        indices = np.clip(raw_indices, 0, max_bounds)
        material = self._material_distribution[indices[:, 0], indices[:, 1], indices[:, 2]]
        return np.ones(material.shape, dtype=bool), material

    def add_child(self, child: SpatialNode) -> None:
        """Запрещает вложение физических объемов внутрь монолитной воксельной сетки."""
        if isinstance(child, Volume):
            raise TypeError(
                "WoodcockVoxelVolume является монолитной воксельной сеткой и не может содержать дочерние объемы Volume."
            )
        super().add_child(child)


__all__ = ["WoodcockVoxelVolume"]

