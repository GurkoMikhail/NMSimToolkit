import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union
import numpy as np

from core.geometry.voxel_volumes import WoodcockVoxelVolume
from core.materials.materials import Material, MaterialArray
from core.data.distribution_loader import DistributionLoader
import settings.database_setting as database_setting
from gui.viewmodels.decorators import gui_field
from gui.viewmodels.nodes.base_node_vm import NodeViewModel

_logger = logging.getLogger(__name__)


class VoxelVolumeViewModel(NodeViewModel):
    """
    ViewModel для воксельного фантома WoodcockVoxelVolume.
    """
    colormap_name = gui_field(default='Hot Iron')
    lod_factor = gui_field(default=1.0)
    opacity_threshold = gui_field(default=0.05)
    max_opacity = gui_field(default=0.4)
    opacity_preset = gui_field(default='air_cutoff')
    file_path = gui_field(default='')

    def __init__(self, core_node: WoodcockVoxelVolume, parent_vm: Optional[NodeViewModel] = None) -> None:
        super().__init__(core_node, parent_vm)
        if core_node.distribution_path:
            self.file_path = str(core_node.distribution_path)

    @property
    def size(self) -> np.ndarray:
        return np.asarray(self.core_node.size, dtype=float)

    @size.setter
    def size(self, new_size: Sequence[float]) -> None:
        new_size_arr = np.asarray(new_size, dtype=float)
        self.core_node.size = new_size_arr
        dims = self.dimensions
        if all(d > 0 for d in dims):
            new_voxel_size = new_size_arr / np.asarray(dims, dtype=float)
            self.core_node.voxel_size = new_voxel_size
        self.core_node.invalidate_geometry()
        self.property_changed.emit('size', new_size_arr)

    @property
    def voxel_size(self) -> np.ndarray:
        return np.asarray(self.core_node.voxel_size)

    @voxel_size.setter
    def voxel_size(self, value: Union[float, Sequence[float]]) -> None:
        val_arr = np.asarray(value, dtype=float)
        dist = self.core_node.material_distribution
        if dist is not None:
            self.core_node.geometry.size = np.asarray(dist.shape, dtype=float) * val_arr
        self.core_node.voxel_size = val_arr
        self.core_node.invalidate_geometry()
        self.property_changed.emit('voxel_size', val_arr)
        self.property_changed.emit('size', self.size)

    @property
    def dimensions(self) -> Tuple[int, ...]:
        dist = self.core_node.material_distribution
        if dist is not None:
            return tuple(dist.shape)
        return (0, 0, 0)

    @property
    def origin(self) -> Tuple[float, float, float]:
        """
        Возвращает смещение начала координат сетки фантома для центрирования в локальной СК.
        """
        sp = self.voxel_size
        dims = self.dimensions
        return tuple(-0.5 * d * s for d, s in zip(dims, sp))

    def reload_distribution(self, path: str, shape: Optional[Tuple[int, ...]] = None, order: str = 'F') -> bool:
        """
        Перезагрузка матрицы фантома из файла (.npy, .dat, .raw) с использованием DistributionLoader.
        """
        target_path = Path(path)
        if not target_path.is_file():
            return False
        try:
            target_shape = shape or self.dimensions
            data = DistributionLoader.load(target_path, target_shape=target_shape, order=order)

            if isinstance(data, MaterialArray):
                mat_arr = data
            else:
                mat_arr = MaterialArray(data.shape)
                existing_list: List[Material] = []
                if self.core_node.material_distribution is not None:
                    existing_list = list(self.core_node.material_distribution.element_list)

                mdb = database_setting.material_database
                if not existing_list or len(existing_list) <= 1:
                    existing_list = [
                        Material(name='Vacuum', ID=0),
                        mdb.get('Water, Liquid', Material(name='Water', ID=1)),
                        mdb.get('Tissue, Soft (ICRU-44)', Material(name='Tissue', ID=2)),
                        mdb.get('Bone, Cortical (ICRU-44)', Material(name='Bone', ID=3)),
                        mdb.get('Lung (ICRP)', Material(name='Lung', ID=4)),
                        mdb.get('Adipose Tissue (ICRU-44)', Material(name='Adipose', ID=5)),
                    ]

                int_data = data.astype(int)
                max_val = int(np.nanmax(int_data)) if int_data.size > 0 else 0
                all_mats = list(mdb.values())
                mat_idx = 0
                while len(existing_list) <= max_val:
                    if mat_idx < len(all_mats):
                        cand = all_mats[mat_idx]
                        if cand not in existing_list:
                            existing_list.append(cand)
                        mat_idx += 1
                    else:
                        new_id = len(existing_list)
                        existing_list.append(Material(name=f"Material_{new_id}", ID=new_id))

                mat_arr.element_list = existing_list
                mat_arr.view(np.ndarray)[:] = int_data

            self.core_node.material_distribution = mat_arr
            self.core_node.distribution_path = str(target_path)
            new_size = np.asarray(mat_arr.shape, dtype=float) * np.asarray(self.core_node.voxel_size, dtype=float)
            self.core_node.size = new_size
            self.core_node.invalidate_geometry()

            self.file_path = str(target_path)
            self.property_changed.emit('file_path', self.file_path)
            self.property_changed.emit('size', new_size)
            self.property_changed.emit('voxel_size', self.voxel_size)
            return True
        except (OSError, ValueError, TypeError, KeyError) as err:
            _logger.warning(f"Ошибка загрузки файла фантома: {err}")
            return False
