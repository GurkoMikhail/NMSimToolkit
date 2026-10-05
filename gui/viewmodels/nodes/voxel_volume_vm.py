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
    colormap_name = gui_field(default='Physical Materials')
    lod_factor = gui_field(default=1.0)
    opacity_threshold = gui_field(default=0.05)
    max_opacity = gui_field(default=0.4)
    opacity_preset = gui_field(default='air_cutoff')
    file_path = gui_field(default='')

    def __init__(self, core_node: WoodcockVoxelVolume, parent_vm: Optional[NodeViewModel] = None, file_path: str = "") -> None:
        super().__init__(core_node, parent_vm)
        if file_path:
            self.file_path = str(file_path)

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
    def material_list(self) -> List[Material]:
        """
        Возвращает список материалов фантома (element_list) из material_distribution.
        """
        distribution_instance = self.core_node.material_distribution
        if distribution_instance is not None:
            return list(distribution_instance.element_list)
        return []

    def set_material_mapping(self, material_id: int, material_name: str) -> None:
        """
        Устанавливает соответствие целочисленного ID ткани материалу базы NIST.
        """
        if material_id < 0:
            raise ValueError(f"material_id должен быть неотрицательным: {material_id}")
        if self.core_node.material_distribution is None:
            raise ValueError("Отсутствует распределение материалов фантома.")

        if material_name == 'Vacuum':
            target_material = Material(name='Vacuum', ID=0)
        else:
            mdb = database_setting.material_database
            if material_name not in mdb:
                raise ValueError(f"Материал '{material_name}' не найден в базе NIST.")
            target_material = mdb[material_name]

        existing_list = list(self.core_node.material_distribution.element_list)
        while len(existing_list) <= material_id:
            new_id = len(existing_list)
            existing_list.append(Material(name=f"Material_{new_id}", ID=new_id))

        existing_list[material_id] = target_material
        self.core_node.material_distribution.element_list = existing_list
        self.core_node.invalidate_geometry()
        self.property_changed.emit('material_distribution', self.material_list)

    @property
    def origin(self) -> Tuple[float, float, float]:
        """
        Возвращает смещение начала координат сетки фантома для центрирования в локальной СК.
        """
        spacing_array = self.voxel_size
        dims = self.dimensions
        return tuple(-0.5 * float(dimension_len) * float(voxel_step) for dimension_len, voxel_step in zip(dims, spacing_array))

    def reload_distribution(
        self,
        path: str,
        shape: Optional[Tuple[int, ...]] = None,
        order: str = 'F',
        dtype: Any = np.float32,
        encoding: Optional[str] = None,
        voxel_size: Optional[Union[float, Sequence[float]]] = None,
        mapping: Optional[Dict[float, str]] = None,
        fill_value: str = 'Air, Dry (near sea level)',
    ) -> bool:
        """
        Перезагрузка матрицы фантома из файла (.npy, .dat, .raw) с использованием DistributionLoader.
        Поддерживает произвольные вещественные (float) значения меток материалов без потери точности.
        Материал фона по умолчанию — Air.
        """
        target_path = Path(path)
        if not target_path.is_file():
            return False
        try:
            target_shape = shape or self.dimensions
            data = DistributionLoader.load(
                target_path,
                target_shape=target_shape,
                order=order,
                dtype=dtype,
                encoding=encoding,
            )

            mdb = database_setting.material_database

            if isinstance(data, MaterialArray):
                mat_arr = data
            else:
                mat_arr = MaterialArray(data.shape)
                underlying_buffer = mat_arr.view(np.ndarray)

                # Инициализация фонового материала (по умолчанию сухой воздух Air)
                air_name = "Air, Dry (near sea level)"
                effective_fill_name = air_name if fill_value in ("Air", air_name) else fill_value
                fallback_mat = mdb.get(effective_fill_name, Material(name=effective_fill_name, ID=0))
                element_list: List[Material] = [fallback_mat]
                underlying_buffer[:] = 0

                if mapping is not None and len(mapping) > 0:
                    for float_key, material_name in mapping.items():
                        target_material = mdb.get(str(material_name), Material(name=str(material_name), ID=len(element_list)))
                        if target_material not in element_list:
                            element_list.append(target_material)
                        material_index = element_list.index(target_material)
                        mask = np.isclose(data, float(float_key), atol=1e-5)
                        underlying_buffer[mask] = material_index
                else:
                    unique_vals = np.unique(data)
                    default_names = [
                        "Air, Dry (near sea level)", "Water, Liquid", "Tissue, Soft (ICRU-44)",
                        "Bone, Cortical (ICRU-44)", "Lung (ICRP)", "Adipose Tissue (ICRU-44)"
                    ]
                    all_mats = list(mdb.values())
                    for idx_val, raw_val in enumerate(unique_vals):
                        if idx_val < len(default_names):
                            mat_cand = mdb.get(default_names[idx_val], Material(name=default_names[idx_val], ID=len(element_list)))
                        elif idx_val - len(default_names) < len(all_mats):
                            mat_cand = all_mats[idx_val - len(default_names)]
                        else:
                            mat_cand = Material(name=f"Material_{len(element_list)}", ID=len(element_list))

                        if mat_cand not in element_list:
                            element_list.append(mat_cand)
                        material_index = element_list.index(mat_cand)
                        mask = np.isclose(data, float(raw_val), atol=1e-5)
                        underlying_buffer[mask] = material_index

                mat_arr.element_list = element_list

            if voxel_size is not None:
                self.voxel_size = voxel_size

            self.core_node.material_distribution = mat_arr
            new_size = np.asarray(mat_arr.shape, dtype=float) * np.asarray(self.core_node.voxel_size, dtype=float)
            self.core_node.size = new_size
            self.core_node.invalidate_geometry()

            self.file_path = str(target_path)
            self.property_changed.emit('file_path', self.file_path)
            self.property_changed.emit('size', new_size)
            self.property_changed.emit('voxel_size', self.voxel_size)
            self.property_changed.emit('material_distribution', self.material_list)
            return True
        except (OSError, ValueError, TypeError, KeyError) as err:
            _logger.warning(f"Ошибка загрузки файла фантома: {err}")
            return False
