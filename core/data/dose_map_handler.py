"""
Модуль сеточного накопления поглощенной дозы (энерговыделения) в трехмерных воксельных объемах
и потоковой синхронизации с памятью SharedMemory и файлами HDF5.
"""

import logging
from collections.abc import Mapping
from multiprocessing import shared_memory
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np

from core.data.data_handlers import BaseDataHandler
from core.physics.interaction_buffers import ChunkType
from core.scene.dose_grid_node import DoseGridNode

_logger = logging.getLogger(__name__)


class DoseGridEntry:
    """
    Дескриптор и буфер данных для отдельной воксельной сетки дозы.
    Содержит геометрические параметры сетки, инвертированную матрицу глобальной трансформации
    и буфер SharedMemory для zero-copy передачи среза дозы в графический интерфейс.
    """

    def __init__(
        self,
        name: str,
        grid_shape: Tuple[int, int, int],
        voxel_size: Union[float, Tuple[float, float, float]],
        origin: Tuple[float, float, float],
        inverse_global_matrix: Optional[np.ndarray] = None,
        shm_name: Optional[str] = None,
        create_shm: bool = True,
        node: Optional[DoseGridNode] = None,
        dtype: Optional[Union[str, np.dtype[Any]]] = None,
    ) -> None:
        self.name = name
        self.grid_shape: Tuple[int, int, int] = (
            int(grid_shape[0]),
            int(grid_shape[1]),
            int(grid_shape[2]),
        )
        if isinstance(voxel_size, (int, float)):
            self.voxel_size = (float(voxel_size), float(voxel_size), float(voxel_size))
        else:
            self.voxel_size = (float(voxel_size[0]), float(voxel_size[1]), float(voxel_size[2]))
        self.origin = (float(origin[0]), float(origin[1]), float(origin[2]))
        if inverse_global_matrix is not None:
            self.inverse_global_matrix = np.asarray(inverse_global_matrix, dtype=np.float64)
        else:
            self.inverse_global_matrix = np.eye(4, dtype=np.float64)
        self.shm_name: Optional[str] = shm_name if shm_name else None
        self.create_shm: bool = create_shm
        self._owns_shm: bool = False
        self.node = node
        resolved_dtype = dtype if dtype is not None else (node.dtype if node is not None else "float32")
        self.dtype: np.dtype[Any] = np.dtype(resolved_dtype)

        self._shm: Optional[shared_memory.SharedMemory] = None
        self._dose_grid: Optional[np.ndarray] = None
        self._init_shared_memory()

    def _init_shared_memory(self) -> None:
        if not self.shm_name:
            self._shm = None
            self._owns_shm = False
            self._dose_grid = np.zeros(self.grid_shape, dtype=self.dtype, order="F")
            return

        nbytes = int(np.prod(self.grid_shape) * self.dtype.itemsize)
        if self.create_shm:
            try:
                self._shm = shared_memory.SharedMemory(name=self.shm_name, create=True, size=nbytes)
                self._owns_shm = True
                self._dose_grid = np.ndarray(self.grid_shape, dtype=self.dtype, buffer=self._shm.buf, order="F")
                self._dose_grid.fill(0.0)
            except FileExistsError:
                existing_shm = shared_memory.SharedMemory(name=self.shm_name, create=False)
                if existing_shm.size >= nbytes:
                    self._shm = existing_shm
                    self._owns_shm = False
                    self._dose_grid = np.ndarray(self.grid_shape, dtype=self.dtype, buffer=self._shm.buf, order="F")
                else:
                    try:
                        existing_shm.close()
                        existing_shm.unlink()
                    except (OSError, RuntimeError, ValueError, KeyError, BufferError):
                        pass
                    self._shm = shared_memory.SharedMemory(name=self.shm_name, create=True, size=nbytes)
                    self._owns_shm = True
                    self._dose_grid = np.ndarray(self.grid_shape, dtype=self.dtype, buffer=self._shm.buf, order="F")
                    self._dose_grid.fill(0.0)
        else:
            self._owns_shm = False
            try:
                existing_shm = shared_memory.SharedMemory(name=self.shm_name, create=False)
                if existing_shm.size >= nbytes:
                    self._shm = existing_shm
                    self._dose_grid = np.ndarray(self.grid_shape, dtype=self.dtype, buffer=self._shm.buf, order="F")
                else:
                    existing_shm.close()
                    self._shm = None
                    self._dose_grid = np.zeros(self.grid_shape, dtype=self.dtype, order="F")
            except FileNotFoundError:
                _logger.warning(f"SharedMemory '{self.shm_name}' для сетки '{self.name}' не найден.")
                self._shm = None
                self._dose_grid = np.zeros(self.grid_shape, dtype=self.dtype, order="F")

    def accumulate(self, pos_x: np.ndarray, pos_y: np.ndarray, pos_z: np.ndarray, edep: np.ndarray) -> None:
        """
        Векторизованное накопление энерговыделения для данной сетки.
        Глобальные координаты взаимодействий (pos_x, pos_y, pos_z) переводятся в локальную
        систему координат сетки через inverse_global_matrix.
        """
        if self._dose_grid is None or len(pos_x) == 0:
            return

        n_pts = len(pos_x)
        p_homo = np.ones((n_pts, 4), dtype=np.float64)
        p_homo[:, 0] = pos_x
        p_homo[:, 1] = pos_y
        p_homo[:, 2] = pos_z

        p_local = np.matmul(p_homo, self.inverse_global_matrix.T)[:, :3]

        ox, oy, oz = self.origin
        sx, sy, sz = self.voxel_size
        nx, ny, nz = self.grid_shape

        ix = np.floor((p_local[:, 0] - ox) / sx).astype(np.int64)
        iy = np.floor((p_local[:, 1] - oy) / sy).astype(np.int64)
        iz = np.floor((p_local[:, 2] - oz) / sz).astype(np.int64)

        in_bounds = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)
        if not np.any(in_bounds):
            return

        np.add.at(
            self._dose_grid,
            (ix[in_bounds], iy[in_bounds], iz[in_bounds]),
            np.asarray(edep[in_bounds], dtype=self.dtype),
        )

    def clear(self) -> None:
        """Сброс накопленной дозы в ноль."""
        if self._dose_grid is not None:
            self._dose_grid.fill(0.0)

    def close(self) -> None:
        """Закрытие и удаление сегмента SharedMemory."""
        self._dose_grid = None
        if self._shm is not None:
            try:
                self._shm.close()
                if self._owns_shm:
                    self._shm.unlink()
            except (OSError, RuntimeError, ValueError, KeyError, BufferError) as shm_error:
                _logger.debug(f"Ошибка закрытия SharedMemory '{self.shm_name}': {shm_error}")
            finally:
                self._shm = None


class DoseMapHandler(BaseDataHandler):
    """
    Обработчик данных моделирования для сеточного накопления дозы (энерговыделения)
    в трехмерных воксельных объемах произвольного количества узлов DoseGridNode.
    Координаты взаимодействий переводятся в локальную систему координат каждого узла
    через inverse_global_matrix.
    """

    _CONSUMED_CHUNK_TYPES: FrozenSet[ChunkType] = frozenset({ChunkType.INTERACTIONS})

    def __init__(
        self,
        grid_nodes: Optional[Sequence[DoseGridNode]] = None,
        shm_name: Optional[str] = "nmsim_dose_shm",
        create_shm: bool = True,
    ) -> None:
        super().__init__()
        self.entries: List[DoseGridEntry] = []
        self._entries_by_name: Dict[str, DoseGridEntry] = {}

        if grid_nodes is not None:
            active_nodes = [node for node in grid_nodes if node.is_active]
            for idx, node in enumerate(active_nodes):
                node_name = node.name or f"DoseGrid_{idx}"
                if shm_name:
                    node_shm: Optional[str] = f"{shm_name}_{node_name}" if len(active_nodes) > 1 else shm_name
                else:
                    node_shm = None
                inv_mat = node.inverse_global_matrix
                entry = DoseGridEntry(
                    name=node_name,
                    grid_shape=node.grid_shape,
                    voxel_size=node.dose_voxel_size,
                    origin=node.origin,
                    inverse_global_matrix=inv_mat,
                    shm_name=node_shm,
                    create_shm=create_shm,
                    node=node,
                    dtype=node.dtype,
                )
                self.entries.append(entry)
                self._entries_by_name[node_name] = entry

    @property
    def consumed_chunk_types(self) -> FrozenSet[ChunkType]:
        return self._CONSUMED_CHUNK_TYPES

    def add_grid_node(self, node: DoseGridNode, shm_prefix: Optional[str] = None) -> None:
        """Динамическое добавление узла DoseGridNode в обработчик."""
        node_name = node.name or f"DoseGrid_{len(self.entries)}"
        prefix = shm_prefix or "nmsim_dose_shm"
        node_shm = f"{prefix}_{node_name}"
        inv_mat = node.inverse_global_matrix
        entry = DoseGridEntry(
            name=node_name,
            grid_shape=node.grid_shape,
            voxel_size=node.dose_voxel_size,
            origin=node.origin,
            inverse_global_matrix=inv_mat,
            shm_name=node_shm,
            create_shm=True,
            node=node,
            dtype=node.dtype,
        )
        self.entries.append(entry)
        self._entries_by_name[node_name] = entry

    def process_chunk(self, chunk: Dict[str, Any]) -> None:
        """
        Векторизованное накопление энерговыделения из чанка взаимодействий
        для всех зарегистрированных сеток дозы.
        """
        if chunk.get('type') != ChunkType.INTERACTIONS:
            return

        data = chunk.get('data')
        if not isinstance(data, (dict, Mapping)):
            return

        pos_x = data.get('pos_x')
        pos_y = data.get('pos_y')
        pos_z = data.get('pos_z')
        energy_deposit = data.get('energy_deposit')

        if pos_x is None or energy_deposit is None or len(pos_x) == 0:
            return

        pos_x_arr = np.asarray(pos_x)
        pos_y_arr = np.asarray(pos_y) if pos_y is not None else np.zeros_like(pos_x_arr)
        pos_z_arr = np.asarray(pos_z) if pos_z is not None else np.zeros_like(pos_x_arr)
        energy_deposit_arr = np.asarray(energy_deposit)

        # Отбираем события с депонированием энергии > 0
        valid_mask = energy_deposit_arr > 0.0
        if not np.any(valid_mask):
            return

        filtered_px = pos_x_arr[valid_mask]
        filtered_py = pos_y_arr[valid_mask]
        filtered_pz = pos_z_arr[valid_mask]
        filtered_edep = energy_deposit_arr[valid_mask]

        for entry in self.entries:
            entry.accumulate(filtered_px, filtered_py, filtered_pz, filtered_edep)

    def get_dose_snapshot(self, name: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Возвращает локальную копию 3D-массива дозы.
        Если указано имя name, возвращает массив указанной сетки.
        Если name не указано и зарегистрирована хотя бы одна сетка, возвращает массив первой сетки.
        """
        if name is not None:
            entry = self._entries_by_name.get(name)
            return entry._dose_grid.copy() if entry is not None and entry._dose_grid is not None else None
        if len(self.entries) >= 1:
            entry = self.entries[0]
            return entry._dose_grid.copy() if entry._dose_grid is not None else None
        return None

    def get_dose_snapshots(self) -> Dict[str, np.ndarray]:
        """
        Возвращает словарь локальных копий всех сеток {имя: ndarray}.
        """
        return {entry.name: entry._dose_grid.copy() for entry in self.entries if entry._dose_grid is not None}

    def clear(self) -> None:
        """
        Сброс накопленной дозы во всех сетках в ноль.
        """
        for entry in self.entries:
            entry.clear()

    def finalize(self) -> None:
        """
        Сохранение и аккумуляция накопленных дозовых карт в группу /dose HDF5 файла.
        """
        if self.writer_callback is not None:
            def write_func(h5_file: h5py.File) -> None:
                dose_group = h5_file.require_group('dose')
                for entry in self.entries:
                    if entry._dose_grid is not None:
                        if entry.name in dose_group:
                            existing_dset = dose_group[entry.name]
                            if isinstance(existing_dset, h5py.Dataset) and existing_dset.shape == entry._dose_grid.shape:
                                existing_dset[...] = (
                                    np.asarray(existing_dset[...], dtype=entry.dtype) + entry._dose_grid
                                )
                                existing_dset.attrs['voxel_size'] = entry.voxel_size
                                existing_dset.attrs['origin'] = entry.origin
                                existing_dset.attrs['grid_shape'] = entry.grid_shape
                                continue
                            del dose_group[entry.name]
                        dset = dose_group.create_dataset(entry.name, data=entry._dose_grid)
                        dset.attrs['voxel_size'] = entry.voxel_size
                        dset.attrs['origin'] = entry.origin
                        dset.attrs['grid_shape'] = entry.grid_shape
            self.writer_callback(write_func)

    def close(self) -> None:
        """
        Закрытие и удаление дескрипторов SharedMemory всех сеток.
        """
        for entry in self.entries:
            entry.close()
        self.entries.clear()
        self._entries_by_name.clear()

    def __del__(self) -> None:
        self.close()


__all__ = ["DoseGridEntry", "DoseMapHandler"]


