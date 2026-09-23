import logging
from collections.abc import Mapping
from multiprocessing import shared_memory
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np

from core.data.data_handlers import BaseDataHandler
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
    ) -> None:
        self.name = name
        self.grid_shape = tuple(int(s) for s in grid_shape)
        if isinstance(voxel_size, (int, float)):
            self.voxel_size = (float(voxel_size), float(voxel_size), float(voxel_size))
        else:
            self.voxel_size = (float(voxel_size[0]), float(voxel_size[1]), float(voxel_size[2]))
        self.origin = (float(origin[0]), float(origin[1]), float(origin[2]))
        if inverse_global_matrix is not None:
            self.inverse_global_matrix = np.asarray(inverse_global_matrix, dtype=np.float64)
        else:
            self.inverse_global_matrix = np.eye(4, dtype=np.float64)
        self.shm_name = shm_name or f"nmsim_dose_{name}_{abs(id(self))}"
        self.create_shm = create_shm
        self._owns_shm = create_shm
        self.node = node

        self._shm: Optional[shared_memory.SharedMemory] = None
        self._dose_grid: Optional[np.ndarray] = None
        self._init_shared_memory()

    def _init_shared_memory(self) -> None:
        nbytes = int(np.prod(self.grid_shape) * np.dtype(np.float64).itemsize)
        if self.create_shm:
            try:
                self._shm = shared_memory.SharedMemory(name=self.shm_name, create=True, size=nbytes)
            except FileExistsError:
                try:
                    temp_shm = shared_memory.SharedMemory(name=self.shm_name, create=False)
                    temp_shm.close()
                    temp_shm.unlink()
                except Exception:
                    pass
                try:
                    self._shm = shared_memory.SharedMemory(name=self.shm_name, create=True, size=nbytes)
                except Exception:
                    self._shm = shared_memory.SharedMemory(name=self.shm_name, create=False)
        else:
            try:
                self._shm = shared_memory.SharedMemory(name=self.shm_name, create=False)
            except FileNotFoundError:
                _logger.warning(f"SharedMemory '{self.shm_name}' для сетки '{self.name}' не найден.")
                self._shm = None

        if self._shm is not None:
            self._dose_grid = np.ndarray(self.grid_shape, dtype=np.float64, buffer=self._shm.buf)
            if self.create_shm:
                self._dose_grid.fill(0.0)
        else:
            self._dose_grid = np.zeros(self.grid_shape, dtype=np.float64)

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

        np.add.at(self._dose_grid, (ix[in_bounds], iy[in_bounds], iz[in_bounds]), edep[in_bounds])

    def clear(self) -> None:
        """Сброс накопленной дозы в ноль."""
        if self._dose_grid is not None:
            self._dose_grid.fill(0.0)

    def close(self) -> None:
        """Закрытие и удаление сегмента SharedMemory."""
        if self._shm is not None:
            try:
                self._shm.close()
                if self._owns_shm:
                    self._shm.unlink()
            except Exception as e:
                _logger.debug(f"Ошибка закрытия SharedMemory '{self.shm_name}': {e}")
            finally:
                self._shm = None
                self._dose_grid = None


class DoseMapHandler(BaseDataHandler):
    """
    Обработчик данных моделирования для сеточного накопления дозы (энерговыделения)
    в трехмерных воксельных объемах произвольного количества узлов DoseGridNode.
    Координаты взаимодействий переводятся в локальную систему координат каждого узла
    через inverse_global_matrix.
    """

    def __init__(
        self,
        grid_nodes: Optional[Sequence[DoseGridNode]] = None,
        shm_name: str = "nmsim_dose_shm",
        create_shm: bool = True,
    ) -> None:
        super().__init__()
        self.entries: List[DoseGridEntry] = []
        self._entries_by_name: Dict[str, DoseGridEntry] = {}

        if grid_nodes is not None:
            for node in grid_nodes:
                if not node.is_active:
                    continue
                node_name = node.name
                node_shm = f"{shm_name}_{node_name}_{abs(id(node))}"
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
                )
                self.entries.append(entry)
                self._entries_by_name[node_name] = entry

    def add_grid_node(self, node: DoseGridNode, shm_prefix: Optional[str] = None) -> None:
        """Динамическое добавление узла DoseGridNode в обработчик."""
        node_name = node.name
        prefix = shm_prefix or "nmsim_dose_shm"
        node_shm = f"{prefix}_{node_name}_{abs(id(node))}"
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
        )
        self.entries.append(entry)
        self._entries_by_name[node_name] = entry


    def process_chunk(self, chunk: Dict[str, Any]) -> None:
        """
        Векторизованное накопление энерговыделения из чанка взаимодействий
        для всех зарегистрированных сеток дозы.
        """
        if chunk.get('type') != 'interactions':
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

        pos_x = np.asarray(pos_x)
        pos_y = np.asarray(pos_y)
        pos_z = np.asarray(pos_z) if pos_z is not None else np.zeros_like(pos_x)
        energy_deposit = np.asarray(energy_deposit)

        # Отбираем события с депонированием энергии > 0
        valid_mask = energy_deposit > 0.0
        if not np.any(valid_mask):
            return

        px = pos_x[valid_mask]
        py = pos_y[valid_mask]
        pz = pos_z[valid_mask]
        edep = energy_deposit[valid_mask]

        for entry in self.entries:
            entry.accumulate(px, py, pz, edep)

    def get_dose_snapshot(self, name: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Возвращает локальную копию 3D-массива дозы.
        Если указано имя name, возвращает массив указанной сетки.
        Если name не указано и зарегистрирована одна сетка, возвращает ее массив.
        """
        if name is not None:
            entry = self._entries_by_name.get(name)
            return entry._dose_grid.copy() if entry and entry._dose_grid is not None else None
        if len(self.entries) == 1:
            entry = self.entries[0]
            return entry._dose_grid.copy() if entry and entry._dose_grid is not None else None
        if len(self.entries) > 1:
            return {e.name: e._dose_grid.copy() for e in self.entries if e._dose_grid is not None}
        return None

    def get_dose_snapshots(self) -> Dict[str, np.ndarray]:
        """
        Возвращает словарь локальных копий всех сеток {имя: ndarray}.
        """
        return {e.name: e._dose_grid.copy() for e in self.entries if e._dose_grid is not None}

    def clear(self) -> None:
        """
        Сброс накопленной дозы во всех сетках в ноль.
        """
        for entry in self.entries:
            entry.clear()

    def finalize(self) -> None:
        """
        Сохранение накопленных дозовых карт в группу /dose HDF5 файла.
        """
        if self.writer_callback is not None:
            def write_func(f: h5py.File) -> None:
                if 'dose' not in f:
                    dose_group = f.create_group('dose')
                else:
                    dose_group = f['dose']
                for entry in self.entries:
                    if entry._dose_grid is not None:
                        if entry.name in dose_group:
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

