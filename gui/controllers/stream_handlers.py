import logging
import queue
import time
from multiprocessing import Queue as MpQueue
from multiprocessing import shared_memory
from collections.abc import Mapping
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from core.data.data_handlers import BaseDataHandler, SensitiveVolumeHandler
from core.geometry.volumes import Volume
from core.physics.interaction_buffers import ChunkType
from core.scene.gamma_camera_node import GammaCameraNode
from core.scene.nodes import CompositeNode, SpatialNode

_logger = logging.getLogger(__name__)


DEFAULT_CTRL_SHM_NAME = "nmsim_ctrl_shm"


def get_task_projection_shm_name(base_shm_name: str, task_id: int) -> str:
    """Формирует имя сегмента SharedMemory 2D-проекции для конкретной задачи."""
    return f"{base_shm_name}_task_{int(task_id)}"


def get_task_telemetry_shm_name(base_shm_name: str, task_id: int) -> str:
    """Формирует имя сегмента SharedMemory кольцевого буфера треков и спектра для задачи."""
    return f"{base_shm_name}_telem_{int(task_id)}"


def get_task_dose_shm_name(
    base_dose_shm: str,
    task_id: int,
    grid_name: Optional[str] = None,
    multi_grid: bool = False,
) -> str:
    """Формирует имя сегмента SharedMemory 3D-карты дозы для конкретной задачи."""
    task_prefix = f"{base_dose_shm}_task_{int(task_id)}"
    if multi_grid and grid_name:
        return f"{task_prefix}_{grid_name}"
    return task_prefix


class ControlBufferSHM:
    """
    Легковесный буфер управления фокусом и двухуровневой паузой в SharedMemory.
    Разметка памяти (32 байта, int64[4]):
      - flags[0]: focused_task_id (индекс текущей сфокусированной задачи)
      - flags[1]: global_run_flag (1 — запущено, 0 — глобальная пауза всех воркеров)
      - flags[2]: focused_run_flag (1 — фокус запущен, 0 — локальная пауза сфокусированной задачи)
      - flags[3]: step_trigger_flag (1 — запрошен одиночный шаг для сфокусированной задачи, 0 — нет)
    """

    BYTE_SIZE: int = 32

    def __init__(
        self,
        shm_name: str = DEFAULT_CTRL_SHM_NAME,
        create: bool = False,
        initial_focused_job: int = 0,
    ) -> None:
        self.shm_name: str = shm_name
        self._owns_shm: bool = create
        self._shm: Optional[shared_memory.SharedMemory] = None

        if create:
            try:
                self._shm = shared_memory.SharedMemory(name=shm_name, create=True, size=self.BYTE_SIZE)
            except FileExistsError:
                existing_shm = shared_memory.SharedMemory(name=shm_name, create=False)
                if existing_shm.size >= self.BYTE_SIZE:
                    self._shm = existing_shm
                    self._owns_shm = False
                else:
                    try:
                        existing_shm.close()
                        existing_shm.unlink()
                    except (FileNotFoundError, OSError, ValueError, BufferError):
                        pass
                    self._shm = shared_memory.SharedMemory(name=shm_name, create=True, size=self.BYTE_SIZE)
                    self._owns_shm = True
            self.flags: Optional[np.ndarray] = np.ndarray((4,), dtype=np.int64, buffer=self._shm.buf)
            self.flags[0] = int(initial_focused_job)
            self.flags[1] = 1
            self.flags[2] = 1
            self.flags[3] = 0
        else:
            self._shm = shared_memory.SharedMemory(name=shm_name, create=False)
            self.flags = np.ndarray((4,), dtype=np.int64, buffer=self._shm.buf)

    @property
    def focused_job_index(self) -> int:
        """Индекс текущей сфокусированной задачи."""
        if self.flags is None:
            return 0
        return int(self.flags[0])

    @focused_job_index.setter
    def focused_job_index(self, value: int) -> None:
        if self.flags is not None:
            self.flags[0] = int(value)

    @property
    def focused_task_id(self) -> int:
        """Синоним индекса текущей сфокусированной задачи."""
        if self.flags is None:
            return 0
        return int(self.flags[0])

    @focused_task_id.setter
    def focused_task_id(self, value: int) -> None:
        if self.flags is not None:
            self.flags[0] = int(value)

    @property
    def global_run(self) -> bool:
        """Флаг глобального разрешения работы всех воркеров пула."""
        if self.flags is None:
            return True
        return bool(int(self.flags[1]) > 0)

    @global_run.setter
    def global_run(self, value: bool) -> None:
        if self.flags is not None:
            self.flags[1] = 1 if value else 0

    @property
    def stop_requested(self) -> bool:
        """Флаг запроса кооперативной остановки всех воркеров пула."""
        if self.flags is None:
            return False
        return bool(int(self.flags[1]) < 0)

    @stop_requested.setter
    def stop_requested(self, value: bool) -> None:
        if self.flags is not None:
            self.flags[1] = -1 if value else 1

    @property
    def focused_run(self) -> bool:
        """Флаг разрешения работы сфокусированного воркера."""
        if self.flags is None:
            return True
        return bool(int(self.flags[2]) != 0)

    @focused_run.setter
    def focused_run(self, value: bool) -> None:
        if self.flags is not None:
            self.flags[2] = 1 if value else 0

    def trigger_step(self) -> None:
        """Взводит триггер одиночного шага для сфокусированной задачи."""
        if self.flags is not None:
            self.flags[3] = 1

    def consume_step_trigger(self) -> bool:
        """Потребляет триггер одиночного шага, если он был взведен."""
        if self.flags is not None and int(self.flags[3]) != 0:
            self.flags[3] = 0
            return True
        return False

    def close(self, unlink: bool = False) -> None:
        """Закрывает сегмент SharedMemory и при необходимости удаляет его."""
        self.flags = None
        if self._shm is not None:
            try:
                self._shm.close()
                if unlink:
                    try:
                        self._shm.unlink()
                    except FileNotFoundError:
                        pass
            except (FileNotFoundError, OSError, ValueError, BufferError) as err:
                _logger.debug(f"Ошибка при закрытии ControlBufferSHM '{self.shm_name}': {err}")
            finally:
                self._shm = None


class TelemetryRingBufferSHM:
    """
    Кольцевой SoA-буфер 3D-треков и энергетического спектра с курсором в SharedMemory.
    Выровненная разметка памяти от 8-байтовых к 4- и 2-байтовым полям:
      - Offset 0: header — int64[4] (32 байта: track_write_cursor, spectrum_write_cursor, C_t, C_s)
      - Offset 32: particle_id — int64[C_t] (8 * C_t байт)
      - Offset 32 + 8 * C_t: pos_x — float32[C_t] (4 * C_t байт)
      - Offset 32 + 12 * C_t: pos_y — float32[C_t] (4 * C_t байт)
      - Offset 32 + 16 * C_t: pos_z — float32[C_t] (4 * C_t байт)
      - Offset 32 + 20 * C_t: detector_energies — float32[C_s] (4 * C_s байт, МэВ)
      - Offset 32 + 20 * C_t + 4 * C_s: process_id — int16[C_t] (2 * C_t байт)
    """

    DEFAULT_TRACK_CAPACITY: int = 10000
    DEFAULT_SPECTRUM_CAPACITY: int = 25000
    HEADER_LONGS: int = 4

    @classmethod
    def compute_byte_size(
        cls,
        track_capacity: int = DEFAULT_TRACK_CAPACITY,
        spectrum_capacity: int = DEFAULT_SPECTRUM_CAPACITY,
    ) -> int:
        """Вычисляет общий размер сегмента SharedMemory в байтах."""
        return 32 + 22 * int(track_capacity) + 4 * int(spectrum_capacity)

    def __init__(
        self,
        shm_name: str,
        create: bool = False,
        track_capacity: int = DEFAULT_TRACK_CAPACITY,
        spectrum_capacity: int = DEFAULT_SPECTRUM_CAPACITY,
    ) -> None:
        self.shm_name: str = shm_name
        self.track_capacity: int = int(track_capacity)
        self.spectrum_capacity: int = int(spectrum_capacity)
        self._owns_shm: bool = create
        self._shm: Optional[shared_memory.SharedMemory] = None

        total_bytes = self.compute_byte_size(self.track_capacity, self.spectrum_capacity)

        if create:
            try:
                self._shm = shared_memory.SharedMemory(name=shm_name, create=True, size=total_bytes)
            except FileExistsError:
                existing_shm = shared_memory.SharedMemory(name=shm_name, create=False)
                if existing_shm.size >= total_bytes:
                    self._shm = existing_shm
                    self._owns_shm = False
                else:
                    try:
                        existing_shm.close()
                        existing_shm.unlink()
                    except (FileNotFoundError, OSError, ValueError, BufferError):
                        pass
                    self._shm = shared_memory.SharedMemory(name=shm_name, create=True, size=total_bytes)
                    self._owns_shm = True
        else:
            self._shm = shared_memory.SharedMemory(name=shm_name, create=False)
            header_preview = np.ndarray((self.HEADER_LONGS,), dtype=np.int64, buffer=self._shm.buf, offset=0)
            stored_track_cap = int(header_preview[2])
            stored_spec_cap = int(header_preview[3])
            del header_preview
            if stored_track_cap > 0 and stored_spec_cap > 0:
                self.track_capacity = stored_track_cap
                self.spectrum_capacity = stored_spec_cap

        c_t = self.track_capacity
        c_s = self.spectrum_capacity
        buf = self._shm.buf

        offset_header = 0
        offset_particle_id = 32
        offset_pos_x = offset_particle_id + 8 * c_t
        offset_pos_y = offset_pos_x + 4 * c_t
        offset_pos_z = offset_pos_y + 4 * c_t
        offset_energies = offset_pos_z + 4 * c_t
        offset_process_id = offset_energies + 4 * c_s

        self.header: Optional[np.ndarray] = np.ndarray((self.HEADER_LONGS,), dtype=np.int64, buffer=buf, offset=offset_header)
        self.particle_id: Optional[np.ndarray] = np.ndarray((c_t,), dtype=np.int64, buffer=buf, offset=offset_particle_id)
        self.pos_x: Optional[np.ndarray] = np.ndarray((c_t,), dtype=np.float32, buffer=buf, offset=offset_pos_x)
        self.pos_y: Optional[np.ndarray] = np.ndarray((c_t,), dtype=np.float32, buffer=buf, offset=offset_pos_y)
        self.pos_z: Optional[np.ndarray] = np.ndarray((c_t,), dtype=np.float32, buffer=buf, offset=offset_pos_z)
        self.detector_energies: Optional[np.ndarray] = np.ndarray((c_s,), dtype=np.float32, buffer=buf, offset=offset_energies)
        self.process_id: Optional[np.ndarray] = np.ndarray((c_t,), dtype=np.int16, buffer=buf, offset=offset_process_id)

        if create:
            self.header[0] = 0
            self.header[1] = 0
            self.header[2] = c_t
            self.header[3] = c_s
            self.particle_id.fill(0)
            self.pos_x.fill(0.0)
            self.pos_y.fill(0.0)
            self.pos_z.fill(0.0)
            self.detector_energies.fill(0.0)
            self.process_id.fill(0)

    @property
    def track_write_cursor(self) -> int:
        """Монотонно растущий курсор числа записанных точек треков."""
        if self.header is None:
            return 0
        return int(self.header[0])

    @property
    def spectrum_write_cursor(self) -> int:
        """Монотонно растущий курсор числа записанных сэмплов спектра."""
        if self.header is None:
            return 0
        return int(self.header[1])

    def write_tracks(
        self,
        pos_x: np.ndarray,
        pos_y: np.ndarray,
        pos_z: np.ndarray,
        particle_id: Optional[np.ndarray] = None,
        process_id: Optional[np.ndarray] = None,
    ) -> int:
        """Записывает батч точек 3D-треков в кольцевой буфер и возвращает новый курсор."""
        if (
            self.header is None
            or self.pos_x is None
            or self.pos_y is None
            or self.pos_z is None
            or self.particle_id is None
            or self.process_id is None
        ):
            return 0
        px = np.asarray(pos_x, dtype=np.float32).ravel()
        num_points = len(px)
        cur = int(self.header[0])
        if num_points == 0:
            return cur

        py = np.asarray(pos_y, dtype=np.float32).ravel()
        pz = np.asarray(pos_z, dtype=np.float32).ravel()
        pids = (
            np.asarray(particle_id, dtype=np.int64).ravel()
            if particle_id is not None
            else np.zeros(num_points, dtype=np.int64)
        )
        procs = (
            np.asarray(process_id, dtype=np.int16).ravel()
            if process_id is not None
            else np.zeros(num_points, dtype=np.int16)
        )

        cap = self.track_capacity
        if num_points >= cap:
            px = px[-cap:]
            py = py[-cap:]
            pz = pz[-cap:]
            pids = pids[-cap:]
            procs = procs[-cap:]
            indices = np.arange(cur + num_points - cap, cur + num_points, dtype=np.int64) % cap
        else:
            indices = np.arange(cur, cur + num_points, dtype=np.int64) % cap

        self.pos_x[indices] = px
        self.pos_y[indices] = py
        self.pos_z[indices] = pz
        self.particle_id[indices] = pids
        self.process_id[indices] = procs
        self.header[0] = cur + num_points
        return cur + num_points

    def write_spectrum(self, energies: np.ndarray) -> int:
        """Записывает положительные энерговыделения детектора в кольцо спектра."""
        if self.header is None or self.detector_energies is None:
            return 0
        cur = int(self.header[1])
        if energies is None:
            return cur
        valid = np.asarray(energies, dtype=np.float32).ravel()
        valid = valid[valid > 0.0]
        num_samples = len(valid)
        if num_samples == 0:
            return cur

        cap = self.spectrum_capacity
        if num_samples >= cap:
            valid = valid[-cap:]
            indices = np.arange(cur + num_samples - cap, cur + num_samples, dtype=np.int64) % cap
        else:
            indices = np.arange(cur, cur + num_samples, dtype=np.int64) % cap

        self.detector_energies[indices] = valid
        self.header[1] = cur + num_samples
        return cur + num_samples

    def read_tracks_since(self, last_cursor: int) -> Tuple[int, Optional[Dict[str, Any]]]:
        """Читает непрочитанные точки треков, появившиеся после last_cursor."""
        if (
            self.header is None
            or self.pos_x is None
            or self.pos_y is None
            or self.pos_z is None
            or self.particle_id is None
            or self.process_id is None
        ):
            return 0, None
        cur = int(self.header[0])
        if cur <= 0 or cur <= last_cursor:
            return cur, None
        n_unread = min(cur - max(0, int(last_cursor)), self.track_capacity)
        start = cur - n_unread
        indices = np.arange(start, cur, dtype=np.int64) % self.track_capacity
        return (
            cur,
            {
                "pos_x": self.pos_x[indices].copy(),
                "pos_y": self.pos_y[indices].copy(),
                "pos_z": self.pos_z[indices].copy(),
                "particle_id": self.particle_id[indices].copy(),
                "process_id": self.process_id[indices].copy(),
            },
        )

    def read_all_tracks(self) -> Tuple[int, Optional[Dict[str, Any]]]:
        """Читает все доступные в кольце точки треков."""
        return self.read_tracks_since(0)

    def read_spectrum(self) -> Tuple[int, np.ndarray]:
        """Читает накопленные значения энерговыделений детектора из кольцевого буфера."""
        if self.header is None or self.detector_energies is None:
            return 0, np.empty(0, dtype=np.float32)
        cur = int(self.header[1])
        if cur <= 0:
            return 0, np.empty(0, dtype=np.float32)
        if cur <= self.spectrum_capacity:
            return cur, self.detector_energies[:cur].copy()
        indices = np.arange(cur - self.spectrum_capacity, cur, dtype=np.int64) % self.spectrum_capacity
        return cur, self.detector_energies[indices].copy()

    def read_spectrum_since(self, last_cursor: int) -> Tuple[int, np.ndarray]:
        """Читает новые значения энерговыделений детектора, записанные после last_cursor."""
        if self.header is None or self.detector_energies is None:
            return 0, np.empty(0, dtype=np.float32)
        cur = int(self.header[1])
        if cur <= 0 or cur <= last_cursor:
            return cur, np.empty(0, dtype=np.float32)
        n_unread = min(cur - max(0, int(last_cursor)), self.spectrum_capacity)
        start = cur - n_unread
        indices = np.arange(start, cur, dtype=np.int64) % self.spectrum_capacity
        return cur, self.detector_energies[indices].copy()

    def clear(self) -> None:
        """Сбрасывает курсоры и обнуляет массивы кольцевого буфера."""
        if (
            self.header is None
            or self.particle_id is None
            or self.pos_x is None
            or self.pos_y is None
            or self.pos_z is None
            or self.detector_energies is None
            or self.process_id is None
        ):
            return
        self.header[0] = 0
        self.header[1] = 0
        self.particle_id.fill(0)
        self.pos_x.fill(0.0)
        self.pos_y.fill(0.0)
        self.pos_z.fill(0.0)
        self.detector_energies.fill(0.0)
        self.process_id.fill(0)

    def close(self, unlink: bool = False) -> None:
        """Закрывает сегмент SharedMemory и при необходимости удаляет его."""
        self.header = None
        self.particle_id = None
        self.pos_x = None
        self.pos_y = None
        self.pos_z = None
        self.detector_energies = None
        self.process_id = None
        if self._shm is not None:
            try:
                self._shm.close()
                if unlink:
                    try:
                        self._shm.unlink()
                    except FileNotFoundError:
                        pass
            except (FileNotFoundError, OSError, ValueError, BufferError) as err:
                _logger.debug(f"Ошибка при закрытии TelemetryRingBufferSHM '{self.shm_name}': {err}")
            finally:
                self._shm = None


class GuiStreamDataHandler(SensitiveVolumeHandler):
    """
    Обработчик потоковых данных для передачи телеметрии в реальном времени в GUI.
    Наследуется от SensitiveVolumeHandler для единого сопоставления чувствительных объемов
    через FlattenedScene. Передает пакеты 3D-треков через TelemetryRingBufferSHM (или multiprocessing.Queue)
    и обновляет накопленную 2D-проекцию детектора в multiprocessing.shared_memory.SharedMemory.
    """

    _CONSUMED_WITH_ESCAPED: FrozenSet[ChunkType] = frozenset({
        ChunkType.INITIAL_STATES,
        ChunkType.INTERACTIONS,
        ChunkType.ESCAPED_PARTICLES,
    })
    _CONSUMED_WITHOUT_ESCAPED: FrozenSet[ChunkType] = frozenset({
        ChunkType.INITIAL_STATES,
        ChunkType.INTERACTIONS,
    })

    def __init__(
        self,
        track_queue: Optional[Any] = None,
        shm_name: Optional[str] = None,
        projection_shape: Tuple[int, int] = (128, 128),
        projection_dtype: Any = np.float32,
        create_shm: bool = False,
        sensitive_volume_ids: Optional[Set[int]] = None,
        max_tracks_per_batch: int = 2000,
        show_escaped_tracks: bool = False,
        detector_volume: Optional[SpatialNode] = None,
        detector_size: Optional[Union[Tuple[float, float], Sequence[float], float]] = None,
        detector_inv_matrix: Optional[np.ndarray] = None,
        sensitive_volumes: Optional[Sequence[SpatialNode]] = None,
        telemetry_shm_name: Optional[str] = None,
        max_points_per_batch: Optional[int] = None,
        pause_proxy: Optional[Any] = None,
    ) -> None:
        """
        Инициализация обработчика потоковых данных GUI.

        :param track_queue: Очередь для отправки 3D треков в UI (legacy/fallback).
        :param shm_name: Имя блока разделяемой памяти SharedMemory для 2D-проекции.
        :param projection_shape: Разрешение матрицы проекции (высота, ширина).
        :param projection_dtype: Тип данных матрицы проекции.
        :param create_shm: Создавать ли блок разделяемой памяти (True) или подключаться к существующему (False).
        :param sensitive_volume_ids: Множество ID чувствительных объемов для физического отбора проекций.
        :param max_tracks_per_batch: Максимальное число точек треков в одном пакете.
        :param show_escaped_tracks: Передавать ли треки вылетевших без взаимодействия частиц.
        :param detector_volume: Экземпляр чувствительного объема (Volume) детектора.
        :param detector_size: Физические размеры кристалла детектора (Ширина, Высота) в мм.
        :param detector_inv_matrix: Обратная матрица трансформации детектора (Мир -> Локальный).
        :param sensitive_volumes: Последовательность чувствительных узлов сцены для фильтрации детектора.
        :param telemetry_shm_name: Имя сегмента SharedMemory кольцевого буфера треков и спектра.
        :param max_points_per_batch: Синоним max_tracks_per_batch для настройки лимита точек в батче.
        :param pause_proxy: Опциональная ссылка на FocusedPauseProxy для закрытия при завершении.
        """
        if sensitive_volumes:
            initial_sensitive_volumes: List[SpatialNode] = [
                node for node in sensitive_volumes if isinstance(node, SpatialNode)
            ]
        elif isinstance(detector_volume, SpatialNode):
            initial_sensitive_volumes = [detector_volume]
        else:
            initial_sensitive_volumes = []

        super().__init__(sensitive_volumes=initial_sensitive_volumes)

        self.track_queue = track_queue
        self.shm_name = shm_name
        self.telemetry_shm_name = telemetry_shm_name
        self.projection_shape = projection_shape
        self.projection_dtype = np.dtype(projection_dtype)
        self.max_tracks_per_batch = (
            int(max_points_per_batch) if max_points_per_batch is not None else int(max_tracks_per_batch)
        )
        self.show_escaped_tracks = show_escaped_tracks
        self._pause_proxy = pause_proxy
        self.detector_volume: Optional[SpatialNode] = (
            detector_volume
            if detector_volume is not None
            else (self.sensitive_volumes[0] if self.sensitive_volumes else None)
        )
        self.detector_inv_matrix = detector_inv_matrix
        self._explicit_detector_size: Optional[Tuple[float, float]] = None

        if sensitive_volume_ids is not None:
            self.sensitive_volume_ids: Optional[Set[int]] = set(sensitive_volume_ids) | set(self.target_volumes)
        else:
            self.sensitive_volume_ids = set(self.target_volumes) if self.target_volumes else None

        if detector_size is not None:
            if isinstance(detector_size, (int, float)):
                self.detector_size = (float(detector_size), float(detector_size))
            else:
                self.detector_size = (float(detector_size[0]), float(detector_size[1]))
            self._explicit_detector_size = self.detector_size
        elif self.detector_volume is not None:
            if isinstance(self.detector_volume, Volume):
                vol_size = self.detector_volume.local_bound
                self.detector_size = (float(vol_size[0]), float(vol_size[1]))
            else:
                self.detector_size = (400.0, 400.0)
        else:
            self.detector_size = (400.0, 400.0)

        if (
            self.detector_volume is not None
            and self.detector_inv_matrix is None
            and isinstance(self.detector_volume, SpatialNode)
        ):
            self.detector_inv_matrix = self.detector_volume.inverse_global_matrix

        self._shm: Optional[shared_memory.SharedMemory] = None
        self._projection_array: Optional[np.ndarray] = None
        self._owns_shm = create_shm

        if shm_name is not None:
            nbytes = int(np.prod(projection_shape) * self.projection_dtype.itemsize)
            if create_shm:
                try:
                    self._shm = shared_memory.SharedMemory(name=shm_name, create=True, size=nbytes)
                except FileExistsError:
                    self._shm = shared_memory.SharedMemory(name=shm_name, create=False)
            else:
                try:
                    self._shm = shared_memory.SharedMemory(name=shm_name, create=False)
                except FileNotFoundError:
                    _logger.warning(f"Блок SharedMemory '{shm_name}' не найден. Проекции не будут сохраняться в SHM.")
                    self._shm = None

            if self._shm is not None:
                self._projection_array = np.ndarray(
                    projection_shape,
                    dtype=self.projection_dtype,
                    buffer=self._shm.buf,
                )
                if create_shm:
                    self._projection_array.fill(0)

        self._telemetry_ring: Optional[TelemetryRingBufferSHM] = None
        if telemetry_shm_name is not None:
            try:
                self._telemetry_ring = TelemetryRingBufferSHM(shm_name=telemetry_shm_name, create=False)
            except FileNotFoundError:
                _logger.warning(
                    f"Кольцевой буфер телеметрии SharedMemory '{telemetry_shm_name}' не найден."
                )
                self._telemetry_ring = None

    def set_writer_callback(self, callback: Any) -> None:
        """
        Отключение записи в HDF5 для потокового GUI-обработчика (no-op).
        """
        pass

    @property
    def consumed_chunk_types(self) -> FrozenSet[ChunkType]:
        return self._CONSUMED_WITH_ESCAPED if self.show_escaped_tracks else self._CONSUMED_WITHOUT_ESCAPED

    def process_chunk(self, chunk: Dict[str, Any]) -> None:
        """
        Обработка чанка данных моделирования из очереди.
        """
        chunk_type = chunk.get('type')
        data = chunk.get('data')

        if not isinstance(data, (dict, Mapping)):
            return

        if chunk_type == ChunkType.INTERACTIONS:
            self._handle_interactions(data)
        elif chunk_type == ChunkType.INITIAL_STATES:
            self._handle_initial_states(data)
        elif chunk_type == ChunkType.ESCAPED_PARTICLES and self.show_escaped_tracks:
            self._handle_escaped_particles(data)

    def _handle_initial_states(self, data: Mapping[str, Any]) -> None:
        """
        Обработка начальных состояний частиц: запись точек рождения в кольцевой буфер SHM
        или отправка в очередь треков для построения полных траекторий от источника к взаимодействиям.
        """
        if self._telemetry_ring is None and self.track_queue is None:
            return

        pos_x = data.get('pos_x')
        pos_y = data.get('pos_y')
        pos_z = data.get('pos_z')
        particle_id = data.get('particle_ID') if 'particle_ID' in data else data.get('particle_id')

        if pos_x is None or len(pos_x) == 0:
            return

        total_particles = len(pos_x)
        batch_limit = min(total_particles, self.max_tracks_per_batch)
        batch_slice = slice(0, batch_limit)

        batch_x = np.asarray(pos_x)[batch_slice]
        batch_y = np.asarray(pos_y)[batch_slice]
        batch_z = np.asarray(pos_z)[batch_slice] if pos_z is not None else np.zeros_like(batch_x)
        particle_ids = np.asarray(particle_id)[batch_slice] if particle_id is not None else None

        if self._telemetry_ring is not None:
            proc_arr = np.full(len(batch_x), -2, dtype=np.int16)
            pids_arr = (
                np.asarray(particle_ids, dtype=np.int64)
                if particle_ids is not None
                else np.zeros(len(batch_x), dtype=np.int64)
            )
            self._telemetry_ring.write_tracks(
                pos_x=batch_x,
                pos_y=batch_y,
                pos_z=batch_z,
                particle_id=pids_arr,
                process_id=proc_arr,
            )
        elif self.track_queue is not None:
            birth_payload = {
                'type': 'tracks',
                'pos_x': np.array(batch_x, copy=True),
                'pos_y': np.array(batch_y, copy=True),
                'pos_z': np.array(batch_z, copy=True),
                'process_id': np.full(len(batch_x), -2, dtype=np.int32),
                'particle_id': np.array(particle_ids, copy=True) if particle_ids is not None else None,
                'energy_deposit': np.zeros(len(batch_x), dtype=np.float32),
                'detector_energy_deposit': None,
            }
            try:
                self.track_queue.put_nowait(birth_payload)
            except queue.Full:
                pass

    _handle_born = _handle_initial_states

    def _handle_escaped_particles(self, data: Mapping[str, Any]) -> None:
        """
        Обработка вылетевших частиц: запись граничной точки выхода в кольцевой буфер SHM
        или отправка в очередь треков для завершения траекторий вылетевших квантов.
        """
        if self._telemetry_ring is None and self.track_queue is None:
            return

        pos_x = data.get('pos_x')
        pos_y = data.get('pos_y')
        pos_z = data.get('pos_z')
        particle_id = data.get('particle_ID') if 'particle_ID' in data else data.get('particle_id')

        if pos_x is None or len(pos_x) == 0:
            return

        total_escaped = len(pos_x)
        limit = min(total_escaped, self.max_tracks_per_batch)
        sub = slice(0, limit)

        px = np.asarray(pos_x)[sub]
        py = np.asarray(pos_y)[sub]
        pz = np.asarray(pos_z)[sub] if pos_z is not None else np.zeros_like(px)
        pids = np.asarray(particle_id)[sub] if particle_id is not None else None

        if self._telemetry_ring is not None:
            proc_arr = np.full(len(px), -1, dtype=np.int16)
            pids_arr = (
                np.asarray(pids, dtype=np.int64)
                if pids is not None
                else np.zeros(len(px), dtype=np.int64)
            )
            self._telemetry_ring.write_tracks(
                pos_x=px,
                pos_y=py,
                pos_z=pz,
                particle_id=pids_arr,
                process_id=proc_arr,
            )
        elif self.track_queue is not None:
            escape_payload = {
                'type': 'tracks',
                'pos_x': np.array(px, copy=True),
                'pos_y': np.array(py, copy=True),
                'pos_z': np.array(pz, copy=True),
                'process_id': np.full(len(px), -1, dtype=np.int32),
                'particle_id': np.array(pids, copy=True) if pids is not None else None,
                'energy_deposit': np.zeros(len(px), dtype=np.float32),
                'detector_energy_deposit': None,
            }
            try:
                self.track_queue.put_nowait(escape_payload)
            except queue.Full:
                pass

    _handle_escaped = _handle_escaped_particles

    def _project_points_for_node(
        self,
        coords: np.ndarray,
        target_node: Optional[SpatialNode],
    ) -> None:
        """
        Переводит мировые координаты точек в локальную систему координат целевого узла детектора,
        проверяет попадание в 3D-габариты кристалла (lx, ly, lz) и инкрементирует матрицу проекции.
        """
        if self._projection_array is None or len(coords) == 0:
            return

        if isinstance(target_node, SpatialNode):
            local_pos = target_node.convert_to_local_position(coords)
            lx = local_pos[:, 0]
            ly = local_pos[:, 1]
            lz = local_pos[:, 2]
        elif self.detector_inv_matrix is not None:
            local_pos = np.ones((len(coords), 4), dtype=coords.dtype)
            local_pos[:, :3] = coords
            np.matmul(local_pos, self.detector_inv_matrix.T.astype(coords.dtype), out=local_pos)
            lx = local_pos[:, 0]
            ly = local_pos[:, 1]
            lz = local_pos[:, 2]
        else:
            lx = coords[:, 0]
            ly = coords[:, 1]
            lz = coords[:, 2]

        half_z: Optional[float] = None
        if isinstance(target_node, Volume):
            vol_bound = target_node.local_bound
            if self._explicit_detector_size is None:
                size_x = float(vol_bound[0])
                size_y = float(vol_bound[1])
            else:
                size_x, size_y = self.detector_size
            half_z = float(vol_bound[2]) / 2.0
        else:
            size_x, size_y = self.detector_size

        half_x = size_x / 2.0
        half_y = size_y / 2.0

        eps = 1e-3
        in_crystal = (
            (lx >= -half_x - eps)
            & (lx <= half_x + eps)
            & (ly >= -half_y - eps)
            & (ly <= half_y + eps)
        )
        if half_z is not None:
            in_crystal = in_crystal & (np.abs(lz) <= half_z + eps)

        lx = lx[in_crystal]
        ly = ly[in_crystal]

        if len(lx) > 0:
            height, width = self.projection_shape
            ix = np.clip(((lx / size_x) + 0.5) * (width - 1), 0, width - 1).astype(np.int32)
            iy = np.clip(((ly / size_y) + 0.5) * (height - 1), 0, height - 1).astype(np.int32)
            np.add.at(self._projection_array, (iy, ix), 1.0)

    def _handle_interactions(self, data: Mapping[str, Any]) -> None:
        """
        Обработка взаимодействий: запись треков и спектра в кольцевой буфер SHM
        (или очередь треков) и проецирование на детектор.
        """
        pos_x = data.get('pos_x')
        pos_y = data.get('pos_y')
        pos_z = data.get('pos_z')
        process_id = data.get('process_id')
        particle_id = data.get('particle_ID') if 'particle_ID' in data else data.get('particle_id')
        energy_deposit = data.get('energy_deposit')
        volume_id = data.get('volume_id') if 'volume_id' in data else data.get('volume_ID')

        if pos_x is None or len(pos_x) == 0:
            return

        pos_x = np.asarray(pos_x)
        pos_y = np.asarray(pos_y)
        if pos_z is not None:
            pos_z = np.asarray(pos_z)

        n_points = len(pos_x)

        if self._telemetry_ring is not None or self.track_queue is not None:
            if n_points <= self.max_tracks_per_batch:
                sub_indices = slice(0, n_points)
            else:
                sub_indices = slice(0, self.max_tracks_per_batch)

            det_edep: Optional[np.ndarray] = None
            if self.sensitive_volume_ids is not None:
                if volume_id is not None and energy_deposit is not None:
                    det_mask = np.isin(volume_id, list(self.sensitive_volume_ids))
                    if np.any(det_mask):
                        det_edep = np.asarray(energy_deposit, dtype=np.float32)[det_mask]
                    else:
                        det_edep = np.empty(0, dtype=np.float32)
                else:
                    det_edep = np.empty(0, dtype=np.float32)

            if self._telemetry_ring is not None:
                sub_x = pos_x[sub_indices]
                sub_y = pos_y[sub_indices]
                sub_z = (
                    pos_z[sub_indices]
                    if pos_z is not None
                    else np.zeros(len(sub_x), dtype=np.float32)
                )
                sub_pids = (
                    np.asarray(particle_id)[sub_indices]
                    if particle_id is not None
                    else np.zeros(len(sub_x), dtype=np.int64)
                )
                sub_procs = (
                    np.asarray(process_id)[sub_indices]
                    if process_id is not None
                    else np.zeros(len(sub_x), dtype=np.int16)
                )
                self._telemetry_ring.write_tracks(
                    pos_x=sub_x,
                    pos_y=sub_y,
                    pos_z=sub_z,
                    particle_id=sub_pids,
                    process_id=sub_procs,
                )
                if det_edep is not None:
                    if len(det_edep) > 0:
                        self._telemetry_ring.write_spectrum(det_edep)
                elif energy_deposit is not None:
                    self._telemetry_ring.write_spectrum(np.asarray(energy_deposit, dtype=np.float32))
            elif self.track_queue is not None:
                track_payload = {
                    'type': 'tracks',
                    'pos_x': np.array(pos_x[sub_indices], copy=True),
                    'pos_y': np.array(pos_y[sub_indices], copy=True),
                    'pos_z': np.array(pos_z[sub_indices], copy=True) if pos_z is not None else None,
                    'process_id': np.array(process_id[sub_indices], copy=True) if process_id is not None else None,
                    'particle_id': np.array(particle_id[sub_indices], copy=True) if particle_id is not None else None,
                    'energy_deposit': np.array(energy_deposit[sub_indices], copy=True) if energy_deposit is not None else None,
                    'detector_energy_deposit': np.array(det_edep, copy=True) if det_edep is not None else None,
                }
                try:
                    self.track_queue.put_nowait(track_payload)
                except (queue.Full, OSError, RuntimeError, ValueError, BufferError):
                    pass

        if self._projection_array is not None:
            filtered_vids: Optional[np.ndarray] = None
            if self.sensitive_volume_ids is not None:
                if volume_id is None:
                    return
                volume_id_arr = np.asarray(volume_id)
                mask = np.isin(volume_id_arr, list(self.sensitive_volume_ids))
                if not np.any(mask):
                    return
                px = pos_x[mask]
                py = pos_y[mask]
                pz = pos_z[mask] if pos_z is not None else np.zeros_like(px)
                filtered_vids = volume_id_arr[mask]
            else:
                px = pos_x
                py = pos_y
                pz = pos_z if pos_z is not None else np.zeros_like(px)
                if volume_id is not None:
                    filtered_vids = np.asarray(volume_id)

            if len(px) == 0:
                return

            coords = np.column_stack((px, py, pz))

            if filtered_vids is not None and self.volume_mapping:
                for unique_vid in np.unique(filtered_vids):
                    vid_int = int(unique_vid)
                    mapped_node = self.volume_mapping.get(vid_int)
                    if mapped_node is None or (
                        self.sensitive_volumes and mapped_node not in self.sensitive_volumes
                    ):
                        mapped_node = self.detector_volume
                    vid_mask = filtered_vids == unique_vid
                    self._project_points_for_node(coords[vid_mask], mapped_node)
            else:
                self._project_points_for_node(coords, self.detector_volume)

    def get_projection_snapshot(self) -> Optional[np.ndarray]:
        """
        Возвращает локальную копию текущей проекции из SharedMemory.
        """
        if self._projection_array is not None:
            return self._projection_array.copy()
        return None

    def clear(self) -> None:
        """
        Сброс накопленной матрицы 2D-проекции в ноль.
        """
        if self._projection_array is not None:
            self._projection_array.fill(0.0)

    def close(self) -> None:
        """
        Закрытие дескрипторов SharedMemory проекции, кольцевого буфера телеметрии и прокси паузы.
        """
        if self._telemetry_ring is not None:
            try:
                self._telemetry_ring.close(unlink=False)
            except (OSError, ValueError) as err:
                _logger.debug(f"Ошибка при закрытии TelemetryRingBufferSHM: {err}")
            finally:
                self._telemetry_ring = None

        if self._pause_proxy is not None:
            try:
                self._pause_proxy.close()
            except (OSError, ValueError) as err:
                _logger.debug(f"Ошибка при закрытии FocusedPauseProxy: {err}")
            finally:
                self._pause_proxy = None

        self._projection_array = None
        if self._shm is not None:
            try:
                self._shm.close()
                if self._owns_shm:
                    self._shm.unlink()
            except (OSError, ValueError, BufferError) as err:
                _logger.debug(f"Ошибка при закрытии SharedMemory: {err}")
            finally:
                self._shm = None

    def __del__(self) -> None:
        self.close()


class FocusedPauseProxy:
    """
    Межпроцессный прокси-объект управления паузой для воркеров пула.
    Обеспечивает двухуровневую паузу (глобальная пауза пула + локальная пауза сфокусированного воркера),
    динамическое переключение фокуса на лету через ControlBufferSHM и пошаговый расчет порции частиц.
    """

    def __init__(
        self,
        global_pause_event: Optional[Any] = None,
        focused_pause_event: Optional[Any] = None,
        step_trigger_event: Optional[Any] = None,
        ctrl_shm_name: Optional[str] = None,
        task_id: int = 0,
        step_event: Optional[Any] = None,
    ) -> None:
        effective_step_event = step_event if step_event is not None else step_trigger_event
        self.global_pause_event = global_pause_event
        self.focused_pause_event = focused_pause_event
        self.step_trigger_event = effective_step_event
        self.step_event = effective_step_event
        self.ctrl_shm_name = ctrl_shm_name
        self.task_id: int = int(task_id)
        self._single_step_active: bool = False
        self._ctrl_shm: Optional[ControlBufferSHM] = None

        if ctrl_shm_name is not None:
            try:
                self._ctrl_shm = ControlBufferSHM(shm_name=ctrl_shm_name, create=False)
            except FileNotFoundError:
                self._ctrl_shm = None

    def check_stop_requested(self) -> bool:
        """Проверяет, был ли запрошен кооперативный останов сессии через ControlBufferSHM."""
        if self._ctrl_shm is not None:
            return self._ctrl_shm.stop_requested
        return False

    def is_set(self) -> bool:
        """
        Проверка разрешения выполнения шага.
        При использовании ControlBufferSHM проверяет глобальный флаг, принадлежность задачи
        к текущему focused_job_index, локальную паузу фокуса и триггер одиночного шага.
        """
        if self._ctrl_shm is not None:
            if self._ctrl_shm.stop_requested:
                return True
            if not self._ctrl_shm.global_run:
                return False
            is_currently_focused = self.task_id == self._ctrl_shm.focused_job_index
            if not is_currently_focused:
                self._single_step_active = False
                return True
            if self._ctrl_shm.focused_run:
                self._single_step_active = False
                return True
            if self._single_step_active:
                self._single_step_active = False
                return True
            if self._ctrl_shm.consume_step_trigger():
                return True
            return False

        if self._single_step_active:
            self._single_step_active = False
            return True
        if self.step_trigger_event is not None and self.step_trigger_event.is_set():
            self.step_trigger_event.clear()
            return True
        global_is_active = self.global_pause_event.is_set() if self.global_pause_event is not None else True
        focused_is_active = self.focused_pause_event.is_set() if self.focused_pause_event is not None else True
        return bool(global_is_active and focused_is_active)

    def on_step_completed(self) -> None:
        """Сбрасывает состояние одиночного шага после завершения расчетной итерации."""
        self._single_step_active = False
        if self.step_trigger_event is not None:
            self.step_trigger_event.clear()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """
        Ожидание снятия паузы либо взведения триггера пошагового расчета.
        """
        end_time = time.monotonic() + timeout if timeout is not None else None
        while True:
            if self._ctrl_shm is not None:
                if self._ctrl_shm.stop_requested:
                    return True
                if self._ctrl_shm.global_run:
                    is_currently_focused = self.task_id == self._ctrl_shm.focused_job_index
                    if not is_currently_focused or self._ctrl_shm.focused_run:
                        return True
                    if self._ctrl_shm.consume_step_trigger():
                        self._single_step_active = True
                        return True
            else:
                global_is_active = self.global_pause_event.is_set() if self.global_pause_event is not None else True
                focused_is_active = self.focused_pause_event.is_set() if self.focused_pause_event is not None else True
                if global_is_active and focused_is_active:
                    return True
                if self.step_trigger_event is not None and self.step_trigger_event.is_set():
                    self.step_trigger_event.clear()
                    self._single_step_active = True
                    return True
            if end_time is not None and time.monotonic() >= end_time:
                return False
            time.sleep(0.01)

    def set(self) -> None:
        pass

    def clear(self) -> None:
        pass

    def close(self) -> None:
        """Закрывает подключение к сегменту ControlBufferSHM."""
        if self._ctrl_shm is not None:
            self._ctrl_shm.close(unlink=False)
            self._ctrl_shm = None


def _find_scene_nodes_by_names(node: Any, target_names: Sequence[str]) -> List[SpatialNode]:
    found: List[SpatialNode] = []
    target_set = set(target_names)

    def _traverse(current: Any) -> None:
        if isinstance(current, SpatialNode) and current.name in target_set:
            found.append(current)
        if isinstance(current, CompositeNode):
            for child_item in current.childs:
                _traverse(child_item)

    _traverse(node)
    return found


def _find_all_detector_nodes(node: Any) -> List[SpatialNode]:
    found: List[SpatialNode] = []

    def _traverse(current: Any) -> None:
        if isinstance(current, Volume):
            node_name_lower = (current.name or "").lower()
            if (
                ("detector" in node_name_lower or "crystal" in node_name_lower)
                and "detector_box" not in node_name_lower
            ):
                found.append(current)
        if isinstance(current, CompositeNode):
            for child_item in current.childs:
                _traverse(child_item)

    _traverse(node)
    return found


def create_gui_stream_handler(
    root_scene: Any,
    task_dict: Dict[str, Any],
    task_id: int,
    telemetry_queue: Optional[Any] = None,
    focused_job_index: Optional[int] = None,
    focused_shm_name: Optional[str] = None,
    projection_shape: Tuple[int, int] = (128, 128),
    global_pause_event: Optional[Any] = None,
    focused_pause_event: Optional[Any] = None,
    step_trigger_event: Optional[Any] = None,
    extra_args: Optional[Dict[str, Any]] = None,
    *args: Any,
    **kwargs: Any,
) -> List[BaseDataHandler]:
    """
    Фабрика для создания GuiStreamDataHandler и настройки межпроцессного контроля паузы.
    Инжектируется из подсистемы GUI в оркестратор ядра без создания прямых связей.
    В режиме per_task_shm=True создает независимый потоковый обработчик для каждой задачи task_id.
    """
    merged_extra: Dict[str, Any] = {}
    dict_extra = task_dict.get("_extra_handler_args")
    if isinstance(dict_extra, Mapping):
        merged_extra.update(dict_extra)

    if isinstance(telemetry_queue, Mapping):
        merged_extra.update(telemetry_queue)
        telemetry_queue = None
    if isinstance(focused_job_index, Mapping):
        merged_extra.update(focused_job_index)
        focused_job_index = None
    if isinstance(focused_shm_name, Mapping):
        merged_extra.update(focused_shm_name)
        focused_shm_name = None
    if isinstance(projection_shape, Mapping):
        merged_extra.update(projection_shape)
        projection_shape = (128, 128)
    if isinstance(global_pause_event, Mapping):
        merged_extra.update(global_pause_event)
        global_pause_event = None
    if isinstance(focused_pause_event, Mapping):
        merged_extra.update(focused_pause_event)
        focused_pause_event = None
    if isinstance(step_trigger_event, Mapping):
        merged_extra.update(step_trigger_event)
        step_trigger_event = None

    if isinstance(extra_args, Mapping):
        merged_extra.update(extra_args)
    elif isinstance(extra_args, bool):
        merged_extra["per_task_shm"] = extra_args
    elif isinstance(extra_args, str):
        merged_extra["ctrl_shm_name"] = extra_args

    for item in args:
        if isinstance(item, Mapping):
            merged_extra.update(item)
        elif isinstance(item, bool):
            merged_extra["per_task_shm"] = item
        elif isinstance(item, str):
            merged_extra["ctrl_shm_name"] = item

    if kwargs:
        merged_extra.update(kwargs)

    if telemetry_queue is None:
        telemetry_queue = merged_extra.get("telemetry_queue")
    if focused_job_index is None:
        focused_job_index = merged_extra.get("focused_job_index")
    if focused_shm_name is None:
        focused_shm_name = (
            merged_extra.get("focused_shm_name")
            or merged_extra.get("shm_name")
            or task_dict.get("_focused_shm_name")
        )
    if "projection_shape" in merged_extra and projection_shape == (128, 128):
        projection_shape = merged_extra["projection_shape"]
    if global_pause_event is None:
        global_pause_event = merged_extra.get("global_pause_event")
    if focused_pause_event is None:
        focused_pause_event = merged_extra.get("focused_pause_event")
    if step_trigger_event is None:
        step_trigger_event = merged_extra.get("step_trigger_event") or merged_extra.get("step_event")

    per_task_shm: bool = bool(
        merged_extra.get(
            "per_task_shm",
            task_dict.get("_per_task_shm", task_dict.get("per_task_shm", False)),
        )
    )
    ctrl_shm_name: Optional[str] = merged_extra.get(
        "ctrl_shm_name",
        task_dict.get("_ctrl_shm_name", task_dict.get("ctrl_shm_name", None)),
    )
    max_points_per_batch: int = int(
        merged_extra.get(
            "max_points_per_batch",
            merged_extra.get("max_tracks_per_batch", task_dict.get("_max_points_per_batch", 2000)),
        )
    )

    proj_shape = task_dict.get('_projection_shape', projection_shape)
    show_escaped = bool(merged_extra.get("show_escaped_tracks", task_dict.get('_show_escaped_tracks', False)))
    configured_sensitive_names: List[str] = []

    dm_dict = task_dict.get('data_manager')
    if isinstance(dm_dict, Mapping):
        raw_handlers = dm_dict.get('handlers', ())
        if isinstance(raw_handlers, Sequence):
            for handler_cfg in raw_handlers:
                if isinstance(handler_cfg, Mapping):
                    if bool(handler_cfg.get('show_escaped_tracks', False)):
                        show_escaped = True
                    raw_sv = handler_cfg.get('sensitive_volumes')
                    if isinstance(raw_sv, Sequence) and not isinstance(raw_sv, (str, bytes)):
                        for sv_name in raw_sv:
                            if isinstance(sv_name, str) and sv_name and sv_name not in configured_sensitive_names:
                                configured_sensitive_names.append(sv_name)

    if per_task_shm and focused_shm_name:
        task_proj_shm = get_task_projection_shm_name(focused_shm_name, task_id)
        task_telem_shm = get_task_telemetry_shm_name(focused_shm_name, task_id)

        proxy_event = FocusedPauseProxy(
            global_pause_event=global_pause_event,
            focused_pause_event=focused_pause_event,
            step_trigger_event=step_trigger_event,
            ctrl_shm_name=ctrl_shm_name or DEFAULT_CTRL_SHM_NAME,
            task_id=int(task_id),
        )
        task_dict['_run_event'] = proxy_event
        task_dict['_pause_event'] = proxy_event
        task_dict['_stop_check'] = proxy_event.check_stop_requested

        sensitive_volumes: List[SpatialNode] = []
        if configured_sensitive_names:
            sensitive_volumes = _find_scene_nodes_by_names(root_scene, configured_sensitive_names)
        if not sensitive_volumes:
            sensitive_volumes = _find_all_detector_nodes(root_scene)

        det_node: Optional[SpatialNode] = sensitive_volumes[0] if sensitive_volumes else None

        stream_handler = GuiStreamDataHandler(
            track_queue=telemetry_queue,
            shm_name=task_proj_shm,
            telemetry_shm_name=task_telem_shm,
            projection_shape=proj_shape,
            create_shm=False,
            show_escaped_tracks=show_escaped,
            detector_volume=det_node,
            sensitive_volumes=sensitive_volumes,
            max_points_per_batch=max_points_per_batch,
            pause_proxy=proxy_event,
        )
        return [stream_handler]

    is_focused = focused_job_index is None or task_id == focused_job_index
    legacy_proxy: Optional[FocusedPauseProxy] = None

    if ctrl_shm_name is not None:
        legacy_proxy = FocusedPauseProxy(
            global_pause_event=global_pause_event,
            focused_pause_event=focused_pause_event,
            step_trigger_event=step_trigger_event,
            ctrl_shm_name=ctrl_shm_name,
            task_id=int(task_id),
        )
        task_dict['_run_event'] = legacy_proxy
        task_dict['_pause_event'] = legacy_proxy
        task_dict['_stop_check'] = legacy_proxy.check_stop_requested
    elif global_pause_event is not None:
        if is_focused and focused_pause_event is not None and step_trigger_event is not None:
            legacy_proxy = FocusedPauseProxy(
                global_pause_event=global_pause_event,
                focused_pause_event=focused_pause_event,
                step_trigger_event=step_trigger_event,
                task_id=int(task_id),
            )
            task_dict['_run_event'] = legacy_proxy
            task_dict['_pause_event'] = legacy_proxy
        else:
            task_dict['_run_event'] = global_pause_event
            task_dict['_pause_event'] = global_pause_event

    if focused_shm_name and is_focused:
        sensitive_volumes = []
        if configured_sensitive_names:
            sensitive_volumes = _find_scene_nodes_by_names(root_scene, configured_sensitive_names)
        if not sensitive_volumes:
            sensitive_volumes = _find_all_detector_nodes(root_scene)

        det_node = sensitive_volumes[0] if sensitive_volumes else None

        stream_handler = GuiStreamDataHandler(
            track_queue=telemetry_queue,
            shm_name=focused_shm_name,
            telemetry_shm_name=merged_extra.get("telemetry_shm_name"),
            projection_shape=proj_shape,
            create_shm=False,
            show_escaped_tracks=show_escaped,
            detector_volume=det_node,
            sensitive_volumes=sensitive_volumes,
            max_points_per_batch=max_points_per_batch,
            pause_proxy=legacy_proxy,
        )
        return [stream_handler]
    return []


__all__ = [
    'DEFAULT_CTRL_SHM_NAME',
    'get_task_projection_shm_name',
    'get_task_telemetry_shm_name',
    'get_task_dose_shm_name',
    'ControlBufferSHM',
    'TelemetryRingBufferSHM',
    'GuiStreamDataHandler',
    'FocusedPauseProxy',
    'create_gui_stream_handler',
]
