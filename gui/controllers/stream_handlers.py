import logging
import queue
import time
from multiprocessing import Queue as MpQueue
from multiprocessing import shared_memory
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from core.data.data_handlers import BaseDataHandler
from core.scene.nodes import SpatialNode, CompositeNode
from core.geometry.volumes import Volume
from core.scene.gamma_camera_node import GammaCameraNode

_logger = logging.getLogger(__name__)


class GuiStreamDataHandler(BaseDataHandler):
    """
    Обработчик потоковых данных для передачи телеметрии в реальном времени в GUI.
    Передает пакеты 3D-треков через multiprocessing.Queue и обновляет
    накопленную 2D-проекцию детектора в multiprocessing.shared_memory.SharedMemory.
    """

    def __init__(
        self,
        track_queue: Optional[Any] = None,
        shm_name: Optional[str] = None,
        projection_shape: Tuple[int, int] = (128, 128),
        projection_dtype: np.dtype = np.float32,
        create_shm: bool = False,
        sensitive_volume_ids: Optional[Set[int]] = None,
        max_tracks_per_batch: int = 2000,
        show_escaped_tracks: bool = False,
        detector_volume: Optional[Any] = None,
        detector_size: Optional[Union[Tuple[float, float], Sequence[float], float]] = None,
        detector_inv_matrix: Optional[np.ndarray] = None,
    ) -> None:
        """
        Инициализация обработчика потоковых данных GUI.

        :param track_queue: Очередь для отправки 3D треков в UI.
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
        """
        super().__init__()
        self.track_queue = track_queue
        self.shm_name = shm_name
        self.projection_shape = projection_shape
        self.projection_dtype = np.dtype(projection_dtype)
        self.sensitive_volume_ids = sensitive_volume_ids
        self.max_tracks_per_batch = max_tracks_per_batch
        self.show_escaped_tracks = show_escaped_tracks
        self.detector_volume = detector_volume
        self.detector_inv_matrix = detector_inv_matrix

        if detector_size is not None:
            if isinstance(detector_size, (int, float)):
                self.detector_size = (float(detector_size), float(detector_size))
            else:
                self.detector_size = (float(detector_size[0]), float(detector_size[1]))
        elif detector_volume is not None:
            if isinstance(detector_volume, Volume):
                vol_size = detector_volume.local_bound
                self.detector_size = (float(vol_size[0]), float(vol_size[1]))
            else:
                self.detector_size = (400.0, 400.0)
        else:
            self.detector_size = (400.0, 400.0)

        if detector_volume is not None and self.detector_inv_matrix is None and isinstance(detector_volume, SpatialNode):
            self.detector_inv_matrix = detector_volume.inverse_global_matrix

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
                    buffer=self._shm.buf
                )
                if create_shm:
                    self._projection_array.fill(0)

    def process_chunk(self, chunk: Dict[str, Any]) -> None:
        """
        Обработка чанка данных моделирования из очереди.
        """
        chunk_type = chunk.get('type')
        data = chunk.get('data')

        if not isinstance(data, (dict, Mapping)):
            return

        if chunk_type == 'interactions':
            self._handle_interactions(data)
        elif chunk_type == 'initial_states':
            self._handle_initial_states(data)
        elif chunk_type == 'escaped_particles' and self.show_escaped_tracks:
            self._handle_escaped_particles(data)

    def _handle_initial_states(self, data: Dict[str, np.ndarray]) -> None:
        """
        Обработка начальных состояний частиц: отправка точек рождения
        в очередь треков для построения полных траекторий от источника к взаимодействиям.
        """
        if self.track_queue is None:
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

    def _handle_escaped_particles(self, data: Dict[str, np.ndarray]) -> None:
        """
        Обработка вылетевших частиц: отправка начальной точки рождения и граничной точки выхода
        в очередь треков для построения сквозных лучей вылетевших квантов.
        """
        if self.track_queue is None:
            return

        pos_x = data.get('pos_x')
        pos_y = data.get('pos_y')
        pos_z = data.get('pos_z')
        birth_x = data.get('birth_x')
        birth_y = data.get('birth_y')
        birth_z = data.get('birth_z')
        particle_id = data.get('particle_id')
        has_interacted = data.get('has_interacted')

        if pos_x is None or len(pos_x) == 0:
            return

        n = len(pos_x)
        limit = min(n, self.max_tracks_per_batch)
        sub = slice(0, limit)

        if birth_x is not None:
            if has_interacted is not None:
                uncollided_mask = ~np.asarray(has_interacted)[sub]
            else:
                uncollided_mask = np.ones(limit, dtype=bool)

            if np.any(uncollided_mask):
                bx = np.asarray(birth_x)[sub][uncollided_mask]
                by = np.asarray(birth_y)[sub][uncollided_mask]
                bz = np.asarray(birth_z)[sub][uncollided_mask] if birth_z is not None else np.zeros_like(bx)
                pids = np.asarray(particle_id)[sub][uncollided_mask] if particle_id is not None else None

                birth_payload = {
                    'type': 'tracks',
                    'pos_x': np.array(bx, copy=True),
                    'pos_y': np.array(by, copy=True),
                    'pos_z': np.array(bz, copy=True),
                    'process_id': np.full(len(bx), -2, dtype=np.int32),
                    'particle_id': np.array(pids, copy=True) if pids is not None else None,
                    'energy_deposit': np.zeros(len(bx), dtype=np.float32),
                    'detector_energy_deposit': None,
                }
                try:
                    self.track_queue.put_nowait(birth_payload)
                except queue.Full:
                    pass

        px = np.asarray(pos_x)[sub]
        py = np.asarray(pos_y)[sub]
        pz = np.asarray(pos_z)[sub] if pos_z is not None else np.zeros_like(px)
        pids = np.asarray(particle_id)[sub] if particle_id is not None else None

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

    def _handle_interactions(self, data: Dict[str, np.ndarray]) -> None:
        """
        Обработка взаимодействий: отправка треков в очередь и проецирование на детектор.
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

        if self.track_queue is not None:
            if n_points <= self.max_tracks_per_batch:
                sub_indices = slice(0, n_points)
            else:
                sub_indices = slice(0, self.max_tracks_per_batch)

            det_edep = None
            if self.sensitive_volume_ids is not None and volume_id is not None:
                det_mask = np.isin(volume_id, list(self.sensitive_volume_ids))
                if np.any(det_mask) and energy_deposit is not None:
                    det_edep = energy_deposit[det_mask]

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
            except Exception:
                pass

        if self._projection_array is not None:
            if self.sensitive_volume_ids is not None and volume_id is not None:
                mask = np.isin(volume_id, list(self.sensitive_volume_ids))
                if not np.any(mask):
                    return
                px = pos_x[mask]
                py = pos_y[mask]
                pz = pos_z[mask] if pos_z is not None else np.zeros_like(px)
            else:
                px = pos_x
                py = pos_y
                pz = pos_z if pos_z is not None else np.zeros_like(px)

            if len(px) == 0:
                return

            coords = np.column_stack((px, py, pz))

            if isinstance(self.detector_volume, SpatialNode):
                local_pos = self.detector_volume.convert_to_local_position(coords)
                lx = local_pos[:, 0]
                ly = local_pos[:, 1]
            elif self.detector_inv_matrix is not None:
                local_pos = np.ones((len(coords), 4), dtype=coords.dtype)
                local_pos[:, :3] = coords
                np.matmul(local_pos, self.detector_inv_matrix.T.astype(coords.dtype), out=local_pos)
                lx = local_pos[:, 0]
                ly = local_pos[:, 1]
            else:
                lx = px
                ly = py

            size_x, size_y = self.detector_size
            half_x = size_x / 2.0
            half_y = size_y / 2.0

            eps = 1e-3
            in_crystal = (lx >= -half_x - eps) & (lx <= half_x + eps) & (ly >= -half_y - eps) & (ly <= half_y + eps)
            lx = lx[in_crystal]
            ly = ly[in_crystal]

            if len(lx) > 0:
                h, w = self.projection_shape
                ix = np.clip(((lx / size_x) + 0.5) * (w - 1), 0, w - 1).astype(np.int32)
                iy = np.clip(((ly / size_y) + 0.5) * (h - 1), 0, h - 1).astype(np.int32)
                np.add.at(self._projection_array, (iy, ix), 1.0)

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
        Закрытие дескриптора SharedMemory.
        """
        if self._shm is not None:
            try:
                self._shm.close()
                if self._owns_shm:
                    self._shm.unlink()
            except (OSError, ValueError) as err:
                _logger.debug(f"Ошибка при закрытии SharedMemory: {err}")
            finally:
                self._shm = None
                self._projection_array = None

    def __del__(self) -> None:
        self.close()


class FocusedPauseProxy:
    """
    Межпроцессный прокси-объект управления паузой для сфокусированного воркера.
    Обеспечивает двухуровневую паузу (глобальная пауза пула + локальная пауза воркера)
    и покадровый пошаговый расчет порции частиц (Particle Batch Step).
    """

    def __init__(self, global_pause_event: Any, focused_pause_event: Any, step_trigger_event: Any) -> None:
        self.global_pause_event = global_pause_event
        self.focused_pause_event = focused_pause_event
        self.step_trigger_event = step_trigger_event
        self._single_step_active: bool = False

    def is_set(self) -> bool:
        """
        Проверка разрешения выполнения шага.
        Если взведен триггер шага, он потребляется и возвращает True ровно на одну итерацию.
        """
        if self._single_step_active:
            self._single_step_active = False
            return True
        if self.step_trigger_event is not None and self.step_trigger_event.is_set():
            self.step_trigger_event.clear()
            return True
        global_is_active = self.global_pause_event.is_set() if self.global_pause_event is not None else True
        focused_is_active = self.focused_pause_event.is_set() if self.focused_pause_event is not None else True
        return bool(global_is_active and focused_is_active)

    def wait(self, timeout: Optional[float] = None) -> bool:
        """
        Ожидание снятия паузы либо взведения триггера пошагового расчета.
        """
        end_time = time.monotonic() + timeout if timeout is not None else None
        while True:
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
) -> List[BaseDataHandler]:
    """
    Фабрика для создания GuiStreamDataHandler и настройки межпроцессного контроля паузы.
    Инжектируется из подсистемы GUI в оркестратор ядра без создания прямых связей.
    """
    is_focused = (focused_job_index is None or task_id == focused_job_index)

    # Настройка межпроцессных событий хода/паузы для процесса
    if global_pause_event is not None:
        if is_focused and focused_pause_event is not None and step_trigger_event is not None:
            proxy_event = FocusedPauseProxy(
                global_pause_event=global_pause_event,
                focused_pause_event=focused_pause_event,
                step_trigger_event=step_trigger_event,
            )
            task_dict['_run_event'] = proxy_event
            task_dict['_pause_event'] = proxy_event
        else:
            task_dict['_run_event'] = global_pause_event
            task_dict['_pause_event'] = global_pause_event

    if focused_shm_name and is_focused:
        def _find_detector_node(node: Any) -> Any:
            if isinstance(node, Volume):
                node_name_lower = (node.name or "").lower()
                if ("detector" in node_name_lower or "crystal" in node_name_lower) and "detector_box" not in node_name_lower:
                    return node
            if isinstance(node, CompositeNode):
                for child_item in node.childs:
                    found_node = _find_detector_node(child_item)
                    if found_node is not None:
                        return found_node
            return None

        det_node = _find_detector_node(root_scene)
        proj_shape = task_dict.get('_projection_shape', projection_shape)
        stream_handler = GuiStreamDataHandler(
            track_queue=telemetry_queue,
            shm_name=focused_shm_name,
            projection_shape=proj_shape,
            create_shm=False,
            detector_volume=det_node
        )
        return [stream_handler]
    return []


__all__ = [
    'GuiStreamDataHandler',
    'FocusedPauseProxy',
    'create_gui_stream_handler',
]
