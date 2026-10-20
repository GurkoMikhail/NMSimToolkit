import logging
import queue
import threading
import time
from multiprocessing import shared_memory
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import hepunits as units
from PySide6.QtCore import QThread, Signal

from gui.controllers.stream_handlers import (
    TelemetryRingBufferSHM,
    get_task_dose_shm_name,
    get_task_projection_shm_name,
    get_task_telemetry_shm_name,
    safe_close_shm,
)

_logger = logging.getLogger(__name__)


class IPCReceiver(QThread):
    """
    Контроллер фонового приема данных межпроцессного взаимодействия (IPC Receiver).
    Поддерживает как одиночный режим (через очередь треков), так и пер-тасковый режим
    разделяемой памяти (SharedMemory) с мгновенным переключением фокуса между задачами
    без сброса накопленной статистики. Поддерживает как 4D гиперспектральный куб
    (N_detectors, E_bins, Rows, Cols), так и 2D-буферы проекций (Rows, Cols).
    """

    tracks_received = Signal(dict)
    projection_received = Signal(object)
    spectrum_received = Signal(object)
    stats_updated = Signal(int, float)
    dose_volume_received = Signal(object)
    task_event = Signal(dict)
    status_received = Signal(dict)
    _shm_poll_requested = Signal()

    def __init__(
        self,
        track_queue: Optional[Any] = None,
        shm_name: Optional[str] = None,
        projection_shape: Union[Tuple[int, int], Tuple[int, int, int, int], Tuple[int, ...]] = (128, 128),
        projection_dtype: Any = np.float32,
        fps: float = 30.0,
        dose_shm_name: Optional[str] = None,
        dose_grid_shape: Tuple[int, int, int] = (64, 64, 64),
        dose_dtype: Any = np.float32,
        dose_fps: float = 2.0,
        dose_grids_info: Optional[List[Dict[str, Any]]] = None,
        focused_job_index: int = 0,
        num_tasks: int = 1,
        per_task_shm: bool = False,
        base_dose_shm_name: str = "nmsim_dose_shm",
        telemetry_queue: Optional[Any] = None,
        parent: Optional[Any] = None,
    ) -> None:
        super().__init__(parent)
        self.track_queue = track_queue if track_queue is not None else telemetry_queue
        self.telemetry_queue = self.track_queue
        self.shm_name = shm_name
        self.projection_shape = projection_shape
        self.projection_dtype = np.dtype(projection_dtype)
        self.poll_interval = 1.0 / max(1.0, fps)

        self.dose_shm_name = dose_shm_name
        self.dose_grid_shape = dose_grid_shape
        self.dose_dtype = np.dtype(dose_dtype)
        self.dose_fps = dose_fps
        self.dose_poll_interval = 1.0 / max(0.5, dose_fps)
        self.dose_grids_info = dose_grids_info or []

        self._focused_job_index: int = max(0, int(focused_job_index))
        self._num_tasks: int = max(1, int(num_tasks))
        self._per_task_shm: bool = bool(per_task_shm)
        self._base_dose_shm_name: str = base_dose_shm_name
        self._last_track_cursor: int = 0
        self._last_spec_cursor: int = 0
        self._last_proj_sum: float = -1.0
        self._last_dose_sum: float = -1.0
        self._focus_switched: bool = True

        self._dose_in_flight: bool = False
        self._proj_in_flight: bool = False
        self._spec_in_flight: bool = False
        self._shm_poll_pending: bool = False

        self._shm: Optional[shared_memory.SharedMemory] = None
        self._projection_buf: Optional[np.ndarray] = None
        self._telem_ring: Optional[TelemetryRingBufferSHM] = None
        self._dose_shm: Optional[shared_memory.SharedMemory] = None
        self._dose_buf: Optional[np.ndarray] = None

        self._multi_dose_shms: Dict[str, shared_memory.SharedMemory] = {}
        self._multi_dose_bufs: Dict[str, np.ndarray] = {}

        # Кэш открытых пер-тасковых сегментов SharedMemory для мгновенного переключения фокуса без close/reopen
        self._task_proj_shms: Dict[int, shared_memory.SharedMemory] = {}
        self._task_proj_bufs: Dict[int, np.ndarray] = {}
        self._task_telem_rings: Dict[int, TelemetryRingBufferSHM] = {}
        self._task_dose_shms: Dict[int, shared_memory.SharedMemory] = {}
        self._task_dose_bufs: Dict[int, np.ndarray] = {}
        self._task_multi_dose_shms: Dict[int, Dict[str, shared_memory.SharedMemory]] = {}
        self._task_multi_dose_bufs: Dict[int, Dict[str, np.ndarray]] = {}

        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._last_proj_emit_time = 0.0
        self._last_spec_emit_time = 0.0
        self._last_dose_emit_time = 0.0
        self._last_stats_time = 0.0
        self._counts_at_last_stats = 0
        self._accumulated_energies: List[np.ndarray] = []
        self._spectrum_dirty = False
        self.max_spectrum_samples = 50000
        self.spectrum_poll_interval = 0.2

        self._shm_poll_requested.connect(self._on_shm_poll_in_main_thread)
        self._init_shared_memory()

    def mark_dose_delivered(self) -> None:
        """Сбрасывает флаг занятости GUI-потока после отрисовки 3D-карты дозы (backpressure)."""
        self._dose_in_flight = False

    def mark_proj_delivered(self) -> None:
        """Сбрасывает флаг занятости GUI-потока после отрисовки 2D-проекции."""
        self._proj_in_flight = False

    def mark_spec_delivered(self) -> None:
        """Сбрасывает флаг занятости GUI-потока после отрисовки спектра."""
        self._spec_in_flight = False

    @property
    def focused_job_index(self) -> int:
        return self._focused_job_index

    def _close_task_shms_unlocked(self) -> None:
        """Закрывает все открытые дескрипторы SharedMemory без удаления сегментов (unlink=False)."""
        self._projection_buf = None
        self._telem_ring = None
        self._dose_buf = None
        self._multi_dose_bufs = {}
        self._task_proj_bufs.clear()
        self._task_dose_bufs.clear()
        for m_buf_dict in self._task_multi_dose_bufs.values():
            m_buf_dict.clear()
        self._task_multi_dose_bufs.clear()

        proj_shms = list(self._task_proj_shms.values())
        self._task_proj_shms.clear()
        single_shm = self._shm
        self._shm = None
        for t_shm in proj_shms:
            safe_close_shm(t_shm, unlink=False)
        if single_shm is not None and single_shm not in proj_shms:
            safe_close_shm(single_shm, unlink=False)

        telem_rings = list(self._task_telem_rings.values())
        self._task_telem_rings.clear()
        for t_ring in telem_rings:
            try:
                t_ring.close(unlink=False)
            except (OSError, BufferError, ValueError):
                pass

        dose_shms = list(self._task_dose_shms.values())
        self._task_dose_shms.clear()
        single_dose_shm = self._dose_shm
        self._dose_shm = None
        for t_dshm in dose_shms:
            safe_close_shm(t_dshm, unlink=False)
        if single_dose_shm is not None and single_dose_shm not in dose_shms:
            safe_close_shm(single_dose_shm, unlink=False)

        seen_multi_shms: List[shared_memory.SharedMemory] = []
        for m_dict in list(self._task_multi_dose_shms.values()):
            for m_shm in list(m_dict.values()):
                if m_shm not in seen_multi_shms:
                    seen_multi_shms.append(m_shm)
            m_dict.clear()
        self._task_multi_dose_shms.clear()
        for multi_shm in list(self._multi_dose_shms.values()):
            if multi_shm not in seen_multi_shms:
                seen_multi_shms.append(multi_shm)
        self._multi_dose_shms = {}
        for m_shm in seen_multi_shms:
            safe_close_shm(m_shm, unlink=False)

    def _bind_focused_task_shm(self, task_id: int) -> None:
        """
        Переключает активные указатели на пер-тасковые сегменты SharedMemory (проекция, кольцо телеметрии, доза)
        для задачи task_id с кэшированием открытых дескрипторов. Должен вызываться под self._lock.
        """
        if self.shm_name is not None:
            if task_id not in self._task_proj_bufs:
                proj_shm_name = get_task_projection_shm_name(self.shm_name, task_id)
                try:
                    shm = shared_memory.SharedMemory(name=proj_shm_name, create=False)
                    buf = np.ndarray(
                        self.projection_shape,
                        dtype=self.projection_dtype,
                        buffer=shm.buf,
                    )
                    self._task_proj_shms[task_id] = shm
                    self._task_proj_bufs[task_id] = buf
                except (OSError, ValueError, BufferError, TypeError) as shm_err:
                    _logger.debug(f"IPCReceiver: пер-тасковый SharedMemory '{proj_shm_name}' пока недоступен: {shm_err}")
            self._shm = self._task_proj_shms.get(task_id)
            self._projection_buf = self._task_proj_bufs.get(task_id)

            if task_id not in self._task_telem_rings:
                telem_shm_name = get_task_telemetry_shm_name(self.shm_name, task_id)
                try:
                    ring = TelemetryRingBufferSHM(shm_name=telem_shm_name, create=False)
                    self._task_telem_rings[task_id] = ring
                except (OSError, ValueError, BufferError, TypeError) as ring_err:
                    _logger.debug(f"IPCReceiver: кольцо телеметрии '{telem_shm_name}' пока недоступно: {ring_err}")
            self._telem_ring = self._task_telem_rings.get(task_id)

        effective_base_dose = (
            self._base_dose_shm_name
            if self._base_dose_shm_name != "nmsim_dose_shm"
            else (self.dose_shm_name or self._base_dose_shm_name)
        )

        if len(self.dose_grids_info) > 1:
            t_shms = self._task_multi_dose_shms.setdefault(task_id, {})
            t_bufs = self._task_multi_dose_bufs.setdefault(task_id, {})
            for info in self.dose_grids_info:
                name = info["name"]
                if name not in t_bufs:
                    shape = info["grid_shape"]
                    grid_dtype = np.dtype(info.get("dtype", self.dose_dtype))
                    t_dose_name = get_task_dose_shm_name(
                        effective_base_dose, task_id, grid_name=name, multi_grid=True
                    )
                    try:
                        shm = shared_memory.SharedMemory(name=t_dose_name, create=False)
                        buf = np.ndarray(shape, dtype=grid_dtype, buffer=shm.buf, order="F")
                        t_shms[name] = shm
                        t_bufs[name] = buf
                    except (OSError, ValueError, BufferError, TypeError, KeyError) as multi_err:
                        _logger.debug(f"IPCReceiver: пер-тасковый Dose SHM '{t_dose_name}' пока недоступен: {multi_err}")
            self._multi_dose_shms = t_shms
            self._multi_dose_bufs = t_bufs
            self._dose_shm = None
            self._dose_buf = None
        elif len(self.dose_grids_info) == 1 or self.dose_shm_name is not None or bool(effective_base_dose):
            if task_id not in self._task_dose_bufs:
                grid_shape = (
                    self.dose_grids_info[0]["grid_shape"]
                    if len(self.dose_grids_info) == 1
                    else self.dose_grid_shape
                )
                grid_dtype = (
                    np.dtype(self.dose_grids_info[0].get("dtype", self.dose_dtype))
                    if len(self.dose_grids_info) == 1
                    else self.dose_dtype
                )
                grid_name = (
                    self.dose_grids_info[0].get("name")
                    if len(self.dose_grids_info) == 1
                    else None
                )
                t_dose_name = get_task_dose_shm_name(
                    effective_base_dose, task_id, grid_name=grid_name, multi_grid=False
                )
                try:
                    d_shm = shared_memory.SharedMemory(name=t_dose_name, create=False)
                    d_buf = np.ndarray(
                        grid_shape,
                        dtype=grid_dtype,
                        buffer=d_shm.buf,
                        order="F",
                    )
                    self._task_dose_shms[task_id] = d_shm
                    self._task_dose_bufs[task_id] = d_buf
                except (OSError, ValueError, BufferError, TypeError) as dose_err:
                    _logger.debug(f"IPCReceiver: пер-тасковый Dose SHM '{t_dose_name}' пока недоступен: {dose_err}")
            self._dose_shm = self._task_dose_shms.get(task_id)
            self._dose_buf = self._task_dose_bufs.get(task_id)

        self._last_track_cursor = 0
        self._last_spec_cursor = -1
        self._last_proj_sum = -1.0
        self._last_dose_sum = -1.0
        self._focus_switched = True
        self._last_proj_emit_time = 0.0
        self._last_spec_emit_time = 0.0
        self._last_dose_emit_time = 0.0

    def set_focused_job(self, index: int) -> None:
        """
        Мгновенно переключает чтение телеметрии на буферы задачи с указанным индексом.
        """
        with self._lock:
            target_idx = max(0, int(index))
            if target_idx != self._focused_job_index or self._per_task_shm:
                self._focused_job_index = target_idx
                if self._per_task_shm:
                    self._bind_focused_task_shm(self._focused_job_index)
        if self.isRunning() and not self._shm_poll_pending:
            self._shm_poll_pending = True
            self._shm_poll_requested.emit()

    def clear_accumulation(self) -> None:
        """
        Сброс накопленного массива энергий для спектра, проекции, треков и карты дозы
        текущей сфокусированной задачи.
        """
        with self._lock:
            self._accumulated_energies = []
            self._spectrum_dirty = False
            if self._telem_ring is not None:
                try:
                    self._telem_ring.clear()
                except (OSError, ValueError, BufferError):
                    pass
                self._last_track_cursor = 0
                self._last_spec_cursor = -1
            if self._projection_buf is not None:
                self._projection_buf.fill(0.0)
                self._last_proj_sum = -1.0
            if self._dose_buf is not None:
                self._dose_buf.fill(0.0)
                self._last_dose_sum = -1.0
            for buf in self._multi_dose_bufs.values():
                buf.fill(0.0)
            self._last_dose_sum = -1.0

    @property
    def _stop_requested(self) -> bool:
        return self._stop_event.is_set()

    @_stop_requested.setter
    def _stop_requested(self, val: bool) -> None:
        if val:
            self._stop_event.set()
        else:
            self._stop_event.clear()

    def _init_shared_memory(self) -> None:
        if self._per_task_shm:
            with self._lock:
                self._bind_focused_task_shm(self._focused_job_index)
            return

        if self.shm_name is not None and self._projection_buf is None:
            try:
                self._shm = shared_memory.SharedMemory(name=self.shm_name, create=False)
                self._projection_buf = np.ndarray(
                    self.projection_shape,
                    dtype=self.projection_dtype,
                    buffer=self._shm.buf,
                )
            except (OSError, ValueError, BufferError, TypeError) as shm_err:
                _logger.debug(f"IPCReceiver: SharedMemory '{self.shm_name}' пока недоступен: {shm_err}")
                self._shm = None
                self._projection_buf = None

        if self.dose_shm_name is not None and self._dose_buf is None:
            try:
                self._dose_shm = shared_memory.SharedMemory(name=self.dose_shm_name, create=False)
                self._dose_buf = np.ndarray(
                    self.dose_grid_shape,
                    dtype=self.dose_dtype,
                    buffer=self._dose_shm.buf,
                    order="F",
                )
            except (OSError, ValueError, BufferError, TypeError) as dose_err:
                _logger.debug(f"IPCReceiver: Dose SharedMemory '{self.dose_shm_name}' пока недоступен: {dose_err}")
                self._dose_shm = None
                self._dose_buf = None

        if self.dose_grids_info:
            for info in self.dose_grids_info:
                name = info["name"]
                shm_n = info["shm_name"]
                shape = info["grid_shape"]
                grid_dtype = np.dtype(info.get("dtype", self.dose_dtype))
                if name not in self._multi_dose_bufs:
                    try:
                        shm = shared_memory.SharedMemory(name=shm_n, create=False)
                        buf = np.ndarray(shape, dtype=grid_dtype, buffer=shm.buf, order="F")
                        self._multi_dose_shms[name] = shm
                        self._multi_dose_bufs[name] = buf
                    except (OSError, ValueError, BufferError, TypeError, KeyError) as multi_err:
                        _logger.debug(f"IPCReceiver: Dose SharedMemory '{shm_n}' для '{name}' пока недоступен: {multi_err}")

    def _process_track_item(self, item: Any) -> bool:
        """
        Обрабатывает один элемент из очереди телеметрии.
        Возвращает True, если получен маркер останова 'stop'.
        """
        if item == "stop":
            return True

        if isinstance(item, dict):
            item_type = item.get("type")
            if item_type == "tracks" and not self._per_task_shm:
                self.tracks_received.emit(item)
                det_edep = item.get("detector_energy_deposit")
                edep = det_edep if det_edep is not None else item.get("energy_deposit")
                if edep is not None and len(edep) > 0:
                    edep_arr = np.asarray(edep)
                    valid_edep = edep_arr[edep_arr > 0]
                    if len(valid_edep) > 0:
                        energies = (valid_edep / units.keV).astype(np.float32)
                        with self._lock:
                            self._accumulated_energies.append(energies)
                            self._spectrum_dirty = True
                            total_len = sum(len(arr_chunk) for arr_chunk in self._accumulated_energies)
                            if total_len > self.max_spectrum_samples * 2:
                                merged = np.concatenate(self._accumulated_energies)
                                step = int(np.ceil(len(merged) / self.max_spectrum_samples))
                                self._accumulated_energies = [merged[::step]]
            elif item_type in ("task_started", "task_finished", "task_progress", "task_error"):
                self.task_event.emit(item)
                self.status_received.emit(item)
        return False

    def _on_shm_poll_in_main_thread(self) -> None:
        """
        Слот, выполняющийся в главном потоке GUI по триггеру _shm_poll_requested.
        Читает сегменты SharedMemory в главном потоке и передает снимки в виджеты и рендереры.
        """
        self._shm_poll_pending = False
        self._poll_shm_buffers(direct_call=False)

    def _poll_shm_buffers(self, direct_call: bool = False) -> None:
        """
        Читает сегменты SharedMemory в главном потоке с троттлингом, dirty-check и backpressure,
        передавая безопасные локальные копии снимков без удержания буферов SharedMemory.buf.
        """
        if self._per_task_shm:
            with self._lock:
                if (self._projection_buf is None or self._telem_ring is None) and self.shm_name is not None:
                    self._bind_focused_task_shm(self._focused_job_index)

                switched = self._focus_switched
                self._focus_switched = False
                now = time.time()

                # Чтение треков из кольцевого буфера
                if self._telem_ring is not None:
                    try:
                        if switched:
                            new_cur, track_batch = self._telem_ring.read_all_tracks()
                            self._last_track_cursor = new_cur
                            if track_batch is not None:
                                track_batch["reset"] = True
                                self.tracks_received.emit(track_batch)
                            else:
                                self.tracks_received.emit({
                                    "pos_x": np.empty(0, dtype=np.float32),
                                    "pos_y": np.empty(0, dtype=np.float32),
                                    "pos_z": np.empty(0, dtype=np.float32),
                                    "particle_id": np.empty(0, dtype=np.int64),
                                    "process_id": np.empty(0, dtype=np.int16),
                                    "reset": True,
                                })
                        else:
                            new_cur, track_batch = self._telem_ring.read_tracks_since(self._last_track_cursor)
                            self._last_track_cursor = new_cur
                            if track_batch is not None:
                                self.tracks_received.emit(track_batch)
                    except (OSError, RuntimeError, ValueError, BufferError, TypeError) as trk_err:
                        _logger.debug(f"Ошибка чтения треков из кольца SHM: {trk_err}")

                # Чтение спектра из кольцевого буфера (с троттлингом и проверкой курсора)
                if self._telem_ring is not None and (
                    switched
                    or direct_call
                    or ((now - self._last_spec_emit_time) >= self.spectrum_poll_interval and not self._spec_in_flight)
                ):
                    try:
                        cur_spec = self._telem_ring.spectrum_write_cursor
                        if switched or cur_spec != self._last_spec_cursor:
                            spec_cur, raw_energies = self._telem_ring.read_spectrum()
                            self._last_spec_cursor = spec_cur
                            self._last_spec_emit_time = now
                            if len(raw_energies) > 0:
                                scaled_energies = (raw_energies / np.float32(units.keV)).astype(np.float32)
                                if not direct_call:
                                    self._spec_in_flight = True
                                self.spectrum_received.emit(scaled_energies)
                            elif switched:
                                if not direct_call:
                                    self._spec_in_flight = True
                                self.spectrum_received.emit(np.empty(0, dtype=np.float32))
                    except (OSError, RuntimeError, ValueError, BufferError, TypeError) as spec_err:
                        _logger.debug(f"Ошибка чтения спектра из кольца SHM: {spec_err}")

                # Чтение 2D-проекции / 4D-куба (копия снимка без экспорта указателей на SharedMemory.buf)
                if self._projection_buf is not None and (
                    switched
                    or direct_call
                    or ((now - self._last_proj_emit_time) >= self.poll_interval and not self._proj_in_flight)
                ):
                    try:
                        current_proj_sum = float(np.sum(self._projection_buf))
                        if switched or current_proj_sum != self._last_proj_sum:
                            self._last_proj_sum = current_proj_sum
                            proj_snapshot = self._projection_buf.copy()
                            if not direct_call:
                                self._proj_in_flight = True
                            self.projection_received.emit(proj_snapshot)
                            self._last_proj_emit_time = now

                        if switched:
                            total_counts = int(current_proj_sum)
                            self.stats_updated.emit(total_counts, 0.0)
                            self._counts_at_last_stats = total_counts
                            self._last_stats_time = now
                        elif (now - self._last_stats_time) >= 1.0:
                            total_counts = int(current_proj_sum)
                            cps = (total_counts - self._counts_at_last_stats) / (now - self._last_stats_time)
                            self.stats_updated.emit(total_counts, max(0.0, cps))
                            self._counts_at_last_stats = total_counts
                            self._last_stats_time = now
                    except (OSError, RuntimeError, ValueError, BufferError, TypeError) as proj_err:
                        _logger.debug(f"Ошибка чтения снимка проекции: {proj_err}")

                # Чтение 3D-карты дозы (копия снимка без экспорта указателей на SharedMemory.buf)
                if (
                    switched
                    or direct_call
                    or ((now - self._last_dose_emit_time) >= self.dose_poll_interval and not self._dose_in_flight)
                ):
                    if self.dose_grids_info and len(self._multi_dose_bufs) > 0:
                        try:
                            current_dose_sum = sum(float(np.sum(buf)) for buf in self._multi_dose_bufs.values())
                            if switched or (current_dose_sum != self._last_dose_sum and current_dose_sum > 0.0):
                                self._last_dose_sum = current_dose_sum
                                dose_views = {
                                    grid_key: grid_buf.copy(order="F")
                                    for grid_key, grid_buf in self._multi_dose_bufs.items()
                                }
                                if not direct_call:
                                    self._dose_in_flight = True
                                self.dose_volume_received.emit(dose_views)
                                self._last_dose_emit_time = now
                        except (OSError, RuntimeError, ValueError, BufferError, TypeError) as multi_dose_err:
                            _logger.debug(f"Ошибка чтения снимков дозы: {multi_dose_err}")
                    elif self._dose_buf is not None:
                        try:
                            current_dose_sum = float(np.sum(self._dose_buf))
                            if switched or (current_dose_sum != self._last_dose_sum and current_dose_sum > 0.0):
                                self._last_dose_sum = current_dose_sum
                                dose_snapshot = self._dose_buf.copy(order="F")
                                if not direct_call:
                                    self._dose_in_flight = True
                                self.dose_volume_received.emit(dose_snapshot)
                                self._last_dose_emit_time = now
                        except (OSError, RuntimeError, ValueError, BufferError, TypeError) as single_dose_err:
                            _logger.debug(f"Ошибка чтения снимка дозы: {single_dose_err}")
            return

        # Режим одиночных буферов (_per_task_shm == False)
        if (
            (self._projection_buf is None and self.shm_name is not None)
            or (self._dose_buf is None and self.dose_shm_name is not None)
            or (bool(self.dose_grids_info) and len(self._multi_dose_bufs) < len(self.dose_grids_info))
        ):
            self._init_shared_memory()

        now = time.time()

        if direct_call or (now - self._last_spec_emit_time) >= self.spectrum_poll_interval:
            all_energies: Optional[np.ndarray] = None
            with self._lock:
                if self._spectrum_dirty and len(self._accumulated_energies) > 0:
                    all_energies = (
                        np.concatenate(self._accumulated_energies)
                        if len(self._accumulated_energies) > 1
                        else self._accumulated_energies[0].copy()
                    )
                    self._last_spec_emit_time = now
                    self._spectrum_dirty = False
            if all_energies is not None:
                self.spectrum_received.emit(all_energies)

        if self._projection_buf is not None and (
            direct_call or (now - self._last_proj_emit_time) >= self.poll_interval
        ):
            try:
                self.projection_received.emit(self._projection_buf.copy())
                self._last_proj_emit_time = now

                if (now - self._last_stats_time) >= 1.0:
                    total_counts = int(np.sum(self._projection_buf))
                    cps = (total_counts - self._counts_at_last_stats) / (now - self._last_stats_time)
                    self.stats_updated.emit(total_counts, max(0.0, cps))
                    self._counts_at_last_stats = total_counts
                    self._last_stats_time = now
            except (OSError, RuntimeError, ValueError, BufferError, TypeError) as proj_err:
                _logger.debug(f"Ошибка чтения снимка проекции: {proj_err}")

        if direct_call or (now - self._last_dose_emit_time) >= self.dose_poll_interval:
            if self.dose_grids_info and len(self._multi_dose_bufs) > 0:
                try:
                    self.dose_volume_received.emit({
                        grid_key: grid_buf.copy(order="F")
                        for grid_key, grid_buf in self._multi_dose_bufs.items()
                    })
                    self._last_dose_emit_time = now
                except (OSError, RuntimeError, ValueError, BufferError, TypeError) as multi_dose_err:
                    _logger.debug(f"Ошибка чтения снимков дозы: {multi_dose_err}")
            elif self._dose_buf is not None:
                try:
                    self.dose_volume_received.emit(self._dose_buf.copy(order="F"))
                    self._last_dose_emit_time = now
                except (OSError, RuntimeError, ValueError, BufferError, TypeError) as single_dose_err:
                    _logger.debug(f"Ошибка чтения снимка дозы: {single_dose_err}")

    def _poll_once(self, wait_timeout: float = 0.0) -> bool:
        """
        Выполняет одну итерацию опроса очереди статусов и сегментов SharedMemory.
        При вызове из фонового потока run() делегирует чтение SharedMemory в главный поток GUI
        через легкий пустой сигнал _shm_poll_requested (0 байт в QueuedConnection).
        При вызове из главного потока читает SharedMemory синхронно на месте.
        """
        poll_start = time.monotonic()
        if self.track_queue is not None:
            try:
                if wait_timeout > 0.0:
                    first_item = self.track_queue.get(timeout=wait_timeout)
                    if self._process_track_item(first_item):
                        return True
                while not self._stop_event.is_set():
                    try:
                        item = self.track_queue.get_nowait()
                    except (queue.Empty, OSError, EOFError, ValueError):
                        break
                    if self._process_track_item(item):
                        return True
            except queue.Empty:
                pass
            except (OSError, EOFError, RuntimeError, ValueError, TypeError) as queue_err:
                _logger.debug(f"Ошибка чтения из очереди статусов: {queue_err}")

            if self._per_task_shm and wait_timeout > 0.0 and not self._focus_switched:
                elapsed = time.monotonic() - poll_start
                remaining = wait_timeout - elapsed
                if remaining > 0.002:
                    time.sleep(remaining)
        elif wait_timeout > 0.0:
            time.sleep(wait_timeout)

        if threading.current_thread() is threading.main_thread():
            self._poll_shm_buffers(direct_call=(wait_timeout == 0.0))
        else:
            if not self._shm_poll_pending:
                self._shm_poll_pending = True
                self._shm_poll_requested.emit()

        return False

    def run(self) -> None:
        """
        Рабочий цикл приема IPC-данных без busy-wait (блокирующее чтение с таймаутом кадра).
        """
        self._stop_event.clear()
        with self._lock:
            self._accumulated_energies = []
            self._spectrum_dirty = False
            if self._per_task_shm:
                self._bind_focused_task_shm(self._focused_job_index)
        self._last_spec_emit_time = 0.0
        self._last_proj_emit_time = 0.0
        self._last_dose_emit_time = 0.0
        self._last_stats_time = time.time()
        self._counts_at_last_stats = 0

        while not self._stop_event.is_set():
            if self._poll_once(wait_timeout=self.poll_interval):
                break

        # Финальный дренаж очереди перед завершением
        if not self._per_task_shm and self.track_queue is not None:
            while True:
                try:
                    item = self.track_queue.get_nowait()
                except (queue.Empty, OSError, EOFError, ValueError):
                    break
                if self._process_track_item(item):
                    break

        if not self._shm_poll_pending:
            self._shm_poll_pending = True
            self._shm_poll_requested.emit()

    def stop(self, timeout_ms: int = 200) -> None:
        """
        Запрос остановки потока опроса без длительной блокировки GUI.
        """
        self._stop_event.set()
        if self.track_queue is not None:
            try:
                self.track_queue.put_nowait("stop")
            except (queue.Full, OSError, RuntimeError, ValueError):
                pass
        if self.isRunning() and timeout_ms > 0:
            self.wait(timeout_ms)
        if threading.current_thread() is threading.main_thread() and self._shm_poll_pending:
            self._on_shm_poll_in_main_thread()

    def close(self) -> None:
        """
        Освобождение дескрипторов IPC.
        """
        self.stop()
        with self._lock:
            self._close_task_shms_unlocked()


__all__ = [
    "IPCReceiver",
]

