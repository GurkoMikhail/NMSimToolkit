import logging
import queue
import threading
import time
from multiprocessing import shared_memory
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QThread, Signal

_logger = logging.getLogger(__name__)


class IPCReceiver(QThread):
    """
    Контроллер фонового приема данных межпроцессного взаимодействия (IPC Receiver).
    1. Опрашивает multiprocessing.Queue на наличие новых 3D-треков частиц с блокирующим таймаутом кадра.
    2. Периодически считывает матрицу накопленной 2D-проекции из SharedMemory.
    3. Транслирует данные в Qt-сигналы для мгновенного отображения во вьюпорте и панели результатов.
    """

    tracks_received = Signal(dict)
    projection_received = Signal(object)
    spectrum_received = Signal(object)
    stats_updated = Signal(int, float)
    dose_volume_received = Signal(object)
    task_event = Signal(dict)

    def __init__(
        self,
        track_queue: Optional[Any] = None,
        shm_name: Optional[str] = None,
        projection_shape: Tuple[int, int] = (128, 128),
        projection_dtype: np.dtype = np.float32,
        fps: float = 30.0,
        dose_shm_name: Optional[str] = None,
        dose_grid_shape: Tuple[int, int, int] = (64, 64, 64),
        dose_fps: float = 2.0,
        dose_grids_info: Optional[List[Dict[str, Any]]] = None,
        parent: Optional[Any] = None
    ) -> None:
        super().__init__(parent)
        self.track_queue = track_queue
        self.shm_name = shm_name
        self.projection_shape = projection_shape
        self.projection_dtype = np.dtype(projection_dtype)
        self.poll_interval = 1.0 / max(1.0, fps)

        self.dose_shm_name = dose_shm_name
        self.dose_grid_shape = dose_grid_shape
        self.dose_fps = dose_fps
        self.dose_poll_interval = 1.0 / max(0.5, dose_fps)
        self.dose_grids_info = dose_grids_info or []

        self._shm: Optional[shared_memory.SharedMemory] = None
        self._projection_buf: Optional[np.ndarray] = None
        self._dose_shm: Optional[shared_memory.SharedMemory] = None
        self._dose_buf: Optional[np.ndarray] = None

        self._multi_dose_shms: Dict[str, shared_memory.SharedMemory] = {}
        self._multi_dose_bufs: Dict[str, np.ndarray] = {}

        self._stop_event = threading.Event()
        self._last_proj_emit_time = 0.0
        self._last_spec_emit_time = 0.0
        self._last_dose_emit_time = 0.0
        self._accumulated_energies: list = []
        self._spectrum_dirty = False
        self.max_spectrum_samples = 50000
        self.spectrum_poll_interval = 0.2

        self._init_shared_memory()

    def clear_accumulation(self) -> None:
        """
        Сброс накопленного массива энергий для спектра, проекции и карты дозы.
        """
        self._accumulated_energies = []
        self._spectrum_dirty = False
        if self._projection_buf is not None:
            self._projection_buf.fill(0.0)
        if self._dose_buf is not None:
            self._dose_buf.fill(0.0)
        for buf in self._multi_dose_bufs.values():
            buf.fill(0.0)

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
        if self.shm_name is not None and self._projection_buf is None:
            try:
                self._shm = shared_memory.SharedMemory(name=self.shm_name, create=False)
                self._projection_buf = np.ndarray(
                    self.projection_shape,
                    dtype=self.projection_dtype,
                    buffer=self._shm.buf
                )
            except Exception as e:
                _logger.debug(f"IPCReceiver: SharedMemory '{self.shm_name}' пока недоступен: {e}")
                self._shm = None
                self._projection_buf = None

        if self.dose_shm_name is not None and self._dose_buf is None:
            try:
                self._dose_shm = shared_memory.SharedMemory(name=self.dose_shm_name, create=False)
                self._dose_buf = np.ndarray(
                    self.dose_grid_shape,
                    dtype=np.float64,
                    buffer=self._dose_shm.buf
                )
            except Exception as e:
                _logger.debug(f"IPCReceiver: Dose SharedMemory '{self.dose_shm_name}' пока недоступен: {e}")
                self._dose_shm = None
                self._dose_buf = None

        if self.dose_grids_info:
            for info in self.dose_grids_info:
                name = info['name']
                shm_n = info['shm_name']
                shape = info['grid_shape']
                if name not in self._multi_dose_bufs:
                    try:
                        shm = shared_memory.SharedMemory(name=shm_n, create=False)
                        buf = np.ndarray(shape, dtype=np.float64, buffer=shm.buf)
                        self._multi_dose_shms[name] = shm
                        self._multi_dose_bufs[name] = buf
                    except Exception as e:
                        _logger.debug(f"IPCReceiver: Dose SharedMemory '{shm_n}' для '{name}' пока недоступен: {e}")

    def _process_track_item(self, item: Any) -> bool:
        """
        Обрабатывает один элемент из очереди треков.
        Возвращает True, если получен маркер останова 'stop'.
        """
        if item == 'stop':
            return True

        if isinstance(item, dict):
            item_type = item.get('type')
            if item_type == 'tracks':
                self.tracks_received.emit(item)
                det_edep = item.get('detector_energy_deposit')
                edep = det_edep if (det_edep is not None and len(det_edep) > 0) else item.get('energy_deposit')
                if edep is not None and len(edep) > 0:
                    edep_arr = np.asarray(edep)
                    valid_edep = edep_arr[edep_arr > 0]
                    if len(valid_edep) > 0:
                        energies = (valid_edep * 1000.0).astype(np.float32)
                        self._accumulated_energies.append(energies)
                        self._spectrum_dirty = True
                        # Предотвращение лавинообразного роста памяти: периодическая компактификация
                        total_len = sum(len(a) for a in self._accumulated_energies)
                        if total_len > self.max_spectrum_samples * 2:
                            merged = np.concatenate(self._accumulated_energies)
                            step = int(np.ceil(len(merged) / self.max_spectrum_samples))
                            self._accumulated_energies = [merged[::step]]
            elif item_type in ('task_started', 'task_finished', 'task_progress', 'task_error'):
                self.task_event.emit(item)
        return False

    def run(self) -> None:
        """
        Рабочий цикл приема IPC-данных без busy-wait (блокирующее чтение с таймаутом кадра).
        """
        self._stop_event.clear()
        self._accumulated_energies = []
        self._spectrum_dirty = False
        self._last_spec_emit_time = 0.0
        self._last_proj_emit_time = 0.0
        last_stats_time = time.time()
        counts_at_last_stats = 0

        while not self._stop_event.is_set():
            # 1. Попытка подключения к SharedMemory, если не подключено
            if (self._projection_buf is None and self.shm_name is not None) or \
               (self._dose_buf is None and self.dose_shm_name is not None) or \
               (bool(self.dose_grids_info) and len(self._multi_dose_bufs) < len(self.dose_grids_info)):
                self._init_shared_memory()

            # 2. Опрос очереди треков с блокирующим ожиданием с таймаутом кадра
            if self.track_queue is not None:
                try:
                    item = self.track_queue.get(timeout=self.poll_interval)
                    if self._process_track_item(item):
                        break
                    while not self._stop_event.is_set():
                        try:
                            item = self.track_queue.get_nowait()
                        except (queue.Empty, Exception):
                            break
                        if self._process_track_item(item):
                            break
                except queue.Empty:
                    pass
                except Exception as e:
                    _logger.debug(f"Ошибка чтения из очереди треков: {e}")
            else:
                time.sleep(self.poll_interval)

            now = time.time()

            # 3. Периодическая передача накопленного спектра по таймингу
            if self._spectrum_dirty and (now - self._last_spec_emit_time) >= self.spectrum_poll_interval:
                if len(self._accumulated_energies) > 0:
                    all_energies = np.concatenate(self._accumulated_energies) if len(self._accumulated_energies) > 1 else self._accumulated_energies[0]
                    self.spectrum_received.emit(all_energies)
                    self._last_spec_emit_time = now
                    self._spectrum_dirty = False

            # 4. Периодическая передача среза проекции из SharedMemory по таймингу кадра
            if self._projection_buf is not None and (now - self._last_proj_emit_time) >= self.poll_interval:
                try:
                    snapshot = self._projection_buf.copy()
                    self.projection_received.emit(snapshot)
                    self._last_proj_emit_time = now

                    # Статистика скорости счета (CPS)
                    if (now - last_stats_time) >= 1.0:
                        total_counts = int(np.sum(snapshot))
                        cps = (total_counts - counts_at_last_stats) / (now - last_stats_time)
                        self.stats_updated.emit(total_counts, max(0.0, cps))
                        counts_at_last_stats = total_counts
                        last_stats_time = now
                except Exception as e:
                    _logger.debug(f"Ошибка чтения снимка проекции: {e}")

            # 5. Периодическая передача 3D-среза дозы из SharedMemory (с частотой dose_fps ~ 2 FPS)
            if (now - self._last_dose_emit_time) >= self.dose_poll_interval:
                if self.dose_grids_info and len(self._multi_dose_bufs) > 0:
                    try:
                        dose_dict = {name: buf.copy() for name, buf in self._multi_dose_bufs.items()}
                        self.dose_volume_received.emit(dose_dict)
                        self._last_dose_emit_time = now
                    except Exception as e:
                        _logger.debug(f"Ошибка чтения снимков дозы: {e}")
                elif self._dose_buf is not None:
                    try:
                        dose_snapshot = self._dose_buf.copy()
                        self.dose_volume_received.emit(dose_snapshot)
                        self._last_dose_emit_time = now
                    except Exception as e:
                        _logger.debug(f"Ошибка чтения снимка дозы: {e}")

        # Финальный дренаж очереди треков перед завершением
        if self.track_queue is not None:
            while True:
                try:
                    item = self.track_queue.get_nowait()
                except (queue.Empty, Exception):
                    break
                if self._process_track_item(item):
                    break

        # Финальный сброс спектра
        if len(self._accumulated_energies) > 0:
            all_energies = np.concatenate(self._accumulated_energies) if len(self._accumulated_energies) > 1 else self._accumulated_energies[0]
            self.spectrum_received.emit(all_energies)
            self._spectrum_dirty = False

        # Финальный снимок проекции
        if self._projection_buf is not None:
            try:
                snapshot = self._projection_buf.copy()
                self.projection_received.emit(snapshot)
                total_counts = int(np.sum(snapshot))
                self.stats_updated.emit(total_counts, 0.0)
            except Exception:
                pass

        # Финальный снимок дозы
        if self._dose_buf is not None:
            try:
                self.dose_volume_received.emit(self._dose_buf.copy())
            except Exception:
                pass

    def stop(self, timeout_ms: int = 200) -> None:
        """
        Запрос остановки потока опроса без длительной блокировки GUI.
        """
        self._stop_event.set()
        if self.track_queue is not None:
            try:
                self.track_queue.put_nowait('stop')
            except Exception:
                pass
        if self.isRunning() and timeout_ms > 0:
            self.wait(timeout_ms)

    def close(self) -> None:
        """
        Освобождение дескрипторов IPC.
        """
        self.stop()
        if self._shm is not None:
            try:
                self._shm.close()
            except Exception:
                pass
            self._shm = None
            self._projection_buf = None

        if self._dose_shm is not None:
            try:
                self._dose_shm.close()
            except Exception:
                pass
            self._dose_shm = None
            self._dose_buf = None
