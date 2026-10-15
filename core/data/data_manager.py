"""
Модуль диспетчера данных моделирования (DataManager).
Управляет сохранением чанков взаимодействия частиц в файл HDF5,
а также обеспечивает раннюю инициализацию, изоляцию метаданных задач
и финализацию жизненного цикла симуляции.
"""

import json
import logging
import threading
import time
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, List, Optional, Union

import h5py
import hepunits as units
import numpy as np

from core.data.data_handlers import BaseDataHandler

_logger = logging.getLogger(__name__)


class DataManager(threading.Thread):
    """
    Поток-потребитель для сохранения чанков InteractionBuffer из SoA-движка
    в файл HDF5 с использованием делегированных обработчиков BaseDataHandler.
    Обеспечивает раннюю запись метаданных и изолированное сохранение метаданных задач.
    """

    def __init__(
        self,
        filename: Union[str, Path],
        handlers: List[BaseDataHandler],
        queue: Any = None,
        lock: Optional[Any] = None,
        swmr: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
        task_id: Optional[Any] = None,
    ) -> None:
        super().__init__()
        filepath_obj = Path(filename)
        if filepath_obj.is_absolute():
            self.filename = filepath_obj
        else:
            self.filename = Path(f"output data/{filename}")
        self.filename.parent.mkdir(parents=True, exist_ok=True)

        self.queue = queue
        self.lock = lock
        self.metadata: Dict[str, Any] = dict(metadata) if metadata is not None else {}

        # Разрешение task_id
        if task_id is not None:
            self.task_id: Any = task_id
        elif "task_id" in self.metadata:
            self.task_id = self.metadata["task_id"]
        else:
            self.task_id = 0
        self.metadata["task_id"] = self.task_id

        # Режим SWMR (Single Writer Multiple Reader) несовместим со сценарием
        # нескольких процессов-писателей (наличие внешнего файлового мьютекса lock)
        self.swmr = swmr and (lock is None)
        self.daemon = True

        self.handlers: List[BaseDataHandler] = []
        for handler_instance in handlers:
            handler_instance.set_writer_callback(self._write_with_retry)
            self.handlers.append(handler_instance)

        # Состояние жизненного цикла метаданных
        self._status: str = "pending"
        self._is_initialized: bool = False
        self._is_finalized: bool = False
        self._start_perf_time: Optional[float] = None
        self._start_time_str: Optional[str] = None
        self._completion_time_str: Optional[str] = None
        self._elapsed_real_time: Optional[float] = None
        self._error_message: Optional[str] = None
        self._runtime_summary: Optional[Dict[str, Any]] = None

    @staticmethod
    def _contexts_match(saved_context: Dict[str, Any], target_context: Optional[Dict[str, Any]]) -> bool:
        """
        Проверяет эквивалентность сохраненного в HDF5 контекста задачи и целевого контекста
        с поддержкой сравнения чисел с плавающей точкой через np.isclose.
        """
        if target_context is None:
            return True
        if set(saved_context.keys()) != set(target_context.keys()):
            return False
        for context_key, target_val in target_context.items():
            saved_val = saved_context[context_key]
            if isinstance(saved_val, (bytes, np.bytes_)):
                saved_val = saved_val.decode("utf-8")
            if (
                isinstance(saved_val, (int, float, np.number))
                and not isinstance(saved_val, bool)
                and isinstance(target_val, (int, float, np.number))
                and not isinstance(target_val, bool)
            ):
                if not bool(np.isclose(float(saved_val), float(target_val), rtol=1e-7, atol=1e-9)):
                    return False
            elif saved_val != target_val:
                return False
        return True

    @classmethod
    def inspect_resume_state(
        cls,
        filename: Union[str, Path],
        lock: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Инспектирует существующий HDF5-файл перед запуском задачи симуляции.
        Если в группе /metadata/tasks присутствуют завершенные задачи с сохраненным
        состоянием чекпоинта (actual_stop_time, final_rng_state), возвращает словарь
        параметров для бесшовного досчёта с учетом контекста задачи (context) и
        сквозного счетчика particle_ID по всему файлу.
        """
        filepath_obj = Path(filename)
        if filepath_obj.is_absolute():
            resolved_path = filepath_obj
        else:
            candidate_path = Path(f"output data/{filename}")
            resolved_path = candidate_path if candidate_path.exists() else filepath_obj

        if not resolved_path.exists():
            return None

        def _read_checkpoint() -> Optional[Dict[str, Any]]:
            retries_count = 20
            for retry_index in range(retries_count):
                try:
                    with h5py.File(resolved_path, "r") as h5_file:
                        if "metadata" not in h5_file:
                            return None
                        meta_group = h5_file["metadata"]
                        if not isinstance(meta_group, h5py.Group) or "tasks" not in meta_group:
                            return None
                        tasks_group = meta_group["tasks"]
                        if not isinstance(tasks_group, h5py.Group):
                            return None

                        has_completed_tasks = False
                        max_particle_id_end = 0
                        max_task_index = -1

                        best_key: Optional[tuple[float, int]] = None
                        best_stop_time: Optional[float] = None
                        best_seed: Optional[int] = None
                        best_rng_state: Optional[Dict[str, Any]] = None

                        for task_group_key, task_item in tasks_group.items():
                            if not isinstance(task_item, h5py.Group):
                                continue
                            if str(task_item.attrs.get("status", "")) != "completed":
                                continue
                            if (
                                "actual_stop_time" not in task_item.attrs
                                or "final_rng_state" not in task_item.attrs
                            ):
                                continue

                            stop_time_val = float(task_item.attrs["actual_stop_time"])
                            start_time_val = float(task_item.attrs.get("actual_start_time", 0.0))
                            if stop_time_val <= start_time_val:
                                continue

                            has_completed_tasks = True
                            particle_end_val = int(
                                task_item.attrs.get(
                                    "particle_id_end",
                                    task_item.attrs.get("simulated_histories", 0),
                                )
                            )
                            max_particle_id_end = max(max_particle_id_end, particle_end_val)

                            raw_task_id = task_item.attrs.get("task_id")
                            if isinstance(raw_task_id, (int, np.integer)):
                                max_task_index = max(max_task_index, int(raw_task_id))
                            elif str(task_group_key).startswith("task_"):
                                suffix_str = str(task_group_key)[len("task_"):]
                                if suffix_str.isdigit():
                                    max_task_index = max(max_task_index, int(suffix_str))

                            saved_context: Dict[str, Any] = {}
                            if "context" in task_item and isinstance(task_item["context"], h5py.Group):
                                saved_context = dict(task_item["context"].attrs)

                            if not cls._contexts_match(saved_context, context):
                                continue

                            raw_rng = task_item.attrs["final_rng_state"]
                            rng_json_str = (
                                raw_rng.decode("utf-8")
                                if isinstance(raw_rng, (bytes, np.bytes_))
                                else str(raw_rng)
                            )
                            rng_state_dict = json.loads(rng_json_str)
                            seed_val = int(task_item.attrs.get("seed", 0))

                            candidate_key = (stop_time_val, particle_end_val)
                            if best_key is None or candidate_key > best_key:
                                best_key = candidate_key
                                best_stop_time = stop_time_val
                                best_seed = seed_val
                                best_rng_state = rng_state_dict

                        if not has_completed_tasks:
                            return None

                        next_task_id = max_task_index + 1
                        while f"task_{next_task_id}" in tasks_group:
                            next_task_id += 1

                        return {
                            "seed": best_seed,
                            "actual_stop_time": best_stop_time,
                            "next_particle_id": max_particle_id_end,
                            "rng_state": best_rng_state,
                            "next_task_id": next_task_id,
                        }
                except (OSError, BlockingIOError):
                    if retry_index == retries_count - 1:
                        return None
                    time.sleep(0.05)
                except (ValueError, KeyError, TypeError):
                    return None
            return None

        if lock is not None:
            with lock:
                return _read_checkpoint()
        return _read_checkpoint()

    @property
    def task_group_name(self) -> str:
        """
        Формирует имя изолированной группы задачи в HDF5 (например, task_0, task_view_3).
        """
        string_identifier = str(self.task_id)
        if string_identifier.startswith("task_"):
            return string_identifier
        return f"task_{string_identifier}"

    @staticmethod
    def _write_dict_recursive(target_group: h5py.Group, data_dict: Dict[str, Any]) -> None:
        """
        Рекурсивно записывает словарь в группу HDF5 (вложенные словари в подгруппы,
        скаляры и массивы в HDF5 attrs).
        """
        for item_key, item_value in data_dict.items():
            if isinstance(item_value, dict):
                nested_group = target_group.require_group(str(item_key))
                DataManager._write_dict_recursive(nested_group, item_value)
            elif item_value is not None:
                if isinstance(item_value, (int, float, bool, str, bytes, np.number, np.ndarray)):
                    target_group.attrs[str(item_key)] = item_value
                else:
                    target_group.attrs[str(item_key)] = str(item_value)

    @staticmethod
    def _write_canonical_section(target_group: h5py.Group, data_dict: Dict[str, Any]) -> None:
        """
        Записывает структурированные параметры канонической группы метаданных
        (/metadata/gantry, geometry, source, simulation) как атрибуты и датасеты с единицами hepunits.
        """
        units_map: Dict[str, str] = {
            "gantry_angle": "rad",
            "angular_step": "deg",
            "orbit_radius": "mm",
            "frame_duration": "s",
            "exposure_time": "s",
            "elapsed_real_time": "s",
            "crystal_dimensions": "mm",
            "collimator_hole_diameter": "mm",
            "collimator_hole_length": "mm",
            "collimator_septa_thickness": "mm",
            "half_life": "s",
            "simulated_activity": "Bq",
        }

        for item_key, item_value in data_dict.items():
            if item_value is None:
                continue

            unit = units_map.get(str(item_key))

            # Если значение - словарь (например, запись объема), рекурсивно создаем подгруппу
            if isinstance(item_value, dict):
                sub_group = target_group.require_group(str(item_key))
                DataManager._write_canonical_section(sub_group, item_value)
                continue

            # Запись атрибута в группу
            if isinstance(item_value, (int, float, bool, str, bytes, np.number, np.ndarray)):
                target_group.attrs[str(item_key)] = item_value
            else:
                target_group.attrs[str(item_key)] = str(item_value)

            if unit is not None:
                target_group.attrs[f"{item_key}_units"] = unit

            # Создание датасета для массивов или числовых параметров
            if isinstance(item_value, np.ndarray):
                if str(item_key) not in target_group:
                    dset = target_group.create_dataset(str(item_key), data=item_value)
                    if unit is not None:
                        dset.attrs["units"] = unit
                else:
                    target_group[str(item_key)][...] = item_value
            elif isinstance(item_value, (list, tuple)):
                array_value = np.array(item_value)
                if str(item_key) not in target_group:
                    dset = target_group.create_dataset(str(item_key), data=array_value)
                    if unit is not None:
                        dset.attrs["units"] = unit
                else:
                    target_group[str(item_key)][...] = array_value
            elif isinstance(item_value, (int, float, np.integer, np.floating)):
                if str(item_key) not in target_group:
                    dset = target_group.create_dataset(str(item_key), data=item_value)
                    if unit is not None:
                        dset.attrs["units"] = unit
                else:
                    target_group[str(item_key)][()] = item_value
            elif isinstance(item_value, str):
                if str(item_key) not in target_group:
                    target_group.create_dataset(str(item_key), data=np.bytes_(item_value.encode("utf-8")))
                else:
                    target_group[str(item_key)][()] = np.bytes_(item_value.encode("utf-8"))

    def initialize_metadata(self) -> None:
        """
        Выполняет раннюю фиксацию метаданных задачи моделирования ДО фактического старта симуляции.
        Создает канонические группы /metadata/{gantry, geometry, source, simulation, protocol},
        а также изолированную подгруппу /metadata/tasks/task_{task_id} со статусом 'in_progress',
        отметкой времени старта и параметрами протокола/детекторов для обратной совместимости.
        """
        if self._is_initialized:
            return

        self._start_perf_time = time.perf_counter()
        self._start_time_str = str(time.strftime("%Y-%m-%d %H:%M:%S"))
        self._status = "in_progress"
        self._is_initialized = True

        def do_write_initial_metadata(file_handle: h5py.File) -> None:
            metadata_group = file_handle.require_group("metadata")
            if "created_time" not in metadata_group.attrs:
                metadata_group.attrs["created_time"] = self._start_time_str

            # 1. Запись в канонические корневые разделы /metadata/{gantry, geometry, source, simulation, protocol}
            for section_name in ("gantry", "geometry", "source", "simulation", "protocol"):
                if section_name in self.metadata and isinstance(self.metadata[section_name], dict):
                    section_group = metadata_group.require_group(section_name)
                    self._write_canonical_section(section_group, self.metadata[section_name])

            # 2. Изолированная запись для текущей задачи в /metadata/tasks/task_{task_id}
            tasks_group = metadata_group.require_group("tasks")
            if (
                self.task_group_name in tasks_group
                and str(tasks_group[self.task_group_name].attrs.get("status", "")) == "completed"
            ):
                candidate_idx = 0
                while f"task_{candidate_idx}" in tasks_group:
                    candidate_idx += 1
                self.task_id = candidate_idx
                self.metadata["task_id"] = candidate_idx

            task_group = tasks_group.require_group(self.task_group_name)

            task_group.attrs["task_id"] = self.task_id
            task_group.attrs["status"] = self._status
            task_group.attrs["start_time"] = self._start_time_str

            if self.metadata:
                task_specific_metadata = {
                    meta_key: meta_val
                    for meta_key, meta_val in self.metadata.items()
                    if meta_key not in ("gantry", "geometry", "source", "simulation")
                }
                self._write_dict_recursive(task_group, task_specific_metadata)

        try:
            self._write_with_retry(do_write_initial_metadata)
        except (OSError, RuntimeError, KeyError, ValueError) as write_error:
            _logger.error(f"Ошибка ранней записи метаданных: {write_error}", exc_info=True)

    def finalize_metadata(
        self,
        status: str = "completed",
        error: Optional[str] = None,
        runtime_summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Обновляет статус выполнения задачи ('completed' / 'failed'), фиксирует время завершения,
        реальное затраченное время выполнения, чекпоинт ГСЧ и кумулятивные метрики файла.
        """
        if self._is_finalized and self._status == "failed" and status == "completed":
            return

        if runtime_summary is not None:
            self._runtime_summary = dict(runtime_summary)

        self._status = status
        self._is_finalized = True
        self._completion_time_str = str(time.strftime("%Y-%m-%d %H:%M:%S"))
        if self._start_perf_time is not None:
            self._elapsed_real_time = float(time.perf_counter() - self._start_perf_time)
        else:
            self._elapsed_real_time = 0.0

        if error is not None:
            self._error_message = str(error)

        def do_write_final_metadata(file_handle: h5py.File) -> None:
            metadata_group = file_handle.require_group("metadata")
            tasks_group = metadata_group.require_group("tasks")
            task_group = tasks_group.require_group(self.task_group_name)

            task_group.attrs["status"] = self._status
            task_group.attrs["completion_time"] = self._completion_time_str
            task_group.attrs["elapsed_real_time"] = self._elapsed_real_time
            task_group.attrs["elapsed_real_time_units"] = "s"
            if self._error_message is not None:
                task_group.attrs["error"] = self._error_message

            if self._runtime_summary is not None:
                self._write_dict_recursive(task_group, self._runtime_summary)

            # Обновление статуса в канонической группе simulation
            if "simulation" in metadata_group:
                metadata_group["simulation"].attrs["status"] = self._status

            # Кумулятивный пересчет длительности экспозиции и числа историй по всем завершенным задачам
            has_runtime_durations = False
            total_duration = 0.0
            has_runtime_histories = False
            max_particle_id_end = 0
            sum_simulated_histories = 0

            for _, candidate_task in tasks_group.items():
                if not isinstance(candidate_task, h5py.Group):
                    continue
                if str(candidate_task.attrs.get("status", "")) != "completed":
                    continue

                if "actual_stop_time" in candidate_task.attrs and "actual_start_time" in candidate_task.attrs:
                    has_runtime_durations = True
                    total_duration += (
                        float(candidate_task.attrs["actual_stop_time"])
                        - float(candidate_task.attrs["actual_start_time"])
                    ) / float(units.s)
                elif "frame_duration" in candidate_task.attrs:
                    has_runtime_durations = True
                    total_duration += float(candidate_task.attrs["frame_duration"])

                if "particle_id_end" in candidate_task.attrs:
                    has_runtime_histories = True
                    max_particle_id_end = max(max_particle_id_end, int(candidate_task.attrs["particle_id_end"]))
                if "simulated_histories" in candidate_task.attrs:
                    has_runtime_histories = True
                    sum_simulated_histories += int(candidate_task.attrs["simulated_histories"])

            if has_runtime_durations:
                cumulative_frame_duration = round(total_duration, 6)
                gantry_group = metadata_group.require_group("gantry")
                self._write_canonical_section(gantry_group, {"frame_duration": cumulative_frame_duration})

            if has_runtime_histories:
                cumulative_histories = max(max_particle_id_end, sum_simulated_histories)
                sim_group = metadata_group.require_group("simulation")
                self._write_canonical_section(sim_group, {"simulated_histories": int(cumulative_histories)})

        try:
            self._write_with_retry(do_write_final_metadata)
        except (OSError, RuntimeError, KeyError, ValueError) as write_error:
            _logger.error(f"Ошибка финализации метаданных: {write_error}", exc_info=True)

    def run(self) -> None:
        """
        Извлекает чанки из очереди до получения сигнала 'stop'.
        """
        if self.queue is None:
            return

        if not self._is_initialized:
            self.initialize_metadata()

        while True:
            chunk = self.queue.get()
            if isinstance(chunk, str) and chunk == "stop":
                break
            elif isinstance(chunk, dict):
                frozen_chunk = self._freeze_chunk(chunk)
                for handler_instance in self.handlers:
                    try:
                        handler_instance.process_chunk(frozen_chunk)
                    except (OSError, RuntimeError, ValueError, KeyError, TypeError, IndexError) as handler_error:
                        _logger.error(
                            f"Ошибка обработки чанка обработчиком {handler_instance}: {handler_error}",
                            exc_info=True,
                        )
                        self._error_message = str(handler_error)

        for handler_instance in self.handlers:
            try:
                handler_instance.finalize()
            except (OSError, RuntimeError, ValueError, KeyError, TypeError, IndexError) as finalize_error:
                _logger.error(
                    f"Ошибка финализации обработчика {handler_instance}: {finalize_error}",
                    exc_info=True,
                )
                if self._error_message is None:
                    self._error_message = str(finalize_error)

        if not self._is_finalized:
            completion_status = "failed" if self._error_message is not None else "completed"
            self.finalize_metadata(status=completion_status)

    def stop(self, timeout: Optional[float] = 1.0) -> None:
        """
        Завершение фонового потока диспетчера данных и закрытие ресурсов.
        """
        if self.queue is not None:
            try:
                self.queue.put("stop")
            except (OSError, ValueError) as queue_error:
                _logger.debug(f"Не удалось отправить сигнал остановки в очередь DataManager: {queue_error}")
        if self.is_alive():
            self.join(timeout=timeout)

    @staticmethod
    def _freeze_chunk(chunk: dict) -> dict:
        """
        Устанавливает флаг только для чтения на все массивы numpy внутри данных чанка
        для предотвращения случайной модификации при широковещательной рассылке обработчикам.
        Возвращает защищенную копию словаря данных для защиты ключей.
        """
        data = chunk.get("data")
        if isinstance(data, dict):
            frozen_data = {}
            for field_name, field_array in data.items():
                if isinstance(field_array, np.ndarray):
                    field_array.flags.writeable = False
                frozen_data[field_name] = field_array
            chunk["data"] = MappingProxyType(frozen_data)
        elif isinstance(data, np.ndarray):
            data.flags.writeable = False

        return chunk

    def _write_with_retry(self, write_func: Any) -> None:
        """
        Выполняет функцию записи HDF5 с логикой повторных попыток и опциональной файловой блокировкой.
        """
        retries_count = 100

        def do_write() -> None:
            for retry_index in range(retries_count):
                try:
                    with h5py.File(self.filename, "a", libver="latest") as file_handle:
                        if self.swmr and not file_handle.swmr_mode:
                            try:
                                file_handle.swmr_mode = True
                            except (OSError, RuntimeError) as swmr_error:
                                _logger.debug(
                                    f"Не удалось активировать режим SWMR для файла {self.filename}: {swmr_error}"
                                )
                        write_func(file_handle)
                        file_handle.flush()
                    return
                except (OSError, BlockingIOError):
                    if retry_index == retries_count - 1:
                        raise
                    time.sleep(0.1)

        if self.lock is not None:
            with self.lock:
                do_write()
        else:
            do_write()


__all__ = ["DataManager"]
