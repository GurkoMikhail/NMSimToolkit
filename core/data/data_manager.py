"""
Модуль диспетчера данных моделирования (DataManager).
Управляет сохранением чанков взаимодействия частиц в файл HDF5,
а также обеспечивает раннюю инициализацию, изоляцию метаданных задач
и финализацию жизненного цикла симуляции.
"""

import logging
import threading
import time
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, List, Optional, Union

import h5py
import numpy as np

from core.data.data_handlers import BaseDataHandler

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


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
            if not isinstance(handler_instance, BaseDataHandler):
                raise TypeError(f"Handler {handler_instance} must be an instance of BaseDataHandler.")
            handler_instance.set_writer_callback(self._write_with_retry)
            self.handlers.append(handler_instance)

        # Состояние жизненного цикла метаданных
        self._status: str = "pending"
        self._is_initialized: bool = False
        self._is_finalized: bool = False
        self._start_perf_time: Optional[float] = None
        self._start_time_str: Optional[str] = None
        self._completion_time_str: Optional[str] = None
        self._elapsed_real_time_seconds: Optional[float] = None
        self._error_message: Optional[str] = None

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
                try:
                    target_group.attrs[str(item_key)] = item_value
                except Exception:
                    try:
                        target_group.attrs[str(item_key)] = str(item_value)
                    except Exception as attribute_error:
                        _logger.debug(
                            f"Не удалось записать атрибут {item_key} со значением {item_value}: {attribute_error}"
                        )

    def initialize_metadata(self) -> None:
        """
        Выполняет раннюю фиксацию метаданных задачи моделирования ДО фактического старта симуляции.
        Создает изолированную подгруппу /metadata/tasks/task_{task_id} со статусом 'in_progress',
        отметкой времени старта и параметрами протокола/детекторов.
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

            tasks_group = metadata_group.require_group("tasks")
            task_group = tasks_group.require_group(self.task_group_name)

            # Изолированная запись для текущей задачи
            task_group.attrs["task_id"] = self.task_id
            task_group.attrs["status"] = self._status
            task_group.attrs["start_time"] = self._start_time_str

            if self.metadata:
                self._write_dict_recursive(task_group, self.metadata)

        try:
            self._write_with_retry(do_write_initial_metadata)
        except (OSError, RuntimeError, KeyError, ValueError) as write_error:
            _logger.error(f"Ошибка ранней записи метаданных: {write_error}", exc_info=True)

    def finalize_metadata(self, status: str = "completed", error: Optional[str] = None) -> None:
        """
        Обновляет статус выполнения задачи ('completed' / 'failed'), фиксирует время завершения
        и реальное затраченное время выполнения в секундах.
        """
        if self._is_finalized and self._status == "failed" and status == "completed":
            return

        self._status = status
        self._is_finalized = True
        self._completion_time_str = str(time.strftime("%Y-%m-%d %H:%M:%S"))
        if self._start_perf_time is not None:
            self._elapsed_real_time_seconds = float(time.perf_counter() - self._start_perf_time)
        else:
            self._elapsed_real_time_seconds = 0.0

        if error is not None:
            self._error_message = str(error)

        def do_write_final_metadata(file_handle: h5py.File) -> None:
            metadata_group = file_handle.require_group("metadata")
            tasks_group = metadata_group.require_group("tasks")
            task_group = tasks_group.require_group(self.task_group_name)

            task_group.attrs["status"] = self._status
            task_group.attrs["completion_time"] = self._completion_time_str
            task_group.attrs["elapsed_real_time_seconds"] = self._elapsed_real_time_seconds
            if self._error_message is not None:
                task_group.attrs["error"] = self._error_message

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
                    except Exception as handler_error:
                        _logger.error(
                            f"Ошибка обработки чанка обработчиком {handler_instance}: {handler_error}",
                            exc_info=True,
                        )
                        self._error_message = str(handler_error)

        for handler_instance in self.handlers:
            try:
                handler_instance.finalize()
            except Exception as finalize_error:
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
            except (OSError, ValueError):
                pass
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
                    with h5py.File(self.filename, "a", libver="latest") as h5_file:
                        if self.swmr:
                            try:
                                h5_file.swmr_mode = True
                            except (OSError, RuntimeError):
                                pass
                        write_func(h5_file)
                        h5_file.flush()
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
