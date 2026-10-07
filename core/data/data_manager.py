import logging
import threading
import time
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, List, Optional

import h5py
import numpy as np

from core.data.data_handlers import BaseDataHandler

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


class DataManager(threading.Thread):
    """
    Поток-потребитель для сохранения чанков InteractionBuffer из SoA-движка
    в файл HDF5 с использованием делегированных обработчиков BaseDataHandler.
    """

    def __init__(
        self,
        filename: str,
        handlers: List[BaseDataHandler],
        queue: Any = None,
        lock: Optional[Any] = None,
        swmr: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        fn_path = Path(filename)
        if fn_path.is_absolute():
            self.filename = fn_path
        else:
            self.filename = Path(f'output data/{filename}')
        self.filename.parent.mkdir(parents=True, exist_ok=True)

        self.queue = queue
        self.lock = lock
        self.metadata = metadata
        # Режим SWMR (Single Writer Multiple Reader) несовместим со сценарием
        # нескольких процессов-писателей (наличие внешнего файлового мьютекса lock)
        self.swmr = swmr and (lock is None)
        self.daemon = True

        self.handlers = []
        for h in handlers:
            if not isinstance(h, BaseDataHandler):
                raise TypeError(f"Handler {h} must be an instance of BaseDataHandler.")
            h.set_writer_callback(self._write_with_retry)
            self.handlers.append(h)

    def run(self):
        """
        Извлекает чанки из очереди до получения сигнала 'stop'.
        """
        if self.queue is None:
            return

        while True:
            chunk = self.queue.get()
            if isinstance(chunk, str) and chunk == 'stop':
                break
            elif isinstance(chunk, dict):
                frozen_chunk = self._freeze_chunk(chunk)
                for h in self.handlers:
                    try:
                        h.process_chunk(frozen_chunk)
                    except Exception as e:
                        _logger.error(f"Ошибка обработки чанка обработчиком {h}: {e}", exc_info=True)

        for h in self.handlers:
            try:
                h.finalize()
            except Exception as e:
                _logger.error(f"Ошибка финализации обработчика {h}: {e}", exc_info=True)

        self.write_metadata()

    def write_metadata(self) -> None:
        """
        Записывает глобальные метаданные моделирования, процедурный контекст и отметку времени в HDF5.
        """
        def do_write_metadata(f: h5py.File) -> None:
            if 'metadata' not in f:
                meta = f.create_group('metadata')
            else:
                meta = f['metadata']
            meta.attrs['completion_time'] = str(time.strftime('%Y-%m-%d %H:%M:%S'))
            if self.metadata:
                for meta_key, meta_val in self.metadata.items():
                    if isinstance(meta_val, dict):
                        sub_group = meta.require_group(meta_key)
                        for sub_k, sub_v in meta_val.items():
                            try:
                                sub_group.attrs[sub_k] = sub_v
                            except Exception:
                                sub_group.attrs[sub_k] = str(sub_v)
                    else:
                        try:
                            meta.attrs[meta_key] = meta_val
                        except Exception:
                            meta.attrs[meta_key] = str(meta_val)

        try:
            self._write_with_retry(do_write_metadata)
        except (OSError, RuntimeError, KeyError, ValueError) as e:
            _logger.debug(f"Запись метаданных пропущена: {e}")

    def stop(self, timeout: Optional[float] = 1.0) -> None:
        """
        Завершение фонового потока диспетчера данных и закрытие ресурсов.
        """
        if self.queue is not None:
            try:
                self.queue.put('stop')
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
        data = chunk.get('data')
        if isinstance(data, dict):
            frozen_data = {}
            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    v.flags.writeable = False
                frozen_data[k] = v
            # Return MappingProxyType to prevent adding/removing keys
            chunk['data'] = MappingProxyType(frozen_data)
        elif isinstance(data, np.ndarray):
            data.flags.writeable = False

        return chunk

    def _write_with_retry(self, write_func: Any) -> None:
        """
        Выполняет функцию записи HDF5 с логикой повторных попыток и опциональной файловой блокировкой.
        """
        retries = 100

        def do_write():
            for i in range(retries):
                try:
                    with h5py.File(self.filename, 'a', libver='latest') as f:
                        if self.swmr:
                            try:
                                f.swmr_mode = True
                            except (OSError, RuntimeError):
                                pass
                        write_func(f)
                        f.flush()
                    return
                except (OSError, BlockingIOError):
                    if i == retries - 1:
                        raise
                    time.sleep(0.1)

        if self.lock is not None:
            with self.lock:
                do_write()
        else:
            do_write()
