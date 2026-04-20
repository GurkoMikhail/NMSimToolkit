import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np

from core.data.data_handlers import BaseDataHandler

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


class DataManager(threading.Thread):
    """
    Consumer Thread for saving InteractionBuffer chunks from SoA engine
    to HDF5 file using delegated BaseDataHandlers.
    """

    def __init__(self, filename: str, handlers: List[BaseDataHandler], queue: Any = None, lock: Optional[Any] = None) -> None:
        super().__init__()
        self.filename = Path(f'output data/{filename}')
        self.filename.parent.mkdir(parents=True, exist_ok=True)

        self.queue = queue
        self.lock = lock
        self.daemon = True

        self.handlers = []
        for h in handlers:
            if not isinstance(h, BaseDataHandler):
                raise TypeError(f"Handler {h} must be an instance of BaseDataHandler.")
            h.set_writer_callback(self._write_with_retry)
            self.handlers.append(h)

    def run(self):
        """
        Consumes chunks from the queue until 'stop' signal.
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
                    h.process_chunk(frozen_chunk)

    @staticmethod
    def _freeze_chunk(chunk: dict) -> dict:
        """
        Sets all numpy arrays within the chunk data to read-only
        to prevent accidental modification during multi-handler broadcast.
        Returns a shallow copy of the data dictionary to protect keys from .pop().
        """
        from types import MappingProxyType

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
        Executes an HDF5 write function with retry logic and optional mutex locking.
        """
        import time
        retries = 100

        def do_write():
            for i in range(retries):
                try:
                    with h5py.File(self.filename, 'a') as f:
                        write_func(f)
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
