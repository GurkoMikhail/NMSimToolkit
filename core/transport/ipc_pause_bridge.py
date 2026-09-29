"""
Модуль межпроцессного адаптера паузы/возобновления расчетного ядра.
"""

import threading
from typing import Any

from core.transport.simulation_managers import SimulationManager, SimulationState


class IpcPauseBridge(threading.Thread):
    """
    Легковесный фоновый поток-адаптер для трансляции межпроцессных сигналов паузы (IPC Event)
    в публичные вызовы методов SimulationManager.pause() и SimulationManager.resume().
    Обеспечивает полную изоляцию расчетного ядра от подсистем multiprocessing и IPC.
    """

    def __init__(
        self,
        manager: SimulationManager,
        ipc_event: Any,
        poll_interval_seconds: float = 0.02,
    ) -> None:
        super().__init__(name="IpcPauseBridge", daemon=True)
        self.manager = manager
        self.ipc_event = ipc_event
        self.poll_interval_seconds = poll_interval_seconds
        self._stop_event = threading.Event()
        self._last_state_was_set: bool = bool(ipc_event.is_set())

    def stop_bridge(self) -> None:
        """Останавливает рабочий цикл потока-адаптера."""
        self._stop_event.set()

    def run(self) -> None:
        """Основной цикл опроса флага межпроцессного события паузы."""
        while not self._stop_event.is_set():
            if self.manager.state == SimulationState.STOPPED:
                break

            current_is_set = bool(self.ipc_event.is_set())
            if current_is_set != self._last_state_was_set:
                self._last_state_was_set = current_is_set
                if current_is_set:
                    self.manager.resume()
                else:
                    self.manager.pause()

            self._stop_event.wait(timeout=self.poll_interval_seconds)


__all__ = ["IpcPauseBridge"]
