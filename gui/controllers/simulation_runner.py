import logging
import threading
from typing import Any, List, Optional

from PySide6.QtCore import QThread, Signal

from core.transport.simulation_managers import SimulationManager, SimulationState
from core.config.models import SimulationConfig
from core.config.orchestrator import Orchestrator
from core.data.data_handlers import BaseDataHandler

_logger = logging.getLogger(__name__)


class SimulationRunner(QThread):
    """
    Контроллер выполнения моделирования на базе QThread.
    Изолирует длительные вычисления в отдельном системном потоке,
    предотвращая блокировку основного графического цикла событий Qt.
    Поддерживает запуск как одиночного SimulationManager, так и пула воркеров Orchestrator
    с возможностью внешней инъекции обработчиков данных (extra_handlers).
    """

    simulation_started = Signal()
    simulation_paused = Signal()
    simulation_resumed = Signal()
    simulation_stopped = Signal()
    simulation_finished = Signal()
    simulation_error = Signal(str)
    status_updated = Signal(str)

    def __init__(
        self,
        manager: Optional[SimulationManager] = None,
        config: Optional[SimulationConfig] = None,
        parent: Optional[Any] = None,
        extra_handlers: Optional[List[BaseDataHandler]] = None,
        telemetry_queue: Optional[Any] = None,
    ) -> None:
        super().__init__(parent)
        self.manager = manager
        self.config = config
        self.orchestrator: Optional[Orchestrator] = None
        self.extra_handlers = extra_handlers
        self.telemetry_queue = telemetry_queue
        self._running_event = threading.Event()

    @property
    def is_running(self) -> bool:
        """
        Флаг активности рабочего потока симуляции.
        """
        return self._running_event.is_set()

    def set_simulation_manager(self, manager: SimulationManager) -> None:
        """
        Установка одиночного экземпляра менеджера моделирования.
        """
        self.manager = manager
        self.config = None

    def set_config(self, config: SimulationConfig) -> None:
        """
        Установка конфигурации для многопроцессного оркестратора.
        """
        self.config = config
        self.manager = None

    def run(self) -> None:
        """
        Точка входа рабочего потока QThread.
        """
        self._running_event.set()
        self.simulation_started.emit()
        self.status_updated.emit("Моделирование запущено")

        try:
            if self.manager is not None:
                # Одиночный SimulationManager: вызов публичного интерфейса run()
                self.manager.run()
            elif self.config is not None:
                # Многопроцессный пул Orchestrator
                self.orchestrator = Orchestrator(self.config)
                self.orchestrator.run(
                    telemetry_queue=self.telemetry_queue,
                    extra_handlers=self.extra_handlers,
                )
            else:
                raise ValueError("Не задан менеджер или конфигурация для моделирования.")

            self.status_updated.emit("Моделирование успешно завершено")
            self.simulation_finished.emit()

        except Exception as e:
            _logger.error(f"Ошибка в SimulationRunner: {e}", exc_info=True)
            self.simulation_error.emit(str(e))
            self.status_updated.emit(f"Ошибка: {e}")

        finally:
            self._running_event.clear()

    def pause(self) -> None:
        """
        Приостановка процесса моделирования.
        """
        if self.manager is not None:
            self.manager.pause()
            self.simulation_paused.emit()
            self.status_updated.emit("Моделирование приостановлено")

    def resume(self) -> None:
        """
        Возобновление процесса моделирования.
        """
        if self.manager is not None:
            self.manager.resume()
            self.simulation_resumed.emit()
            self.status_updated.emit("Моделирование возобновлено")

    def stop(self) -> None:
        """
        Кооперативная остановка процесса моделирования.
        """
        if self.manager is not None:
            self.manager.stop()
            self.simulation_stopped.emit()
            self.status_updated.emit("Моделирование остановлено пользователем")

    def step_once(self) -> None:
        """
        Покадровый шаг моделирования (при паузе).
        """
        if self.manager is not None:
            self.manager.step_once()
            self.status_updated.emit(f"Выполнен шаг {self.manager.step}")
