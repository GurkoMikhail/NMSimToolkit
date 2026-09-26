import logging
from typing import Any, Dict, List, Optional
from PySide6.QtCore import QObject, Signal

from core.config.models import (
    AnyDataHandlerConfig,
    DirectStreamHandlerConfig,
    SensitiveVolumeHandlerConfig,
    HistoryAssemblerHandlerConfig,
    DoseMapHandlerConfig,
    DataManagerConfig,
)

_logger = logging.getLogger(__name__)


class BaseDataHandlerViewModel(QObject):
    """
    Базовая модель представления для обработчика данных симуляции (DataHandler).
    """

    changed = Signal()

    def __init__(
        self,
        name: str = "DataHandler",
        handler_type: str = "Base",
        enabled: bool = True,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._name: str = name
        self._handler_type: str = handler_type
        self._enabled: bool = enabled

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        if self._name != value:
            self._name = str(value)
            self.changed.emit()

    @property
    def handler_type(self) -> str:
        return self._handler_type

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        b = bool(value)
        if self._enabled != b:
            self._enabled = b
            self.changed.emit()

    def to_config(self) -> Optional[AnyDataHandlerConfig]:
        """Конвертация в конфигурационную модель ядра. Возвращает None, если обработчик выключен."""
        raise NotImplementedError("to_config() должен быть реализован в подклассе.")


class DirectStreamHandlerViewModel(BaseDataHandlerViewModel):
    """
    Модель представления обработчика прямого стриминга треков и проекций в интерфейс.
    """

    def __init__(self, show_escaped_tracks: bool = False, enabled: bool = True, parent: Optional[QObject] = None) -> None:
        super().__init__(
            name="Потоковый стриминг в GUI (DirectStreamHandler)",
            handler_type="DirectStreamHandler",
            enabled=enabled,
            parent=parent,
        )
        self._show_escaped_tracks: bool = bool(show_escaped_tracks)

    @property
    def show_escaped_tracks(self) -> bool:
        """Отображать ли покинувшие расчетную область треки."""
        return self._show_escaped_tracks

    @show_escaped_tracks.setter
    def show_escaped_tracks(self, value: bool) -> None:
        b = bool(value)
        if self._show_escaped_tracks != b:
            self._show_escaped_tracks = b
            self.changed.emit()

    def to_config(self) -> Optional[DirectStreamHandlerConfig]:
        if not self._enabled:
            return None
        return DirectStreamHandlerConfig(
            type="DirectStreamHandler",
            show_escaped_tracks=self._show_escaped_tracks
        )


class SensitiveVolumeHandlerViewModel(BaseDataHandlerViewModel):
    """
    Модель представления обработчика сбора взаимодействий в чувствительных объемах.
    """

    def __init__(
        self,
        sensitive_volumes: Optional[List[str]] = None,
        enabled: bool = True,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(
            name="Сбор попаданий в объемы (SensitiveVolumeHandler)",
            handler_type="SensitiveVolumeHandler",
            enabled=enabled,
            parent=parent,
        )
        self._sensitive_volumes: List[str] = list(sensitive_volumes) if sensitive_volumes else []

    @property
    def sensitive_volumes(self) -> List[str]:
        return list(self._sensitive_volumes)

    @sensitive_volumes.setter
    def sensitive_volumes(self, volumes: List[str]) -> None:
        self._sensitive_volumes = [str(v) for v in volumes]
        self.changed.emit()

    def to_config(self) -> Optional[SensitiveVolumeHandlerConfig]:
        if not self._enabled:
            return None
        return SensitiveVolumeHandlerConfig(
            type="SensitiveVolumeHandler",
            sensitive_volumes=self._sensitive_volumes,
        )


class HistoryAssemblerHandlerViewModel(BaseDataHandlerViewModel):
    """
    Модель представления обработчика сохранения истории взаимодействий и начальных состояний.
    """

    def __init__(
        self,
        sensitive_volumes: Optional[List[str]] = None,
        save_initial_states: bool = True,
        enabled: bool = True,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(
            name="История треков детектора (HistoryAssemblerHandler)",
            handler_type="HistoryAssemblerHandler",
            enabled=enabled,
            parent=parent,
        )
        self._sensitive_volumes: List[str] = list(sensitive_volumes) if sensitive_volumes else []
        self._save_initial_states: bool = bool(save_initial_states)

    @property
    def sensitive_volumes(self) -> List[str]:
        return list(self._sensitive_volumes)

    @sensitive_volumes.setter
    def sensitive_volumes(self, volumes: List[str]) -> None:
        self._sensitive_volumes = [str(v) for v in volumes]
        self.changed.emit()

    @property
    def save_initial_states(self) -> bool:
        return self._save_initial_states

    @save_initial_states.setter
    def save_initial_states(self, value: bool) -> None:
        b = bool(value)
        if self._save_initial_states != b:
            self._save_initial_states = b
            self.changed.emit()

    def to_config(self) -> Optional[HistoryAssemblerHandlerConfig]:
        if not self._enabled:
            return None
        return HistoryAssemblerHandlerConfig(
            type="HistoryAssemblerHandler",
            sensitive_volumes=self._sensitive_volumes,
            save_initial_states=self._save_initial_states,
        )


class DoseMapHandlerViewModel(BaseDataHandlerViewModel):
    """
    Модель представления обработчика накопления 3D-карты поглощенной дозы в DoseGridNode.
    """

    def __init__(
        self,
        grid_names: Optional[List[str]] = None,
        shm_name: str = "nmsim_dose_shm",
        enabled: bool = True,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(
            name="Накопление карты дозы (DoseMapHandler)",
            handler_type="DoseMapHandler",
            enabled=enabled,
            parent=parent,
        )
        self._grid_names: List[str] = list(grid_names) if grid_names else []
        self._shm_name: str = str(shm_name)

    @property
    def grid_names(self) -> List[str]:
        return list(self._grid_names)

    @grid_names.setter
    def grid_names(self, names: List[str]) -> None:
        self._grid_names = [str(n) for n in names]
        self.changed.emit()

    @property
    def shm_name(self) -> str:
        return self._shm_name

    @shm_name.setter
    def shm_name(self, name: str) -> None:
        if self._shm_name != name:
            self._shm_name = str(name)
            self.changed.emit()

    def to_config(self) -> Optional[DoseMapHandlerConfig]:
        if not self._enabled:
            return None
        return DoseMapHandlerConfig(
            type="DoseMapHandler",
            grid_names=self._grid_names,
            shm_name=self._shm_name,
        )


class DataManagerViewModel(QObject):
    """
    Модель представления диспетчера данных DataManager.
    Управляет путем к итоговому HDF5-файлу, емкостью буфера и списком активных обработчиков данных.
    """

    changed = Signal()
    handlers_changed = Signal()

    def __init__(
        self,
        filename: str = "simulation_results.h5",
        buffer_capacity: Optional[int] = None,
        handlers: Optional[List[BaseDataHandlerViewModel]] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._filename: str = filename
        self._min_buffer_capacity: int = 1
        self._buffer_capacity: int = buffer_capacity if buffer_capacity is not None else 1
        self._handlers: List[BaseDataHandlerViewModel] = []

        if handlers:
            for h in handlers:
                self.add_handler(h)
        else:
            # Набор обработчиков по умолчанию
            self.add_handler(DirectStreamHandlerViewModel(enabled=True))
            self.add_handler(HistoryAssemblerHandlerViewModel(enabled=True, save_initial_states=True))
            self.add_handler(SensitiveVolumeHandlerViewModel(enabled=True))
            self.add_handler(DoseMapHandlerViewModel(enabled=True))

    @property
    def filename(self) -> str:
        return self._filename

    @filename.setter
    def filename(self, path: str) -> None:
        p = str(path).strip()
        if self._filename != p:
            self._filename = p
            self.changed.emit()

    @property
    def min_buffer_capacity(self) -> int:
        return self._min_buffer_capacity

    @min_buffer_capacity.setter
    def min_buffer_capacity(self, value: int) -> None:
        self._min_buffer_capacity = max(1, int(value))
        if self._buffer_capacity < self._min_buffer_capacity:
            self.buffer_capacity = self._min_buffer_capacity

    @property
    def buffer_capacity(self) -> int:
        return self._buffer_capacity

    @buffer_capacity.setter
    def buffer_capacity(self, cap: int) -> None:
        c = max(self._min_buffer_capacity, int(cap))
        if self._buffer_capacity != c:
            self._buffer_capacity = c
            self.changed.emit()

    @property
    def handlers(self) -> List[BaseDataHandlerViewModel]:
        return list(self._handlers)

    def add_handler(self, handler_vm: BaseDataHandlerViewModel) -> None:
        """Добавление нового обработчика данных."""
        if handler_vm not in self._handlers:
            self._handlers.append(handler_vm)
            handler_vm.changed.connect(self._on_handler_changed)
            self.handlers_changed.emit()
            self.changed.emit()

    def remove_handler(self, handler_vm: BaseDataHandlerViewModel) -> None:
        """Удаление обработчика данных."""
        if handler_vm in self._handlers:
            self._handlers.remove(handler_vm)
            try:
                handler_vm.changed.disconnect(self._on_handler_changed)
            except RuntimeError:
                pass
            self.handlers_changed.emit()
            self.changed.emit()

    def _on_handler_changed(self) -> None:
        self.changed.emit()

    def to_config(self) -> DataManagerConfig:
        """Сборка Pydantic-конфигурации DataManagerConfig ядра."""
        configs = self.to_core_handlers()
        return DataManagerConfig(
            filename=self._filename,
            buffer_capacity=self._buffer_capacity,
            handlers=configs,
        )

    def to_core_handlers(self) -> List[AnyDataHandlerConfig]:
        """Сборка списка Pydantic-конфигураций обработчиков данных ядра."""
        configs: List[AnyDataHandlerConfig] = []
        for h in self._handlers:
            cfg = h.to_config()
            if cfg is not None:
                configs.append(cfg)
        return configs

    def load_from_config(self, config: DataManagerConfig) -> None:
        """
        Загрузка диспетчера обработчиков данных из конфигурационной модели ядра.
        """
        self._filename = config.filename
        raw_capacity = config.buffer_capacity
        self.buffer_capacity = self._min_buffer_capacity if raw_capacity is None else max(self._min_buffer_capacity, int(raw_capacity))
        for h in list(self._handlers):
            self.remove_handler(h)

        for h_cfg in config.handlers:
            if isinstance(h_cfg, DirectStreamHandlerConfig):
                self.add_handler(DirectStreamHandlerViewModel(
                    show_escaped_tracks=h_cfg.show_escaped_tracks,
                    enabled=True,
                ))
            elif isinstance(h_cfg, SensitiveVolumeHandlerConfig):
                self.add_handler(SensitiveVolumeHandlerViewModel(
                    sensitive_volumes=h_cfg.sensitive_volumes,
                    enabled=True,
                ))
            elif isinstance(h_cfg, HistoryAssemblerHandlerConfig):
                self.add_handler(HistoryAssemblerHandlerViewModel(
                    sensitive_volumes=h_cfg.sensitive_volumes,
                    save_initial_states=h_cfg.save_initial_states,
                    enabled=True,
                ))
            elif isinstance(h_cfg, DoseMapHandlerConfig):
                self.add_handler(DoseMapHandlerViewModel(
                    grid_names=h_cfg.grid_names,
                    shm_name=h_cfg.shm_name,
                    enabled=True,
                ))
        self.handlers_changed.emit()
        self.changed.emit()

