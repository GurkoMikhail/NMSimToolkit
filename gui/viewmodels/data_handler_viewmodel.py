import logging
from typing import Any, Dict, List, Literal, Optional, Union
from PySide6.QtCore import QObject, Signal

from core.config.models import (
    AnyDataHandlerConfig,
    DirectStreamHandlerConfig,
    SensitiveVolumeHandlerConfig,
    HistoryAssemblerHandlerConfig,
    DoseMapHandlerConfig,
    DataManagerConfig,
)
from core.config.units import data_size_validator, format_data_size
from gui.viewmodels.nodes.base_node_vm import NodeViewModel

_logger = logging.getLogger(__name__)

# Размер выходного буфера записи по умолчанию (совпадает со значением DataManagerConfig.buffer_size), байт
_DEFAULT_BUFFER_SIZE: int = data_size_validator("3 MiB")


def _parse_buffer_size(value: Union[int, str]) -> int:
    """
    Переводит введённый объём буфера ('3 MiB', '512 KiB', целое число байт) в целое число байт.
    Строковые макросы ${var} в GUI не поддерживаются.
    Raises:
        ValueError: значение не является положительным объёмом данных.
    """
    parsed_size = data_size_validator(value)
    if not isinstance(parsed_size, int):
        raise ValueError(f"Макрос '{value}' недопустим для объёма буфера в GUI")
    return parsed_size


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
        enabled_flag = bool(value)
        if self._enabled != enabled_flag:
            self._enabled = enabled_flag
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
        show_escaped_flag = bool(value)
        if self._show_escaped_tracks != show_escaped_flag:
            self._show_escaped_tracks = show_escaped_flag
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
    Является единственным владельцем списка чувствительных узлов (Volume / CompositeNode).
    """

    volumes_changed = Signal()

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
        self._sensitive_volumes: List[str] = []
        if sensitive_volumes:
            for vol in sensitive_volumes:
                name_str = str(vol).strip()
                if name_str and name_str not in self._sensitive_volumes:
                    self._sensitive_volumes.append(name_str)

    @property
    def sensitive_volumes(self) -> List[str]:
        return list(self._sensitive_volumes)

    @sensitive_volumes.setter
    def sensitive_volumes(self, volumes: List[str]) -> None:
        cleaned: List[str] = []
        for vol in volumes:
            name_str = str(vol).strip()
            if name_str and name_str not in cleaned:
                cleaned.append(name_str)
        if self._sensitive_volumes != cleaned:
            self._sensitive_volumes = cleaned
            self.volumes_changed.emit()
            self.changed.emit()

    def add_volume(self, node_or_name: Union[str, NodeViewModel]) -> bool:
        """
        Добавляет имя узла или NodeViewModel в список чувствительных объемов.
        Возвращает True, если узел был добавлен, False если уже присутствовал или имя пустое.
        """
        if isinstance(node_or_name, str):
            volume_name = node_or_name.strip()
        elif isinstance(node_or_name, NodeViewModel):
            volume_name = node_or_name.name.strip()
        else:
            raise TypeError(f"Ожидался NodeViewModel или str, получено: {type(node_or_name).__name__}")

        if not volume_name or volume_name in self._sensitive_volumes:
            return False

        self._sensitive_volumes.append(volume_name)
        self.volumes_changed.emit()
        self.changed.emit()
        return True

    def remove_volume(self, volume_name: str) -> bool:
        """
        Удаляет объем из списка по имени.
        Возвращает True при успешном удалении, False если объем не найден.
        """
        name_clean = str(volume_name).strip()
        if name_clean in self._sensitive_volumes:
            self._sensitive_volumes.remove(name_clean)
            self.volumes_changed.emit()
            self.changed.emit()
            return True
        return False

    def clear_volumes(self) -> None:
        """Полная очистка списка чувствительных объемов."""
        if self._sensitive_volumes:
            self._sensitive_volumes.clear()
            self.volumes_changed.emit()
            self.changed.emit()

    def on_node_renamed(self, old_name: str, new_name: str) -> None:
        """
        Атомарно обновляет имя узла в списке при переименовании в сцене без сброса настроек.
        """
        old_clean = str(old_name).strip()
        new_clean = str(new_name).strip()
        if old_clean in self._sensitive_volumes and new_clean:
            idx = self._sensitive_volumes.index(old_clean)
            self._sensitive_volumes[idx] = new_clean
            self.volumes_changed.emit()
            self.changed.emit()

    def on_node_removed(self, node_name: str) -> None:
        """
        Автоматически удаляет узел из списка, если он был удален из дерева сцены.
        """
        self.remove_volume(node_name)

    def to_config(self) -> Optional[Union[SensitiveVolumeHandlerConfig, HistoryAssemblerHandlerConfig]]:
        if not self._enabled:
            return None
        return SensitiveVolumeHandlerConfig(
            type="SensitiveVolumeHandler",
            sensitive_volumes=list(self._sensitive_volumes),
        )


class HistoryAssemblerHandlerViewModel(SensitiveVolumeHandlerViewModel):
    """
    Модель представления обработчика сохранения истории взаимодействий и начальных состояний.
    Наследует реестр чувствительных объемов от SensitiveVolumeHandlerViewModel без дублирования кода.
    """

    def __init__(
        self,
        sensitive_volumes: Optional[List[str]] = None,
        save_initial_states: bool = True,
        enabled: bool = True,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(
            sensitive_volumes=sensitive_volumes,
            enabled=enabled,
            parent=parent,
        )
        self._name = "История треков детектора (HistoryAssemblerHandler)"
        self._handler_type = "HistoryAssemblerHandler"
        self._save_initial_states: bool = bool(save_initial_states)

    @property
    def save_initial_states(self) -> bool:
        return self._save_initial_states

    @save_initial_states.setter
    def save_initial_states(self, value: bool) -> None:
        bool_value = bool(value)
        if self._save_initial_states != bool_value:
            self._save_initial_states = bool_value
            self.changed.emit()

    def to_config(self) -> Optional[HistoryAssemblerHandlerConfig]:
        if not self._enabled:
            return None
        return HistoryAssemblerHandlerConfig(
            type="HistoryAssemblerHandler",
            sensitive_volumes=list(self._sensitive_volumes),
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
        self._grid_names = [str(grid_name) for grid_name in names]
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
    Управляет путем к итоговому HDF5-файлу, объёмом выходного буфера записи и списком активных обработчиков данных.
    """

    changed = Signal()
    handlers_changed = Signal()

    def __init__(
        self,
        filename: str = "simulation_results.h5",
        buffer_size: Union[int, str] = _DEFAULT_BUFFER_SIZE,
        handlers: Optional[List[BaseDataHandlerViewModel]] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._filename: str = filename
        self._buffer_size: int = _parse_buffer_size(buffer_size)
        self._floating_point_precision: Literal["float32", "float64"] = "float64"
        self._compression: Optional[Literal["gzip", "lzf"]] = "gzip"
        self._compression_opts: int = 4
        self._handlers: List[BaseDataHandlerViewModel] = []

        if handlers:
            for handler_item in handlers:
                self.add_handler(handler_item)
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
        clean_path = str(path).strip()
        if self._filename != clean_path:
            self._filename = clean_path
            self.changed.emit()

    @property
    def buffer_size(self) -> int:
        """Объём выходного буфера записи на группу датасетов, байт."""
        return self._buffer_size

    @buffer_size.setter
    def buffer_size(self, value: Union[int, str]) -> None:
        """
        Принимает целое число байт или строку с единицей ('3 MiB', '512 KiB').
        Raises:
            ValueError: значение не является положительным объёмом данных (состояние не меняется).
        """
        parsed_size = _parse_buffer_size(value)
        if self._buffer_size != parsed_size:
            self._buffer_size = parsed_size
            self.changed.emit()

    @property
    def buffer_size_text(self) -> str:
        """Объём выходного буфера в читаемом виде со строкой единицы ('3 MiB')."""
        return format_data_size(self._buffer_size)

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
            buffer_size=self._buffer_size,
            floating_point_precision=self._floating_point_precision,
            compression=self._compression,
            compression_opts=self._compression_opts,
            handlers=configs,
        )

    def to_core_handlers(self) -> List[AnyDataHandlerConfig]:
        """Сборка списка Pydantic-конфигураций обработчиков данных ядра."""
        configs: List[AnyDataHandlerConfig] = []
        for handler in self._handlers:
            handler_config = handler.to_config()
            if handler_config is not None:
                configs.append(handler_config)
        return configs

    def load_from_config(self, config: DataManagerConfig) -> None:
        """
        Загрузка диспетчера обработчиков данных из конфигурационной модели ядра.
        """
        # Предусловие проверяется до изменения состояния: макросы ${var} недопустимы для объёма буфера в GUI
        self.buffer_size = config.buffer_size
        self._filename = config.filename
        self._floating_point_precision = config.floating_point_precision
        self._compression = config.compression
        self._compression_opts = int(config.compression_opts) if config.compression_opts is not None else 4
        for handler in list(self._handlers):
            self.remove_handler(handler)

        for handler_config in config.handlers:
            if isinstance(handler_config, DirectStreamHandlerConfig):
                self.add_handler(DirectStreamHandlerViewModel(
                    show_escaped_tracks=handler_config.show_escaped_tracks,
                    enabled=True,
                ))
            elif isinstance(handler_config, SensitiveVolumeHandlerConfig):
                self.add_handler(SensitiveVolumeHandlerViewModel(
                    sensitive_volumes=handler_config.sensitive_volumes,
                    enabled=True,
                ))
            elif isinstance(handler_config, HistoryAssemblerHandlerConfig):
                self.add_handler(HistoryAssemblerHandlerViewModel(
                    sensitive_volumes=handler_config.sensitive_volumes,
                    save_initial_states=handler_config.save_initial_states,
                    enabled=True,
                ))
            elif isinstance(handler_config, DoseMapHandlerConfig):
                self.add_handler(DoseMapHandlerViewModel(
                    grid_names=handler_config.grid_names,
                    shm_name=handler_config.shm_name,
                    enabled=True,
                ))
        self.handlers_changed.emit()
        self.changed.emit()

    def on_node_renamed(self, old_name: str, new_name: str) -> None:
        """Транслирует переименование узла сцены во все обработчики чувствительных объемов."""
        for handler in self._handlers:
            if isinstance(handler, SensitiveVolumeHandlerViewModel):
                handler.on_node_renamed(old_name, new_name)

    def on_node_removed(self, node_name: str) -> None:
        """Транслирует удаление узла сцены во все обработчики чувствительных объемов."""
        for handler in self._handlers:
            if isinstance(handler, SensitiveVolumeHandlerViewModel):
                handler.on_node_removed(node_name)

    def is_sensitive_volume(self, node_or_name: Union[str, NodeViewModel]) -> bool:
        """
        Проверяет, зарегистрировано ли имя узла (или узел/его предки)
        как чувствительный объем в любом включенном обработчике.
        """
        if isinstance(node_or_name, str):
            target_name = node_or_name.strip()
            for handler in self._handlers:
                if isinstance(handler, SensitiveVolumeHandlerViewModel) and handler.enabled:
                    if target_name in handler.sensitive_volumes:
                        return True
            return False

        if isinstance(node_or_name, NodeViewModel):
            curr: Optional[NodeViewModel] = node_or_name
            while curr is not None:
                if self.is_sensitive_volume(curr.name):
                    return True
                curr = curr.parent_vm
            return False

        return False


__all__ = [
    "BaseDataHandlerViewModel",
    "DirectStreamHandlerViewModel",
    "SensitiveVolumeHandlerViewModel",
    "HistoryAssemblerHandlerViewModel",
    "DoseMapHandlerViewModel",
    "DataManagerViewModel",
]

