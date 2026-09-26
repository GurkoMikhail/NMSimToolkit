import logging
from typing import Any, Callable, Optional, Protocol, runtime_checkable
import numpy as np

_logger = logging.getLogger(__name__)


@runtime_checkable
class IViewModelWithPropertyChanged(Protocol):
    """
    Контракт модели представления с поддержкой сигнала property_changed.
    Исключает утиную типизацию и проверку hasattr/getattr при эмиссии сигналов.
    """
    property_changed: Any


def _is_equal(first_value: Any, second_value: Any) -> bool:
    """
    Безопасное поэлементное сравнение значений, включая массивы numpy.
    """
    if first_value is second_value:
        return True
    if first_value is None or second_value is None:
        return False
    if isinstance(first_value, np.ndarray) or isinstance(second_value, np.ndarray):
        try:
            return bool(np.array_equal(first_value, second_value))
        except (ValueError, TypeError):
            return False
    try:
        return bool(first_value == second_value)
    except (ValueError, TypeError):
        return False


def _emit_property_change(
    instance: Any,
    name: Optional[str],
    value: Any,
    on_change: Optional[Callable[[Any, Any], None]] = None
) -> None:
    """
    Унифицированное оповещение об изменении свойства: вызов on_change и эмиссия сигнала Qt property_changed.
    """
    if on_change is not None:
        try:
            on_change(instance, value)
        except Exception as err:
            _logger.error(f"Ошибка в колбэке on_change для поля {name}: {err}", exc_info=True)

    if isinstance(instance, IViewModelWithPropertyChanged) and name is not None:
        try:
            instance.property_changed.emit(name, value)
        except Exception as err:
            _logger.error(f"Ошибка эмиссии сигнала property_changed для поля {name}: {err}", exc_info=True)


class core_field:
    """
    Дескриптор-прокси для атрибутов расчетного ядра (core_node).
    Обеспечивает принцип Single Source of Truth: данные хранятся исключительно в core_node
    и никогда не дублируются в словаре состояния ViewModel (instance.__dict__).
    При изменении значения генерирует Qt-сигнал property_changed.
    """

    def __init__(
        self,
        core_attr: Optional[str] = None,
        default: Any = None,
        on_change: Optional[Callable[[Any, Any], None]] = None
    ) -> None:
        self.core_attr = core_attr
        self.default = default
        self.on_change = on_change
        self.name: Optional[str] = None

    def __set_name__(self, owner: Any, name: str) -> None:
        self.name = name
        if self.core_attr is None:
            self.core_attr = name

    def __get__(self, instance: Any, owner: Optional[Any] = None) -> Any:
        if instance is None:
            return self
        try:
            return instance.core_node.__getattribute__(self.core_attr)
        except AttributeError:
            return self.default

    def __set__(self, instance: Any, value: Any) -> None:
        if self.name is None:
            return

        old_value = self.__get__(instance)
        if _is_equal(old_value, value):
            return

        try:
            setattr(instance.core_node, self.core_attr, value)
        except AttributeError:
            raise AttributeError(
                f"Невозможно установить атрибут '{self.core_attr}': core_node не содержит данного поля"
            )

        # Single Source of Truth: данные НЕ кэшируются в instance.__dict__!
        _emit_property_change(instance, self.name, value, self.on_change)


class gui_field:
    """
    Дескриптор локального свойства отображения ViewModel (цвет, видимость, LOD и т.д.),
    отсутствующего в расчетном ядре.
    Значение хранится в instance.__dict__ и при изменении генерирует сигнал Qt.
    """

    def __init__(
        self,
        default: Any = None,
        on_change: Optional[Callable[[Any, Any], None]] = None
    ) -> None:
        self.default = default
        self.on_change = on_change
        self.name: Optional[str] = None

    def __set_name__(self, owner: Any, name: str) -> None:
        self.name = name

    def __get__(self, instance: Any, owner: Optional[Any] = None) -> Any:
        if instance is None:
            return self
        return instance.__dict__.get(self.name, self.default)

    def __set__(self, instance: Any, value: Any) -> None:
        if self.name is None:
            return

        old_value = self.__get__(instance)
        if _is_equal(old_value, value):
            return

        instance.__dict__[self.name] = value
        _emit_property_change(instance, self.name, value, self.on_change)
