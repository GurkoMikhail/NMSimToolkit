"""
Модуль настройки и управления логированием симуляции (logging_setup).
Обеспечивает создание изолированных файловых хэндлеров для параллельных воркеров и оркестратора,
предотвращает конфликты блокировок файлов в ОС Windows и исключает оверхед на горячем пути.
"""

import logging
from pathlib import Path
from typing import List, Optional

from core.config.models import LoggingConfig


def _resolve_log_level(level_name: str) -> int:
    """Преобразует строковое имя уровня логирования в числовую константу logging."""
    levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    normalized_name = level_name.upper()
    if normalized_name not in levels:
        raise ValueError(f"Неизвестный уровень логирования: '{level_name}'.")
    return levels[normalized_name]


def setup_task_logging(
    logging_config: Optional[LoggingConfig],
    task_id: int,
) -> List[logging.Handler]:
    """
    Настраивает изолированное логирование для единичной задачи воркера симуляции.
    Привязывает файловый и консольный хэндлеры к логгеру 'core'.

    :param logging_config: Конфигурация логирования или None (если логирование отключено).
    :param task_id: Уникальный целочисленный идентификатор задачи воркера.
    :return: Список созданных хэндлеров для последующего закрытия через close_logging_handlers.
    """
    if logging_config is None:
        return []

    handlers_list: List[logging.Handler] = []
    log_level = _resolve_log_level(logging_config.level)
    log_formatter = logging.Formatter(logging_config.format)

    # 1. Файловый хэндлер
    if logging_config.output_dir:
        output_directory = Path(logging_config.output_dir)
        output_directory.mkdir(parents=True, exist_ok=True)

        if logging_config.per_task_files:
            log_filename = logging_config.filename_pattern.format(task_id=task_id)
        else:
            log_filename = "simulation.log"

        log_file_path = output_directory / log_filename
        file_handler = logging.FileHandler(
            str(log_file_path),
            mode="a",
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(log_formatter)
        handlers_list.append(file_handler)

    # 2. Консольный хэндлер
    if logging_config.console:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(log_level)
        stream_handler.setFormatter(log_formatter)
        handlers_list.append(stream_handler)

    if handlers_list:
        core_logger = logging.getLogger("core")
        core_logger.setLevel(log_level)
        for handler in handlers_list:
            core_logger.addHandler(handler)

    return handlers_list


def setup_orchestrator_logging(
    logging_config: Optional[LoggingConfig],
) -> List[logging.Handler]:
    """
    Настраивает логирование для диспетчера оркестратора симуляций.

    :param logging_config: Конфигурация логирования или None.
    :return: Список созданных хэндлеров для последующего закрытия.
    """
    if logging_config is None:
        return []

    handlers_list: List[logging.Handler] = []
    log_level = _resolve_log_level(logging_config.level)
    log_formatter = logging.Formatter(logging_config.format)

    # Выбор целевого логгера: при per_task_files изолируем лог оркестратора
    target_logger_name = (
        "core.config.orchestrator" if logging_config.per_task_files else "core"
    )
    target_logger = logging.getLogger(target_logger_name)

    # 1. Файловый хэндлер
    if logging_config.output_dir:
        output_directory = Path(logging_config.output_dir)
        output_directory.mkdir(parents=True, exist_ok=True)

        log_filename = (
            "orchestrator.log" if logging_config.per_task_files else "simulation.log"
        )
        log_file_path = output_directory / log_filename
        file_handler = logging.FileHandler(
            str(log_file_path),
            mode="a",
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(log_formatter)
        handlers_list.append(file_handler)

    # 2. Консольный хэндлер
    if logging_config.console:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(log_level)
        stream_handler.setFormatter(log_formatter)
        handlers_list.append(stream_handler)

    if handlers_list:
        target_logger.setLevel(log_level)
        if logging_config.per_task_files:
            target_logger.propagate = False
        for handler in handlers_list:
            target_logger.addHandler(handler)

    return handlers_list


def close_logging_handlers(
    handlers_to_close: List[logging.Handler],
    target_logger: Optional[logging.Logger] = None,
) -> None:
    """
    Корректно закрывает и удаляет файловые/консольные хэндлеры,
    предотвращая утечки дескрипторов файлов в ОС Windows.

    :param handlers_to_close: Список хэндлеров для закрытия и удаления.
    :param target_logger: Логгер, от которого нужно открепить хэндлеры.
                         Если не указан, открепляет от 'core' и 'core.config.orchestrator'.
    """
    if not handlers_to_close:
        return

    registered_loggers = (
        [target_logger]
        if target_logger is not None
        else [
            logging.getLogger("core"),
            logging.getLogger("core.config.orchestrator"),
        ]
    )

    for handler in handlers_to_close:
        for current_logger in registered_loggers:
            if handler in current_logger.handlers:
                current_logger.removeHandler(handler)
        try:
            handler.flush()
        except (OSError, ValueError):
            pass
        try:
            handler.close()
        except (OSError, ValueError):
            pass

    orch_logger = logging.getLogger("core.config.orchestrator")
    if not orch_logger.handlers:
        orch_logger.propagate = True


__all__ = [
    "setup_task_logging",
    "setup_orchestrator_logging",
    "close_logging_handlers",
]
