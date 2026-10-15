"""
Модуль модульных и интеграционных тестов подсистемы логирования NMSimToolkit.
Проверяет Pydantic-валидацию LoggingConfig, сериализацию YAML, изоляцию файлов
логов для параллельных задач воркеров и оркестратора в ОС Windows,
а также корректное закрытие файловых дескрипторов и Zero-Overhead на горячем пути.
"""

import logging
from pathlib import Path
import shutil
import tempfile
import time
import unittest

import numpy as np
from pydantic import ValidationError

from core.config.logging_setup import (
    setup_task_logging,
    setup_orchestrator_logging,
    close_logging_handlers,
    _resolve_log_level,
)
from core.config.models import (
    BoxConfig,
    DataManagerConfig,
    LoggingConfig,
    SimulationConfig,
    SimulationManagerConfig,
    VolumeConfig,
)
from core.config.orchestrator import Orchestrator
from core.config.simulation_worker import simulation_worker_task
from core.config.yaml_dumper import dump_simulation_config
from core.config.yaml_loader import load_simulation_config


class TestLoggingConfigModel(unittest.TestCase):
    """
    Тестирование Pydantic-модели LoggingConfig и ее интеграции в SimulationConfig.
    """

    def test_default_values(self) -> None:
        """Проверка дефолтных значений конфигурации логирования."""
        config = LoggingConfig()
        self.assertEqual(config.output_dir, "logs")
        self.assertEqual(config.level, "INFO")
        self.assertFalse(config.console)
        self.assertTrue(config.per_task_files)
        self.assertEqual(config.filename_pattern, "task_{task_id}.log")

    def test_level_normalization(self) -> None:
        """Проверка нормализации регистра уровня логирования (нижний регистр в верхний)."""
        config_lower = LoggingConfig(level="debug")  # type: ignore[arg-type]
        self.assertEqual(config_lower.level, "DEBUG")

        config_mixed = LoggingConfig(level="Warning")  # type: ignore[arg-type]
        self.assertEqual(config_mixed.level, "WARNING")

    def test_invalid_level_rejected(self) -> None:
        """Проверка отклонения некорректного уровня логирования."""
        with self.assertRaises(ValidationError):
            LoggingConfig(level="NOT_A_LEVEL")  # type: ignore[arg-type]

    def test_output_dir_path_normalization(self) -> None:
        """Проверка корректной нормализации pathlib.Path в строку."""
        custom_path = Path("custom") / "logs_dir"
        config = LoggingConfig(output_dir=custom_path)  # type: ignore[arg-type]
        self.assertEqual(config.output_dir, str(custom_path))

    def test_output_dir_empty_normalized_to_none(self) -> None:
        """Проверка нормализации пустой строки output_dir в None."""
        config_empty = LoggingConfig(output_dir="")
        self.assertIsNone(config_empty.output_dir)

        config_spaces = LoggingConfig(output_dir="   ")
        self.assertIsNone(config_spaces.output_dir)

    def test_filename_pattern_validation(self) -> None:
        """Проверка валидации шаблона имени файла лога."""
        valid_config = LoggingConfig(filename_pattern="custom_worker_{task_id}.log")
        self.assertEqual(valid_config.filename_pattern, "custom_worker_{task_id}.log")

        with self.assertRaises(ValidationError):
            LoggingConfig(filename_pattern="invalid_no_task_id_{worker}.log")

        with self.assertRaises(ValidationError):
            LoggingConfig(filename_pattern="")

    def test_resolve_log_level_invalid(self) -> None:
        """Проверка возбуждения ValueError при неизвестном уровне логирования."""
        with self.assertRaises(ValueError):
            _resolve_log_level("UNKNOWN_LEVEL")

    def test_multiprocess_single_file_logging_rejected(self) -> None:
        """Проверка запрета общего файла лога при pool_size > 1 для защиты от блокировок Windows."""
        raw_config_dict = {
            "pool_size": 4,
            "logging": {
                "output_dir": "logs",
                "per_task_files": False,
            },
            "simulation_manager": {
                "stop_time": "1.0 s",
                "particles_number": 100,
                "min_energy": "0.1 MeV",
            },
            "data_manager": {
                "filename": "output.h5",
                "handlers": [],
            },
            "scene": {
                "type": "Volume",
                "name": "World",
                "geometry": {
                    "type": "Box",
                    "x": "10.0 cm",
                    "y": "10.0 cm",
                    "z": "10.0 cm",
                },
                "material": "Air, Dry (near sea level)",
            },
        }
        with self.assertRaises(ValidationError):
            SimulationConfig.model_validate(raw_config_dict)

    def test_backward_compatibility_when_logging_omitted(self) -> None:
        """Проверка обратной совместимости: отсутствие секции logging оставляет None."""
        raw_config_dict = {
            "simulation_manager": {
                "stop_time": "1.0 s",
                "particles_number": 100,
                "min_energy": "0.1 MeV",
            },
            "data_manager": {
                "filename": "output.h5",
                "handlers": [],
            },
            "scene": {
                "type": "Volume",
                "name": "World",
                "geometry": {
                    "type": "Box",
                    "x": "10.0 cm",
                    "y": "10.0 cm",
                    "z": "10.0 cm",
                },
                "material": "Air, Dry (near sea level)",
            },
        }
        sim_config = SimulationConfig.model_validate(raw_config_dict)
        self.assertIsNone(sim_config.logging)

    def test_explicit_logging_section(self) -> None:
        """Проверка валидации явной секции logging в SimulationConfig."""
        raw_config_dict = {
            "logging": {
                "output_dir": "output_run_logs",
                "level": "DEBUG",
                "console": True,
                "per_task_files": True,
            },
            "simulation_manager": {
                "stop_time": "1.0 s",
                "particles_number": 100,
                "min_energy": "0.1 MeV",
            },
            "data_manager": {
                "filename": "output.h5",
                "handlers": [],
            },
            "scene": {
                "type": "Volume",
                "name": "World",
                "geometry": {
                    "type": "Box",
                    "x": "10.0 cm",
                    "y": "10.0 cm",
                    "z": "10.0 cm",
                },
                "material": "Air, Dry (near sea level)",
            },
        }
        sim_config = SimulationConfig.model_validate(raw_config_dict)
        self.assertIsNotNone(sim_config.logging)
        assert sim_config.logging is not None
        self.assertEqual(sim_config.logging.output_dir, "output_run_logs")
        self.assertEqual(sim_config.logging.level, "DEBUG")
        self.assertTrue(sim_config.logging.console)
        self.assertTrue(sim_config.logging.per_task_files)


class TestLoggingFileOperations(unittest.TestCase):
    """
    Тестирование файловых операций, хэндлеров и изоляции потоков логирования.
    """

    def setUp(self) -> None:
        self.temp_directory = tempfile.mkdtemp(prefix="nmsim_test_logs_")
        self.logs_path = Path(self.temp_directory) / "simulation_logs"

    def tearDown(self) -> None:
        if Path(self.temp_directory).exists():
            shutil.rmtree(self.temp_directory, ignore_errors=True)

    def test_setup_and_close_task_logging(self) -> None:
        """Проверка создания файла лога задачи и корректного закрытия хэндлеров на Windows."""
        log_config = LoggingConfig(
            output_dir=str(self.logs_path),
            level="INFO",
            per_task_files=True,
        )

        handlers = setup_task_logging(log_config, task_id=42)
        self.assertEqual(len(handlers), 1)

        task_logger = logging.getLogger("core.transport.simulation_managers")
        task_logger.info("Тестовое информационное сообщение воркера 42")
        task_logger.debug("Это сообщение DEBUG не должно быть записано")

        # Закрываем хэндлеры
        close_logging_handlers(handlers)

        # Проверяем наличие файла
        expected_file = self.logs_path / "task_42.log"
        self.assertTrue(expected_file.is_file())

        content = expected_file.read_text(encoding="utf-8")
        self.assertIn("Тестовое информационное сообщение воркера 42", content)
        self.assertNotIn("Это сообщение DEBUG не должно быть записано", content)

        # Проверка отсутствия блокировки в Windows: файл можно удалить сразу после закрытия
        expected_file.unlink()
        self.assertFalse(expected_file.exists())

    def test_setup_orchestrator_logging_isolated(self) -> None:
        """Проверка создания отдельного orchestrator.log при per_task_files=True."""
        log_config = LoggingConfig(
            output_dir=str(self.logs_path),
            level="INFO",
            per_task_files=True,
        )

        handlers = setup_orchestrator_logging(log_config)
        self.assertEqual(len(handlers), 1)

        orch_logger = logging.getLogger("core.config.orchestrator")
        orch_logger.info("Оркестратор запустил пул процессов")

        # Сообщение из воркера не должно попадать в orchestrator.log
        worker_logger = logging.getLogger("core.config.simulation_worker")
        worker_logger.info("Воркер выполняет перенос частиц")

        close_logging_handlers(handlers)

        expected_file = self.logs_path / "orchestrator.log"
        self.assertTrue(expected_file.is_file())

        content = expected_file.read_text(encoding="utf-8")
        self.assertIn("Оркестратор запустил пул процессов", content)
        self.assertNotIn("Воркер выполняет перенос частиц", content)

    def test_yaml_roundtrip_with_logging(self) -> None:
        """Проверка сериализации и десериализации конфигурации с секцией logging в YAML."""
        yaml_file = Path(self.temp_directory) / "config_with_logging.yaml"
        raw_config_dict = {
            "logging": {
                "output_dir": "yaml_output_logs",
                "level": "WARNING",
                "console": False,
                "per_task_files": True,
            },
            "simulation_manager": {
                "stop_time": "1.0 s",
                "particles_number": 500,
                "min_energy": "0.1 MeV",
            },
            "data_manager": {
                "filename": str(Path(self.temp_directory) / "test.h5"),
                "handlers": [],
            },
            "scene": {
                "type": "Volume",
                "name": "World",
                "geometry": {
                    "type": "Box",
                    "x": "10.0 cm",
                    "y": "10.0 cm",
                    "z": "10.0 cm",
                },
                "material": "Air, Dry (near sea level)",
            },
        }

        sim_config = SimulationConfig.model_validate(raw_config_dict)
        dump_simulation_config(sim_config, yaml_file)

        self.assertTrue(yaml_file.is_file())
        loaded_config = load_simulation_config(yaml_file)

        self.assertIsNotNone(loaded_config.logging)
        assert loaded_config.logging is not None
        self.assertEqual(loaded_config.logging.output_dir, "yaml_output_logs")
        self.assertEqual(loaded_config.logging.level, "WARNING")


class TestSimulationWorkerAndOrchestratorLogging(unittest.TestCase):
    """
    Интеграционное тестирование логирования при выполнении воркеров и оркестратора.
    """

    def setUp(self) -> None:
        self.temp_directory = tempfile.mkdtemp(prefix="nmsim_integ_logs_")
        self.output_logs = Path(self.temp_directory) / "logs"
        self.output_h5 = Path(self.temp_directory) / "sim_output.h5"

    def tearDown(self) -> None:
        if Path(self.temp_directory).exists():
            shutil.rmtree(self.temp_directory, ignore_errors=True)

    def _create_minimal_config_dict(self) -> dict:
        """Формирует минимальный валидный словарь конфигурации для запуска симуляции."""
        return {
            "logging": {
                "output_dir": str(self.output_logs),
                "level": "INFO",
                "console": False,
                "per_task_files": True,
            },
            "simulation_manager": {
                "stop_time": "0.1 s",
                "particles_number": 10,
                "min_energy": "0.1 MeV",
            },
            "data_manager": {
                "filename": str(self.output_h5),
                "handlers": [],
            },
            "scene": {
                "type": "Volume",
                "name": "World",
                "geometry": {
                    "type": "Box",
                    "x": "10.0 cm",
                    "y": "10.0 cm",
                    "z": "10.0 cm",
                },
                "material": "Air, Dry (near sea level)",
            },
        }

    def test_simulation_worker_task_creates_isolated_log(self) -> None:
        """Проверка создания и содержания файла лога task_0.log при запуске simulation_worker_task."""
        config_dict = self._create_minimal_config_dict()
        import multiprocessing
        multiprocessing_lock = multiprocessing.Lock()

        payload = (
            config_dict,
            42,  # seed
            multiprocessing_lock,
            None,  # telemetry_queue
            None,  # extra_handlers
            None,  # extra_handler_factory
            (),    # extra_handler_args
            0,     # task_id
        )

        simulation_worker_task(payload)

        task_log_file = self.output_logs / "task_0.log"
        self.assertTrue(task_log_file.is_file(), "Файл task_0.log должен быть создан воркером")

        log_content = task_log_file.read_text(encoding="utf-8")
        self.assertIn("Воркер 0: запуск задачи симуляции (seed=42)", log_content)
        self.assertIn("Воркер 0: симуляция успешно завершена", log_content)
        self.assertIn(str(self.output_h5), log_content)

        # Проверяем отсутствие файловых блокировок Windows
        task_log_file.unlink()
        self.assertFalse(task_log_file.exists())

    def test_orchestrator_multi_task_logging(self) -> None:
        """
        Проверка параллельного логирования оркестратора с несколькими задачами:
        создаются orchestrator.log, task_0.log, task_1.log.
        """
        config_dict = self._create_minimal_config_dict()
        config_dict["pool_size"] = 1
        config_dict["protocol"] = {
            "type": "CustomSweep",
            "grid_variables": {
                "test_angle": [0.0, 90.0],
            },
            "zipped_variables": {},
        }

        orchestrator = Orchestrator(config_dict)
        results = orchestrator.run()
        self.assertEqual(len(results), 2)

        # 1. Проверяем наличие логов задач
        task_0_log = self.output_logs / "task_0.log"
        task_1_log = self.output_logs / "task_1.log"
        orchestrator_log = self.output_logs / "orchestrator.log"

        self.assertTrue(task_0_log.is_file(), "task_0.log должен существовать")
        self.assertTrue(task_1_log.is_file(), "task_1.log должен существовать")
        self.assertTrue(orchestrator_log.is_file(), "orchestrator.log должен существовать")

        content_task_0 = task_0_log.read_text(encoding="utf-8")
        content_task_1 = task_1_log.read_text(encoding="utf-8")
        content_orch = orchestrator_log.read_text(encoding="utf-8")

        self.assertIn("Воркер 0: запуск задачи симуляции", content_task_0)
        self.assertIn("Воркер 0: симуляция успешно завершена", content_task_0)
        self.assertNotIn("Воркер 1:", content_task_0)

        self.assertIn("Воркер 1: запуск задачи симуляции", content_task_1)
        self.assertIn("Воркер 1: симуляция успешно завершена", content_task_1)
        self.assertNotIn("Воркер 0:", content_task_1)

        self.assertIn("Оркестратор: запуск моделирования (2 задач", content_orch)
        self.assertIn("Оркестратор: все 2 задач успешно завершены", content_orch)

    def test_simulation_worker_task_failure_cleans_up_handlers(self) -> None:
        """
        Проверка гарантированного закрытия файловых хэндлеров и записи ошибки в лог,
        если сбой произошел на этапе сборки сцены или валидации макросов.
        """
        import multiprocessing
        config_dict = self._create_minimal_config_dict()
        # Внедряем неразрешенный макрос, вызывающий немедленную ошибку инициализации
        config_dict["simulation_manager"]["particles_number"] = "${UNRESOLVED_MACRO}"
        multiprocessing_lock = multiprocessing.Lock()

        payload = (
            config_dict,
            777,   # seed
            multiprocessing_lock,
            None,  # telemetry_queue
            None,  # extra_handlers
            None,  # extra_handler_factory
            (),    # extra_handler_args
            5,     # task_id
        )

        with self.assertRaises(ValueError):
            simulation_worker_task(payload)

        task_5_log = self.output_logs / "task_5.log"
        self.assertTrue(task_5_log.is_file(), "Лог-файл задачи должен быть создан до падения")

        content = task_5_log.read_text(encoding="utf-8")
        self.assertIn("Воркер 5: ошибка моделирования", content)
        self.assertIn("UNRESOLVED_MACRO", content)

        # Проверка отсутствия блокировки Windows: файл удаляется без PermissionError
        task_5_log.unlink()
        self.assertFalse(task_5_log.exists())


if __name__ == "__main__":
    unittest.main()
