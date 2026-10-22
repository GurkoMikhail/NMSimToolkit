"""
Тесты конфигурации объёма данных и ёмкости буферов:
- валидатор и тип DataSizeConfig (core/config/units.py);
- поле data_manager.buffer_size и его сериализация (core/config/models.py);
- поле simulation_manager.buffer_capacity и его связывание в воркере (simulation_worker.py).
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import hepunits as units
from pydantic import ValidationError

from core.config.models import DataManagerConfig, SimulationManagerConfig
from core.config.simulation_worker import simulation_worker_task
from core.config.units import data_size_validator


class TestDataSizeValidator(unittest.TestCase):
    """Проверка перевода объёма данных в целое число байт."""

    def test_binary_and_decimal_prefixes_converted_to_bytes(self) -> None:
        self.assertEqual(data_size_validator("3 MiB"), 3145728)
        self.assertEqual(data_size_validator("512 KiB"), 524288)
        self.assertEqual(data_size_validator("1 GiB"), 1073741824)
        self.assertEqual(data_size_validator("4 MB"), 4000000)
        self.assertEqual(data_size_validator("4096 B"), 4096)

    def test_result_is_exact_int(self) -> None:
        for raw_value in ("3 MiB", "4 MB", "4096 B"):
            self.assertIs(type(data_size_validator(raw_value)), int)

    def test_positive_int_passes_as_bytes(self) -> None:
        self.assertEqual(data_size_validator(4096), 4096)

    def test_strict_macro_is_skipped(self) -> None:
        self.assertEqual(data_size_validator("${buffer_var}"), "${buffer_var}")

    def test_integer_valued_float_from_macro_substitution_passes(self) -> None:
        result = data_size_validator(3145728.0)
        self.assertEqual(result, 3145728)
        self.assertIs(type(result), int)

    def test_invalid_values_raise_value_error(self) -> None:
        invalid_values = (
            "0 B", "1.5 B", "4 s", "", "   ", "-1 KiB", "3", "4 rad", "abc",
            0, -5, True, 1.5, 0.0, -4096.0, float("nan"), float("inf"), None,
        )
        for invalid_value in invalid_values:
            with self.subTest(value=invalid_value):
                with self.assertRaises(ValueError):
                    data_size_validator(invalid_value)


class TestDataManagerBufferSize(unittest.TestCase):
    """Проверка поля data_manager.buffer_size."""

    def test_default_is_validated_to_bytes(self) -> None:
        self.assertEqual(DataManagerConfig(filename="out.hdf").buffer_size, 3145728)

    def test_string_and_int_inputs(self) -> None:
        self.assertEqual(DataManagerConfig(filename="out.hdf", buffer_size="512 KiB").buffer_size, 524288)
        self.assertEqual(DataManagerConfig(filename="out.hdf", buffer_size=2048).buffer_size, 2048)

    def test_invalid_values_raise_validation_error(self) -> None:
        for invalid_value in ("0 B", "1.5 B", "4 s", "", True):
            with self.subTest(value=invalid_value):
                with self.assertRaises(ValidationError):
                    DataManagerConfig(filename="out.hdf", buffer_size=invalid_value)

    def test_macro_is_preserved(self) -> None:
        config = DataManagerConfig(filename="out.hdf", buffer_size="${buffer_var}")
        self.assertEqual(config.buffer_size, "${buffer_var}")
        self.assertEqual(config.model_dump()["buffer_size"], "${buffer_var}")

    def test_serialization_uses_binary_prefixes(self) -> None:
        expected_by_bytes = {
            3145728: "3 MiB",
            524288: "512 KiB",
            1073741824: "1 GiB",
            4000000: "4000000 B",
            1000: "1000 B",
            1536: "1536 B",
        }
        for byte_count, expected_text in expected_by_bytes.items():
            with self.subTest(byte_count=byte_count):
                config = DataManagerConfig(filename="out.hdf", buffer_size=byte_count)
                self.assertEqual(config.model_dump()["buffer_size"], expected_text)

    def test_roundtrip_preserves_buffer_size(self) -> None:
        for byte_count in (3145728, 524288, 1073741824, 4000000, 1000):
            with self.subTest(byte_count=byte_count):
                config = DataManagerConfig(filename="out.hdf", buffer_size=byte_count)
                restored = DataManagerConfig.model_validate(config.model_dump())
                self.assertEqual(restored, config)
                self.assertIs(type(restored.buffer_size), int)
                restored_from_json = DataManagerConfig.model_validate_json(config.model_dump_json())
                self.assertEqual(restored_from_json, config)

    def test_buffer_capacity_field_removed(self) -> None:
        self.assertNotIn("buffer_capacity", DataManagerConfig.model_fields)


class TestSimulationManagerBufferCapacity(unittest.TestCase):
    """Проверка поля simulation_manager.buffer_capacity и его связывания в воркере."""

    def test_default_is_none(self) -> None:
        self.assertIsNone(SimulationManagerConfig().buffer_capacity)

    def test_value_roundtrip(self) -> None:
        config = SimulationManagerConfig(buffer_capacity=100000)
        self.assertEqual(config.buffer_capacity, 100000)
        self.assertEqual(SimulationManagerConfig.model_validate(config.model_dump()), config)
        none_config = SimulationManagerConfig(buffer_capacity=None)
        self.assertEqual(SimulationManagerConfig.model_validate(none_config.model_dump()), none_config)

    def test_invalid_value_raises_validation_error(self) -> None:
        with self.assertRaises(ValidationError):
            SimulationManagerConfig(buffer_capacity=0)

    def _run_worker_with_stubbed_manager(self, simulation_manager_section: dict) -> dict:
        """Запускает воркер с подменённым SimulationManager и возвращает его именованные аргументы."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            task_config = {
                "simulation_manager": {
                    "particles_number": 10,
                    "min_energy": 0.05 * units.keV,
                    "stop_time": 10.0 * units.s,
                    **simulation_manager_section,
                },
                "data_manager": {"filename": str(Path(tmp_dir) / "output.hdf")},
                "scene": {
                    "type": "Volume",
                    "name": "World",
                    "geometry": {"type": "Box", "x": 100.0, "y": 100.0, "z": 100.0},
                    "material": "Vacuum",
                },
            }
            with patch(
                "core.config.simulation_worker.SimulationManager",
                side_effect=RuntimeError("остановка после создания менеджера"),
            ) as manager_cls:
                with self.assertRaises(RuntimeError):
                    simulation_worker_task((task_config, 42, None))
            return dict(manager_cls.call_args.kwargs)

    def test_worker_passes_simulation_manager_buffer_capacity(self) -> None:
        manager_kwargs = self._run_worker_with_stubbed_manager({"buffer_capacity": 12345})
        self.assertEqual(manager_kwargs["buffer_capacity"], 12345)

    def test_worker_passes_none_when_buffer_capacity_not_set(self) -> None:
        manager_kwargs = self._run_worker_with_stubbed_manager({})
        self.assertIsNone(manager_kwargs["buffer_capacity"])


if __name__ == "__main__":
    unittest.main()
