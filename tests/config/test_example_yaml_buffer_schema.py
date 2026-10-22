"""
Тесты схемы буферизации в YAML-примерах: simulation_manager.buffer_capacity и data_manager.buffer_size.
"""
import tempfile
import unittest
from pathlib import Path

from core.config.yaml_dumper import dump_simulation_config
from core.config.yaml_loader import load_raw_config, load_simulation_config

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
EXAMPLE_FILENAMES = ("tb161_nema_protocol.yaml", "nema_1_cam.yaml")


class TestExampleYamlBufferSchema(unittest.TestCase):
    """Примеры конфигураций используют новую схему и проходят валидацию загрузчиком проекта."""

    def test_examples_use_new_buffer_fields(self) -> None:
        for example_filename in EXAMPLE_FILENAMES:
            with self.subTest(example=example_filename):
                raw_config = load_raw_config(EXAMPLES_DIR / example_filename)
                self.assertNotIn("buffer_capacity", raw_config["data_manager"])
                self.assertEqual(raw_config["data_manager"]["buffer_size"], "3 MiB")
                self.assertEqual(raw_config["simulation_manager"]["buffer_capacity"], 100000)

    def test_examples_load_and_validate(self) -> None:
        for example_filename in EXAMPLE_FILENAMES:
            with self.subTest(example=example_filename):
                config = load_simulation_config(EXAMPLES_DIR / example_filename)
                assert config.data_manager is not None
                assert config.simulation_manager is not None
                self.assertEqual(config.data_manager.buffer_size, 3 * 1024 ** 2)
                self.assertEqual(config.simulation_manager.buffer_capacity, 100000)

    def test_buffer_fields_roundtrip_through_yaml(self) -> None:
        """export(build(config)) == config для data_manager.buffer_size и simulation_manager.buffer_capacity."""
        for example_filename in EXAMPLE_FILENAMES:
            with self.subTest(example=example_filename):
                original_config = load_simulation_config(EXAMPLES_DIR / example_filename)
                with tempfile.TemporaryDirectory() as temp_dir:
                    dumped_path = Path(temp_dir) / "dumped.yaml"
                    dump_simulation_config(original_config, dumped_path)
                    dumped_raw = load_raw_config(dumped_path)
                    restored_config = load_simulation_config(dumped_path)
                self.assertEqual(dumped_raw["data_manager"]["buffer_size"], "3 MiB")
                self.assertEqual(dumped_raw["simulation_manager"]["buffer_capacity"], 100000)
                assert original_config.data_manager is not None and restored_config.data_manager is not None
                assert original_config.simulation_manager is not None and restored_config.simulation_manager is not None
                self.assertEqual(restored_config.data_manager.buffer_size, original_config.data_manager.buffer_size)
                self.assertEqual(
                    restored_config.simulation_manager.buffer_capacity,
                    original_config.simulation_manager.buffer_capacity,
                )


if __name__ == "__main__":
    unittest.main()
