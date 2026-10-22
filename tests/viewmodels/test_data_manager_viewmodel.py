"""
Тесты DataManagerViewModel: объём выходного буфера записи (buffer_size) и roundtrip конфигурации.
"""
import unittest

from core.config.models import DataManagerConfig
from gui.viewmodels.data_handler_viewmodel import DataManagerViewModel


class TestDataManagerViewModelBufferSize(unittest.TestCase):
    """Проверка свойства buffer_size (байты) и его текстового представления."""

    def test_default_buffer_size_is_three_mebibytes(self) -> None:
        view_model = DataManagerViewModel()
        self.assertEqual(view_model.buffer_size, 3 * 1024 ** 2)
        self.assertEqual(view_model.buffer_size_text, "3 MiB")

    def test_parses_values_with_units(self) -> None:
        view_model = DataManagerViewModel()
        view_model.buffer_size = "3 MiB"
        self.assertEqual(view_model.buffer_size, 3145728)
        view_model.buffer_size = "512 KiB"
        self.assertEqual(view_model.buffer_size, 524288)
        self.assertEqual(view_model.buffer_size_text, "512 KiB")
        view_model.buffer_size = 1000
        self.assertEqual(view_model.buffer_size, 1000)
        self.assertEqual(view_model.buffer_size_text, "1000 B")

    def test_rejects_invalid_values_and_keeps_state(self) -> None:
        view_model = DataManagerViewModel(buffer_size="1 MiB")
        for invalid_value in ("0 B", "4 s", "abc", "", "${buffer_var}", 0, -5):
            with self.assertRaises(ValueError, msg=repr(invalid_value)):
                view_model.buffer_size = invalid_value
            self.assertEqual(view_model.buffer_size, 1024 ** 2, msg=repr(invalid_value))

    def test_changed_signal_emitted_only_on_actual_change(self) -> None:
        view_model = DataManagerViewModel(buffer_size="1 MiB")
        emitted_count = []
        view_model.changed.connect(lambda: emitted_count.append(1))
        view_model.buffer_size = "1 MiB"
        self.assertEqual(len(emitted_count), 0)
        view_model.buffer_size = "2 MiB"
        self.assertEqual(len(emitted_count), 1)


class TestDataManagerViewModelRoundtrip(unittest.TestCase):
    """Roundtrip to_config / load_from_config для buffer_size."""

    def test_buffer_size_roundtrip_through_config(self) -> None:
        source_view_model = DataManagerViewModel(filename="out.h5", buffer_size="512 KiB")
        exported_config = source_view_model.to_config()
        self.assertEqual(exported_config.buffer_size, 524288)

        restored_view_model = DataManagerViewModel()
        restored_view_model.load_from_config(exported_config)
        self.assertEqual(restored_view_model.buffer_size, 524288)
        self.assertEqual(restored_view_model.to_config(), exported_config)

    def test_export_yaml_value_is_human_readable(self) -> None:
        exported_config = DataManagerViewModel(filename="out.h5", buffer_size="3 MiB").to_config()
        dumped_data = exported_config.model_dump()
        self.assertEqual(dumped_data["buffer_size"], "3 MiB")
        self.assertEqual(DataManagerConfig.model_validate(dumped_data), exported_config)

    def test_load_from_config_rejects_macro(self) -> None:
        view_model = DataManagerViewModel()
        macro_config = DataManagerConfig(filename="out.h5", buffer_size="${buffer_var}")
        with self.assertRaises(ValueError):
            view_model.load_from_config(macro_config)


if __name__ == "__main__":
    unittest.main()
