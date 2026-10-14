import os
import tempfile
import unittest
from pathlib import Path
import numpy as np

from core.data.distribution_loader import DistributionLoader


class TestDistributionLoader(unittest.TestCase):
    """
    Модульные тесты для сервиса DistributionLoader.
    Проверяют корректность LBYL-валидации, диспетчеризации форматов и защиты от поврежденных данных.
    """

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_load_npy(self) -> None:
        """Проверка загрузки формата .npy."""
        sample_data = np.arange(24, dtype=np.float32).reshape((2, 3, 4))
        file_path = self.dir_path / "distribution.npy"
        np.save(file_path, sample_data)

        loaded = DistributionLoader.load(file_path)
        np.testing.assert_array_equal(sample_data, loaded)

    def test_load_binary_raw_and_bin(self) -> None:
        """Проверка загрузки бинарных форматов .raw и .bin с LBYL-валидацией."""
        sample_data = np.linspace(0.1, 10.0, 60, dtype=np.float32).reshape((3, 4, 5))
        file_path_raw = self.dir_path / "phantom.raw"
        sample_data.ravel(order='F').tofile(file_path_raw)

        loaded_raw = DistributionLoader.load(file_path_raw, target_shape=(3, 4, 5), order='F', dtype=np.float32)
        np.testing.assert_allclose(sample_data, loaded_raw)

        file_path_bin = self.dir_path / "phantom.bin"
        sample_data.ravel(order='F').tofile(file_path_bin)
        loaded_bin = DistributionLoader.load(file_path_bin, target_shape=(3, 4, 5), order='F', dtype=np.float32)
        np.testing.assert_allclose(sample_data, loaded_bin)

    def test_load_binary_mismatched_size_raises_value_error(self) -> None:
        """Проверка, что несовпадение объема бинарного буфера байт вызывает ValueError."""
        sample_data = np.zeros(50, dtype=np.float32)
        file_path = self.dir_path / "corrupted.raw"
        sample_data.tofile(file_path)

        with self.assertRaises(ValueError) as ctx:
            DistributionLoader.load(file_path, target_shape=(4, 4, 4), dtype=np.float32)
        self.assertIn("не соответствует требуемой форме", str(ctx.exception))

    def test_load_text_txt(self) -> None:
        """Проверка загрузки текстового файла распределения .txt."""
        sample_data = np.arange(12, dtype=np.float32).reshape((3, 4))
        file_path = self.dir_path / "matrix.txt"
        np.savetxt(file_path, sample_data)

        loaded = DistributionLoader.load(file_path, target_shape=(3, 4))
        np.testing.assert_allclose(sample_data, loaded)

    def test_load_dat_binary_and_text(self) -> None:
        """Проверка автоматической диспетчеризации .dat между бинарным и текстовым форматами."""
        # 1. Бинарный .dat
        sample_binary = np.ones((4, 4, 4), dtype=np.float32) * 42.0
        file_path_bin = self.dir_path / "binary.dat"
        sample_binary.tofile(file_path_bin)

        loaded_bin = DistributionLoader.load(file_path_bin, target_shape=(4, 4, 4), dtype=np.float32)
        np.testing.assert_allclose(sample_binary, loaded_bin)

        # 2. Текстовый .dat (размер в байтах больше 4 * N из-за ASCII символов)
        sample_text = np.array([0.04, 0.15, 0.04, 0.15], dtype=np.float32)
        file_path_txt = self.dir_path / "text.dat"
        np.savetxt(file_path_txt, sample_text)

        loaded_txt = DistributionLoader.load(file_path_txt, target_shape=(2, 2), order='F')
        np.testing.assert_allclose(sample_text.reshape((2, 2), order='F'), loaded_txt)

    def test_load_nonexistent_file_raises_filenotfound(self) -> None:
        """Проверка возбуждения FileNotFoundError при отсутствии файла на диске."""
        missing_path = self.dir_path / "nonexistent.raw"
        with self.assertRaises(FileNotFoundError):
            DistributionLoader.load(missing_path, target_shape=(2, 2))

    def test_load_unsupported_extension_raises_value_error(self) -> None:
        """Проверка возбуждения ValueError при неподдерживаемом расширении."""
        unsupported_path = self.dir_path / "image.png"
        unsupported_path.touch()
        with self.assertRaises(ValueError) as ctx:
            DistributionLoader.load(unsupported_path, target_shape=(10, 10))
        self.assertIn("Неподдерживаемое расширение", str(ctx.exception))

    def test_load_missing_shape_raises_value_error(self) -> None:
        """Проверка возбуждения ValueError при отсутствии обязательного target_shape."""
        file_path = self.dir_path / "test.raw"
        file_path.touch()
        with self.assertRaises(ValueError):
            DistributionLoader.load(file_path, target_shape=None)
        with self.assertRaises(ValueError):
            DistributionLoader.load(file_path, target_shape=(10, -5))

    def test_load_real_nema_phantom(self) -> None:
        """Проверка загрузки реального NEMA фантома anema_voxel_size_4.2_mm.dat из репозитория."""
        nema_path = Path("phantoms/nema/anema_voxel_size_4.2_mm.dat")
        if nema_path.is_file():
            # Загружаем срез или весь объем
            loaded = DistributionLoader.load(nema_path, target_shape=(128, 128, 92), order='F')
            self.assertEqual(loaded.shape, (128, 128, 92))
            self.assertGreater(float(np.max(loaded)), 0.0)
