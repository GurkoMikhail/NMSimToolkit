import os
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PySide6.QtWidgets import QApplication

from core.data.distribution_loader import DistributionLoader
from core.geometry.voxel_volumes import WoodcockVoxelVolume
from core.materials.materials import MaterialArray
from core.source.sources import Source
from gui.models.distribution_import_params import DistributionImportParameters, ImportTargetKind
from gui.viewmodels.nodes.source_vm import SourceViewModel
from gui.viewmodels.nodes.voxel_volume_vm import VoxelVolumeViewModel
from gui.views.distribution_import_dialog import DistributionImportDialog


class TestDistributionImportDialog(unittest.TestCase):
    """
    Модульные и интеграционные тесты для диалога импорта распределений,
    инспекции метаданных и поддержки нецелочисленных данных.
    """

    @classmethod
    def setUpClass(cls) -> None:
        if QApplication.instance() is None:
            cls.app = QApplication([])
        else:
            cls.app = QApplication.instance()

    def test_inspect_metadata_numpy(self) -> None:
        """
        Проверка инспекции метаданных numpy-контейнеров без полной загрузки в память.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_phantom.npy"
            original_array = np.zeros((10, 20, 30), dtype=np.float32, order='F')
            np.save(file_path, original_array)

            meta = DistributionLoader.inspect_metadata(file_path)
            self.assertEqual(meta["shape"], (10, 20, 30))
            self.assertEqual(meta["order"], "F")
            self.assertEqual(meta["dtype"], "float32")
            self.assertGreater(meta["file_size"], 0)

    def test_inspect_metadata_raw(self) -> None:
        """
        Проверка инспекции сырых бинарных файлов (dat/raw/bin).
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_source.dat"
            raw_bytes = bytes(10 * 10 * 10 * 4)  # 4000 байт
            file_path.write_bytes(raw_bytes)

            meta = DistributionLoader.inspect_metadata(file_path)
            self.assertIsNone(meta["shape"])
            self.assertIsNone(meta["order"])
            self.assertIsNone(meta["dtype"])
            self.assertEqual(meta["file_size"], 4000)

    def test_distribution_import_parameters_validation(self) -> None:
        """
        Проверка LBYL-валидации параметров импорта распределения.
        """
        # Корректные параметры
        params = DistributionImportParameters(
            file_path="/tmp/phantom.dat",
            target_kind=ImportTargetKind.PHANTOM,
            shape=(16, 16, 16),
            order='C',
            dtype='float32',
            voxel_size=(2.0, 2.0, 2.0),
        )
        self.assertEqual(params.shape, (16, 16, 16))

        # Ошибка: неположительные размеры сетки
        with self.assertRaises(ValueError):
            DistributionImportParameters(
                file_path="/tmp/phantom.dat",
                target_kind=ImportTargetKind.PHANTOM,
                shape=(0, 16, 16),
                order='C',
                dtype='float32',
                voxel_size=(2.0, 2.0, 2.0),
            )

        # Ошибка: неположительный размер вокселя
        with self.assertRaises(ValueError):
            DistributionImportParameters(
                file_path="/tmp/phantom.dat",
                target_kind=ImportTargetKind.PHANTOM,
                shape=(16, 16, 16),
                order='C',
                dtype='float32',
                voxel_size=(2.0, 0.0, 2.0),
            )

    def test_voxel_volume_vm_non_integer_mapping(self) -> None:
        """
        Проверка загрузки нецелочисленных данных (float values) в фантом
        без потери точности и усечения до целых чисел.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "float_phantom.npy"
            # Массив с дробными значениями: 0.0 (фон), 0.25 (вода), 1.5 (кость)
            shape = (4, 4, 4)
            array_data = np.zeros(shape, dtype=np.float32)
            array_data[0:2, :, :] = 0.25
            array_data[2:4, :, :] = 1.5
            np.save(file_path, array_data)

            core_phantom = WoodcockVoxelVolume(
                voxel_size=2.0,
                material_distribution=MaterialArray(shape),
                name="FloatPhantom",
            )
            phantom_vm = VoxelVolumeViewModel(core_phantom)

            mapping_config = {
                0.0: "Vacuum",
                0.25: "Water, Liquid",
                1.5: "Bone, Cortical (ICRU-44)",
            }

            success = phantom_vm.reload_distribution(
                path=str(file_path),
                shape=shape,
                mapping=mapping_config,
                voxel_size=(1.5, 1.5, 1.5),
            )
            self.assertTrue(success)
            np.testing.assert_allclose(phantom_vm.voxel_size, (1.5, 1.5, 1.5))

            # Проверяем, что в material_list попали заданные материалы
            material_names = [mat.name for mat in phantom_vm.material_list]
            self.assertIn("Water, Liquid", material_names)
            self.assertIn("Bone, Cortical (ICRU-44)", material_names)

    def test_source_vm_raw_reload_distribution(self) -> None:
        """
        Проверка загрузки сырых бинарных данных источника с геометрией и активностью.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "source_activity.raw"
            shape = (5, 5, 5)
            # 125 float32 чисел
            activity_data = np.full(shape, 10.0, dtype=np.float32)
            activity_data[0, 0, 0] = 0.01  # Ниже порога шума
            activity_data.tofile(file_path)

            source_core = Source(distribution=np.ones(shape, dtype=float), voxel_size=2.0)
            source_vm = SourceViewModel(source_core)

            success = source_vm.reload_distribution(
                path=str(file_path),
                shape=shape,
                order='C',
                dtype=np.float32,
                voxel_size=3.0,
                total_activity=250.0,
                noise_threshold=0.1,
            )
            self.assertTrue(success)
            self.assertEqual(source_vm.voxel_size, 3.0)
            self.assertEqual(source_vm.activity, 250.0)
            self.assertEqual(source_vm.dimensions, shape)
            # Проверяем фильтрацию фонового шума и нормировку вероятностей эмиссии
            self.assertAlmostEqual(source_core.distribution[0, 0, 0], 0.0)
            self.assertGreater(source_core.distribution[1, 1, 1], 0.0)
            self.assertAlmostEqual(float(np.sum(source_core.distribution)), 1.0)

    def test_distribution_import_dialog_phantom_mode(self) -> None:
        """
        Проверка интерфейса и параметров диалога в режиме импорта фантома.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "dialog_phantom.npy"
            array_data = np.zeros((8, 8, 8), dtype=np.float32)
            array_data[2:6, 2:6, 2:6] = 1.0
            np.save(file_path, array_data)

            dialog = DistributionImportDialog(
                file_path=str(file_path),
                target_kind=ImportTargetKind.PHANTOM,
                default_voxel_size=(2.5, 2.5, 2.5),
            )

            # Проверка автодетекции параметров npy
            self.assertEqual(dialog.spin_dim_x.value(), 8)
            self.assertEqual(dialog.spin_dim_y.value(), 8)
            self.assertEqual(dialog.spin_dim_z.value(), 8)

            # Проверка, что материал фона по умолчанию — Air, а не Vacuum
            self.assertEqual(dialog.combo_default_material.currentText(), "Air, Dry (near sea level)")

            # Запуск сканирования уникальных меток
            dialog._on_auto_scan_materials()
            self.assertGreaterEqual(dialog.tbl_mapping.rowCount(), 2)

            # Проверяем, что первая метка по умолчанию получила Air
            first_combo = dialog.tbl_mapping.cellWidget(0, 4)
            self.assertEqual(first_combo.currentText(), "Air, Dry (near sea level)")

            params = dialog.get_parameters()
            self.assertEqual(params.target_kind, ImportTargetKind.PHANTOM)
            self.assertEqual(params.shape, (8, 8, 8))
            self.assertEqual(params.voxel_size, (2.5, 2.5, 2.5))
            self.assertEqual(params.fill_value, "Air, Dry (near sea level)")
            dialog.close()

    def test_distribution_import_dialog_source_mode_with_phantom_sync(self) -> None:
        """
        Проверка интерфейса и кнопки синхронизации геометрии в режиме источника.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "dialog_source.raw"
            shape = (12, 12, 12)
            np.ones(shape, dtype=np.float32).tofile(file_path)

            phantom_core = WoodcockVoxelVolume(
                voxel_size=(3.5, 3.5, 3.5),
                material_distribution=MaterialArray(shape),
                name="ScenePhantom",
            )
            phantom_vm = VoxelVolumeViewModel(phantom_core)

            dialog = DistributionImportDialog(
                file_path=str(file_path),
                target_kind=ImportTargetKind.SOURCE,
                scene_phantom_node=phantom_vm,
            )

            # Нажимаем синхронизацию геометрии с фантомом
            dialog._on_copy_phantom_geometry()
            self.assertEqual(dialog.spin_dim_x.value(), 12)
            self.assertEqual(dialog.spin_source_voxel_step.value(), 3.5)

            # Установка параметров активности
            dialog.spin_source_total_activity.setValue(500.0)
            dialog.spin_source_threshold.setValue(0.05)

            params = dialog.get_parameters()
            self.assertEqual(params.target_kind, ImportTargetKind.SOURCE)
            self.assertEqual(params.shape, (12, 12, 12))
            self.assertEqual(params.total_activity, 500.0)
            self.assertEqual(params.noise_threshold, 0.05)
            dialog.close()

    def test_property_inspector_distribution_dialog_integration(self) -> None:
        """
        Интеграционный тест вызова DistributionImportDialog из PropertyInspector
        для фантома и источника с реактивным обновлением полей интерфейса.
        """
        from unittest.mock import patch
        from PySide6.QtWidgets import QDialog
        from gui.views.property_inspector import PropertyInspector
        from gui.viewmodels.scene_viewmodel import SceneViewModel

        with tempfile.TemporaryDirectory() as temp_dir:
            phantom_file = Path(temp_dir) / "test_phantom.npy"
            source_file = Path(temp_dir) / "test_source.npy"

            phantom_shape = (6, 6, 6)
            source_shape = (6, 6, 6)
            np.save(phantom_file, np.zeros(phantom_shape, dtype=np.float32))
            np.save(source_file, np.ones(source_shape, dtype=np.float32))

            phantom_core = WoodcockVoxelVolume(
                voxel_size=(2.0, 2.0, 2.0),
                material_distribution=MaterialArray(phantom_shape),
                name="InspectorPhantom",
            )
            phantom_vm = VoxelVolumeViewModel(phantom_core)

            source_core = Source(distribution=np.ones(source_shape, dtype=float), voxel_size=2.0)
            source_vm = SourceViewModel(source_core)

            from core.geometry.geometries import Box
            from core.geometry.volumes import Volume
            from core.materials.materials import Material

            world_core = Volume(geometry=Box(1000.0, 1000.0, 1000.0), material=Material(name="Vacuum", ID=0), name="World")
            scene_vm = SceneViewModel()
            scene_vm.load_scene(world_core)
            scene_vm.add_node(scene_vm.root_vm, phantom_vm)
            scene_vm.add_node(scene_vm.root_vm, source_vm)

            inspector = PropertyInspector(scene_vm=scene_vm)

            # 1. Проверка выбора файла фантома
            inspector.set_target_viewmodel(phantom_vm)

            mock_phantom_params = DistributionImportParameters(
                file_path=str(phantom_file),
                target_kind=ImportTargetKind.PHANTOM,
                shape=phantom_shape,
                order='C',
                dtype='float32',
                voxel_size=(3.0, 3.0, 3.0),
                material_mapping={0.0: "Water, Liquid"},
                fill_value="Air, Dry (near sea level)",
            )

            with patch("PySide6.QtWidgets.QFileDialog.getOpenFileName", return_value=(str(phantom_file), "All Files")), \
                 patch.object(DistributionImportDialog, "exec", return_value=QDialog.Accepted), \
                 patch.object(DistributionImportDialog, "get_parameters", return_value=mock_phantom_params):
                inspector._on_browse_phantom_file()

            self.assertEqual(inspector.txt_voxel_path.text(), str(phantom_file))
            self.assertEqual(inspector.spin_voxel_size_x.value(), 3.0)
            self.assertEqual(inspector.spin_voxel_size_y.value(), 3.0)
            self.assertEqual(inspector.spin_voxel_size_z.value(), 3.0)

            # 2. Проверка выбора файла источника
            inspector.set_target_viewmodel(source_vm)

            mock_source_params = DistributionImportParameters(
                file_path=str(source_file),
                target_kind=ImportTargetKind.SOURCE,
                shape=source_shape,
                order='C',
                dtype='float32',
                voxel_size=2.5,
                total_activity=150.0,
                noise_threshold=0.01,
            )

            with patch("PySide6.QtWidgets.QFileDialog.getOpenFileName", return_value=(str(source_file), "All Files")), \
                 patch.object(DistributionImportDialog, "exec", return_value=QDialog.Accepted), \
                 patch.object(DistributionImportDialog, "get_parameters", return_value=mock_source_params):
                inspector._on_browse_source_file()

            self.assertEqual(inspector.txt_source_path.text(), str(source_file))
            self.assertEqual(inspector.spin_source_voxel_size.value(), 2.5)
            self.assertEqual(inspector.spin_source_activity.value(), 150.0)
            inspector.close()

    def test_distribution_import_dialog_text_mode_validation(self) -> None:
        """
        Проверка того, что текстовый режим выбран по умолчанию для .dat/.txt файлов,
        а валидация мгновенно подсчитывает точное количество чисел без зависаний.
        """
        from PySide6.QtWidgets import QDialogButtonBox

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_phantom.dat"
            # 64 числа (4x4x4)
            numbers = [f"{idx * 0.1:.1f}" for idx in range(64)]
            file_path.write_text(" ".join(numbers), encoding="utf-8")

            dialog = DistributionImportDialog(
                file_path=str(file_path),
                target_kind=ImportTargetKind.PHANTOM,
                default_voxel_size=(2.0, 2.0, 2.0),
            )

            # Проверяем, что режим кодирования по умолчанию — текстовый
            self.assertEqual(dialog.combo_encoding.currentData(), "text")

            # Устанавливаем точную форму 4x4x4
            dialog.spin_dim_x.setValue(4)
            dialog.spin_dim_y.setValue(4)
            dialog.spin_dim_z.setValue(4)

            self.assertIn("Точное совпадение: в файле 64 чисел (4×4×4)", dialog.lbl_lbyl_status.text())
            self.assertTrue(dialog.button_box.button(QDialogButtonBox.Ok).isEnabled())

            # Меняем форму на 5x4x4 (требуется 80 чисел)
            dialog.spin_dim_x.setValue(5)
            self.assertIn("Несовпадение: в файле 64 чисел, ожидается 80 (разница: -16)", dialog.lbl_lbyl_status.text())
            self.assertFalse(dialog.button_box.button(QDialogButtonBox.Ok).isEnabled())

            dialog.close()


if __name__ == '__main__':
    unittest.main()
