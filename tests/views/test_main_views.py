"""
Модульные тесты графических представлений (Views): дерево сцены, инспектор свойств, панель результатов и главное окно.
"""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

try:
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication(['-platform', 'offscreen'])
except ImportError:
    app = None

from core.config.models import BaseCompositeNodeConfig, LoggingConfig, SimulationConfig, SimulationManagerConfig
from core.config.builder import SceneBuilder
from core.config.yaml_loader import load_simulation_config
from core.geometry.geometries import Box
from core.geometry.volumes import Volume
from core.materials.materials import Material
from core.scene.nodes import CompositeNode
from core.source.sources import PointSource
from gui.viewmodels.nodes.volume_vm import VolumeViewModel
from gui.viewmodels.scene_viewmodel import SceneViewModel
from gui.views.main_window import MainWindow
from gui.views.property_inspector import PropertyInspector
from gui.views.results_viewer import ResultsViewer
from gui.views.scene_tree_widget import SceneTreeWidget


class TestStage4Views(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if app is None:
            raise unittest.SkipTest("PySide6 не установлен в окружении")

    def test_scene_tree_widget(self) -> None:
        """Проверка инициализации и построения дерева сцены."""
        root = CompositeNode(name="RootNode")
        child = Volume(geometry=Box(10.0, 10.0, 10.0), material=Material(name="Water"), name="Target")
        root.add_child(child)

        scene_vm = SceneViewModel(root)
        tree_widget = SceneTreeWidget(scene_vm)
        self.assertEqual(tree_widget.tree.topLevelItemCount(), 1)
        root_item = tree_widget.tree.topLevelItem(0)
        assert root_item is not None
        self.assertEqual(root_item.text(0), "RootNode")
        self.assertEqual(root_item.childCount(), 1)
        child_item = root_item.child(0)
        assert child_item is not None
        self.assertEqual(child_item.text(0), "Target")

    def test_property_inspector_binding(self) -> None:
        """Проверка инспектора свойств и двусторонней реактивности."""
        geo = Box(50.0, 60.0, 70.0)
        vol = Volume(geometry=geo, material=Material(name="Water"), name="WaterBox")
        vm = VolumeViewModel(vol)

        inspector = PropertyInspector()
        inspector.set_target_viewmodel(vm)

        self.assertEqual(inspector.general_section.txt_name.text(), "WaterBox")
        self.assertAlmostEqual(inspector.volume_section.spin_size_x.value(), 50.0)

        # Модификация через UI
        inspector.general_section.txt_name.setText("RenamedBox")
        inspector.general_section._on_name_changed()
        self.assertEqual(vm.name, "RenamedBox")
        self.assertEqual(vol.name, "RenamedBox")

    def test_results_viewer(self) -> None:
        """Проверка панели результатов: обновление 2D проекции и спектра."""
        viewer = ResultsViewer()
        dummy_proj = np.ones((64, 64), dtype=np.float32) * 5.0
        viewer.set_projection_data(dummy_proj)

        self.assertEqual(viewer.lbl_stats.text(), f"Всего отсчетов: {int(np.sum(dummy_proj)):,}")

        rng = np.random.default_rng(42)
        dummy_energies = rng.normal(loc=140.5, scale=5.0, size=1000)
        viewer.set_spectrum_data(dummy_energies)

        viewer.clear_results()
        self.assertEqual(viewer.lbl_stats.text(), "Всего отсчетов: 0")

    def test_main_window_instantiation(self) -> None:
        """Проверка корректной сборки главного окна и наличия док-панелей."""
        main_win = MainWindow()
        self.assertIsNotNone(main_win.viewport)
        self.assertIsNotNone(main_win.dock_tree)
        self.assertIsNotNone(main_win.dock_inspector)
        main_win.close()

    def test_main_window_save_yaml_logic(self) -> None:
        """Проверка логики сохранения конфигурации симуляции из MainWindow в YAML."""
        main_win = MainWindow()
        root = CompositeNode(name="World")
        main_win.scene_vm.load_scene(root)

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "test_saved_config.yaml"
            with patch(
                "PySide6.QtWidgets.QFileDialog.getSaveFileName",
                return_value=(str(save_path), "YAML files (*.yaml *.yml)"),
            ):
                with patch("PySide6.QtWidgets.QMessageBox.information"), patch(
                    "PySide6.QtWidgets.QMessageBox.critical"
                ) as mock_crit:
                    main_win._on_save_yaml()
                    self.assertFalse(mock_crit.called, f"QMessageBox.critical был вызван: {mock_crit.call_args}")

            self.assertTrue(save_path.exists())
            loaded_cfg = load_simulation_config(str(save_path))
            self.assertIsNotNone(loaded_cfg.simulation_manager)
            self.assertEqual(loaded_cfg.simulation_manager.particles_number, main_win.sim_settings.particles_number)
        main_win.close()

    def test_main_window_load_yaml_logic(self) -> None:
        """Проверка логики загрузки конфигурации симуляции из YAML в MainWindow."""
        main_win = MainWindow()
        config_path = "simulation_config.yaml"

        with patch(
            "PySide6.QtWidgets.QFileDialog.getOpenFileName",
            return_value=(config_path, "YAML files (*.yaml *.yml)"),
        ):
            with patch("PySide6.QtWidgets.QMessageBox.critical"):
                main_win._on_open_yaml()

        assert main_win.scene_vm.root_vm is not None
        self.assertEqual(main_win.scene_vm.root_vm.name, "Simulation_volume")
        main_win.close()

    def test_main_window_save_preserves_manager_fields_and_created_nodes(self) -> None:
        """
        Проверка сохранения seed, physics_list, start_time, logging и созданных с нуля
        узлов (PointSource, Source, WoodcockVoxelVolume) с последующей сборкой через SceneBuilder.
        """
        main_win = MainWindow()
        root = CompositeNode(name="World")
        main_win.scene_vm.load_scene(root)

        # Добавляем созданные с нуля узлы через SceneTreeWidget
        main_win.scene_tree._add_point_source(main_win.scene_vm.root_vm)
        main_win.scene_tree._add_voxel_source(main_win.scene_vm.root_vm)
        main_win.scene_tree._add_voxel_volume(main_win.scene_vm.root_vm)

        # Задаем начальную конфигурацию с seed, physics_list, start_time и logging
        main_win.current_config = SimulationConfig(
            simulation_manager=SimulationManagerConfig(
                start_time=5.0,
                stop_time=1e9,
                particles_number=1000,
                min_energy=0.01,
                physics_list="em_standard",
                seed=42,
            ),
            data_manager=main_win.data_manager_vm.to_config(),
            scene=BaseCompositeNodeConfig(name="World"),
            logging=LoggingConfig(output_dir="custom_logs", level="DEBUG", console=True),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "saved_full.yaml"
            with patch(
                "PySide6.QtWidgets.QFileDialog.getSaveFileName",
                return_value=(str(save_path), "YAML files (*.yaml *.yml)"),
            ):
                with patch("PySide6.QtWidgets.QMessageBox.information"), patch(
                    "PySide6.QtWidgets.QMessageBox.critical"
                ) as mock_crit:
                    main_win._on_save_yaml()
                    self.assertFalse(mock_crit.called, f"QMessageBox.critical был вызван: {mock_crit.call_args}")

            loaded_cfg = load_simulation_config(str(save_path))
            self.assertEqual(loaded_cfg.simulation_manager.seed, 42)
            self.assertEqual(loaded_cfg.simulation_manager.physics_list, "em_standard")
            self.assertAlmostEqual(float(loaded_cfg.simulation_manager.start_time), 5.0)
            self.assertIsNotNone(loaded_cfg.logging)
            assert loaded_cfg.logging is not None
            self.assertEqual(loaded_cfg.logging.output_dir, "custom_logs")
            self.assertEqual(loaded_cfg.logging.level, "DEBUG")

            # Проверяем, что SceneBuilder успешно собирает сцену без FileNotFoundError
            builder = SceneBuilder(base_dir=Path(tmp_dir))
            rebuilt_root = builder.build_scene(loaded_cfg.scene)
            self.assertEqual(len(rebuilt_root.childs), 3)
            self.assertIsInstance(rebuilt_root.childs[0], PointSource)
            self.assertEqual(float(rebuilt_root.childs[0].voxel_size), 0.0)

        main_win.close()


if __name__ == '__main__':
    unittest.main()
