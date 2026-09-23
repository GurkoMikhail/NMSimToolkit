"""Unit-тесты для проверки исправлений из GUI_REVIEW.md.

Покрывает:
- BUG-1: Отключение сигналов узлов в MainWindow
- BUG-2: Защита от бесконечного цикла и NaN/Inf в углах Эйлера в PropertyInspector
- BUG-3: VolumeViewModel.size.setter не ломает геометрию не-Box узлов
- BUG-4: Потокобезопасность флагов остановки в IPCReceiver и SimulationRunner
- BUG-5: TrackRenderer кольцевой буфер и флаг render_as_lines
- BUG-6: IPCReceiver.stop() таймаут и неблокирующее завершение
- BUG-7: DicomColormaps.to_vtk_piecewise_function порог и скалярный диапазон
- CLEAN-1 / CLEAN-2: Унификация формулы орбиты GammaCameraViewModel.compute_orbit_matrix
- CLEAN-3: Поддержка PetScannerViewModel и фабрики create_node_viewmodel
- CLEAN-5: Очистка сигналов в SceneTreeWidget при rebuild_tree
- CLEAN-6 / CLEAN-7: Логирование ошибок в decorators.py
- CLEAN-8: Отложенный запуск DataManager в SimulationSession
- CLEAN-9: SimulationRunner.run() вызывает публичный manager.run()
- STYLE-5: Точечное обновление PropertyInspector._on_property_changed_externally
- STYLE-8: Детерминированная генерация имен в SceneTreeWidget
- STYLE-10: Конфигурация темы pyqtgraph
"""

import sys
import time
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import hepunits as units

from PySide6.QtWidgets import QApplication

# Инициализируем QApplication для тестов виджетов, если еще не создан
APP = QApplication.instance() or QApplication(sys.argv)

from core.scene.nodes import SpatialNode, CompositeNode
from core.geometry.volumes import Volume
from core.geometry.geometries import Geometry, Box
from core.materials.materials import Material
from gui.viewmodels.node_viewmodel import (
    NodeViewModel,
    VolumeViewModel,
    VoxelVolumeViewModel,
    GammaCameraViewModel,
    PetScannerViewModel,
    create_node_viewmodel,
)
from gui.viewmodels.scene_viewmodel import SceneViewModel
from gui.viewmodels.decorators import core_field, gui_field
from gui.viewport_3d.dicom_colormaps import to_vtk_piecewise_function
from gui.viewport_3d.track_renderer import TrackRenderer
from gui.viewport_3d.spect_manipulator import SPECTManipulator
from gui.controllers.ipc_receiver import IPCReceiver
from gui.controllers.simulation_runner import SimulationRunner
from gui.controllers.simulation_session import SimulationSession
from gui.views.property_inspector import PropertyInspector
from gui.views.scene_tree_widget import SceneTreeWidget
from gui.views.results_viewer import configure_pyqtgraph_theme
from gui.views.main_window import MainWindow


class TestGuiReviewFixes(unittest.TestCase):
    """Набор тестов для валидации замечаний GUI_REVIEW.md."""

    # -------------------------------------------------------------------------
    # BUG-3: VolumeViewModel.size полиморфизм
    # -------------------------------------------------------------------------
    def test_bug_3_volume_size_setter_preserves_non_box_geometry(self):
        """Проверка, что изменение size не подменяет произвольную геометрию на Box."""
        class CustomGeometry(Geometry):
            def write_shape_data(self, shape_data_array, index):
                pass

        geom = CustomGeometry([10.0, 10.0, 20.0])
        vol = Volume(geometry=geom, material=Material(name="Water"), name="CustomVol")
        vm = VolumeViewModel(core_node=vol)

        self.assertIsInstance(vm.core_node.geometry, CustomGeometry)
        # Устанавливаем новый размер (x, y, z)
        vm.size = [15.0, 15.0, 30.0]

        # Геометрия не должна стать Box!
        self.assertIsInstance(vm.core_node.geometry, CustomGeometry)
        np.testing.assert_array_equal(vm.core_node.geometry.size, [15.0, 15.0, 30.0])

    # -------------------------------------------------------------------------
    # BUG-4 & BUG-6: Потокобезопасность и таймаут IPCReceiver / SimulationRunner
    # -------------------------------------------------------------------------
    def test_bug_4_and_6_ipc_receiver_thread_safety_and_stop_timeout(self):
        """Проверка потокобезопасности _stop_requested и неблокирующего stop()."""
        receiver = IPCReceiver()
        self.assertFalse(receiver._stop_requested)
        self.assertFalse(receiver._stop_event.is_set())

        # Проверяем остановку ненапущенного потока (не должна висеть 1000мс)
        start = time.time()
        receiver.stop(timeout_ms=100)
        elapsed = time.time() - start

        self.assertTrue(receiver._stop_requested)
        self.assertTrue(receiver._stop_event.is_set())
        # Не должно зависать надолго
        self.assertLess(elapsed, 0.5)

    def test_bug_4_simulation_runner_thread_safe_running_event(self):
        """Проверка потокобезопасности _running_event в SimulationRunner."""
        runner = SimulationRunner()
        self.assertFalse(runner.is_running)

        runner._running_event.set()
        self.assertTrue(runner.is_running)

        runner._running_event.clear()
        self.assertFalse(runner.is_running)

        # Проверка свойства обратной совместимости _is_running
        runner._is_running = True
        self.assertTrue(runner.is_running)
        runner._is_running = False
        self.assertFalse(runner.is_running)

    # -------------------------------------------------------------------------
    # BUG-5: TrackRenderer и render_as_lines
    # -------------------------------------------------------------------------
    def test_bug_5_track_renderer_line_buffer_mode(self):
        """Проверка кольцевого буфера и режима render_as_lines."""
        mock_vp = MagicMock()
        renderer = TrackRenderer(viewport=mock_vp, max_points=50, render_as_lines=False)
        self.assertFalse(renderer.render_as_lines)

        batch = {
            'pos_x': np.array([0.0, 10.0]),
            'pos_y': np.array([0.0, 10.0]),
            'pos_z': np.array([0.0, 10.0]),
            'process_id': np.array([1, 1]),
            'particle_id': np.array([1, 1]),
        }
        renderer.add_tracks_batch(batch)
        self.assertEqual(len(renderer._point_buffer), 2)
        self.assertEqual(len(renderer._lines_buffer), 0)

        # Режим с линиями
        renderer_lines = TrackRenderer(viewport=mock_vp, max_points=50, render_as_lines=True)
        self.assertTrue(renderer_lines.render_as_lines)
        renderer_lines.add_tracks_batch(batch)
        self.assertEqual(len(renderer_lines._point_buffer), 2)
        self.assertEqual(len(renderer_lines._lines_buffer), 1)

    # -------------------------------------------------------------------------
    # BUG-7: DicomColormaps с порогом и скалярным диапазоном
    # -------------------------------------------------------------------------
    def test_bug_7_dicom_colormaps_piecewise_function_with_threshold(self):
        """Проверка вычисления непрозрачности с учетом скалярного диапазона и порога."""
        pwf = to_vtk_piecewise_function(
            scalar_range=(-1000.0, 2000.0),
            threshold=0.2,
        )
        self.assertIsNotNone(pwf)
        val = pwf.GetValue(-1000.0)
        self.assertEqual(val, 0.0)

    # -------------------------------------------------------------------------
    # CLEAN-1 & CLEAN-2: Орбита GammaCameraViewModel и SPECTManipulator
    # -------------------------------------------------------------------------
    def test_clean_1_and_2_orbit_matrix_unification(self):
        """Проверка дедупликации формулы матрицы орбиты."""
        mat_vm = GammaCameraViewModel.compute_orbit_matrix(radius=250.0, angle_deg=90.0, z=10.0)
        self.assertEqual(mat_vm.shape, (4, 4))
        self.assertAlmostEqual(mat_vm[0, 3], 0.0, places=4)
        self.assertAlmostEqual(mat_vm[1, 3], 250.0, places=4)
        self.assertAlmostEqual(mat_vm[2, 3], 10.0, places=4)

        manipulator = SPECTManipulator(viewport=None, initial_radius=250.0, initial_angle=90.0, initial_z=10.0)
        mat_manip = manipulator.get_orientation_matrix()
        np.testing.assert_allclose(mat_vm, mat_manip, rtol=1e-5, atol=1e-5)

    # -------------------------------------------------------------------------
    # CLEAN-3: PetScannerViewModel
    # -------------------------------------------------------------------------
    def test_clean_3_pet_scanner_viewmodel_factory(self):
        """Проверка создания PetScannerViewModel фабрикой create_node_viewmodel."""
        class PetScanner(CompositeNode):
            pass

        node = PetScanner(name="PET_Ring_1")
        vm = create_node_viewmodel(node)
        self.assertIsInstance(vm, PetScannerViewModel)
        self.assertEqual(vm.name, "PET_Ring_1")

        # Проверка распознавания по классу PETScanner
        class PETScanner(CompositeNode):
            pass

        node2 = PETScanner(name="PET_Sector")
        vm2 = create_node_viewmodel(node2)
        self.assertIsInstance(vm2, PetScannerViewModel)
        self.assertEqual(vm2.node_type, "PETScanner")

    # -------------------------------------------------------------------------
    # CLEAN-5: Отключение сигналов в SceneTreeWidget
    # -------------------------------------------------------------------------
    def test_clean_5_scene_tree_widget_disconnects_signals_on_rebuild(self):
        """Проверка отсоединения сигналов viewmodel при перестроении дерева сцены."""
        tree_widget = SceneTreeWidget()
        root_node = Volume(geometry=Box(500, 500, 500), material=Material(name="Air"), name="World")
        scene_vm = SceneViewModel(root_core_node=root_node)

        box_node = Volume(geometry=Box(10, 10, 10), material=Material(name="Water"), name="TestBox")
        vm = VolumeViewModel(core_node=box_node)
        scene_vm.add_node(scene_vm.root_vm, vm)

        tree_widget.set_scene_viewmodel(scene_vm)
        self.assertIn(id(vm), tree_widget._connected_vms)

        # Перестраиваем дерево
        tree_widget.rebuild_tree()
        self.assertIn(id(vm), tree_widget._connected_vms)

        # Очищаем дочерние узлы
        scene_vm.remove_node(vm)
        tree_widget.rebuild_tree()
        self.assertNotIn(id(vm), tree_widget._connected_vms)

    # -------------------------------------------------------------------------
    # CLEAN-6 & CLEAN-7: Логирование ошибок в decorators.py
    # -------------------------------------------------------------------------
    def test_clean_6_and_7_decorators_log_exceptions(self):
        """Проверка логирования исключений в gui_field."""
        def bad_on_change(instance, val):
            raise ValueError("Ошибка внутри колбэка")

        class DummyVM:
            val = gui_field(default=0, on_change=bad_on_change)

        vm = DummyVM()
        with patch("gui.viewmodels.decorators._logger.error") as mock_log:
            vm.val = 100
            mock_log.assert_called_once()
            self.assertIn("Ошибка в колбэке on_change для поля val", mock_log.call_args[0][0])

    # -------------------------------------------------------------------------
    # CLEAN-8: DataManager отложенный запуск
    # -------------------------------------------------------------------------
    def test_clean_8_simulation_session_data_manager_deferred_start(self):
        """Проверка, что _setup_pipeline не стартует data_manager до вызова start()."""
        with patch("gui.controllers.simulation_session.GuiStreamDataHandler"), \
             patch("gui.controllers.simulation_session.SimulationManager"), \
             patch("gui.controllers.simulation_session.DataManager") as mock_dm_cls, \
             patch("gui.controllers.simulation_session.IPCReceiver"), \
             patch("gui.controllers.simulation_session.SimulationRunner"):
            mock_dm_instance = mock_dm_cls.return_value
            mock_dm_instance.is_alive.return_value = False

            session = SimulationSession(scene_root=MagicMock())
            mock_dm_instance.start.assert_not_called()

            session.start()
            mock_dm_instance.start.assert_called_once()
            session.stop()

    # -------------------------------------------------------------------------
    # CLEAN-9: SimulationRunner.run вызывает manager.run
    # -------------------------------------------------------------------------
    def test_clean_9_simulation_runner_calls_public_manager_run(self):
        """Проверка вызова публичного manager.run() вместо _run()."""
        mock_manager = MagicMock()
        runner = SimulationRunner(manager=mock_manager)

        runner.run()
        mock_manager.run.assert_called_once()

    # -------------------------------------------------------------------------
    # BUG-2 & STYLE-5: PropertyInspector защита от Gimbal Lock и точечные апдейты
    # -------------------------------------------------------------------------
    def test_bug_2_and_style_5_property_inspector(self):
        """Проверка устойчивости к бесконечным углам и точечного обновления свойств."""
        inspector = PropertyInspector()
        node = Volume(geometry=Box(10, 10, 10), material=Material(name="Water"), name="TestVol")
        vm = VolumeViewModel(core_node=node)
        inspector.set_target_viewmodel(vm)

        # Проверка установки NaN в матрицу
        vm.local_matrix = np.full((4, 4), np.nan)
        inspector._update_transform_fields()
        # Не должно бросать исключение

        # Проверка точечного обновления
        with patch.object(inspector, "update_all_fields") as mock_all:
            inspector._on_property_changed_externally("name", "NewName")
            mock_all.assert_not_called()
            self.assertEqual(inspector.txt_name.text(), "NewName")

    # -------------------------------------------------------------------------
    # STYLE-8: Детерминированная генерация имен в SceneTreeWidget
    # -------------------------------------------------------------------------
    def test_style_8_deterministic_name_generation(self):
        """Проверка генерации имен Volume_1, Volume_2..."""
        tree = SceneTreeWidget()
        root_node = Volume(geometry=Box(500, 500, 500), material=Material(name="Air"), name="World")
        scene_vm = SceneViewModel(root_core_node=root_node)
        tree.set_scene_viewmodel(scene_vm)

        tree._on_add_box_clicked()
        names1 = [n.name for n in scene_vm.all_nodes()]
        self.assertIn("Volume_1", names1)

        tree._on_add_box_clicked()
        names2 = [n.name for n in scene_vm.all_nodes()]
        self.assertIn("Volume_2", names2)

    # -------------------------------------------------------------------------
    # BUG-1: MainWindow _sync_viewport_scene очищает _node_connections для удаленных узлов
    # -------------------------------------------------------------------------
    def test_bug_1_sync_viewport_disconnects_removed_nodes(self):
        """Проверка отключения подписок на узлы, удаленные из сцены перед _sync_viewport_scene."""
        win = MainWindow()
        root = Volume(geometry=Box(500, 500, 500), material=Material(name="Air"), name="World")
        scene_vm = SceneViewModel(root_core_node=root)
        box1 = Volume(geometry=Box(10, 10, 10), material=Material(name="Water"), name="Box1")
        box2 = Volume(geometry=Box(10, 10, 10), material=Material(name="Water"), name="Box2")
        vm1 = VolumeViewModel(core_node=box1)
        vm2 = VolumeViewModel(core_node=box2)
        scene_vm.add_node(scene_vm.root_vm, vm1)
        scene_vm.add_node(scene_vm.root_vm, vm2)

        win.scene_vm = scene_vm
        win._sync_viewport_scene()

        self.assertIn(id(vm1), win._node_connections)
        self.assertIn(id(vm2), win._node_connections)

        # Удаляем узел vm2 и синхронизируем сцену
        scene_vm.remove_node(vm2)
        win._sync_viewport_scene()

        self.assertIn(id(vm1), win._node_connections)
        self.assertNotIn(id(vm2), win._node_connections)
        win.close()

    # -------------------------------------------------------------------------
    # STYLE-1 & STYLE-7: MainWindow считывает simulation_manager из конфигурации
    # -------------------------------------------------------------------------
    def test_style_1_and_7_main_window_reads_simulation_manager_config(self):
        """Проверка извлечения particles_number и stop_time из simulation_manager конфигурации."""
        win = MainWindow()
        root = Volume(geometry=Box(500, 500, 500), material=Material(name="Air"), name="World")
        win.scene_vm = SceneViewModel(root_core_node=root)

        mock_config = MagicMock()
        mock_sim_manager = MagicMock()
        mock_sim_manager.particles_number = 7777
        mock_sim_manager.stop_time = 42.0 * units.s
        mock_sim_manager.min_energy = 0.05 * units.keV
        mock_config.simulation_manager = mock_sim_manager
        mock_config.simulation = None
        win.current_config = mock_config

        with patch.object(win.orchestrator_session, "start") as mock_start:
            win._on_start_simulation()
            mock_start.assert_called_once()
            self.assertEqual(win.orchestrator_session.particles_number, 7777)
            self.assertEqual(win.orchestrator_session.stop_time, 42.0)
            self.assertEqual(win.orchestrator_session.min_energy, 0.05)

        win.close()

    # -------------------------------------------------------------------------
    # BUG-7: Проверка PropertyInspector и порога прозрачности вокселей
    # -------------------------------------------------------------------------
    def test_bug_7_property_inspector_opacity_threshold_and_lod(self):
        """Проверка наличия контрола порога прозрачности и синхронизации с VoxelVolumeViewModel."""
        inspector = PropertyInspector()
        mock_core = MagicMock()
        mock_core.name = "TestPhantom"
        mock_core.material_distribution.shape = (32, 32, 32)
        mock_core.voxel_size = (1.0, 1.0, 1.0)
        vm = VoxelVolumeViewModel(core_node=mock_core)

        inspector.set_target_viewmodel(vm)
        self.assertTrue(hasattr(inspector, "spin_opacity_thresh"))
        self.assertAlmostEqual(inspector.spin_opacity_thresh.value(), 0.05, places=2)

        # Проверка точечного обновления при изменении свойства
        vm.opacity_threshold = 0.35
        self.assertAlmostEqual(inspector.spin_opacity_thresh.value(), 0.35, places=2)

        # Проверка обновления LOD
        vm.lod_factor = 1.4
        self.assertEqual(inspector.slider_lod.value(), 7)

    # -------------------------------------------------------------------------
    # BUG-7: MainWindow транслирует изменения вокселей в VoxelVolumeRenderer
    # -------------------------------------------------------------------------
    def test_bug_7_voxel_properties_propagated_to_renderer(self):
        """Проверка вызова set_colormap и set_opacity_threshold в VoxelVolumeRenderer при смене свойств."""
        win = MainWindow()
        mock_renderer = MagicMock()
        win.voxel_renderer = mock_renderer

        mock_core = MagicMock()
        mock_core.material_distribution.shape = (32, 32, 32)
        mock_core.voxel_size = (1.0, 1.0, 1.0)
        vm = VoxelVolumeViewModel(core_node=mock_core)

        win._on_node_property_changed(vm, "colormap_name", "Hot Iron")
        mock_renderer.set_colormap.assert_called_with("Hot Iron")

        win._on_node_property_changed(vm, "opacity_threshold", 0.45)
        mock_renderer.set_opacity_threshold.assert_called_with(0.45)
        win.close()

    # -------------------------------------------------------------------------
    # BUG-7: to_vtk_piecewise_function линейный режим с порогом и защита от диапазона 0
    # -------------------------------------------------------------------------
    def test_bug_7_to_vtk_piecewise_function_linear_and_equal_range(self):
        """Проверка линейной рампы с порогом и защиты от диапазона нулевой ширины."""
        # Нулевой диапазон не должен вызывать сбой
        pwf_zero = to_vtk_piecewise_function(scalar_range=(50.0, 50.0), threshold=0.1)
        self.assertIsNotNone(pwf_zero)

        # Линейная рампа с порогом 0.2 на диапазоне 0..100:
        # от 0 до 20: 0.0, от 20 до 100: линейно до 1.0 (в 60 -> 0.5)
        pwf_linear = to_vtk_piecewise_function(
            scalar_range=(0.0, 100.0),
            min_alpha=0.0,
            max_alpha=1.0,
            ramp_type='linear',
            threshold=0.2
        )
        self.assertAlmostEqual(pwf_linear.GetValue(0.0), 0.0, places=3)
        self.assertAlmostEqual(pwf_linear.GetValue(20.0), 0.0, places=3)
        self.assertAlmostEqual(pwf_linear.GetValue(60.0), 0.5, places=2)
        self.assertAlmostEqual(pwf_linear.GetValue(100.0), 1.0, places=3)

    # -------------------------------------------------------------------------
    # CLEAN-5: SceneTreeWidget отключает сигналы старого SceneViewModel
    # -------------------------------------------------------------------------
    def test_clean_5_scene_tree_widget_disconnects_previous_scene_vm(self):
        """Проверка отсоединения от старого SceneViewModel при установке нового."""
        tree = SceneTreeWidget()
        s1 = SceneViewModel(root_core_node=Volume(geometry=Box(10, 10, 10), material=Material(name="Air")))
        tree.set_scene_viewmodel(s1)

        s2 = SceneViewModel(root_core_node=Volume(geometry=Box(20, 20, 20), material=Material(name="Water")))
        tree.set_scene_viewmodel(s2)

        # Эмиссия сигнала из s1 не должна вызывать повторную перестройку дерева
        with patch.object(tree, "rebuild_tree") as mock_rebuild:
            s1.scene_loaded.emit(s1)
            mock_rebuild.assert_not_called()


    # -------------------------------------------------------------------------
    # STYLE-10: Конфигурация pyqtgraph
    # -------------------------------------------------------------------------
    def test_style_10_configure_pyqtgraph_theme(self):
        """Проверка вызова configure_pyqtgraph_theme без исключений."""
        try:
            configure_pyqtgraph_theme()
        except Exception as e:
            self.fail(f"configure_pyqtgraph_theme() вызвал ошибку: {e}")


if __name__ == "__main__":
    unittest.main()
