import unittest
import threading
import numpy as np
import hepunits as units

from core.geometry.volumes import Volume
from core.geometry.geometries import Box
from core.materials.materials import Material
from settings.database_setting import material_database
from core.source.sources import PointSource
from core.scene.nodes import CompositeNode, SpatialNode
from core.transport.simulation_managers import SimulationManager, SimulationState
from gui.viewmodels.nodes.base_node_vm import NodeViewModel
from gui.viewmodels.nodes.volume_vm import VolumeViewModel
from gui.viewmodels.nodes.gamma_camera_vm import GammaCameraViewModel
from gui.viewmodels.nodes.factory import create_node_viewmodel
from gui.viewmodels.scene_viewmodel import SceneViewModel
from gui.views.property_inspector import PropertyInspector
from gui.views.main_window import MainWindow

try:
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication(['-platform', 'offscreen'])
except ImportError:
    app = None


class TestGuiIntegrationAndFixes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if app is None:
            raise unittest.SkipTest("PySide6 не доступен в текущем окружении")

    def test_simulation_manager_thread_instantiation(self):
        """Проверка безопасной инициализации SimulationManager во вторичном потоке (без сбоя signal)."""
        error_holder = []

        def target():
            try:
                root = CompositeNode(name="TestNode")
                mgr = SimulationManager(scene=root, particles_number=10)
                self.assertEqual(mgr.state, SimulationState.IDLE)
            except Exception as e:
                error_holder.append(e)

        t = threading.Thread(target=target)
        t.start()
        t.join(timeout=2.0)
        self.assertEqual(len(error_holder), 0, f"Ошибка в потоке: {error_holder}")

    def test_simulation_manager_termination_no_hang(self):
        """Проверка завершения цикла симуляции без бесконечного зацикливания."""
        root = Volume(
            geometry=Box(10 * units.cm, 10 * units.cm, 10 * units.cm),
            material=material_database['Air, Dry (near sea level)'],
            name='TestVol'
        )
        root.add_child(PointSource(activity=50 * units.Bq, energy=140 * units.keV))
        mgr = SimulationManager(scene=root, stop_time=0.005 * units.s, particles_number=10)

        mgr.start()
        mgr.join(timeout=3.0)
        self.assertFalse(mgr.is_alive(), "SimulationManager завис в бесконечном цикле!")
        self.assertEqual(mgr.state, SimulationState.STOPPED)

    def test_node_viewmodel_remove_child_parent_reset(self):
        """Проверка очистки parent у core_node при удалении узла из ViewModel."""
        root_core = CompositeNode(name="Root")
        child_core = SpatialNode(name="Child")
        root_core.add_child(child_core)

        scene_vm = SceneViewModel(root_core)
        child_vm = scene_vm.find_by_name("Child")
        self.assertIsNotNone(child_vm)
        self.assertIs(child_core.parent, root_core)

        scene_vm.remove_node(child_vm)
        self.assertIsNone(child_core.parent)
        self.assertNotIn(child_core, root_core.childs)

    def test_property_inspector_rotation_sync(self):
        """Проверка двусторонней синхронизации вращения Эйлера в инспекторе свойств."""
        geo = Box(50.0, 50.0, 50.0)
        vol = Volume(geometry=geo, material=Material(name="Lead"), name="BoxRot")
        vm = VolumeViewModel(vol)

        inspector = PropertyInspector()
        inspector.set_target_viewmodel(vm)

        # Проверяем начальные углы поворота
        self.assertAlmostEqual(inspector.spin_rot_x.value(), 0.0)
        self.assertAlmostEqual(inspector.spin_rot_y.value(), 0.0)
        self.assertAlmostEqual(inspector.spin_rot_z.value(), 0.0)

        # Задаем поворот вокруг Z на 45 градусов
        inspector.spin_rot_z.setValue(45.0)
        inspector._on_transform_changed()

        # Проверяем матрицу во ViewModel
        rot_z_angle = np.degrees(np.arctan2(vm.local_matrix[1, 0], vm.local_matrix[0, 0]))
        self.assertAlmostEqual(rot_z_angle, 45.0, places=3)

    def test_main_window_components_and_actions(self):
        """Проверка полной интеграции MainWindow со всеми манипуляторами и контроллерами."""
        win = MainWindow()
        self.assertIsNotNone(win.viewport_controller.track_renderer)
        self.assertIsNotNone(win.viewport_controller.spect_manipulator)
        self.assertIsNotNone(win.viewport_controller.pet_manipulator)
        self.assertIsNotNone(win.viewport_controller.voxel_renderer)
        self.assertIsNotNone(win.viewport_controller.dose_renderer)

        # Проверка отсутствия паразитной отрисовки манипуляторов в пустой/дефолтной сцене
        self.assertNotIn("spect_orbit_trajectory", win.viewport._actors)
        self.assertNotIn("spect_detector_indicator", win.viewport._actors)
        self.assertNotIn("pet_ring_geometry", win.viewport._actors)

        # Проверка начальных состояний кнопок управления
        self.assertTrue(win.act_run.isEnabled())
        self.assertFalse(win.act_pause.isEnabled())
        self.assertFalse(win.act_stop.isEnabled())

        # Синхронизация 3D-сцены и проверка устойчивости к повторным вызовам
        win.viewport_controller.sync_viewport_scene()
        win.viewport_controller.sync_viewport_scene()

        # Проверка вызова showEvent
        win.showEvent(None)

        # Проверка смены состояний кнопок
        win._update_action_states(running=True, paused=False)
        self.assertFalse(win.act_run.isEnabled())
        self.assertTrue(win.act_pause.isEnabled())
        self.assertTrue(win.act_stop.isEnabled())

        win._update_action_states(running=True, paused=True)
        self.assertTrue(win.act_resume.isEnabled())
        self.assertTrue(win.act_step.isEnabled())

        # Проверка программного вызова _on_open_yaml без диалогового окна
        win._on_open_yaml("nema_1_cam.yaml")
        self.assertIsNotNone(win.current_config)
        self.assertIn("voxel_volume", win.viewport._actors)

        # Проверка создания новой сцены
        win._on_new_scene()
        self.assertNotIn("voxel_volume", win.viewport._actors)

        # Проверка add_mesh_actor с rgb и kwargs
        import pyvista as pv
        sample_poly = pv.PolyData([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        sample_poly.point_data['RGB'] = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
        actor_kw = win.viewport.add_mesh_actor("rgb_kw_actor", sample_poly, **{'rgb': True})
        self.assertIsNotNone(actor_kw)
        self.assertIn("rgb_kw_actor", win.viewport._actors)

        # Закрытие окна
        win.close()

    def test_nema_time_step_and_source_activity(self):
        """
        Критерий приемки 1: Проверка расчета шага по времени для nema_1_cam.yaml.
        dt должно составлять ~10 мкс (а не 10 фс), что подтверждает корректность единиц активности (* units.Bq).
        """
        from core.config.yaml_loader import load_simulation_config
        from core.config.builder import SceneBuilder

        cfg = load_simulation_config("nema_1_cam.yaml")
        root = SceneBuilder().build_scene(cfg.scene)
        mgr = SimulationManager(scene=root, particles_number=1000)

        dt = mgr._calculate_time_step(1000)
        # 1000 частиц / (100 MBq * 10^-9 ns^-1) = 10 000 нс = 10 мкс
        self.assertAlmostEqual(dt / units.microsecond, 10.0, places=1)

    def test_orchestrator_session_telemetry_setup(self):
        """
        Проверка сборки конфигурации и генерации задач в OrchestratorSession.
        """
        from core.config.yaml_loader import load_simulation_config
        from core.config.builder import SceneBuilder
        from gui.controllers.orchestrator_session import OrchestratorSession
        from gui.viewmodels.scene_viewmodel import SceneViewModel
        from gui.viewmodels.procedure_viewmodel import SpectProcedureViewModel

        cfg = load_simulation_config("nema_1_cam.yaml")
        root = SceneBuilder().build_scene(cfg.scene)
        scene_vm = SceneViewModel(root)

        proc_vm = SpectProcedureViewModel()
        proc_vm.views = 4
        session = OrchestratorSession(
            scene_vm=scene_vm,
            procedure_vm=proc_vm,
            particles_number=2000,
            stop_time=0.1,
            projection_shape=(128, 128)
        )

        try:
            jobs = session.generate_jobs()
            self.assertGreater(len(jobs), 0)
            self.assertEqual(session.projection_shape, (128, 128))
            self.assertFalse(session.is_running)
        finally:
            session.close()

    def test_detector_projection_accumulation_direct(self):
        """
        Верификация прямого накопления 2D-проекции в GuiStreamDataHandler.
        """
        import time
        from core.data.stream_handlers import GuiStreamDataHandler

        shm_name = f"test_accum_shm_{int(time.time()*1000)}"
        detector = Volume(
            geometry=Box(40 * units.cm, 40 * units.cm, 1 * units.cm),
            material=material_database['Sodium Iodide'],
            name='Detector'
        )

        handler = GuiStreamDataHandler(
            shm_name=shm_name,
            projection_shape=(64, 64),
            create_shm=True,
            sensitive_volume_ids={0},
            detector_volume=detector,
            detector_size=(400.0, 400.0),
        )

        try:
            chunk = {
                'type': 'interactions',
                'data': {
                    'particle_ID': np.array([1, 2], dtype=np.uint64),
                    'volume_id': np.array([0, 0], dtype=np.uint32),
                    'pos_x': np.array([0.0, 10.0], dtype=np.float32),
                    'pos_y': np.array([0.0, 10.0], dtype=np.float32),
                    'pos_z': np.array([0.0, 0.0], dtype=np.float32),
                    'process_id': np.array([1, 1], dtype=np.uint16),
                    'energy_deposit': np.array([140.0, 140.0], dtype=np.float32),
                }
            }
            handler.process_chunk(chunk)
            snapshot = handler.get_projection_snapshot()
            self.assertIsNotNone(snapshot)
            self.assertEqual(snapshot.shape, (64, 64))
            self.assertGreater(float(np.sum(snapshot)), 0.0)
        finally:
            handler.close()


if __name__ == '__main__':
    unittest.main()
