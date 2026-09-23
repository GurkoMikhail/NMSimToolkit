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
from gui.viewmodels.node_viewmodel import (
    NodeViewModel,
    VolumeViewModel,
    GammaCameraViewModel,
    create_node_viewmodel
)
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
        self.assertIsNotNone(win.track_renderer)
        self.assertIsNotNone(win.spect_manipulator)
        self.assertIsNotNone(win.pet_manipulator)
        self.assertIsNotNone(win.voxel_renderer)

        # Проверка отсутствия паразитной отрисовки манипуляторов в пустой/дефолтной сцене
        self.assertNotIn("spect_orbit_trajectory", win.viewport._actors)
        self.assertNotIn("spect_detector_indicator", win.viewport._actors)
        self.assertNotIn("pet_ring_geometry", win.viewport._actors)

        # Проверка начальных состояний кнопок управления
        self.assertTrue(win.act_run.isEnabled())
        self.assertFalse(win.act_pause.isEnabled())
        self.assertFalse(win.act_stop.isEnabled())

        # Синхронизация 3D-сцены и проверка устойчивости к повторным вызовам
        win._sync_viewport_scene()
        win._sync_viewport_scene()

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

    def test_simulation_session_telemetry_stream(self):
        """
        Критерии приемки 2, 3, 4: Интеграционный тест конвейера телеметрии.
        Проверяет автодетектирование sensitive_volume_ids детектора,
        поступление 3D-треков, масштабирование энергетического спектра в кэВ
        и накопление отсчетов в 2D-проекции SharedMemory.
        """
        from core.config.yaml_loader import load_simulation_config
        from core.config.builder import SceneBuilder
        from gui.controllers.simulation_session import SimulationSession
        import time

        cfg = load_simulation_config("nema_1_cam.yaml")
        root = SceneBuilder().build_scene(cfg.scene)

        session = SimulationSession(
            scene_root=root,
            particles_number=2000,
            stop_time=0.1 * units.s,
            shm_name=f"test_nema_shm_{int(time.time()*1000)}",
            projection_shape=(128, 128)
        )

        try:
            # 1. Проверка автодетекции чувствительного объема
            self.assertIsNotNone(session.stream_handler.sensitive_volume_ids)
            self.assertGreater(len(session.stream_handler.sensitive_volume_ids), 0)
            self.assertIsNotNone(session.stream_handler.detector_volume)
            self.assertEqual(session.stream_handler.detector_volume.name, "Detector")

            tracks_batches = []
            spectra_batches = []
            proj_snapshots = []

            session.tracks_received.connect(lambda t: tracks_batches.append(t))
            session.spectrum_received.connect(lambda s: spectra_batches.append(s))
            session.projection_received.connect(lambda p: proj_snapshots.append(p))

            # 2. Запуск симуляции
            session.start()
            for _ in range(60):
                if app is not None:
                    app.processEvents()
                if len(tracks_batches) > 0 and len(spectra_batches) > 0:
                    break
                time.sleep(0.1)

            session.stop()

            if app is not None:
                app.processEvents()

            # 3. Проверка треков
            self.assertGreater(len(tracks_batches), 0, "Пакеты 3D-треков не поступили в GUI")
            first_track = tracks_batches[0]
            self.assertEqual(first_track['type'], 'tracks')
            self.assertIn('pos_x', first_track)
            self.assertGreater(len(first_track['pos_x']), 0)

            # 4. Проверка спектра в кэВ (энергия пика для Tc-99m / nema ~ 159 кэВ)
            self.assertGreater(len(spectra_batches), 0, "Спектральные данные не поступили в GUI")
            max_energy_kev = float(np.max(spectra_batches[0]))
            self.assertGreater(max_energy_kev, 50.0, f"Энергия не масштабирована в кэВ: max={max_energy_kev}")
            self.assertLessEqual(max_energy_kev, 160.0, f"Энергия превышает максимум источника: max={max_energy_kev}")

            self.assertEqual(session.stream_handler.detector_size, (540.0, 400.0))

            # 5. Проверка 2D проекции
            snapshot = session.stream_handler.get_projection_snapshot()
            self.assertIsNotNone(snapshot)
            self.assertEqual(snapshot.shape, (128, 128))

        finally:
            session.close()

    def test_detector_physical_projection_accumulation(self):
        """
        Критерий приемки 4: Строгая верификация физического накопления отсчетов
        в 2D-проекции и спектре детектора (проверка реального > 0 счета, а не >= 0).
        """
        import time
        from gui.controllers.simulation_session import SimulationSession
        world = Volume(
            geometry=Box(100 * units.cm, 100 * units.cm, 100 * units.cm),
            material=material_database['Air, Dry (near sea level)'],
            name='World'
        )
        detector = Volume(
            geometry=Box(40 * units.cm, 40 * units.cm, 1 * units.cm),
            material=material_database['Sodium Iodide'],
            name='Detector'
        )
        detector.translate(z=15 * units.cm)
        world.add_child(detector)

        source = PointSource(activity=100 * units.MBq, energy=159 * units.keV)
        world.add_child(source)

        session = SimulationSession(
            scene_root=world,
            particles_number=1000,
            stop_time=0.01 * units.s,
            shm_name=f"test_accum_shm_{int(time.time()*1000)}",
            projection_shape=(64, 64)
        )

        try:
            self.assertEqual(session.stream_handler.sensitive_volume_ids, {1})
            self.assertEqual(session.stream_handler.detector_size, (400.0, 400.0))

            spectra = []
            projections = []
            session.spectrum_received.connect(lambda s: spectra.append(s))
            session.projection_received.connect(lambda p: projections.append(p))

            session.start()
            t0 = time.time()
            while time.time() - t0 < 6.0:
                if app is not None:
                    app.processEvents()
                snap = session.stream_handler.get_projection_snapshot()
                if snap is not None and np.sum(snap) > 0.0 and len(spectra) > 0:
                    break
                time.sleep(0.1)

            session.stop()

            if app is not None:
                app.processEvents()

            snapshot = session.stream_handler.get_projection_snapshot()
            self.assertIsNotNone(snapshot)
            total_counts = float(np.sum(snapshot))
            self.assertGreater(total_counts, 0.0, "2D-проекция детектора должна реально накапливать отсчеты (> 0)")

            self.assertGreater(len(spectra), 0, "Спектр должен содержать зарегистрированные события")
            final_spectrum = spectra[-1]
            max_kev = float(np.max(final_spectrum))
            self.assertGreater(max_kev, 100.0, f"Пик поглощения должен быть в районе 159 кэВ: max={max_kev}")
            self.assertLessEqual(max_kev, 160.0)

        finally:
            session.close()


if __name__ == '__main__':
    unittest.main()
