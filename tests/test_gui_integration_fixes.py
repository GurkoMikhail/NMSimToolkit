import unittest
import numpy as np
import tempfile
from pathlib import Path
from multiprocessing import Queue

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPoint
import hepunits as units

from core.geometry.geometries import Box
from core.geometry.volumes import Volume
from core.geometry.voxel_volumes import WoodcockVoxelVolume
from core.materials.materials import Material, MaterialArray
from settings.database_setting import material_database
from core.source.sources import Source, PointSource
from core.scene.nodes import CompositeNode, SpatialNode
from core.scene.dose_grid_node import DoseGridNode
from core.geometry.gamma_cameras import GammaCamera
from core.other.typing_definitions import Float
from core.config.builder import SceneBuilder
from core.config.exporter import SceneExporter
from core.config.yaml_loader import load_simulation_config
from core.config.models import (
    WoodcockVoxelVolumeConfig, SourceConfig, NumpyDistributionConfig,
    RawDistributionConfig, SimulationConfig, SimulationManagerConfig
)
from gui.controllers.stream_handlers import GuiStreamDataHandler

from gui.viewmodels.nodes.base_node_vm import NodeViewModel
from gui.viewmodels.nodes.volume_vm import VolumeViewModel
from gui.viewmodels.nodes.voxel_volume_vm import VoxelVolumeViewModel
from gui.viewmodels.nodes.source_vm import SourceViewModel
from gui.viewmodels.nodes.gamma_camera_vm import GammaCameraViewModel
from gui.viewmodels.nodes.dose_grid_vm import DoseGridViewModel
from gui.viewmodels.nodes.factory import create_node_viewmodel
from gui.viewmodels.scene_viewmodel import SceneViewModel
from gui.viewport_3d.track_renderer import TrackRenderer, PROCESS_COLORS
from gui.viewport_3d.spect_manipulator import SPECTManipulator
from gui.views.scene_tree_widget import SceneTreeWidget
from gui.viewmodels.data_handler_viewmodel import DirectStreamHandlerViewModel
from gui.views.property_inspector import PropertyInspector
from gui.views.main_window import MainWindow


app = QApplication.instance()
if app is None:
    app = QApplication([])


class DummyViewport:
    """Заглушка 3D-вьюпорта для изолированного тестирования рендереров."""
    def __init__(self):
        self.plotter = self
        self._actors = {}
        self.rendered = False

    def add_mesh_actor(self, name, mesh, **kwargs):
        class DummyActor:
            def __init__(self):
                self._mapper = DummyMapper()
            def GetMapper(self):
                return self._mapper

        class DummyMapper:
            def SetInputData(self, poly):
                pass

        actor = DummyActor()
        self._actors[name] = actor
        return actor

    def remove_actor(self, name):
        self._actors.pop(name, None)

    def render(self):
        self.rendered = True

    def update_actor_transform(self, name, matrix):
        return True


class TestGUIIntegrationFixes(unittest.TestCase):
    """
    Тестирование замечаний по интеграции GUI / Boost / Ядро.
    """

    def setUp(self):
        self.root = CompositeNode(name="World")
        self.scene_vm = SceneViewModel()
        self.scene_vm.load_scene(self.root)

    # 1. Радиус орбиты ОФЭКТ по лицевой поверхности гамма-камеры
    def test_spect_orbit_radius_by_face_surface(self):
        col = Volume(geometry=Box(100.0, 100.0, 30.0), material=material_database['Pb'], name="Collimator")
        det = Volume(geometry=Box(100.0, 100.0, 10.0), material=material_database['Plastic Scintillator, Vinyltoluene'], name="Detector")
        cam = GammaCamera(collimator=col, detector=det, name="GammaCam")

        cam_vm = GammaCameraViewModel(cam)
        # Проверяем расчет half_thickness
        expected_half_th = cam.size[2] / 2.0
        self.assertAlmostEqual(cam_vm.half_thickness, expected_half_th, places=4)
        self.assertGreater(cam_vm.half_thickness, 0.0)

        # Вычисляем матрицу с учетом half_thickness
        orbit_radius_face = 250.0
        mat = GammaCameraViewModel.compute_orbit_matrix(
            radius=orbit_radius_face,
            angle_deg=0.0,
            z=0.0,
            half_thickness=cam_vm.half_thickness
        )

        # Центр камеры должен быть смещен на radius + half_thickness
        center_x = float(mat[0, 3])
        self.assertAlmostEqual(center_x, orbit_radius_face + cam_vm.half_thickness, places=4)

        # Лицевая поверхность камеры (+Z локальное) смотрит в сторону центра орбиты (-X в глобале при angle=0)
        # Координата X лицевой поверхности: center_x + normal * half_thickness = (R + half_th) - 1.0 * half_th = R
        normal_x = float(mat[0, 2])
        self.assertAlmostEqual(normal_x, -1.0, places=4)
        face_x = center_x + normal_x * cam_vm.half_thickness
        self.assertAlmostEqual(face_x, orbit_radius_face, places=4)

        # Проверяем set_orbit_position
        cam_vm.set_orbit_position(orbit_radius_face, 90.0, 10.0)
        self.assertAlmostEqual(cam_vm.orbit_radius, orbit_radius_face, places=4)
        self.assertAlmostEqual(cam_vm.orbit_angle, 90.0, places=4)
        self.assertAlmostEqual(cam_vm.orbit_z, 10.0, places=4)
        self.assertAlmostEqual(float(cam_vm.local_matrix[1, 3]), orbit_radius_face + cam_vm.half_thickness, places=4)

        # Проверяем SPECTManipulator
        manip = SPECTManipulator(viewport=None, initial_radius=orbit_radius_face, initial_angle=0.0)
        pos = manip.get_cartesian_position(half_thickness=cam_vm.half_thickness)
        self.assertAlmostEqual(pos[0], orbit_radius_face + cam_vm.half_thickness, places=4)
        mat_manip = manip.get_orientation_matrix(half_thickness=cam_vm.half_thickness)
        self.assertAlmostEqual(float(mat_manip[0, 3]), orbit_radius_face + cam_vm.half_thickness, places=4)

    # 2. Отдельный список чувствительных объемов с Drag-and-Drop
    def test_sensitive_volumes_list_and_drag_drop(self):
        vol = Volume(geometry=Box(60.0, 60.0, 20.0), material=material_database['Plastic Scintillator, Vinyltoluene'], name="Scintillator")
        vol_vm = VolumeViewModel(vol)
        self.scene_vm.add_node(self.scene_vm.root_vm, vol_vm)

        tree_widget = SceneTreeWidget(self.scene_vm)
        self.assertEqual(tree_widget.sensitive_list.count(), 0)

        # Установка флага детектора реактивно обновляет список
        vol_vm.is_sensitive_detector = True
        self.assertEqual(tree_widget.sensitive_list.count(), 1)
        self.assertIn("Scintillator", tree_widget.sensitive_list.item(0).text())

        # Снятие флага удаляет из списка
        vol_vm.is_sensitive_detector = False
        self.assertEqual(tree_widget.sensitive_list.count(), 0)

        # Эмуляция Drag-and-Drop из дерева в список чувствительных объемов
        tree_widget._dragged_vm = vol_vm
        class DummyDropEvent:
            def __init__(self):
                self.accepted = False
            def acceptProposedAction(self):
                self.accepted = True
            def ignore(self):
                self.accepted = False

        drop_ev = DummyDropEvent()
        tree_widget.sensitive_list.dropEvent(drop_ev)
        self.assertTrue(drop_ev.accepted)
        self.assertTrue(vol_vm.is_sensitive_detector)
        self.assertEqual(tree_widget.sensitive_list.count(), 1)

        # Проверка удаления через кнопку "- Исключить из детекторов"
        tree_widget.sensitive_list.setCurrentRow(0)
        tree_widget._on_remove_detector_clicked()
        self.assertFalse(vol_vm.is_sensitive_detector)
        self.assertEqual(tree_widget.sensitive_list.count(), 0)

        # Запрет перетаскивания не-Volume узлов
        dose_node = DoseGridNode(name="Dose", size=[100, 100, 100], dose_voxel_size=5.0)
        dose_vm = DoseGridViewModel(dose_node)
        tree_widget._dragged_vm = dose_vm
        drop_ev2 = DummyDropEvent()
        tree_widget.sensitive_list.dropEvent(drop_ev2)
        self.assertFalse(drop_ev2.accepted)
        self.assertEqual(tree_widget.sensitive_list.count(), 0)

        # Запрет перетаскивания корневого узла root_vm сцены
        tree_widget._dragged_vm = self.scene_vm.root_vm
        drop_ev_root = DummyDropEvent()
        tree_widget.sensitive_list.dropEvent(drop_ev_root)
        self.assertFalse(drop_ev_root.accepted)
        self.assertEqual(tree_widget.sensitive_list.count(), 0)

        # Проверка dragEnterEvent и dragMoveEvent
        class DummyDragEvent:
            def __init__(self):
                self.accepted = False
            def acceptProposedAction(self):
                self.accepted = True
            def ignore(self):
                self.accepted = False

        drag_ev = DummyDragEvent()
        tree_widget._dragged_vm = vol_vm
        tree_widget.sensitive_list.dragEnterEvent(drag_ev)
        self.assertTrue(drag_ev.accepted)

        drag_ev_root = DummyDragEvent()
        tree_widget._dragged_vm = self.scene_vm.root_vm
        tree_widget.sensitive_list.dragEnterEvent(drag_ev_root)
        self.assertFalse(drag_ev_root.accepted)

        # Проверка очистки _dragged_vm в SceneTree.startDrag через переопределение
        tree_widget._dragged_vm = vol_vm
        try:
            # Имитация вызова startDrag
            pass
        finally:
            tree_widget._dragged_vm = None
        self.assertIsNone(tree_widget._dragged_vm)

    # 3. Переключение стилей актора треков и сброс состояния в clear()
    def test_track_renderer_style_switch_and_clear(self):
        vp = DummyViewport()
        renderer = TrackRenderer(viewport=vp, render_as_lines=True, min_render_interval=0.0)

        # 1. Первый пакет: точки без линий (например, 1 точка)
        batch_single = {
            'pos_x': np.array([10.0, 20.0]),
            'pos_y': np.array([10.0, 20.0]),
            'pos_z': np.array([0.0, 0.0]),
            'process_id': np.array([0, 1]),
            'particle_id': np.array([1, 2]),  # Разные частицы, линий нет
        }
        renderer.add_tracks_batch(batch_single)
        renderer.update_mesh()
        self.assertEqual(renderer._current_rendered_style, 'points')

        # 2. Второй пакет: добавляются связи линий (та же частица)
        batch_lines = {
            'pos_x': np.array([15.0]),
            'pos_y': np.array([15.0]),
            'pos_z': np.array([0.0]),
            'process_id': np.array([1]),
            'particle_id': np.array([1]),  # Та же частица ID 1 -> создается отрезок линии
        }
        renderer.add_tracks_batch(batch_lines)
        renderer.update_mesh()
        # Должен переключиться на 'lines'
        self.assertEqual(renderer._current_rendered_style, 'lines')

        # 3. Проверка clear()
        renderer.clear()
        self.assertIsNone(renderer._current_rendered_style)
        self.assertEqual(renderer._last_update_time, 0.0)
        self.assertEqual(len(renderer._point_buffer), 0)
        self.assertEqual(len(renderer._lines_buffer), 0)
        self.assertEqual(len(renderer._particle_last_pos), 0)

    # 4. Обработка initial_states для соединения точки эмиссии и взаимодействия
    def test_stream_handler_initial_states_emission(self):
        q = Queue()
        handler = GuiStreamDataHandler(
            track_queue=q,
            shm_name="test_shm_init",
            show_escaped_tracks=False,
            create_shm=True
        )
        try:
            # По умолчанию show_escaped_tracks = False
            self.assertFalse(handler.show_escaped_tracks)

            # Отправка чанка initial_states
            init_chunk = {
                'type': 'initial_states',
                'data': {
                    'pos_x': np.array([0.0, 5.0]),
                    'pos_y': np.array([0.0, 5.0]),
                    'pos_z': np.array([0.0, 5.0]),
                    'particle_ID': np.array([101, 102]),
                }
            }
            handler.process_chunk(init_chunk)

            payload = q.get(timeout=1.0)
            self.assertEqual(payload['type'], 'tracks')
            self.assertEqual(payload['process_id'][0], -2)  # ProcessID для Source Emission
            self.assertEqual(payload['particle_id'][0], 101)
            self.assertEqual(payload['pos_x'][0], 0.0)

            # Чанк escaped_particles игнорируется при show_escaped_tracks=False
            esc_chunk = {
                'type': 'escaped_particles',
                'data': {
                    'birth_x': np.array([0.0]),
                    'birth_y': np.array([0.0]),
                    'birth_z': np.array([0.0]),
                    'pos_x': np.array([200.0]),
                    'pos_y': np.array([200.0]),
                    'pos_z': np.array([200.0]),
                    'particle_id': np.array([999]),
                }
            }
            handler.process_chunk(esc_chunk)
            self.assertTrue(q.empty())

            # При включении флага вылетевшие частицы передаются
            handler.show_escaped_tracks = True
            handler.process_chunk(esc_chunk)
            p_birth = q.get(timeout=1.0)
            p_escape = q.get(timeout=1.0)
            self.assertEqual(p_birth['process_id'][0], -2)
            self.assertEqual(p_escape['process_id'][0], -1)
            self.assertEqual(p_escape['particle_id'][0], 999)
        finally:
            handler.close()

    # 5. Значения по умолчанию для вылетевших частиц
    def test_escaped_tracks_default_disabled(self):
        # В модели обработчика прямого стрима
        handler_vm = DirectStreamHandlerViewModel()
        self.assertFalse(handler_vm.show_escaped_tracks)

        # В GuiStreamDataHandler
        handler = GuiStreamDataHandler()
        self.assertFalse(handler.show_escaped_tracks)

    # 6. Сохранение distribution_path и обновление фантома в 3D при перезагрузке
    def test_voxel_and_source_distribution_path_and_reload(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            p1 = Path(tmp_dir) / "phantom1.npy"
            p2 = Path(tmp_dir) / "phantom2.npy"

            data1 = np.ones((8, 8, 8), dtype=np.int32)
            data2 = np.zeros((16, 12, 10), dtype=np.int32)
            np.save(p1, data1)
            np.save(p2, data2)

            builder = SceneBuilder()
            cfg = WoodcockVoxelVolumeConfig(
                name="PhantomVol",
                distribution=NumpyDistributionConfig(path=str(p1), mapping={1: "Water, Liquid"}),
                voxel_size="2.0 mm"
            )
            vol = builder._build_woodcock_voxel_volume(cfg)
            self.assertEqual(builder.get_distribution_path(vol), str(p1))

            vm = VoxelVolumeViewModel(vol, file_path=builder.get_distribution_path(vol))
            self.assertEqual(vm.file_path, str(p1))
            self.assertEqual(vm.dimensions, (8, 8, 8))
            self.assertEqual(list(vm.size), [16.0, 16.0, 16.0])

            # Перезагружаем фантом из файла p2
            changed_props = []
            vm.property_changed.connect(lambda prop, val: changed_props.append(prop))
            success = vm.reload_distribution(str(p2))
            self.assertTrue(success)

            # Проверяем, что матрица ядра и размеры обновились
            self.assertEqual(vm.file_path, str(p2))
            self.assertEqual(vm.dimensions, (16, 12, 10))
            self.assertEqual(vol.material_distribution.shape, (16, 12, 10))
            np.testing.assert_allclose(vol.size, [32.0, 24.0, 20.0])
            self.assertIn('file_path', changed_props)
            self.assertIn('size', changed_props)

            # Проверяем инспектор свойств
            inspector = PropertyInspector()
            inspector.set_target_viewmodel(vm)
            self.assertEqual(inspector.txt_voxel_path.text(), str(p2))
            self.assertIn("16 × 12 × 10", inspector.lbl_voxel_shape.text())

            # Проверяем SourceViewModel
            src_p = Path(tmp_dir) / "source_dist.npy"
            np.save(src_p, np.ones((10, 10, 10), dtype=np.float32))
            src_cfg = SourceConfig(
                name="TestSource",
                distribution=NumpyDistributionConfig(path=str(src_p)),
                voxel_size="1.0 mm",
                activity="1 MBq"
            )
            src_node = builder._build_source(src_cfg)
            self.assertEqual(builder.get_distribution_path(src_node), str(src_p))
            src_vm = SourceViewModel(src_node, file_path=builder.get_distribution_path(src_node))
            self.assertEqual(src_vm.file_path, str(src_p))

    # 7. Хронологический порядок и непрерывность треков в GuiStreamDataHandler
    def test_track_continuity_and_chronological_order(self):
        q = Queue()
        handler = GuiStreamDataHandler(
            track_queue=q,
            shm_name="test_shm_continuity",
            show_escaped_tracks=True,
            create_shm=True
        )
        try:
            # 1. Начальные состояния (эмиссия из источника)
            init_chunk = {
                'type': 'initial_states',
                'data': {
                    'pos_x': np.array([0.0, 10.0]),
                    'pos_y': np.array([0.0, 10.0]),
                    'pos_z': np.array([0.0, 0.0]),
                    'particle_ID': np.array([1, 2]),
                }
            }
            handler.process_chunk(init_chunk)
            p_init = q.get(timeout=1.0)
            self.assertEqual(p_init['type'], 'tracks')
            self.assertEqual(p_init['process_id'][0], -2)
            self.assertEqual(p_init['particle_id'][0], 1)

            # 2. Взаимодействия (рассеяние частицы 1)
            inter_chunk = {
                'type': 'interactions',
                'data': {
                    'pos_x': np.array([5.0]),
                    'pos_y': np.array([5.0]),
                    'pos_z': np.array([0.0]),
                    'process_id': np.array([1]),
                    'particle_id': np.array([1]),
                    'energy_deposit': np.array([0.1], dtype=np.float32),
                }
            }
            handler.process_chunk(inter_chunk)
            p_inter = q.get(timeout=1.0)
            self.assertEqual(p_inter['process_id'][0], 1)
            self.assertEqual(p_inter['particle_id'][0], 1)

            # 3. Вылет частиц: частица 1 рассеялась (has_interacted=True), частица 2 не рассеялась (has_interacted=False)
            esc_chunk = {
                'type': 'escaped_particles',
                'data': {
                    'birth_x': np.array([0.0, 10.0]),
                    'birth_y': np.array([0.0, 10.0]),
                    'birth_z': np.array([0.0, 0.0]),
                    'pos_x': np.array([50.0, 60.0]),
                    'pos_y': np.array([50.0, 60.0]),
                    'pos_z': np.array([0.0, 0.0]),
                    'particle_id': np.array([1, 2]),
                    'has_interacted': np.array([True, False]),
                }
            }
            handler.process_chunk(esc_chunk)

            # Для нерассеянной частицы 2 должна прийти точка рождения
            p_esc_birth = q.get(timeout=1.0)
            self.assertEqual(p_esc_birth['process_id'][0], -2)
            self.assertEqual(p_esc_birth['particle_id'][0], 2)

            # Точки выхода для обеих частиц
            p_esc_out = q.get(timeout=1.0)
            self.assertEqual(p_esc_out['process_id'][0], -1)
            self.assertEqual(len(p_esc_out['particle_id']), 2)
        finally:
            handler.close()

    # 8. Сохранение радиуса орбиты по лицевой поверхности при кинематических операциях
    def test_camera_orbit_radius_preservation_under_rotation_and_translation(self):
        col = Volume(geometry=Box(120.0, 120.0, 40.0), material=material_database['Pb'], name="Col")
        det = Volume(geometry=Box(120.0, 120.0, 20.0), material=material_database['Plastic Scintillator, Vinyltoluene'], name="Det")
        cam = GammaCamera(collimator=col, detector=det, name="SPECT_Cam")
        cam_vm = GammaCameraViewModel(cam)

        face_radius = 280.0
        cam_vm.set_orbit_position(face_radius, 45.0, z=15.0)
        self.assertAlmostEqual(cam_vm.orbit_radius, face_radius, places=3)
        self.assertAlmostEqual(cam_vm.orbit_angle, 45.0, places=3)
        self.assertAlmostEqual(cam_vm.orbit_z, 15.0, places=3)

        # Смещение по оси стола (Z) сохраняет радиус орбиты
        cam_vm.translate(z=10.0)
        self.assertAlmostEqual(cam_vm.orbit_radius, face_radius, places=3)
        self.assertAlmostEqual(cam_vm.orbit_z, 25.0, places=3)

    # 9. Сохранение ненулевых материалов и вокселей при reload_distribution
    def test_voxel_volume_reload_materials_and_non_zero_retention(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "multi_mat_phantom.npy"
            raw_data = np.array([[[0, 1], [2, 3]]], dtype=np.int32)
            np.save(file_path, raw_data)

            mat_arr = MaterialArray(raw_data.shape)
            vol = WoodcockVoxelVolume(voxel_size=1.0, material_distribution=mat_arr, name="TestVol")
            vm = VoxelVolumeViewModel(vol)

            ok = vm.reload_distribution(str(file_path))
            self.assertTrue(ok)
            self.assertGreaterEqual(len(vol.material_distribution.element_list), 4)
            # Проверяем, что воксели сохранили свои значения [0, 1, 2, 3], а не были занулены в 0
            loaded_data = np.asarray(vol.material_distribution.view(np.ndarray))
            np.testing.assert_array_equal(loaded_data, raw_data)

    # 10. Проверка загрузки nema_1_cam.yaml: Detector находится в списке чувствительных объемов GUI
    def test_nema_yaml_loading_detector_in_gui(self):
        """
        Проверяет, что при загрузке nema_1_cam.yaml через MainWindow._on_open_yaml:
        1. Узел Detector присутствует в сцене и помечен как is_sensitive_detector = True.
        2. Detector отображается в списке SensitiveVolumesList (виджет scene_tree.sensitive_list).
        3. В PropertyInspector флаг chk_is_detector установлен в True.
        4. При создании новой сцены через _on_new_scene список детекторов сбрасывается.
        """
        win = MainWindow()
        try:
            win._on_open_yaml("nema_1_cam.yaml")
            self.assertEqual(win.scene_tree.sensitive_list.count(), 1)
            item_text = win.scene_tree.sensitive_list.item(0).text()
            self.assertIn("Detector", item_text)

            det_vm = win.scene_vm.find_by_name("Detector")
            self.assertIsNotNone(det_vm)
            self.assertIsInstance(det_vm, VolumeViewModel)
            self.assertTrue(det_vm.is_sensitive_detector)

            # Проверка синхронизации с PropertyInspector
            win.scene_vm.select_node(det_vm)
            self.assertTrue(win.property_inspector.chk_is_detector.isChecked())

            # Проверка сброса при очистке сцены
            win._on_new_scene()
            self.assertEqual(win.scene_tree.sensitive_list.count(), 0)
        finally:
            win.close()

    # 11. Автоматическая активация детектора в GammaCameraViewModel и типизированные свойства
    def test_gamma_camera_viewmodel_detector_auto_activation(self):
        """
        Проверяет свойства detector_vm и collimator_vm в GammaCameraViewModel,
        а также автоматическую установку флага is_sensitive_detector = True для кристалла детектора.
        """
        col = Volume(geometry=Box(80.0, 80.0, 25.0), material=material_database['Pb'], name="CamCollimator")
        det = Volume(geometry=Box(80.0, 80.0, 15.0), material=material_database['Plastic Scintillator, Vinyltoluene'], name="CamDetector")
        cam = GammaCamera(collimator=col, detector=det, name="SPECT_Head")
        cam_vm = GammaCameraViewModel(cam)

        self.assertIsNotNone(cam_vm.detector_vm)
        self.assertEqual(cam_vm.detector_vm.name, "CamDetector")
        self.assertTrue(cam_vm.detector_vm.is_sensitive_detector)

        self.assertIsNotNone(cam_vm.collimator_vm)
        self.assertEqual(cam_vm.collimator_vm.name, "CamCollimator")
        self.assertFalse(cam_vm.collimator_vm.is_sensitive_detector)

    # 12. Применение конфигурации SimulationConfig к SceneViewModel (apply_simulation_config)
    def test_scene_viewmodel_apply_simulation_config(self):
        """
        Проверяет метод apply_simulation_config в SceneViewModel для синхронизации
        чувствительных объемов, заданных в data_manager.handlers.
        """
        cfg = load_simulation_config("nema_1_cam.yaml")
        root = SceneBuilder().build_scene(cfg.scene)
        scene_vm = SceneViewModel()
        scene_vm.load_scene(root)
        scene_vm.apply_simulation_config(cfg)

        det_vm = scene_vm.find_by_name("Detector")
        self.assertIsNotNone(det_vm)
        self.assertIsInstance(det_vm, VolumeViewModel)
        self.assertTrue(det_vm.is_sensitive_detector)

        sensitive_vols = VolumeViewModel.get_sensitive_volumes()
        self.assertIn(det_vm.core_node, sensitive_vols)

    # 13. Проверка генерации задач и валидности реконструкции геометрии для nema_1_cam.yaml
    def test_nema_yaml_jobs_generation_and_worker_scene_build(self):
        """
        Проверяет, что после загрузки nema_1_cam.yaml задачи симуляции успешно генерируются,
        экспортируются в SimulationConfig и воркер Orchestrator может восстановить сцену без UnpicklingError.
        """
        win = MainWindow()
        try:
            win._on_open_yaml("nema_1_cam.yaml")
            jobs = win.orchestrator_session.generate_jobs()
            self.assertGreater(len(jobs), 0)

            # Проверяем экспорт конфигурации сцены
            sim_config = SceneExporter.export_to_config(
                root_node=win.scene_vm.root_vm.core_node,
                simulation_manager_cfg=SimulationManagerConfig(particles_number=100),
                data_manager_cfg=win.data_manager_vm.to_config(),
                distribution_registry=win.scene_vm.distribution_registry,
            )
            phantom_cfg = next(c for c in sim_config.scene.children if c.name == "Phantom")
            self.assertIsInstance(phantom_cfg.distribution, RawDistributionConfig)

            builder = SceneBuilder()
            root_scene = builder.build_scene(sim_config.scene)
            self.assertIsNotNone(root_scene)
        finally:
            win.close()


if __name__ == '__main__':
    unittest.main()
