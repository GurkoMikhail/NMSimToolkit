import unittest
import numpy as np
import tempfile
from pathlib import Path

from multiprocessing import Queue

from PySide6.QtWidgets import QApplication
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

from gui.viewmodels.nodes.base_node_vm import NodeViewModel
from gui.viewmodels.nodes.volume_vm import VolumeViewModel
from gui.viewmodels.nodes.voxel_volume_vm import VoxelVolumeViewModel
from gui.viewmodels.nodes.source_vm import SourceViewModel
from gui.viewmodels.nodes.gamma_camera_vm import GammaCameraViewModel
from gui.viewmodels.nodes.dose_grid_vm import DoseGridViewModel
from gui.viewmodels.nodes.factory import create_node_viewmodel
from gui.viewmodels.scene_viewmodel import SceneViewModel
from gui.viewport_3d.dicom_colormaps import to_vtk_piecewise_function
from gui.controllers.orchestrator_session import OrchestratorSession
from gui.views.scene_tree_widget import SceneTreeWidget
from gui.views.results_viewer import ResultsViewer
from gui.viewmodels.procedure_viewmodel import SpectProcedureViewModel
from gui.viewmodels.data_handler_viewmodel import DirectStreamHandlerViewModel
from gui.views.property_inspector import PropertyInspector
from gui.controllers.stream_handlers import GuiStreamDataHandler


# Гарантируем наличие QApplication для GUI тестов
app = QApplication.instance()
if app is None:
    app = QApplication([])


class TestAll14GUIFixes(unittest.TestCase):
    """
    Комплексное тестирование всех 14 исправлений GUI и симулятора.
    """

    def setUp(self):
        self.root = CompositeNode(name="World")
        self.scene_vm = SceneViewModel(self.root)

    # 1. Свойства объектов и выбор файлов для Phantom и Source
    def test_01_phantom_and_source_file_reload(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "test_array.npy"
            arr = np.ones((8, 8, 8), dtype=np.uint8) * 3
            np.save(file_path, arr)

            # Phantom
            mat_arr = MaterialArray((8, 8, 8))
            vol = WoodcockVoxelVolume(voxel_size=Float(2.0 * units.mm), material_distribution=mat_arr, name="Phantom1")
            vm = VoxelVolumeViewModel(vol)
            self.assertTrue(vm.reload_distribution(str(file_path)))
            self.assertEqual(vm.dimensions, (8, 8, 8))
            self.assertEqual(vm.file_path, str(file_path))

            # Source
            src = Source(distribution=np.ones((4, 4, 4)), activity=Float(1e5 * units.Bq))
            src_vm = SourceViewModel(src)
            self.assertTrue(src_vm.reload_distribution(str(file_path)))
            self.assertEqual(src_vm.dimensions, (8, 8, 8))
            self.assertEqual(src_vm.file_path, str(file_path))

    # 2. Добавление любых типов узлов в сцену
    def test_02_add_various_node_types(self):
        tree_widget = SceneTreeWidget(self.scene_vm)
        root_vm = self.scene_vm.root_vm

        # Box
        tree_widget._add_box_volume(root_vm)
        # Phantom
        tree_widget._add_voxel_volume(root_vm)
        # PointSource
        tree_widget._add_point_source(root_vm)
        # Source
        tree_widget._add_voxel_source(root_vm)
        # Group
        tree_widget._add_composite_node(root_vm)

        node_types = [type(n.core_node).__name__ for n in root_vm.children]
        self.assertIn("Volume", node_types)
        self.assertIn("WoodcockVoxelVolume", node_types)
        self.assertIn("PointSource", node_types)
        self.assertIn("Source", node_types)
        self.assertIn("CompositeNode", node_types)

    # 3. Перемещение узлов по графу сцены и защита от циклов
    def test_03_move_node_and_cycle_prevention(self):
        parent1 = CompositeNode(name="Group1")
        child1 = CompositeNode(name="Child1")
        grandchild1 = CompositeNode(name="Grandchild1")
        parent1.add_child(child1)
        child1.add_child(grandchild1)

        p1_vm = self.scene_vm.load_scene(parent1)
        c1_vm = p1_vm.children[0]
        gc1_vm = c1_vm.children[0]

        # Попытка переместить родителя в своего потомка (цикл) -> должна отклониться
        self.assertFalse(self.scene_vm.move_node(p1_vm, gc1_vm))
        self.assertFalse(self.scene_vm.move_node(c1_vm, gc1_vm))

        # Валидное перемещение: grandchild1 перемещаем напрямую в root
        self.assertTrue(self.scene_vm.move_node(gc1_vm, p1_vm))
        self.assertIn(gc1_vm, p1_vm.children)

    # 4 & 5 & 7 & 8: Пресеты прозрачности, Air Cutoff и LOD
    def test_04_05_07_08_lod_and_opacity_presets(self):
        for preset in ['air_cutoff', 'linear', 'soft_tissue', 'xray_translucent', 'step']:
            pw = to_vtk_piecewise_function(max_alpha=0.6, threshold=0.1, preset=preset)
            self.assertIsNotNone(pw)

        # Проверка рендерера
        mat_arr = MaterialArray((4, 4, 4))
        vol = WoodcockVoxelVolume(voxel_size=Float(2.0 * units.mm), material_distribution=mat_arr, name="P")
        vm = VoxelVolumeViewModel(vol)
        self.assertEqual(vm.lod_factor, 1.0)
        self.assertEqual(vm.opacity_preset, 'air_cutoff')

        vm.lod_factor = 2.5
        vm.opacity_preset = 'soft_tissue'
        vm.max_opacity = 0.8
        self.assertEqual(vm.lod_factor, 2.5)
        self.assertEqual(vm.opacity_preset, 'soft_tissue')
        self.assertEqual(vm.max_opacity, 0.8)

    # 6. Источники в ViewModel и их параметры
    def test_06_source_viewmodel(self):
        ps = PointSource(activity=Float(5e6 * units.Bq), energy=Float(140.5 * units.keV))
        ps.name = "MyPointSource"
        vm = create_node_viewmodel(ps)
        self.assertIsInstance(vm, SourceViewModel)
        self.assertTrue(vm.is_point_source)
        self.assertAlmostEqual(vm.activity, 5.0)
        self.assertAlmostEqual(vm.energy, 140.5)

    # 9. Автоматический расчет границ DoseVolume по всей сцене
    def test_09_dose_volume_scene_coverage(self):
        # Создаем сцену: фантом в центре (Y=0) и коллиматор/детектор на Y=300
        root = CompositeNode(name="World")
        phantom = Volume(geometry=Box(100.0, 100.0, 100.0), material=material_database['Water, Liquid'], name="Phantom")
        root.add_child(phantom)

        detector = Volume(geometry=Box(200.0, 20.0, 200.0), material=material_database['Sodium Iodide'], name="detector")
        detector.translate(y=300.0)
        root.add_child(detector)

        # Без узла DoseGridNode в сцене 3D-накопление дозы не создается
        session_no_grid = OrchestratorSession(scene_vm=SceneViewModel(root))
        try:
            self.assertIsNone(session_no_grid.dose_voxel_size)
            self.assertIsNone(session_no_grid.dose_origin)
        finally:
            session_no_grid.close()

        # При наличии узла DoseGridNode накопитель инициализируется по параметрам узла
        dose_node = DoseGridNode(name="DoseScorer", size=[200.0, 400.0, 200.0], dose_voxel_size=5.0)
        root.add_child(dose_node)
        session = OrchestratorSession(scene_vm=SceneViewModel(root))
        try:
            self.assertEqual(session.dose_voxel_size, 5.0)
            self.assertEqual(session.dose_origin, (-100.0, -200.0, -100.0))
        finally:
            session.close()

    # 9b. Инициализация параметров сетки дозы по узлу DoseGridNode и точное применение dose_voxel_size
    def test_09b_scene_root_box_bounds_and_exact_voxel_size(self):
        world = Volume(
            geometry=Box(600.0, 600.0, 600.0),
            material=material_database['Air, Dry (near sea level)'],
            name="World"
        )
        phantom = Volume(
            geometry=Box(100.0, 100.0, 100.0),
            material=material_database['Water, Liquid'],
            name="Phantom"
        )
        world.add_child(phantom)
        dose_node = DoseGridNode(name="WorldDose", size=[600.0, 600.0, 600.0], dose_voxel_size=2.0)
        world.add_child(dose_node)

        session = OrchestratorSession(scene_vm=SceneViewModel(world))
        try:
            self.assertEqual(session.dose_voxel_size, 2.0)
            self.assertEqual(session.dose_origin, (-300.0, -300.0, -300.0))
        finally:
            session.close()

    # 9c. Отсутствие глобальной сетки на World и настройка размера вокселя дозы через DoseGridViewModel
    def test_09c_property_inspector_dose_voxel_binding(self):
        world = Volume(
            geometry=Box(500.0, 500.0, 500.0),
            material=material_database['Air, Dry (near sea level)'],
            name="World"
        )
        vm = VolumeViewModel(world)
        inspector = PropertyInspector()
        inspector.set_target_viewmodel(vm)

        # Корневой World больше не содержит глобальную группу карты дозы
        self.assertFalse(hasattr(inspector, 'dose_group'))

        # Настройка шага вокселя теперь производится в отдельном узле DoseGridNode
        grid_node = DoseGridNode(name="DoseScorer", size=[500.0, 500.0, 500.0], dose_voxel_size=5.0)
        grid_vm = DoseGridViewModel(grid_node)
        inspector.set_target_viewmodel(grid_vm)

        self.assertFalse(inspector.dose_grid_group.isHidden())
        self.assertEqual(inspector.spin_dose_grid_voxel.value(), 5.0)

        inspector.spin_dose_grid_voxel.setValue(10.0)
        self.assertEqual(grid_vm.dose_voxel_size, 10.0)
        self.assertEqual(grid_node.dose_voxel_size, 10.0)
        self.assertIn("50 × 50 × 50", inspector.lbl_dose_grid_shape.text())

    # 10. Передача треков вылетевших частиц
    def test_10_escaped_particles_stream(self):
        q = Queue()
        handler = GuiStreamDataHandler(
            track_queue=q,
            shm_name="test_esc_shm",
            show_escaped_tracks=True,
            create_shm=True
        )
        try:
            chunk = {
                'type': 'escaped_particles',
                'data': {
                    'birth_x': np.array([0.0]),
                    'birth_y': np.array([0.0]),
                    'birth_z': np.array([0.0]),
                    'pos_x': np.array([150.0]),
                    'pos_y': np.array([200.0]),
                    'pos_z': np.array([50.0]),
                    'particle_id': np.array([42]),
                }
            }
            handler.process_chunk(chunk)
            
            # В очереди должны быть 2 пакета: рождение и вылет
            p1 = q.get(timeout=1.0)
            p2 = q.get(timeout=1.0)
            self.assertEqual(p1['process_id'][0], -2)
            self.assertEqual(p2['process_id'][0], -1)
            self.assertEqual(p2['pos_x'][0], 150.0)
        finally:
            handler.close()

    # 11 & 13. Модульные параметры процедуры и обработчиков данных
    def test_11_13_procedure_and_stream_settings(self):
        proc = SpectProcedureViewModel()
        proc.steps = 32
        proc.stop_time = 2.5
        proc.particles_number = 10000
        self.assertEqual(proc.steps, 32)
        self.assertEqual(proc.stop_time, 2.5)
        self.assertEqual(proc.particles_number, 10000)

        stream_vm = DirectStreamHandlerViewModel()
        stream_vm.buffer_capacity = 25000
        stream_vm.max_tracks_per_batch = 1500
        stream_vm.show_escaped_tracks = True
        self.assertEqual(stream_vm.buffer_capacity, 25000)
        self.assertEqual(stream_vm.max_tracks_per_batch, 1500)
        self.assertTrue(stream_vm.show_escaped_tracks)

    # 12. Настройка чувствительного объема детектора
    def test_12_sensitive_volume_detection(self):
        vol = Volume(geometry=Box(50.0, 50.0, 50.0), material=material_database['Plastic Scintillator, Vinyltoluene'], name="CustomDet")
        vol_vm = VolumeViewModel(vol)
        vol_vm.is_sensitive_detector = True
        self.assertTrue(vol_vm.is_sensitive_detector)
        self.assertFalse(hasattr(vol, 'is_sensitive_detector'))

        root = CompositeNode(name="World")
        root.add_child(vol)
        scene_vm = SceneViewModel(root)
        matching_nodes = [node for node in scene_vm.all_nodes() if isinstance(node, VolumeViewModel) and node.name == "CustomDet"]
        self.assertEqual(len(matching_nodes), 1)
        matching_nodes[0].is_sensitive_detector = True
        self.assertTrue(matching_nodes[0].is_sensitive_detector)
        self.assertFalse(hasattr(vol, 'is_sensitive_detector'))

    # 14. Многопроекционный просмотр в ResultsViewer
    def test_14_results_viewer_multi_projection(self):
        viewer = ResultsViewer()
        stack = np.ones((8, 64, 64), dtype=np.float32)
        for i in range(8):
            stack[i] *= (i + 1)

        viewer.set_projection_stack_data(stack, current_idx=0, total_views=8, angle_deg=0.0)
        self.assertFalse(viewer.proj_nav_widget.isHidden())
        self.assertEqual(viewer.slider_proj.maximum(), 8)
        self.assertEqual(viewer.slider_proj.value(), 1)

        # Переключаем на проекцию 4
        viewer.slider_proj.setValue(4)
        self.assertEqual(viewer._current_view_idx, 3)
        self.assertAlmostEqual(np.mean(viewer._current_projection), 4.0)


if __name__ == '__main__':
    unittest.main()
