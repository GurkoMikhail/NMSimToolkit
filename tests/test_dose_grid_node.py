import unittest
import numpy as np
from PySide6.QtWidgets import QApplication

from core.scene.nodes import SpatialNode, CompositeNode
from core.scene.dose_grid_node import DoseGridNode
from core.geometry.volumes import Volume
from core.geometry.geometries import Box
from core.materials.materials import Material
from core.data.dose_map_handler import DoseMapHandler, DoseGridEntry
from core.config.exporter import SceneExporter
from core.config.builder import SceneBuilder
from gui.viewmodels.node_viewmodel import DoseGridViewModel, VolumeViewModel, create_node_viewmodel
from gui.viewmodels.scene_viewmodel import SceneViewModel
from gui.views.scene_tree_widget import SceneTreeWidget
from gui.views.property_inspector import PropertyInspector
from gui.controllers.simulation_session import SimulationSession

# Инициализация QApplication для тестирования Qt-компонентов
app = QApplication.instance() or QApplication([])


class TestDoseGridNode(unittest.TestCase):
    """
    Набор тестов для узла графа сцены DoseGridNode, его ViewModel,
    интерфейсных компонентов и многосеточного накопления дозы.
    """

    def test_01_dose_grid_node_core_properties(self):
        """Проверка базовых геометрических и математических свойств DoseGridNode."""
        node = DoseGridNode(
            name="TumorDose",
            size=[60.0, 80.0, 100.0],
            dose_voxel_size=2.0,
            is_active=True
        )

        self.assertEqual(node.name, "TumorDose")
        np.testing.assert_array_equal(node.size, np.array([60.0, 80.0, 100.0]))
        self.assertEqual(node.dose_voxel_size, 2.0)
        self.assertTrue(node.is_active)

        # Проверка автоматического расчета формы сетки (Nx, Ny, Nz)
        self.assertEqual(node.grid_shape, (30, 40, 50))

        # Проверка локального origin: центр узла в (0,0,0), нижний угол в (-Lx/2, -Ly/2, -Lz/2)
        self.assertEqual(node.origin, (-30.0, -40.0, -50.0))

        # Проверка расчета объема памяти: 30 * 40 * 50 * 8 байт = 480 000 байт ~ 0.4578 МБ
        expected_mb = (30 * 40 * 50 * 8) / (1024 * 1024)
        self.assertAlmostEqual(node.memory_mb, expected_mb, places=4)

        # Проверка изменения шага вокселя и валидации
        node.dose_voxel_size = 5.0
        self.assertEqual(node.grid_shape, (12, 16, 20))
        with self.assertRaises(ValueError):
            node.dose_voxel_size = -1.0

    def test_02_dose_grid_node_transforms(self):
        """Проверка пространственных трансформаций DoseGridNode в иерархии сцены."""
        world = CompositeNode(name="World")
        phantom = Volume(geometry=Box(200.0, 200.0, 200.0), material=Material("Water"), name="Phantom")
        phantom.translate(x=50.0, y=100.0, z=0.0)
        world.add_child(phantom)

        # Дочерний узел сетки дозы внутри фантома со смещением
        dose_node = DoseGridNode(name="LocalDose", size=[40.0, 40.0, 40.0], dose_voxel_size=2.0)
        dose_node.translate(x=10.0, y=0.0, z=0.0)
        phantom.add_child(dose_node)

        # Глобальная позиция сетки: x = 50 + 10 = 60, y = 100, z = 0
        g_mat = dose_node.global_matrix
        self.assertAlmostEqual(g_mat[0, 3], 60.0)
        self.assertAlmostEqual(g_mat[1, 3], 100.0)
        self.assertAlmostEqual(g_mat[2, 3], 0.0)

        # Проверка преобразования глобальной точки (60, 100, 0) в локальную систему сетки -> (0, 0, 0)
        global_pt = np.array([[60.0, 100.0, 0.0]])
        local_pt = dose_node.convert_to_local_position(global_pt)
        np.testing.assert_allclose(local_pt, [[0.0, 0.0, 0.0]], atol=1e-6)

    def test_03_dose_grid_viewmodel(self):
        """Проверка модели представления DoseGridViewModel и генерации сигналов."""
        node = DoseGridNode(name="Grid1", size=[100.0, 100.0, 100.0], dose_voxel_size=5.0)
        vm = create_node_viewmodel(node)
        self.assertIsInstance(vm, DoseGridViewModel)

        changed_props = []
        vm.property_changed.connect(lambda p, v: changed_props.append((p, v)))

        vm.dose_voxel_size = 2.5
        self.assertEqual(node.dose_voxel_size, 2.5)
        self.assertIn(('dose_voxel_size', 2.5), changed_props)
        self.assertIn(('grid_shape', (40, 40, 40)), changed_props)

        vm.size = [50.0, 50.0, 50.0]
        np.testing.assert_array_equal(node.size, np.array([50.0, 50.0, 50.0]))
        self.assertIn(('grid_shape', (20, 20, 20)), changed_props)

        vm.is_active = False
        self.assertFalse(node.is_active)
        self.assertIn(('is_active', False), changed_props)

    def test_04_scene_tree_widget_dose_grid_actions(self):
        """Проверка добавления DoseGridNode через SceneTreeWidget."""
        world = CompositeNode(name="World")
        scene_vm = SceneViewModel(world)
        tree_widget = SceneTreeWidget(scene_vm)

        # Добавление независимого узла сетки дозы
        tree_widget._add_dose_grid_node(scene_vm.root_vm)
        nodes = scene_vm.all_nodes()
        dose_vms = [n for n in nodes if isinstance(n, DoseGridViewModel)]
        self.assertEqual(len(dose_vms), 1)
        self.assertEqual(dose_vms[0].name, "DoseGrid_1")

        # Добавление сетки дозы для конкретного Volume (Smart Snap)
        vol = Volume(geometry=Box(150.0, 120.0, 90.0), material=Material("Water"), name="TargetOrgan")
        vol_vm = VolumeViewModel(vol)
        scene_vm.add_node(scene_vm.root_vm, vol_vm)

        tree_widget._add_dose_grid_for_volume(vol_vm)
        vol_children = vol_vm.children
        self.assertEqual(len(vol_children), 1)
        self.assertIsInstance(vol_children[0], DoseGridViewModel)
        child_dose_vm = vol_children[0]
        np.testing.assert_array_equal(child_dose_vm.size, np.array([150.0, 120.0, 90.0]))
        self.assertEqual(child_dose_vm.dose_voxel_size, 5.0)

    def test_05_property_inspector_dose_grid_integration(self):
        """Проверка отображения и редактирования DoseGridViewModel в PropertyInspector."""
        node = DoseGridNode(name="DoseScorer", size=[100.0, 100.0, 100.0], dose_voxel_size=5.0)
        vm = DoseGridViewModel(node)

        inspector = PropertyInspector()
        inspector.set_target_viewmodel(vm)

        # Секция сетки дозы должна быть видима, а секции объемов скрыты
        self.assertFalse(inspector.dose_grid_group.isHidden())
        self.assertTrue(inspector.volume_group.isHidden())

        # Проверка начальных значений
        self.assertEqual(inspector.spin_dose_grid_size_x.value(), 100.0)
        self.assertEqual(inspector.spin_dose_grid_voxel.value(), 5.0)
        self.assertIn("20 × 20 × 20", inspector.lbl_dose_grid_shape.text())

        # Редактирование в инспекторе
        inspector.spin_dose_grid_voxel.setValue(2.0)
        self.assertEqual(vm.dose_voxel_size, 2.0)
        self.assertEqual(node.dose_voxel_size, 2.0)
        self.assertIn("50 × 50 × 50", inspector.lbl_dose_grid_shape.text())

    def test_06_dose_map_handler_multiple_grids_accumulation(self):
        """Проверка независимого накопления дозы в несколько узлов DoseGridNode."""
        # Сетка 1: крупная сетка в центре (0, 0, 0)
        grid1 = DoseGridNode(name="CoarseWhole", size=[200.0, 200.0, 200.0], dose_voxel_size=10.0)
        # Сетка 2: мелкая сетка мишени со смещением (50, 0, 0)
        grid2 = DoseGridNode(name="FineTarget", size=[40.0, 40.0, 40.0], dose_voxel_size=2.0)
        grid2.translate(x=50.0, y=0.0, z=0.0)

        handler = DoseMapHandler(grid_nodes=[grid1, grid2], shm_name="test_multi_dose_shm")
        try:
            self.assertEqual(len(handler.entries), 2)
            self.assertEqual(handler.entries[0].grid_shape, (20, 20, 20))
            self.assertEqual(handler.entries[1].grid_shape, (20, 20, 20))

            # Событие взаимодействия в глобальной точке (50, 0, 0) с энерговыделением 100.0 кэВ
            chunk = {
                'type': 'interactions',
                'data': {
                    'pos_x': np.array([50.0]),
                    'pos_y': np.array([0.0]),
                    'pos_z': np.array([0.0]),
                    'energy_deposit': np.array([100.0]),
                }
            }
            handler.process_chunk(chunk)

            # Проверяем накопление в grid1:
            # Локальные координаты: (50, 0, 0), origin: (-100, -100, -100), voxel: 10
            # ix = (50 - (-100)) / 10 = 15, iy = 10, iz = 10
            snap1 = handler.get_dose_snapshot("CoarseWhole")
            self.assertIsNotNone(snap1)
            self.assertEqual(snap1[15, 10, 10], 100.0)
            self.assertEqual(np.sum(snap1), 100.0)

            # Проверяем накопление в grid2:
            # Глобальная точка (50, 0, 0) переводится в локальную (0, 0, 0), origin: (-20, -20, -20), voxel: 2
            # ix = (0 - (-20)) / 2 = 10, iy = 10, iz = 10 (строго центр сетки мишени)
            snap2 = handler.get_dose_snapshot("FineTarget")
            self.assertIsNotNone(snap2)
            self.assertEqual(snap2[10, 10, 10], 100.0)
            self.assertEqual(np.sum(snap2), 100.0)

        finally:
            handler.close()

    def test_07_simulation_session_with_dose_grid_nodes(self):
        """Проверка автоматической инициализации DoseMapHandler из дерева сцены SimulationSession."""
        world = CompositeNode(name="World")
        target_dose = DoseGridNode(name="TargetDose", size=[80.0, 80.0, 80.0], dose_voxel_size=4.0)
        world.add_child(target_dose)

        session = SimulationSession(
            scene_root=world,
            dose_accumulation_enabled=True,
            particles_number=100
        )
        try:
            self.assertIsNotNone(session.dose_handler)
            self.assertEqual(len(session.dose_handler.entries), 1)
            self.assertEqual(session.dose_grid_shape, (20, 20, 20))
            self.assertEqual(session.dose_voxel_size, 4.0)
            self.assertEqual(session.dose_origin, (-40.0, -40.0, -40.0))

            # Имитация завершения моделирования с заполнением данных
            session.dose_handler.entries[0]._dose_grid[10, 10, 10] = 42.0
            session._on_runner_finished()

            # Проверяем, что массив скопировался в target_dose.dose_data
            self.assertIsNotNone(target_dose.dose_data)
            self.assertEqual(target_dose.dose_data[10, 10, 10], 42.0)

            # Проверка очистки накопления
            session.clear_accumulation()
            self.assertEqual(target_dose.dose_data[10, 10, 10], 0.0)
        finally:
            session.close()

    def test_08_gamma_camera_child_dose_grid_auto_bounds(self):
        """Проверка автоматической подгонки размеров создаваемой сетки дозы под BoundingBox гамма-камеры."""
        from core.geometry.gamma_cameras import GammaCamera
        from gui.viewmodels.node_viewmodel import GammaCameraViewModel

        collimator = Volume(geometry=Box(400.0, 400.0, 40.0), material=Material("Pb"), name="Collimator")
        detector = Volume(geometry=Box(400.0, 400.0, 10.0), material=Material("NaI"), name="Detector")
        camera = GammaCamera(collimator=collimator, detector=detector, name="SpectCamera")
        cam_vm = GammaCameraViewModel(camera)

        world = CompositeNode(name="World")
        scene_vm = SceneViewModel(world)
        scene_vm.add_node(scene_vm.root_vm, cam_vm)
        tree_widget = SceneTreeWidget(scene_vm)

        # Добавляем сетку дозы как дочерний элемент гамма-камеры
        tree_widget._add_dose_grid_node(cam_vm)
        dose_children = [c for c in cam_vm.children if isinstance(c, DoseGridViewModel)]
        self.assertEqual(len(dose_children), 1)
        dose_child = dose_children[0]

        # Размеры сетки дозы должны автоматически соответствовать BoundingBox гамма-камеры!
        cam_bounds_size = cam_vm.local_bound
        np.testing.assert_allclose(dose_child.size, cam_bounds_size, atol=1e-5)
        self.assertGreater(dose_child.size[0], 400.0)  # С учетом защитных экранов Pb
        self.assertGreater(dose_child.size[1], 400.0)

    def test_09_property_inspector_fit_to_parent(self):
        """Проверка кнопки подогнать под родителя в PropertyInspector для DoseGridViewModel."""
        from core.geometry.gamma_cameras import GammaCamera
        from gui.viewmodels.node_viewmodel import GammaCameraViewModel

        collimator = Volume(geometry=Box(400.0, 400.0, 40.0), material=Material("Pb"), name="Collimator")
        detector = Volume(geometry=Box(400.0, 400.0, 10.0), material=Material("NaI"), name="Detector")
        camera = GammaCamera(collimator=collimator, detector=detector, name="SpectCamera")
        cam_vm = GammaCameraViewModel(camera)

        # Дочерний узел с произвольными размерами
        grid_node = DoseGridNode(name="ArbitraryGrid", size=[33.0, 44.0, 55.0], dose_voxel_size=5.0)
        grid_vm = DoseGridViewModel(grid_node, parent_vm=cam_vm)
        cam_vm.children.append(grid_vm)

        inspector = PropertyInspector()
        inspector.set_target_viewmodel(grid_vm)

        self.assertTrue(inspector.btn_fit_dose_grid_to_parent.isEnabled())
        self.assertIn("SpectCamera", inspector.btn_fit_dose_grid_to_parent.text())

        # Нажимаем кнопку подгонки под родителя
        inspector._on_fit_dose_grid_to_parent()

        cam_bounds_size = cam_vm.local_bound
        np.testing.assert_allclose(grid_vm.size, cam_bounds_size, atol=1e-5)
        np.testing.assert_allclose(grid_node.size, cam_bounds_size, atol=1e-5)
        self.assertAlmostEqual(inspector.spin_dose_grid_size_x.value(), cam_bounds_size[0], places=3)

    def test_10_volume_local_bound_property(self):
        """Проверка @property local_bound у Volume и VolumeViewModel."""
        vol = Volume(geometry=Box(120.0, 150.0, 180.0), material=Material("Water"), name="TestVol")
        np.testing.assert_allclose(vol.local_bound, [120.0, 150.0, 180.0])

        vol_vm = VolumeViewModel(vol)
        np.testing.assert_allclose(vol_vm.local_bound, [120.0, 150.0, 180.0])

        # Проверка отсутствия get_local_bounds на базовом SpatialNode
        base_node = SpatialNode(name="BaseNode")
        self.assertFalse(hasattr(base_node, 'get_local_bounds'))

    def test_11_dose_grid_node_serialization_and_deserialization(self):
        """Проверка сериализации DoseGridNode в конфигурацию и обратного восстановления через SceneBuilder."""
        root = CompositeNode(name="Root")
        grid = DoseGridNode(name="DoseTarget", size=[150.0, 250.0, 350.0], dose_voxel_size=2.5, is_active=True)
        grid.translate(10.0, 20.0, 30.0)
        root.add_child(grid)

        # Экспорт в конфигурацию
        exported_cfg = SceneExporter.export_node(root)
        self.assertEqual(len(exported_cfg.children), 1)
        child_cfg = exported_cfg.children[0]
        self.assertEqual(child_cfg.type, "DoseGridNode")
        self.assertEqual(child_cfg.name, "DoseTarget")
        self.assertEqual(child_cfg.dose_voxel_size, 2.5)
        np.testing.assert_allclose(child_cfg.size, [150.0, 250.0, 350.0])
        self.assertTrue(child_cfg.is_active)

        # Обратное восстановление через SceneBuilder
        builder = SceneBuilder()
        reconstructed_root = builder.build_scene(exported_cfg)
        self.assertIsInstance(reconstructed_root, CompositeNode)
        self.assertEqual(len(reconstructed_root.childs), 1)

        reconstructed_grid = reconstructed_root.childs[0]
        self.assertIsInstance(reconstructed_grid, DoseGridNode)
        self.assertEqual(reconstructed_grid.name, "DoseTarget")
        self.assertEqual(reconstructed_grid.dose_voxel_size, 2.5)
        np.testing.assert_allclose(reconstructed_grid.size, [150.0, 250.0, 350.0])
        self.assertTrue(reconstructed_grid.is_active)
        self.assertEqual(reconstructed_grid.grid_shape, (60, 100, 140))


if __name__ == '__main__':
    unittest.main()

