import unittest
from multiprocessing import shared_memory
import numpy as np
from PySide6.QtWidgets import QApplication

from core.scene.nodes import SpatialNode, CompositeNode
from core.scene.dose_grid_node import DoseGridNode
from core.geometry.volumes import Volume
from core.geometry.geometries import Box
from core.materials.materials import Material
from core.physics.interaction_buffers import ChunkType
from core.data.dose_map_handler import DoseMapHandler, DoseGridEntry
from core.config.exporter import SceneExporter
from core.config.builder import SceneBuilder
from gui.viewmodels.nodes.gamma_camera_vm import create_default_gamma_camera_vm
from gui.viewmodels.nodes.dose_grid_vm import DoseGridViewModel
from gui.viewmodels.nodes.volume_vm import VolumeViewModel
from gui.viewmodels.nodes.gamma_camera_vm import GammaCameraViewModel
from gui.viewmodels.nodes.factory import create_node_viewmodel
from gui.viewmodels.scene_viewmodel import SceneViewModel
from gui.views.scene_tree_widget import SceneTreeWidget
from gui.views.property_inspector import PropertyInspector
from gui.controllers.orchestrator_session import OrchestratorSession

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

        # Проверка дефолтного типа данных (float32) и расчета объема памяти: 30 * 40 * 50 * 4 байта
        self.assertEqual(node.dtype, "float32")
        self.assertEqual(node.numpy_dtype, np.dtype(np.float32))
        expected_megabytes_f32 = (30 * 40 * 50 * 4) / (1024 * 1024)
        self.assertAlmostEqual(node.memory_mb, expected_megabytes_f32, places=4)

        # Проверка переключения точности на float64 и пересчета памяти
        node.dtype = "float64"
        self.assertEqual(node.dtype, "float64")
        self.assertEqual(node.numpy_dtype, np.dtype(np.float64))
        expected_megabytes_f64 = (30 * 40 * 50 * 8) / (1024 * 1024)
        self.assertAlmostEqual(node.memory_mb, expected_megabytes_f64, places=4)
        with self.assertRaises(ValueError):
            node.dtype = "int16"  # type: ignore[assignment]
        node.dtype = "float32"

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
        vm.property_changed.connect(lambda prop_name, prop_val: changed_props.append((prop_name, prop_val)))

        vm.dose_voxel_size = 2.5
        self.assertEqual(node.dose_voxel_size, 2.5)
        self.assertIn(('dose_voxel_size', 2.5), changed_props)
        self.assertIn(('grid_shape', (40, 40, 40)), changed_props)

        vm.size = [50.0, 50.0, 50.0]
        np.testing.assert_array_equal(node.size, np.array([50.0, 50.0, 50.0]))
        self.assertIn(('grid_shape', (20, 20, 20)), changed_props)

        vm.dtype = "float64"
        self.assertEqual(node.dtype, "float64")
        self.assertIn(('dtype', 'float64'), changed_props)

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
        dose_vms = [node_item for node_item in nodes if isinstance(node_item, DoseGridViewModel)]
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
        self.assertFalse(inspector.dose_grid_section.isHidden())
        self.assertTrue(inspector.volume_section.isHidden())

        # Проверка начальных значений
        self.assertEqual(inspector.dose_grid_section.spin_dose_grid_size_x.value(), 100.0)
        self.assertEqual(inspector.dose_grid_section.spin_dose_grid_voxel.value(), 5.0)
        self.assertEqual(inspector.dose_grid_section.combo_dose_grid_dtype.currentText(), "float32")
        self.assertIn("20 × 20 × 20", inspector.dose_grid_section.lbl_dose_grid_shape.text())
        self.assertIn("float32", inspector.dose_grid_section.lbl_dose_grid_memory.text())

        # Редактирование в инспекторе
        inspector.dose_grid_section.spin_dose_grid_voxel.setValue(2.0)
        self.assertEqual(vm.dose_voxel_size, 2.0)
        self.assertEqual(node.dose_voxel_size, 2.0)
        self.assertIn("50 × 50 × 50", inspector.dose_grid_section.lbl_dose_grid_shape.text())

        inspector.dose_grid_section.combo_dose_grid_dtype.setCurrentText("float64")
        self.assertEqual(vm.dtype, "float64")
        self.assertEqual(node.dtype, "float64")
        self.assertIn("float64", inspector.dose_grid_section.lbl_dose_grid_memory.text())

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
                'type': ChunkType.INTERACTIONS,
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

    def test_07_orchestrator_session_with_dose_grid_nodes(self):
        """Проверка автоматической инициализации параметров сетки дозы из дерева сцены OrchestratorSession."""
        world = CompositeNode(name="World")
        target_dose = DoseGridNode(name="TargetDose", size=[80.0, 80.0, 80.0], dose_voxel_size=4.0)
        world.add_child(target_dose)
        scene_vm = SceneViewModel(world)

        session = OrchestratorSession(scene_vm=scene_vm)
        try:
            self.assertEqual(session.dose_voxel_size, 4.0)
            self.assertEqual(session.dose_origin, (-40.0, -40.0, -40.0))
            self.assertIsNotNone(session.dose_transform_matrix)
        finally:
            session.close()

        handler = DoseMapHandler(grid_nodes=[target_dose], create_shm=False)
        try:
            self.assertEqual(len(handler.entries), 1)
            self.assertEqual(handler.entries[0].grid_shape, (20, 20, 20))
            self.assertEqual(handler.entries[0].voxel_size, (4.0, 4.0, 4.0))
            self.assertEqual(handler.entries[0].origin, (-40.0, -40.0, -40.0))
        finally:
            handler.close()

    def test_08_gamma_camera_child_dose_grid_auto_bounds(self):
        """Проверка автоматической подгонки размеров создаваемой сетки дозы под BoundingBox гамма-камеры."""
        cam_vm = create_default_gamma_camera_vm(name="SpectCamera")

        world = CompositeNode(name="World")
        scene_vm = SceneViewModel(world)
        scene_vm.add_node(scene_vm.root_vm, cam_vm)
        tree_widget = SceneTreeWidget(scene_vm)

        # Добавляем сетку дозы как дочерний элемент гамма-камеры
        tree_widget._add_dose_grid_node(cam_vm)
        dose_children = [child_vm for child_vm in cam_vm.children if isinstance(child_vm, DoseGridViewModel)]
        self.assertEqual(len(dose_children), 1)
        dose_child = dose_children[0]

        # Размеры сетки дозы должны автоматически соответствовать BoundingBox гамма-камеры!
        cam_bounds_size = cam_vm.local_bound
        np.testing.assert_allclose(dose_child.size, cam_bounds_size, atol=1e-5)
        self.assertGreater(dose_child.size[0], 400.0)  # С учетом защитных экранов Pb
        self.assertGreater(dose_child.size[1], 400.0)

    def test_09_property_inspector_fit_to_parent(self):
        """Проверка кнопки подогнать под родителя в PropertyInspector для DoseGridViewModel."""
        cam_vm = create_default_gamma_camera_vm(name="SpectCamera")

        # Дочерний узел с произвольными размерами
        grid_node = DoseGridNode(name="ArbitraryGrid", size=[33.0, 44.0, 55.0], dose_voxel_size=5.0)
        grid_vm = DoseGridViewModel(grid_node, parent_vm=cam_vm)
        cam_vm.children.append(grid_vm)

        inspector = PropertyInspector()
        inspector.set_target_viewmodel(grid_vm)

        self.assertTrue(inspector.dose_grid_section.btn_fit_dose_grid_to_parent.isEnabled())
        self.assertIn("SpectCamera", inspector.dose_grid_section.btn_fit_dose_grid_to_parent.text())

        # Нажимаем кнопку подгонки под родителя
        inspector.dose_grid_section._on_fit_dose_grid_to_parent()

        cam_bounds_size = cam_vm.local_bound
        np.testing.assert_allclose(grid_vm.size, cam_bounds_size, atol=1e-5)
        np.testing.assert_allclose(grid_node.size, cam_bounds_size, atol=1e-5)
        self.assertAlmostEqual(inspector.dose_grid_section.spin_dose_grid_size_x.value(), cam_bounds_size[0], places=3)

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
        grid = DoseGridNode(
            name="DoseTarget",
            size=[150.0, 250.0, 350.0],
            dose_voxel_size=2.5,
            is_active=True,
            dtype="float64",
        )
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
        self.assertEqual(child_cfg.dtype, "float64")

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
        self.assertEqual(reconstructed_grid.dtype, "float64")
        self.assertEqual(reconstructed_grid.grid_shape, (60, 100, 140))

    def test_12_dose_shm_naming_and_multi_worker_additive_accumulation(self):
        """
        Проверка согласования имен SharedMemory между OrchestratorSession и DoseMapHandler,
        а также аддитивного накопления дозы несколькими воркерами в родительский сегмент.
        """
        # 1. Сценарий с одной сеткой дозы
        world_single = CompositeNode(name="World")
        grid_single = DoseGridNode(name="Grid1", size=[20.0, 20.0, 20.0], dose_voxel_size=10.0)
        world_single.add_child(grid_single)
        scene_vm_single = SceneViewModel(world_single)
        session_single = OrchestratorSession(scene_vm=scene_vm_single)

        single_shm, single_shape, single_info = session_single._collect_dose_ipc_config()
        self.assertEqual(single_shm, "nmsim_dose_shm")
        self.assertEqual(single_shape, (2, 2, 2))
        self.assertEqual(len(single_info), 1)
        self.assertEqual(single_info[0]["shm_name"], "nmsim_dose_shm")
        self.assertEqual(single_info[0]["dtype"], "float32")

        handler_single = DoseMapHandler(grid_nodes=[grid_single], shm_name="nmsim_dose_shm", create_shm=False)
        try:
            self.assertEqual(len(handler_single.entries), 1)
            self.assertEqual(handler_single.entries[0].shm_name, single_info[0]["shm_name"])
            self.assertEqual(handler_single.entries[0].dtype, np.dtype(np.float32))
        finally:
            handler_single.close()
            session_single.close()

        # 2. Сценарий с двумя сетками дозы
        world_multi = CompositeNode(name="World")
        grid_first = DoseGridNode(name="Grid1", size=[20.0, 20.0, 20.0], dose_voxel_size=10.0)
        grid_second = DoseGridNode(name="Grid2", size=[40.0, 40.0, 40.0], dose_voxel_size=10.0, dtype="float64")
        world_multi.add_child(grid_first)
        world_multi.add_child(grid_second)
        scene_vm_multi = SceneViewModel(world_multi)
        session_multi = OrchestratorSession(scene_vm=scene_vm_multi)

        multi_shm, _, multi_info = session_multi._collect_dose_ipc_config()
        self.assertIsNone(multi_shm)
        self.assertEqual(len(multi_info), 2)
        self.assertEqual(multi_info[0]["shm_name"], "nmsim_dose_shm_Grid1")
        self.assertEqual(multi_info[0]["dtype"], "float32")
        self.assertEqual(multi_info[1]["shm_name"], "nmsim_dose_shm_Grid2")
        self.assertEqual(multi_info[1]["dtype"], "float64")

        handler_multi = DoseMapHandler(
            grid_nodes=[grid_first, grid_second], shm_name="nmsim_dose_shm", create_shm=False
        )
        try:
            self.assertEqual(len(handler_multi.entries), 2)
            self.assertEqual(handler_multi.entries[0].shm_name, multi_info[0]["shm_name"])
            self.assertEqual(handler_multi.entries[0].dtype, np.dtype(np.float32))
            self.assertEqual(handler_multi.entries[1].shm_name, multi_info[1]["shm_name"])
            self.assertEqual(handler_multi.entries[1].dtype, np.dtype(np.float64))
        finally:
            handler_multi.close()
            session_multi.close()

        # 3. Проверка аддитивного накопления в предварительно созданный родительский сегмент SharedMemory
        parent_shm_name = "nmsim_test_parent_dose_shm"
        grid_shape = grid_single.grid_shape
        grid_dtype = grid_single.numpy_dtype
        nbytes = int(np.prod(grid_shape) * grid_dtype.itemsize)
        try:
            stale_shm = shared_memory.SharedMemory(name=parent_shm_name, create=False)
            stale_shm.close()
            stale_shm.unlink()
        except FileNotFoundError:
            pass

        parent_shm = shared_memory.SharedMemory(name=parent_shm_name, create=True, size=nbytes)
        try:
            parent_buf = np.ndarray(grid_shape, dtype=grid_dtype, buffer=parent_shm.buf, order="F")
            parent_buf.fill(0.0)

            chunk_first = {
                'type': ChunkType.INTERACTIONS,
                'data': {
                    'pos_x': np.array([5.0]),
                    'pos_y': np.array([5.0]),
                    'pos_z': np.array([5.0]),
                    'energy_deposit': np.array([60.0]),
                },
            }
            chunk_second = {
                'type': ChunkType.INTERACTIONS,
                'data': {
                    'pos_x': np.array([5.0]),
                    'pos_y': np.array([5.0]),
                    'pos_z': np.array([5.0]),
                    'energy_deposit': np.array([90.0]),
                },
            }

            # Воркер 1 (ракурс 1)
            worker_handler_1 = DoseMapHandler(grid_nodes=[grid_single], shm_name=parent_shm_name, create_shm=True)
            self.assertFalse(worker_handler_1.entries[0]._owns_shm)
            worker_handler_1.process_chunk(chunk_first)
            worker_handler_1.close()

            # Проверяем, что после закрытия первого воркера сегмент жив и содержит 60.0
            self.assertAlmostEqual(float(np.sum(parent_buf)), 60.0)

            # Воркер 2 (ракурс 2) подключается к тому же сегменту и НЕ затирает его нулями
            worker_handler_2 = DoseMapHandler(grid_nodes=[grid_single], shm_name=parent_shm_name, create_shm=True)
            self.assertFalse(worker_handler_2.entries[0]._owns_shm)
            worker_handler_2.process_chunk(chunk_second)
            worker_handler_2.close()

            # Суммарная доза должна быть 60.0 + 90.0 = 150.0, и сегмент по-прежнему доступен по имени
            check_shm = shared_memory.SharedMemory(name=parent_shm_name, create=False)
            try:
                check_buf = np.ndarray(grid_shape, dtype=grid_dtype, buffer=check_shm.buf, order="F")
                self.assertAlmostEqual(float(np.sum(check_buf)), 150.0)
                self.assertAlmostEqual(float(check_buf[1, 1, 1]), 150.0)
                del check_buf
            finally:
                check_shm.close()
            del parent_buf
        finally:
            parent_shm.close()
            try:
                parent_shm.unlink()
            except FileNotFoundError:
                pass


if __name__ == '__main__':
    unittest.main()

